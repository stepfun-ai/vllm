# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch CSA summaries attached to the attention KV pages."""

from typing import Any

import torch

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend


class Step5SparseSummaryCache:
    """FP8 history and ordered FP32 accumulators for each page's open region."""

    def __init__(
        self, pages: int, block_size: int, device: torch.device, max_num_seqs: int = 1
    ):
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        self.block_size = block_size
        self.pages = pages
        # Padded decode requests write distinct scratch pages, never live KV pages.
        self._mean = torch.zeros(
            (pages + max_num_seqs, block_size // 8, 256),
            dtype=torch.uint8,
            device=device,
        )
        self.mean = self._mean[:pages]
        self.numerator = torch.zeros(
            (pages + max_num_seqs, 256), device=device, dtype=torch.float32
        )
        self.denominator = torch.zeros_like(self.numerator)
        self.maximum = torch.full_like(self.numerator[:, :1], float("-inf"))
        self.last_slot = torch.full(
            (pages + max_num_seqs,), -1, dtype=torch.long, device=device
        )

    @staticmethod
    def _accumulate(numerator, denominator, maximum, key, z):
        updated_max = torch.maximum(maximum, z.amax(-1, keepdim=True))
        scale = torch.where(denominator > 0, (maximum - updated_max).exp(), 0)
        weight = (z - updated_max).exp()
        return (
            numerator * scale + weight * key,
            denominator * scale + weight,
            updated_max,
        )

    def update_decode(self, key, z, slots):
        active = slots >= 0
        pages = torch.where(
            active,
            slots // self.block_size,
            self.pages + torch.arange(slots.numel(), device=slots.device),
        )
        fresh = (slots % 8 == 0) | ~active
        torch._assert_async(
            (fresh | (self.last_slot[pages] == slots - 1)).all(),
            "Step5 CSA tail missing for a partial region",
        )
        numerator = torch.where(fresh[:, None], 0, self.numerator[pages])
        denominator = torch.where(fresh[:, None], 0, self.denominator[pages])
        maximum = torch.where(fresh[:, None], float("-inf"), self.maximum[pages])
        numerator, denominator, maximum = self._accumulate(
            numerator, denominator, maximum, key[:, 0].float(), z[:, 0].float()
        )
        regions = slots.clamp_min(0) % self.block_size // 8
        self._mean[pages, regions] = (
            (numerator / denominator.clamp_min(1e-20))
            .bfloat16()
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        self.numerator[pages] = numerator
        self.denominator[pages] = denominator
        self.maximum[pages] = maximum
        self.last_slot[pages] = torch.where(active & (slots % 8 != 7), slots, -1)

    def update(self, key: torch.Tensor, z: torch.Tensor, slots: list[int]) -> None:
        groups: dict[int, list[int]] = {}
        for row, slot in enumerate(slots):
            if slot >= 0:
                groups.setdefault(slot // 8, [-1] * 8)[slot % 8] = row
        if not groups:
            return
        regions = list(groups)
        rows = torch.tensor(list(groups.values()), device=key.device)
        valid = rows >= 0
        keys = key[:, 0][rows.clamp_min(0)].float()
        logits = z[:, 0][rows.clamp_min(0)].float()
        region_ids = torch.tensor(regions, device=key.device)
        pages = region_ids * 8 // self.block_size
        fresh = valid[:, 0, None]
        first = torch.tensor(
            [next(j for j, row in enumerate(m) if row >= 0) for m in groups.values()],
            device=key.device,
        )
        if not bool(
            (fresh[:, 0] | (self.last_slot[pages] == region_ids * 8 + first - 1)).all()
        ):
            raise RuntimeError("Step5 CSA tail missing for a partial region")
        numerator = torch.where(fresh, 0, self.numerator[pages])
        denominator = torch.where(fresh, 0, self.denominator[pages])
        maximum = torch.where(fresh, float("-inf"), self.maximum[pages])
        for offset in range(8):
            updated = self._accumulate(
                numerator, denominator, maximum, keys[:, offset], logits[:, offset]
            )
            numerator, denominator, maximum = (
                torch.where(valid[:, offset, None], new, old)
                for new, old in zip(updated, (numerator, denominator, maximum))
            )
        mean = numerator / denominator.clamp_min(1e-20)
        self.mean.view(-1, 256)[region_ids] = (
            mean.bfloat16().to(torch.float8_e4m3fn).view(torch.uint8)
        )
        # Only the final region of each physical page can remain open.
        last = {region * 8 // self.block_size: i for i, region in enumerate(regions)}
        dest = list(last)
        src = list(last.values())
        self.numerator[dest] = numerator[src]
        self.denominator[dest] = denominator[src]
        self.maximum[dest] = maximum[src]
        final_slots = [
            regions[i] * 8
            + max(j for j, row in enumerate(groups[regions[i]]) if row >= 0)
            for i in src
        ]
        self.last_slot[dest] = torch.tensor(
            [s if s % 8 != 7 else -1 for s in final_slots], device=key.device
        )

    def reset(self, block_ids: list[int], clear_mean: bool = True) -> None:
        blocks = set(block_ids)
        if clear_mean and blocks:
            self.mean[list(blocks)] = 0
        self.last_slot[list(blocks)] = -1

    def copy(self, copies: list[Any]) -> None:
        copies = [c for c in copies if c.src_block_id != c.dst_block_id]
        if not copies:
            return
        sources = [c.src_block_id for c in copies]
        destinations = [c.dst_block_id for c in copies]
        snapshots = [
            t[sources].clone()
            for t in (
                self.mean,
                self.numerator,
                self.denominator,
                self.maximum,
                self.last_slot,
            )
        ]
        for tensor, snapshot in zip(
            (self.mean, self.numerator, self.denominator, self.maximum), snapshots
        ):
            tensor[destinations] = snapshot
        old_slots = snapshots[-1]
        offsets = torch.tensor(destinations, device=self.mean.device) * self.block_size
        self.last_slot[destinations] = torch.where(
            old_slots >= 0, offsets + old_slots % self.block_size, -1
        )


class Step5SparseSummaryCacheLayer(torch.nn.Module, AttentionLayerBase):
    """Integrate summary allocation, reset and COW with the vLLM KV lifecycle."""

    def __init__(
        self,
        *,
        prefix,
        target_impl,
        main_layer_name,
        static_forward_context,
    ):
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self.layer_name = prefix
        self._target_impl = target_impl
        self.main_layer_name = main_layer_name
        self.kv_cache = torch.tensor([])
        self._cache: Step5SparseSummaryCache | None = None

    def get_attn_backend(self) -> type[AttentionBackend]:
        from .sparse_attention import Step5DSAAttentionBackend

        return Step5DSAAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        return None

    def _set_summary_cache_budget(self, block_size: int) -> int:
        if block_size % 8:
            raise ValueError("Step5 KV blocks must align to eight-token regions")
        return block_size // 8 * 256 + 513 * 4 + 8

    def _fixed_runtime_state_budget(self, block_size: int) -> int:
        return self._target_impl.max_num_seqs * self._set_summary_cache_budget(
            block_size
        )

    def _peers(self, context):
        return [v for v in context.values() if isinstance(v, type(self))]

    def kv_cache_side_storage_memory_profile_key(self, forward_context):
        return self._peers(forward_context)[0]

    def prepare_kv_cache_side_storage_memory_profile(self, forward_context, device):
        from .flashinfer_sparse import Step5FlashInferSparse

        peers = self._peers(forward_context)
        adapter = peers[0]._target_impl.adapter
        if adapter is None:
            adapter = Step5FlashInferSparse(device)
        for peer in peers:
            peer._target_impl.adapter = adapter

    def bind_kv_cache_side_storage(self, forward_context):
        self.prepare_kv_cache_side_storage_memory_profile(
            forward_context, forward_context[self.main_layer_name].kv_cache.device
        )
        kv = forward_context[self.main_layer_name].kv_cache
        self._cache = Step5SparseSummaryCache(
            kv.shape[0], kv.shape[2], kv.device, self._target_impl.max_num_seqs
        )
        self.kv_cache = self._cache.mean.view(kv.shape[0], -1)
        self._target_impl.summary_cache = self._cache

    @property
    def side_kv_cache(self):
        return self.kv_cache.unsqueeze(0)

    def zero_kv_cache_side_storage(self, block_ids):
        if self._cache is not None:
            self._cache.reset(self._ids(block_ids))

    def release_kv_cache_side_storage(self, block_ids):
        if self._cache is not None:
            self._cache.reset(self._ids(block_ids), clear_mean=False)

    def reset_kv_cache_side_storage_runtime_state(self):
        if self._cache is not None:
            self._cache.last_slot.fill_(-1)
            self._cache.mean.zero_()
        return 1, 0

    def copy_kv_cache_side_storage(self, block_copies, num_blocks):
        if self._cache is not None and block_copies is not None:
            self._cache.copy(list(block_copies))

    @staticmethod
    def _ids(block_ids):
        return block_ids.tolist() if isinstance(block_ids, torch.Tensor) else block_ids
