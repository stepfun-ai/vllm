# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 sparse selection and attention using FlashInfer and PyTorch."""

from typing import Any

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config_or_none

from .ops.prefill_indexer_metadata import prepare_prefill_indexer
from .ops.sparse_decode import prepare_sparse_decode, zero_inactive_decode
from .ops.sparse_indexer import (
    PagedDecodeMetadata,
    pack_decode_summary_pages,
    pack_summary_pages,
    select_all_regions,
    select_regions,
)
from .ops.sparse_indexer_scores import decode_summary_scores
from .ops.sparse_prefill import PrefillChunk, gather_prefill_kv
from .ops.sparse_prefill_pages import CompactPagedPrefill


class Step5FlashInferSparse:
    def __init__(self, device: torch.device):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        from flashinfer.attn_scores import fp8_paged_mqa_logits

        config = get_current_vllm_config_or_none()
        self._topk_backend = (
            config.kernel_config.sparse_indexer_topk_backend if config else "auto"
        )
        self._logits = fp8_paged_mqa_logits
        properties = torch.cuda.get_device_properties(device)
        self._num_sms = properties.multi_processor_count
        self._direct_summary_scores = (properties.major, properties.minor) == (10, 3)
        self._workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
        self._attention = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace, "NHD", backend="fa2"
        )
        self._prefill_plan_event = torch.cuda.Event()
        self._paged_prefill_plan: CompactPagedPrefill | None = None
        self._decode_plans: dict[
            tuple, tuple[Any, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        ] = {}

    def prepare_prefill_attention(
        self, query, kv_cache, positions, region_count, scale
    ) -> CompactPagedPrefill | None:
        if not (
            query.shape[1] in (8, 16)
            and query.shape[2] == 192
            and query.dtype == kv_cache.dtype == torch.bfloat16
            and kv_cache.stride(0) == kv_cache.shape[2] * kv_cache.stride(2)
            and kv_cache.stride(-1) == 1
            and kv_cache.shape[1] == 1
            and kv_cache.shape[2] in (16, 64, 128)
        ):
            return None
        if self._paged_prefill_plan is None:
            self._paged_prefill_plan = CompactPagedPrefill(self._workspace)
        self._paged_prefill_plan.plan(
            query, positions, region_count, scale, kv_cache.shape[2]
        )
        return self._paged_prefill_plan

    @staticmethod
    def _use_paged_decode(query, kv_cache):
        batch, heads, dim = query.shape
        return (
            batch >= 32
            and dim == 192
            and heads == 8
            and query.dtype == torch.bfloat16
            and kv_cache.dtype == torch.bfloat16
            and kv_cache.is_contiguous()
        )

    def _get_decode_plan(self, query, kv_cache, region_count, scale, *, compact=False):
        from flashinfer import (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
        )

        batch, heads, dim = query.shape
        capacity = region_count * 8
        # Eight-token virtual pages avoid large KV gathers. Small batches are
        # faster with the ragged kernel, including short-context decode.
        use_paged = self._use_paged_decode(query, kv_cache)
        assert not compact or use_paged
        plan_key = (
            batch,
            heads,
            dim,
            capacity,
            query.dtype,
            kv_cache.dtype,
            scale,
            use_paged,
            compact,
        )
        if plan_key not in self._decode_plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Step5 decode attention must be warmed up before capture"
                )
            cu_q = torch.arange(batch + 1, dtype=torch.int32)
            cu_pages = cu_q * region_count
            mask = torch.zeros(
                batch * region_count, dtype=torch.uint8, device=query.device
            )
            kv_indptr = None
            # Q has one row per request, so use a fixed plan with only valid CTAs.
            # vLLM still captures wrapper.run() in its outer CUDA Graph.
            wrapper_args = dict(kv_layout="NHD")
            plan_args = dict(
                head_dim_vo=dim,
                custom_mask=torch.zeros(
                    batch * capacity, device=query.device, dtype=torch.bool
                ),
                causal=False,
                sm_scale=scale,
                q_data_type=query.dtype,
                kv_data_type=kv_cache.dtype,
                disable_split_kv=True,
            )
            indices = None
            if use_paged:
                indices = torch.zeros(
                    batch * region_count, dtype=torch.int32, device=query.device
                )
                last_page_len = torch.full((batch,), 8, dtype=torch.int32)
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._workspace,
                    backend="fa2",
                    **wrapper_args,
                )
                wrapper.plan(
                    cu_q,
                    cu_pages,
                    indices,
                    last_page_len,
                    heads,
                    1,
                    dim,
                    8,
                    **plan_args,
                )
            else:
                cu_k = cu_q * capacity
                wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    self._workspace,
                    backend="fa2",
                    **wrapper_args,
                )
                wrapper.plan(cu_q, cu_k, heads, 1, dim, **plan_args)
            # Keep the buffers created by the one-time plan alive across replays.
            mask = wrapper._custom_mask_buf
            if not use_paged:
                kv_indptr = wrapper._kv_indptr_buf
            elif compact:
                indices = wrapper._paged_kv_indices_buf
                kv_indptr = wrapper._paged_kv_indptr_buf
            self._decode_plans[plan_key] = (
                wrapper,
                mask,
                indices,
                kv_indptr,
            )
        return self._decode_plans[plan_key]

    def run_decode_attention(
        self,
        query,
        kv_cache,
        regions,
        block_table,
        positions,
        active,
        out,
        scale,
        *,
        clear_inactive=True,
    ):
        wrapper, mask, indices, kv_indptr = self._get_decode_plan(
            query, kv_cache, regions.shape[1], scale
        )
        k, v = prepare_sparse_decode(
            kv_cache,
            regions,
            block_table,
            positions,
            active,
            mask,
            indices,
            kv_indptr,
        )
        if indices is not None:
            wrapper.run(query.contiguous(), (k, v), out=out)
        else:
            wrapper.run(query.contiguous(), k, v, out=out)
        if clear_inactive:
            zero_inactive_decode(out, active)

    def select_and_run_decode_attention(
        self,
        query,
        kv_cache,
        scores,
        current,
        topk,
        block_table,
        positions,
        active,
        out,
        scale,
        *,
        topk_lengths=None,
        clear_inactive=True,
    ):
        if not (
            self._use_paged_decode(query, kv_cache)
            and topk == 512
            and scores.shape[1] > topk
        ):
            regions = self.select_regions(
                scores, current, topk, topk_lengths=topk_lengths
            )
            return self.run_decode_attention(
                query,
                kv_cache,
                regions,
                block_table,
                positions,
                active,
                out,
                scale,
                clear_inactive=clear_inactive,
            )
        wrapper, mask, indices, indptr = self._get_decode_plan(
            query, kv_cache, topk + 1, scale, compact=True
        )
        assert indices is not None and indptr is not None
        select_regions(
            scores,
            current,
            topk,
            self._topk_backend,
            topk_lengths=topk_lengths,
            paged_decode=PagedDecodeMetadata(
                positions,
                active,
                block_table,
                mask,
                indices,
                indptr,
                kv_cache.shape[2],
            ),
        )
        dim = query.shape[-1]
        key, value = kv_cache.view(-1, 8, 1, 2 * dim).split(dim, -1)
        wrapper.run(query.contiguous(), (key, value), out=out)
        if clear_inactive:
            zero_inactive_decode(out, active)

    def prefill_metadata(self, owners, positions, regions, topk, *, paged_prefill=None):
        return prepare_prefill_indexer(
            owners,
            positions,
            regions,
            topk,
            self._num_sms,
            paged_prefill=paged_prefill,
        )

    def pack_summaries(self, mean, block_table, regions):
        return pack_summary_pages(mean, block_table, regions)

    def pack_decode_summaries(self, mean, block_table, regions, slots, seq_lens):
        return pack_decode_summary_pages(
            mean,
            block_table,
            regions,
            slots,
            seq_lens,
            num_sms=self._num_sms,
            live_only=block_table.shape[0] >= 8,
        )

    def decode_logits(self, query, weights, mean, table, regions, slots, seq_lens):
        if (
            self._direct_summary_scores
            and 0 < query.shape[0] <= 128
            and query.shape[1:] == (1, 8, 256)
            and query.dtype == torch.float8_e4m3fn
            and query.is_contiguous()
            and weights.shape == query.shape[:3]
            and weights.dtype == torch.float32
            and weights.stride(-1) == 1
            and (
                mean.shape[1:] == (8, 256)
                or (mean.shape[1:] == (2, 256) and regions > 4096)
            )
            and mean.dtype == torch.uint8
            and mean.is_contiguous()
            and table.stride(1) == 1
            and regions > 0
        ):
            return decode_summary_scores(
                query,
                weights[:, 0],
                mean,
                table,
                slots,
                seq_lens,
                regions,
                num_sms=self._num_sms,
            )
        packed, decode = self.pack_decode_summaries(
            mean, table, regions, slots, seq_lens
        )
        scores = self.logits(
            query,
            weights,
            packed,
            decode.block_table.shape[1],
            None,
            decode.lengths,
            regions,
            block_table=decode.block_table,
            schedule_meta=decode.schedule,
        )
        return scores, decode

    def select_all_regions(self, current, capacity):
        return select_all_regions(current, capacity)

    def select_regions(
        self, scores, current, topk, *, topk_lengths=None, paged_prefill=None
    ):
        return select_regions(
            scores,
            current,
            topk,
            self._topk_backend,
            topk_lengths=topk_lengths,
            paged_prefill=paged_prefill,
        )

    def logits(
        self,
        query,
        weights,
        packed,
        pages,
        requests,
        lengths,
        regions,
        *,
        block_table=None,
        schedule_meta=None,
    ):
        q = query[:, 0]
        w = weights[:, 0]
        if q.dtype != torch.float8_e4m3fn:
            q = q.float()
            w = w.float()
            heads = q.shape[1]
            if heads < 8:
                q = F.pad(q, (0, 0, 0, 8 - heads))
                w = F.pad(w, (0, 8 - heads))
            q = q.to(torch.float8_e4m3fn)
        q = q.unsqueeze(1)
        if block_table is None:
            block_table = (
                requests[:, None] * pages + torch.arange(pages, device=q.device)
            ).int()
        return self._logits(
            q=q,
            kv_fused=packed,
            weights=w.contiguous(),
            context_lens=lengths.int().contiguous(),
            block_table=block_table,
            max_context_len=regions,
            schedule_meta=schedule_meta,
        )

    def run_attention(
        self,
        query,
        kv_cache,
        regions,
        block_table,
        requests,
        positions,
        out,
        scale,
        *,
        prefill_chunks: list[PrefillChunk],
    ):
        for index, chunk in enumerate(prefill_chunks):
            start = index * 32
            end = start + chunk.cu_q.numel() - 1
            k, v = gather_prefill_kv(
                kv_cache,
                regions[start:end],
                block_table,
                requests[start:end],
                positions[start:end],
                chunk.cu_k_device,
                chunk.total_tokens,
                chunk.max_length,
            )
            # FlashInfer reuses pinned host plan data across asynchronous copies.
            self._prefill_plan_event.synchronize()
            self._attention.plan(
                chunk.cu_q,
                chunk.cu_k,
                query.shape[1],
                1,
                query.shape[2],
                head_dim_vo=query.shape[2],
                causal=False,
                sm_scale=scale,
                q_data_type=query.dtype,
                kv_data_type=kv_cache.dtype,
                disable_split_kv=True,
            )
            self._prefill_plan_event.record(torch.cuda.current_stream(query.device))
            self._attention.run(query[start:end].contiguous(), k, v, out=out[start:end])
