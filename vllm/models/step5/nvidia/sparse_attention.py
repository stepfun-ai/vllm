# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B300 sparse attention using FlashInfer and PyTorch."""

from typing import Any, ClassVar, cast

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


class Step5SplitKVFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention using the canonical packed-content KV cache."""

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)


def _validate_step5_dsa_attention_contract(
    *,
    dtype: torch.dtype,
    head_size: int,
    q_heads_per_kv: int,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    attn_type: str,
    kv_sharing_target_layer_name: str | None,
    sinks: torch.Tensor | None,
) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Step5 DSA kernels require float16 or bfloat16 activations, got {dtype}."
        )
    if int(head_size) not in (128, 192):
        raise ValueError(
            "Step5 DSA kernels require head_size in {128, 192}, got "
            f"{int(head_size)}."
        )
    if int(q_heads_per_kv) not in (4, 8, 16):
        raise ValueError(
            "Step5 DSA kernels require q_heads_per_kv in {4, 8, 16}, got "
            f"{int(q_heads_per_kv)}."
        )
    if alibi_slopes is not None:
        raise NotImplementedError("Step5 DSA does not support ALiBi.")
    if tuple(sliding_window) != (-1, -1):
        raise NotImplementedError(
            "Step5 DSA does not support sliding-window attention."
        )
    if float(logits_soft_cap) != 0.0:
        raise NotImplementedError("Step5 DSA does not support logits soft cap.")
    if attn_type != AttentionType.DECODER:
        raise NotImplementedError(
            "Step5 DSA supports causal decoder self-attention only, got "
            f"attn_type={attn_type!r}."
        )
    if kv_sharing_target_layer_name is not None:
        raise NotImplementedError("Step5 DSA does not support KV cache sharing.")
    if sinks is not None:
        raise NotImplementedError("Step5 DSA does not support attention sinks.")


class Step5DSAMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        step5_metadata = cast(Any, metadata)
        step5_metadata.num_actual_reqs = common_attn_metadata.num_reqs
        step5_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        step5_metadata.step5_runtime = None
        return metadata

    def use_cascade_attention(self, *args, **kwargs):
        return False


class Step5DSAAttentionImpl(Step5SplitKVFlashAttentionImpl):
    can_return_lse_for_decode = False
    supports_dcp = False

    def __init__(
        self,
        *args,
        sparse_config=None,
        max_model_len=None,
        max_num_seqs=None,
        num_speculative_tokens=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config_or_none()
        if not current_platform.is_device_capability(103):
            raise NotImplementedError("Step5 DSA currently supports B300 (SM103) only")
        if config is not None:
            # FlashInfer builds host-side ragged plans before CUDA graph capture.
            config.compilation_config.cudagraph_num_of_warmups = max(
                1, config.compilation_config.cudagraph_num_of_warmups
            )
            if config.speculative_config is not None:
                raise NotImplementedError(
                    "Step5 does not currently support speculative decoding"
                )
            if config.parallel_config.decode_context_parallel_size > 1:
                raise NotImplementedError(
                    "Step5 does not support decode context parallelism"
                )
            if config.parallel_config.use_ubatching:
                raise NotImplementedError("Step5 does not support DBO/ubatching")
        if num_speculative_tokens:
            raise NotImplementedError("Step5 does not currently support MTP")
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("Step5 requires BF16 or FP16 KV cache")
        if self.num_kv_heads != 1:
            raise ValueError("Step5 requires one local KV head; use TP4/8/16")
        _validate_step5_dsa_attention_contract(
            dtype=torch.get_default_dtype(),
            head_size=self.head_size,
            q_heads_per_kv=self.num_queries_per_kv,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            attn_type=self.attn_type,
            kv_sharing_target_layer_name=self.kv_sharing_target_layer_name,
            sinks=self.sinks,
        )
        self.sparse_topk = sparse_config.topk
        self.max_num_seqs = max_num_seqs or 1
        self.max_num_regions = ((max_model_len or 1048576) + 7) // 8
        self.adapter = None
        self.summary_cache = None

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        dsa_proxy_query=None,
        dsa_proxy_key=None,
        dsa_proxy_weights=None,
        dsa_proxy_z=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Step5 does not support output quantization")
        if output is None:
            output = torch.empty_like(query)
        output.zero_()
        if attn_metadata is None:
            return output
        assert self.summary_cache is not None and self.adapter is not None
        n = attn_metadata.num_actual_tokens
        if not n:
            return output
        if attn_metadata.max_query_len == 1 and n == attn_metadata.num_actual_reqs:
            return self._forward_decode(
                query[:n],
                kv_cache,
                attn_metadata,
                output[:n],
                dsa_proxy_query[:n],
                dsa_proxy_key[:n],
                dsa_proxy_weights[:n],
                dsa_proxy_z[:n],
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 prefill requires a breakable CUDA graph")
        if attn_metadata.step5_runtime is None:
            count = attn_metadata.num_actual_reqs
            starts = attn_metadata.query_start_loc_cpu[: count + 1].tolist()
            lengths = attn_metadata.seq_lens[:count].tolist()
            slots = attn_metadata.slot_mapping[:n].tolist()
            owners_cpu: list[int] = []
            positions_cpu: list[int] = []
            for req in range(count):
                qlen = starts[req + 1] - starts[req]
                owners_cpu.extend([req] * qlen)
                positions_cpu.extend(range(lengths[req] - qlen, lengths[req]))
            attn_metadata.step5_runtime = (
                slots,
                torch.tensor(owners_cpu, device=query.device, dtype=torch.long),
                torch.tensor(positions_cpu, device=query.device, dtype=torch.long),
                (max(lengths) + 7) // 8,
            )
        slots, owners, positions, num_regions = attn_metadata.step5_runtime
        self.summary_cache.update(dsa_proxy_key[:n], dsa_proxy_z[:n], slots)
        table = attn_metadata.block_table[: attn_metadata.num_actual_reqs]
        packed, pages = self.adapter.pack_summaries(
            self.summary_cache.mean, table, num_regions
        )
        columns = torch.arange(num_regions, device=query.device)
        # Bound logits and gathered KV scratch independently of prefill length.
        for start in range(0, n, 256):
            end = min(start + 256, n)
            current = positions[start:end] // 8
            scores = self.adapter.logits(
                dsa_proxy_query[start:end],
                dsa_proxy_weights[start:end],
                packed,
                pages,
                owners[start:end],
                current + 1,
                num_regions,
            )
            scores.masked_fill_(columns[None, :] >= current[:, None], float("-inf"))
            k = min(self.sparse_topk, num_regions)
            # Stable ties choose lower logical region IDs, as in the validated path.
            selected = scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
            selected = torch.where(selected < current[:, None], selected, -1)
            selected = torch.cat((selected, current[:, None]), dim=-1)
            selected = selected.sort(dim=-1).values
            self.adapter.run_attention(
                query[start:end],
                kv_cache,
                selected,
                table,
                owners[start:end],
                positions[start:end],
                output[start:end],
                self.scale,
            )
        return output

    def _forward_decode(
        self,
        query,
        kv_cache,
        metadata,
        output,
        proxy_query,
        proxy_key,
        proxy_weights,
        proxy_z,
    ):
        assert self.summary_cache is not None and self.adapter is not None
        n = query.shape[0]
        slots = metadata.slot_mapping[:n]
        active = slots >= 0
        positions = torch.where(active, metadata.seq_lens[:n].long() - 1, 0)
        table = metadata.block_table[:n]
        self.summary_cache.update_decode(proxy_key, proxy_z, slots)
        num_regions = min(self.max_num_regions, table.shape[1] * kv_cache.shape[2] // 8)
        packed, pages = self.adapter.pack_summaries(
            self.summary_cache.mean, table, num_regions
        )
        current = positions // 8
        owners = torch.arange(n, device=query.device)
        scores = self.adapter.logits(
            proxy_query, proxy_weights, packed, pages, owners, current + 1, num_regions
        )
        columns = torch.arange(num_regions, device=query.device)
        scores.masked_fill_(columns[None, :] >= current[:, None], float("-inf"))
        selected = scores.argsort(dim=-1, descending=True, stable=True)[
            :, : min(self.sparse_topk, num_regions)
        ]
        selected = torch.where(selected < current[:, None], selected, -1)
        selected = torch.cat((selected, current[:, None]), dim=-1).sort(dim=-1).values
        self.adapter.run_decode_attention(
            query, kv_cache, selected, table, positions, active, output, self.scale
        )
        return output


class Step5SplitKVFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention with token-major KV pages shared by Step5 DSA."""

    @staticmethod
    def get_name() -> str:
        return "STEP5_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[FlashAttentionImpl]:
        return Step5SplitKVFlashAttentionImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)


class Step5DSAAttentionBackend(Step5SplitKVFlashAttentionBackend):
    """Backend entry for Step5 GQA + DSA attention."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return int(head_size) in (128, 192)

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # The Step5 summary sidecar is not part of the generic connector
        # transfer contract.
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(10, 3)

    @staticmethod
    def get_attention_warmup_num_tokens() -> int:
        return 16

    @staticmethod
    def get_attention_warmup_decode_query_len() -> int:
        return 1

    @staticmethod
    def get_name() -> str:
        return "STEP5_DSA"

    @staticmethod
    def get_impl_cls() -> type[Step5DSAAttentionImpl]:
        return Step5DSAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Step5DSAMetadataBuilder
