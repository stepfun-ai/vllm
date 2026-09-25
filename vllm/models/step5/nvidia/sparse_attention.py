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

from .ops.sparse_prefill import make_prefill_chunks


class Step5SplitKVFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention using the canonical packed-content KV cache."""

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def supports_headwise_gate(self) -> bool:
        return (
            self.vllm_flash_attn_version == 2
            and self.head_size == 192
            and any(
                arg.name == "headwise_gate"
                for arg in torch.ops._vllm_fa2_C.varlen_fwd.default._schema.arguments
            )
        )

    def forward_with_headwise_gate(
        self, layer, query, key, value, kv_cache, attn_metadata, output, gate
    ):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from .ops.attention_gate import headwise_sigmoid_gate

        if (
            attn_metadata is None
            or not attn_metadata.num_actual_tokens
            or attn_metadata.use_cascade
            or self.dcp_world_size != 1
            or self.attn_type != AttentionType.DECODER
            or attn_metadata.causal is not True
            or query.dtype != torch.bfloat16
            or gate.dtype != torch.bfloat16
            or attn_metadata.max_num_splits > 1
            or attn_metadata.max_seq_len <= 0
            or is_quantized_kv_cache(self.kv_cache_dtype)
            or self.alibi_slopes is not None
            or self.logits_soft_cap != 0
            or self.sliding_window is None
            or self.sliding_window[0] < 0
        ):
            self.forward(layer, query, key, value, kv_cache, attn_metadata, output)
            gated = headwise_sigmoid_gate(output.flatten(1), gate)
            return output.copy_(gated.view_as(output))

        n = attn_metadata.num_actual_tokens
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        flash_attn_varlen_func(
            q=query[:n],
            k=key_cache,
            v=value_cache,
            out=output[:n],
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=list(self.sliding_window),
            block_table=attn_metadata.block_table,
            num_splits=attn_metadata.max_num_splits,
            headwise_gate=gate[:n],
            fa_version=2,
        )
        return output


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
        dsa_summary_updated=False,
        defer_inactive_clear=False,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Step5 does not support output quantization")
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            return output.zero_()
        assert self.summary_cache is not None and self.adapter is not None
        n = attn_metadata.num_actual_tokens
        if n < output.shape[0]:
            output[n:].zero_()
        if not n:
            return output
        is_decode = (
            attn_metadata.max_query_len == 1 and n == attn_metadata.num_actual_reqs
        )
        if (dsa_summary_updated or defer_inactive_clear) and not is_decode:
            raise ValueError("Decode attention fusions cannot be used for prefill")
        groups = kv_cache.shape[1]
        if groups > 1:
            heads = query.shape[1] // groups
            # FlashInfer's output rows are contiguous even when KV is strided.
            grouped_output = torch.empty(
                (groups, n, heads, query.shape[2]),
                device=output.device,
                dtype=output.dtype,
            )
            for group in range(groups):
                group_query = query[:n, group * heads : (group + 1) * heads]
                group_output = grouped_output[group]
                args = (
                    group_query,
                    kv_cache[:, group : group + 1],
                    attn_metadata,
                    group_output,
                    dsa_proxy_query[:n, group : group + 1],
                    dsa_proxy_key[:n],
                    dsa_proxy_weights[:n, group : group + 1],
                    dsa_proxy_z[:n],
                )
                if is_decode:
                    self._forward_decode(
                        *args,
                        summary_updated=dsa_summary_updated or group > 0,
                        defer_inactive_clear=defer_inactive_clear,
                    )
                else:
                    self._forward_prefill(*args, summary_updated=group > 0)
            output[:n].view(n, groups, heads, query.shape[2]).copy_(
                grouped_output.transpose(0, 1)
            )
            return output
        if is_decode:
            return self._forward_decode(
                query[:n],
                kv_cache,
                attn_metadata,
                output[:n],
                dsa_proxy_query[:n],
                dsa_proxy_key[:n],
                dsa_proxy_weights[:n],
                dsa_proxy_z[:n],
                summary_updated=dsa_summary_updated,
                defer_inactive_clear=defer_inactive_clear,
            )
        return self._forward_prefill(
            query[:n],
            kv_cache,
            attn_metadata,
            output,
            dsa_proxy_query[:n],
            dsa_proxy_key[:n],
            dsa_proxy_weights[:n],
            dsa_proxy_z[:n],
        )

    def _forward_prefill(
        self,
        query,
        kv_cache,
        attn_metadata,
        output,
        dsa_proxy_query,
        dsa_proxy_key,
        dsa_proxy_weights,
        dsa_proxy_z,
        *,
        summary_updated=False,
    ):
        assert self.summary_cache is not None and self.adapter is not None
        n = query.shape[0]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 prefill requires a breakable CUDA graph")
        table = attn_metadata.block_table[: attn_metadata.num_actual_reqs]
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
            owners = torch.tensor(owners_cpu, device=query.device, dtype=torch.long)
            positions = torch.tensor(
                positions_cpu, device=query.device, dtype=torch.long
            )
            num_regions = (max(lengths) + 7) // 8
            paged_plan = self.adapter.prepare_prefill_attention(
                query[:n],
                kv_cache,
                positions_cpu,
                min(num_regions, self.sparse_topk) + 1,
                self.scale,
            )
            chunks = (
                make_prefill_chunks(positions_cpu, self.sparse_topk, query.device)
                if paged_plan is None
                else None
            )
            attn_metadata.step5_runtime = (
                slots,
                owners,
                positions,
                num_regions,
                chunks,
                self.adapter.prefill_metadata(
                    owners,
                    positions,
                    num_regions,
                    self.sparse_topk,
                    paged_prefill=(
                        paged_plan.page_metadata(table, owners)
                        if paged_plan is not None and num_regions <= self.sparse_topk
                        else None
                    ),
                ),
                paged_plan,
            )
        slots, owners, positions, num_regions, chunks, indexer, paged_plan = (
            attn_metadata.step5_runtime
        )
        if not summary_updated:
            self.summary_cache.update(dsa_proxy_key[:n], dsa_proxy_z[:n], slots)
        packed_summaries = None
        if num_regions > self.sparse_topk:
            packed_summaries = self.adapter.pack_summaries(
                self.summary_cache.mean, table, num_regions
            )
        # Bound logits and gathered KV scratch independently of prefill length.
        start = 0
        for indexer_meta in indexer:
            end = start + indexer_meta.current.numel()
            current = indexer_meta.current
            if packed_summaries is None:
                selected = indexer_meta.selected
            else:
                packed, pages = packed_summaries
                scores = self.adapter.logits(
                    dsa_proxy_query[start:end],
                    dsa_proxy_weights[start:end],
                    packed,
                    pages,
                    owners[start:end],
                    indexer_meta.lengths,
                    num_regions,
                    block_table=indexer_meta.block_table,
                    schedule_meta=indexer_meta.schedule,
                )
                selected = self.adapter.select_regions(
                    scores,
                    current,
                    self.sparse_topk,
                    topk_lengths=indexer_meta.topk_lengths,
                    paged_prefill=(
                        paged_plan.page_metadata(table, owners, start, end)
                        if paged_plan is not None
                        else None
                    ),
                )
            if paged_plan is not None:
                if paged_plan.query_buffer is not None:
                    paged_plan.prepare_chunk(
                        start,
                        query[start:end],
                        kv_cache,
                        selected,
                        table,
                        owners[start:end],
                    )
                start = end
                continue
            self.adapter.run_attention(
                query[start:end],
                kv_cache,
                selected,
                table,
                owners[start:end],
                positions[start:end],
                output[start:end],
                self.scale,
                prefill_chunks=chunks[start // 32 : (end + 31) // 32],
            )
            start = end
        if paged_plan is not None:
            paged_plan.finish(query[:n], kv_cache, output[:n])
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
        *,
        summary_updated=False,
        defer_inactive_clear=False,
    ):
        assert self.summary_cache is not None and self.adapter is not None
        n = query.shape[0]
        slots = metadata.slot_mapping[:n]
        table = metadata.block_table[:n]
        if not summary_updated:
            self.summary_cache.update_decode(proxy_key, proxy_z, slots)
        num_regions = min(self.max_num_regions, table.shape[1] * kv_cache.shape[2] // 8)
        scores, decode = self.adapter.decode_logits(
            proxy_query,
            proxy_weights,
            self.summary_cache.mean,
            table,
            num_regions,
            slots,
            metadata.seq_lens[:n],
        )
        self.adapter.select_and_run_decode_attention(
            query,
            kv_cache,
            scores,
            decode.current,
            self.sparse_topk,
            table,
            decode.positions,
            decode.active,
            output,
            self.scale,
            topk_lengths=decode.topk_lengths,
            clear_inactive=not defer_inactive_clear,
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
