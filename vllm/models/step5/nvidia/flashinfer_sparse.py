# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 sparse selection and attention using FlashInfer and PyTorch."""

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config_or_none

from .ops.prefill_indexer_metadata import prepare_prefill_indexer
from .ops.sparse_decode import (
    merge_split_decode,
    prepare_sparse_decode,
    zero_inactive_decode,
)
from .ops.sparse_decode_attention import (
    sparse_decode_attention,
    sparse_decode_splits,
)
from .ops.sparse_indexer import (
    PagedDecodeMetadata,
    pack_decode_summary_pages,
    pack_summary_pages,
    select_all_regions,
    select_regions,
)
from .ops.sparse_indexer_scores import decode_summary_scores
from .ops.sparse_prefill import PrefillChunk, gather_prefill_kv
from .ops.sparse_prefill_pages import CompactPagedPrefill, view_sparse_kv_pages


def _direct_sparse_decode(query, kv_cache, out) -> bool:
    """Whether the direct paged sparse decode kernel applies; it beats both
    the gathered ragged and the paged FlashInfer paths at every batch size."""
    return (
        query.shape[2] == 192
        and query.shape[1] <= 16
        and kv_cache.shape[-1] == 384
        and kv_cache.stride(-1) == 1
        and query.stride(2) == 1
        and out.stride(2) == 1
    )


class _DecodePlan(NamedTuple):
    wrapper: Any
    mask: torch.Tensor
    indices: torch.Tensor | None
    kv_indptr: torch.Tensor | None
    splits: int = 1
    partial: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    split_query: torch.Tensor | None = None


def _decode_splits(batch: int, region_count: int, num_sms: int) -> int:
    """Split-KV factor for ragged sparse decode.

    Without splits each request runs on one CTA over up to 4K tokens. Splits
    must divide the fixed region capacity and keep at least 16 regions each.
    Plans are fixed per batch size, so the split cannot follow context length;
    capping at one CTA per SM bounds the short-context overhead to about 1-2 us.
    """
    target = min(19, num_sms // max(batch, 1))
    return max(
        (
            s
            for s in range(1, target + 1)
            if region_count % s == 0 and region_count // s >= 16
        ),
        default=1,
    )


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
        self._paged_prefill_plans: list[CompactPagedPrefill | None] = [None, None]
        self._paged_prefill_index = 0
        self._decode_plans: dict[tuple, _DecodePlan] = {}

    def prepare_prefill_attention(
        self,
        query_shape,
        dtype,
        device,
        block_size,
        positions,
        region_count,
        scale,
        *,
        contiguous_query=True,
    ) -> CompactPagedPrefill | None:
        """Plan paged prefill over dense model-dtype single-group pages.

        FP8 and multi-group caches qualify once staged by the caller.
        """
        if not (
            query_shape[1] in (8, 16)
            and query_shape[2] == 192
            and dtype == torch.bfloat16
            and block_size in (16, 32, 64, 128)
        ):
            return None
        # Alternate plans: planning ahead of the forward must not overwrite
        # pinned buffers the previous step's copies may still be reading.
        self._paged_prefill_index ^= 1
        plan = self._paged_prefill_plans[self._paged_prefill_index]
        if plan is None:
            plan = CompactPagedPrefill(self._workspace)
            self._paged_prefill_plans[self._paged_prefill_index] = plan
        plan.plan(
            query_shape,
            dtype,
            device,
            positions,
            region_count,
            scale,
            block_size,
            contiguous_query=contiguous_query,
        )
        return plan

    @staticmethod
    def _use_paged_decode(query, kv_cache):
        # FA2's 192-wide FP8 KV path is not supported. FP8 uses the existing
        # ragged gather, which also dequantizes to the query dtype.
        batch, heads, dim = query.shape
        return (
            batch >= 32
            and dim == 192
            and heads in (8, 16)
            and query.dtype == torch.bfloat16
            and kv_cache.dtype == torch.bfloat16
            and kv_cache.shape[1] == 1
            and kv_cache.stride(-1) == 1
            and kv_cache.stride(2) == kv_cache.shape[3]
            and kv_cache.stride(0) % (8 * kv_cache.stride(2)) == 0
        )

    def _get_decode_plan(
        self, query, kv_cache, region_count, scale, *, compact=False, ragged_only=False
    ):
        from flashinfer import (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper,
        )

        batch, heads, dim = query.shape
        capacity = region_count * 8
        # Eight-token virtual pages avoid large KV gathers. Small batches are
        # faster with the ragged kernel, including short-context decode.
        # Paged pages index the block table per row, so multi-token rows
        # (MTP verify) use the ragged gather instead.
        use_paged = not ragged_only and self._use_paged_decode(query, kv_cache)
        assert not compact or use_paged
        splits = 1 if use_paged else _decode_splits(batch, region_count, self._num_sms)
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
            cu_splits = torch.arange(batch * splits + 1, dtype=torch.int32)
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
                kv_data_type=query.dtype,
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
                # One single-row FA2 request per split; masks and gathered
                # KV keep the per-request layout because splits are contiguous.
                wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    self._workspace,
                    backend="fa2",
                    **wrapper_args,
                )
                wrapper.plan(
                    cu_splits,
                    cu_splits * (capacity // splits),
                    heads,
                    1,
                    dim,
                    **plan_args,
                )
            # Keep the buffers created by the one-time plan alive across replays.
            mask = wrapper._custom_mask_buf
            if not use_paged:
                kv_indptr = wrapper._kv_indptr_buf
            elif compact:
                indices = wrapper._paged_kv_indices_buf
                kv_indptr = wrapper._paged_kv_indptr_buf
            partial = lse = split_query = None
            if splits > 1:
                split_query = query.new_empty((batch * splits, heads, dim))
                partial = query.new_empty((batch * splits, heads, dim))
                lse = torch.empty(
                    (batch * splits, heads), dtype=torch.float32, device=query.device
                )
            self._decode_plans[plan_key] = _DecodePlan(
                wrapper, mask, indices, kv_indptr, splits, partial, lse, split_query
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
        k_scale=None,
        v_scale=None,
        query_len=1,
        gate=None,
    ):
        """``gate`` ([tokens, all heads]) applies the headwise sigmoid gate."""
        if _direct_sparse_decode(query, kv_cache, out):
            # Reads selected regions straight from paged KV: no gather scratch,
            # no host plan, split merge fused for small split counts.
            sparse_decode_attention(
                query,
                kv_cache,
                regions,
                block_table,
                positions,
                active,
                out,
                scale=scale,
                query_len=query_len,
                splits=sparse_decode_splits(query.shape[0]),
                clear_inactive=clear_inactive or gate is not None,
                k_scale=k_scale,
                v_scale=v_scale,
                gate=gate,
            )
            return
        assert gate is None, "gate fusion needs the direct sparse decode kernel"
        plan = self._get_decode_plan(
            query, kv_cache, regions.shape[1], scale, ragged_only=query_len > 1
        )
        wrapper, mask, indices, kv_indptr = plan[:4]
        k, v = prepare_sparse_decode(
            kv_cache,
            regions,
            block_table,
            positions,
            active,
            mask,
            indices,
            kv_indptr,
            output_dtype=query.dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            splits=plan.splits,
            query=query,
            split_query=plan.split_query,
            query_len=query_len,
        )
        if plan.splits > 1:
            wrapper.run(
                plan.split_query,
                k,
                v,
                out=plan.partial,
                lse=plan.lse,
                return_lse=True,
            )
            merge_split_decode(
                plan.partial,
                plan.lse,
                out,
                plan.splits,
                active if clear_inactive else None,
            )
            return
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
        k_scale=None,
        v_scale=None,
        query_len=1,
        gate=None,
    ):
        """Decode attention for rows of ``query_len`` tokens per request."""
        direct = _direct_sparse_decode(query, kv_cache, out)
        if direct or not (
            query_len == 1
            and self._use_paged_decode(query, kv_cache)
            and topk == 512
            and scores.shape[1] > topk
        ):
            regions = self.select_regions(
                scores, current, topk, topk_lengths=topk_lengths, sort=not direct
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
                k_scale=k_scale,
                v_scale=v_scale,
                query_len=query_len,
                gate=gate,
            )
        assert gate is None, "gate fusion needs the direct sparse decode kernel"
        wrapper, mask, indices, indptr = self._get_decode_plan(
            query, kv_cache, topk + 1, scale, compact=True
        )[:4]
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
                kv_cache.stride(0) // (8 * kv_cache.stride(2)),
            ),
        )
        key, value = view_sparse_kv_pages(kv_cache)
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

    def decode_logits(
        self,
        query,
        weights,
        mean,
        table,
        regions,
        slots,
        seq_lens,
        row_groups=1,
        query_len=1,
    ):
        """Score regions for (token, KV group) rows.

        ``query`` is ``[tokens, row_groups, heads, 256]`` and ``weights`` is
        ``[tokens, row_groups, heads]``; row ``r`` is group ``r % row_groups``
        of token ``r // row_groups``. ``query_len`` tokens share a request of
        ``table``/``seq_lens``.
        """
        assert query.shape[1] == row_groups
        if (
            self._direct_summary_scores
            and query.shape[0] > 0
            and query.shape[2:] == (8, 256)
            and query.dtype == torch.float8_e4m3fn
            and query.stride(3) == 1
            and query.stride(2) == 256
            and weights.shape == query.shape[:3]
            and weights.dtype == torch.float32
            and weights.stride(-1) == 1
            and (mean.shape[1:] == (8, 256) or mean.shape[1:] in ((2, 256), (4, 256)))
            and mean.dtype == torch.uint8
            and mean.is_contiguous()
            and table.stride(1) == 1
            and regions > 0
        ):
            return decode_summary_scores(
                query,
                weights,
                mean,
                table,
                slots,
                seq_lens,
                regions,
                num_sms=self._num_sms,
                row_groups=row_groups,
                query_len=query_len,
            )
        if query_len > 1:
            offset = torch.arange(slots.shape[0], device=slots.device) % query_len
            seq_lens = seq_lens.repeat_interleave(query_len) - (query_len - 1 - offset)
            table = table.repeat_interleave(query_len, dim=0)
        if row_groups > 1:
            slots, table, seq_lens = (
                t.repeat_interleave(row_groups, dim=0) for t in (slots, table, seq_lens)
            )
        rows = query.shape[0] * row_groups
        query = query.reshape(rows, 1, *query.shape[2:])
        weights = weights.reshape(rows, 1, weights.shape[2])
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
        self,
        scores,
        current,
        topk,
        *,
        topk_lengths=None,
        paged_prefill=None,
        sort=True,
    ):
        return select_regions(
            scores,
            current,
            topk,
            self._topk_backend,
            topk_lengths=topk_lengths,
            paged_prefill=paged_prefill,
            sort=sort,
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
        k_scale=None,
        v_scale=None,
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
                output_dtype=query.dtype,
                k_scale=k_scale,
                v_scale=v_scale,
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
                kv_data_type=query.dtype,
                disable_split_kv=True,
            )
            self._prefill_plan_event.record(torch.cuda.current_stream(query.device))
            self._attention.run(query[start:end].contiguous(), k, v, out=out[start:end])
