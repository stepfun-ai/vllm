# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5/Step5Mini GQA + DSA attention backend."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from typing_extensions import override

from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.step5.nvidia.ops.cute_dsl.indexer_ops import (
    cutedsl_topk_selector_sm90_multi_cta,
    decode_summary_layout_step3p5,
    prewarm_cutedsl_topk_selector_sm90_compilation,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa import (
    batch_decode_logits_wgmma_n as batch_logits_wgmma_n,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa import (
    batch_decode_weighted_relu_logits_sum_paged_sm90_steptron_gqa as batch_logits,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa import (
    build_decode_mtp_union_exact_mask_meta_sm90_gqa,
    build_grouped_union_sparse_work_queue_gqa,
    csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa,
    csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa,
    cutedsl_topk_selector_decode_meta_sm90_gqa,
    decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa,
    merge_dynamic_split_nat_lse_states_sm90_gqa,
    merge_mtp_union_exact_mask_nat_lse_states_sm90_gqa,
    merge_variable_split_nat_lse_states_sm90_gqa,
    prefill_paged_weighted_relu_logits_sm90_steptron_gqa,
    prewarm_csa_compact_prefill_update_with_slots_sm90_gqa,
    prewarm_topk_selector_decode_meta_sm90_gqa,
    token_wise_flash_attn_decode_mtp_union_exact_mask_sm90_gqa_func,
    token_wise_flash_attn_decode_sm90_gqa_func,
    token_wise_flash_attn_decode_sm90_gqa_plan,
    token_wise_flash_attn_prefill_union_sm90_gqa_func,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.indexer_ops import (
    decode_sparse_meta_step3p5,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from .mtp_state_machine import Step5KernelPlan, build_step5_kernel_plan
from .sparse_summary_cache import (
    Step5DSAScratchWorkspace,
    Step5SparseSummaryCache,
    Step5SparseSummaryCacheConfig,
)

logger = init_logger(__name__)

build_decode_summary_layout_step3p5 = (
    decode_summary_layout_step3p5.build_decode_summary_layout_step3p5
)
build_decode_paged_summary_block_table_and_valid_step3p5 = (
    decode_sparse_meta_step3p5.build_decode_paged_summary_block_table_and_valid_step3p5
)
convert_region_block_topk_to_sparse_meta_step3p5 = (
    decode_sparse_meta_step3p5.convert_region_block_topk_to_sparse_meta_step3p5
)


def _step5_dsa_is_cuda_graph_capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


# Diagnostics for CSA active-slot pressure and for stray writes into the shared
# prefill tile scratch.  Both are off by default and cost nothing when off.
#   VLLM_DSA_TSL_REDZONE  1 = check tile_seq_lens for float32-looking garbage
#   VLLM_DSA_OCC_EVERY    sample CSA slot occupancy every N steps (0 = off)
_DSA_TSL_REDZONE = envs.VLLM_DSA_TSL_REDZONE
_DSA_OCC_EVERY = envs.VLLM_DSA_OCC_EVERY
_dsa_occ_peak = 0
_dsa_occ_step = 0


def _dsa_tile_seq_lens_redzone(
    tile_seq_lens: torch.Tensor,
    *,
    where: str,
    layer: object,
    num_prefill_reqs: int,
) -> None:
    """Report stray writes into the shared tile_seq_lens scratch.

    Bind time over-allocates (max_prefill_tiles, max_num_seqs) while a step only
    uses (num_tiles, num_prefill_reqs), so the unused tail is a natural redzone.
    Scan the whole storage rather than the view: an overrun from a neighbouring
    buffer stays inside the same cudaMalloc segment, which compute-sanitizer
    does not flag by default.  Values are reported both as int32 and as the
    float32 they decode to, because the writers seen so far stored real fp32
    payload (0.0008-0.033) that reads back as ~1e9.
    """
    if not _DSA_TSL_REDZONE:
        return
    torch.accelerator.synchronize()
    storage = tile_seq_lens.untyped_storage()
    flat = torch.empty(0, dtype=torch.int32, device=tile_seq_lens.device)
    flat.set_(storage, storage_offset=0, size=(storage.size() // 4,))
    dirty = torch.nonzero(flat > 10**8).flatten().tolist()
    if not dirty:
        return
    values = flat[dirty[:8]].tolist()
    as_float = [round(struct.unpack("<f", struct.pack("<i", v))[0], 6) for v in values]
    payload = int(tile_seq_lens.numel())
    print(
        f"[dsa-tsl-redzone] at={where} "
        f"layer={getattr(layer, 'layer_name', id(layer))} "
        f"dirty={len(dirty)}/{int(flat.numel())} payload_numel={payload} "
        f"num_prefill_reqs={num_prefill_reqs} "
        f"idx_range=({dirty[0]},{dirty[-1]}) "
        f"beyond_payload={[i for i in dirty if i >= payload][:8]} "
        f"as_int32={values} as_float32={as_float} "
        f"storage_ptr={hex(storage.data_ptr())}",
        flush=True,
    )


def _step5_sparse_gqa_region_capacity(
    num_regions: int,
    sparse_topk: int,
) -> int:
    num_regions = int(num_regions)
    if num_regions <= 0:
        return 0
    sparse_topk = int(sparse_topk)
    if sparse_topk <= 0:
        raise ValueError(f"Step5 DSA sparse_topk must be positive, got {sparse_topk}")
    chunk = ((max(4096, sparse_topk * 8) + 255) // 256) * 256
    return ((max(num_regions, chunk) + chunk - 1) // chunk) * chunk


# Triton treats runtime-value specialization and pointer-alignment
# specialization independently.  Keep both lists explicit: the scalar list
# prevents shape-dependent variants, while the pointer list is required for
# slices whose base address is not 16-byte aligned.
_EXPAND_BLOCK_TABLE_DYNAMIC_ARGS = (
    "rows",
    "cols",
    "pages_per_block",
    "out_cols",
    "kv_num_pages",
    "block_table_stride_row",
    "block_table_stride_col",
    "out_stride_row",
    "out_stride_col",
)
_EXPAND_BLOCK_TABLE_ALIGNMENT_DYNAMIC_ARGS = (
    "block_table",
    "out",
)


@triton.jit(
    do_not_specialize=_EXPAND_BLOCK_TABLE_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _EXPAND_BLOCK_TABLE_DYNAMIC_ARGS + _EXPAND_BLOCK_TABLE_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_expand_block_table_kernel(
    block_table,
    out,
    rows,
    cols,
    pages_per_block,
    out_cols,
    kv_num_pages,
    block_table_stride_row,
    block_table_stride_col,
    out_stride_row,
    out_stride_col,
    VALIDATE_PHYSICAL_PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows * out_cols
    row = offsets // out_cols
    out_col = offsets % out_cols
    source_col = out_col // pages_per_block
    sub_page = out_col - source_col * pages_per_block
    safe_source_col = tl.minimum(source_col, cols - 1)
    source_page = tl.load(
        block_table
        + row * block_table_stride_row
        + safe_source_col * block_table_stride_col,
        mask=mask,
        other=0,
    )
    expanded_page = source_page * pages_per_block + sub_page
    if VALIDATE_PHYSICAL_PAGE:
        valid = (source_col < cols) & (source_page >= 0) & (source_page < kv_num_pages)
        expanded_page = tl.where(valid, expanded_page, -1)
    tl.store(
        out + row * out_stride_row + out_col * out_stride_col,
        expanded_page,
        mask=mask,
    )


_PREFILL_REGION_VALID_SHIFT = 24
_PREFILL_LOGITS_WORKSPACE_ELEMENTS = 128 * 1024 * 1024
_PREFILL_Q_TILE_CAPACITY = 4096
_BATCH_DECODE_Q_HEADS_PER_KV = 4
_BATCH_DECODE_Q_RUNTIME_ROWS = 16
_BATCH_DECODE_BLOCK_Q = 32
# Keep this metadata launch shape stable across CUDA Graph captures. A single
# program covers max_num_seqs=64 under MTP3 (four token rows per request), while
# larger configured batches scale through the grid without compiling
# request-count-specific tiles.
_DECODE_ROW_RANGES_BLOCK_SIZE = 256
_DECODE_ROW_RANGES_NUM_WARPS = 4
_MTP_UNION_Q_GROUP = 4
_MTP_UNION_SPLITS = 8


_PREFILL_TILE_REQUEST_META_DYNAMIC_ARGS = (
    "num_reqs",
    "request_offset",
    "query_base",
    "tile_start",
    "tile_rows",
    "max_regions",
)
_PREFILL_TILE_REQUEST_META_ALIGNMENT_DYNAMIC_ARGS = (
    "query_start_loc",
    "seq_lens",
    "tile_query_start_loc",
    "tile_seq_lens",
    "row_owners",
    "history_lengths",
)


@triton.jit(
    do_not_specialize=_PREFILL_TILE_REQUEST_META_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_TILE_REQUEST_META_DYNAMIC_ARGS
        + _PREFILL_TILE_REQUEST_META_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_tile_request_meta_kernel(
    query_start_loc,
    seq_lens,
    tile_query_start_loc,
    tile_seq_lens,
    row_owners,
    history_lengths,
    num_reqs,
    request_offset,
    query_base,
    tile_start,
    tile_rows,
    max_regions,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    request = tl.program_id(0)
    request_mask = request < num_reqs
    source_request = request + request_offset
    source_start = (
        tl.load(
            query_start_loc + source_request,
            mask=request_mask,
            other=query_base,
        )
        - query_base
    )
    source_end = (
        tl.load(
            query_start_loc + source_request + 1,
            mask=request_mask,
            other=query_base,
        )
        - query_base
    )
    tile_end = tile_start + tile_rows
    clipped_start = tl.maximum(tl.minimum(source_start, tile_end), tile_start)
    clipped_end = tl.maximum(tl.minimum(source_end, tile_end), tile_start)
    tile_request_start = clipped_start - tile_start
    tile_request_end = clipped_end - tile_start
    full_query_len = tl.maximum(source_end - source_start, 0)
    request_tile_rows = tl.maximum(tile_request_end - tile_request_start, 0)
    request_tile_offset = tl.maximum(tile_start - source_start, 0)
    seq_len = tl.load(
        seq_lens + source_request,
        mask=request_mask,
        other=0,
    )
    context_len = tl.maximum(seq_len - full_query_len, 0)
    tile_seq_len = context_len + request_tile_offset + request_tile_rows
    tl.store(
        tile_query_start_loc + request,
        tile_request_start,
        mask=request_mask,
    )
    tl.store(tile_seq_lens + request, tile_seq_len, mask=request_mask)
    row_offset = 0
    while row_offset < request_tile_rows:
        local_rows = row_offset + tl.arange(0, BLOCK)
        row_mask = request_mask & (local_rows < request_tile_rows)
        row = tile_request_start + local_rows
        query_position = context_len + request_tile_offset + local_rows
        current_region = query_position // REGION_BLOCK_SIZE
        tl.store(row_owners + row, request, mask=row_mask)
        tl.store(
            history_lengths + row,
            tl.minimum(current_region, max_regions),
            mask=row_mask,
        )
        row_offset += BLOCK

    write_end = request == 0
    tl.store(tile_query_start_loc + num_reqs, tile_rows, mask=write_end)


_PREFILL_BATCHED_LOGITS_CU_DYNAMIC_ARGS = (
    "num_reqs",
    "region_block_size",
    "block_q",
)
_PREFILL_BATCHED_LOGITS_CU_ALIGNMENT_DYNAMIC_ARGS = (
    "tile_query_start_loc",
    "tile_seq_lens",
    "cu_seqlens_q",
    "cu_seqlens_k",
)


@triton.jit(
    do_not_specialize=_PREFILL_BATCHED_LOGITS_CU_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_BATCHED_LOGITS_CU_DYNAMIC_ARGS
        + _PREFILL_BATCHED_LOGITS_CU_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_batched_logits_cu_kernel(
    tile_query_start_loc,
    tile_seq_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    num_reqs,
    region_block_size,
    block_q,
) -> None:
    running_q = tl.full((), 0, tl.int32)
    running_k = tl.full((), 0, tl.int32)
    request = 0
    while request < num_reqs:
        q_start = tl.load(tile_query_start_loc + request)
        q_end = tl.load(tile_query_start_loc + request + 1)
        q_len = tl.maximum(q_end - q_start, 0)
        q_padded = ((q_len + block_q - 1) // block_q) * block_q
        seq_len = tl.load(tile_seq_lens + request)
        k_len = (seq_len + region_block_size - 1) // region_block_size
        tl.store(cu_seqlens_q + request, running_q)
        tl.store(cu_seqlens_k + request, running_k)
        running_q += q_padded
        running_k += k_len
        request += 1
    tl.store(cu_seqlens_q + num_reqs, running_q)
    tl.store(cu_seqlens_k + num_reqs, running_k)


_PREFILL_BATCHED_LOGITS_PACK_DYNAMIC_ARGS = (
    "num_reqs",
    "tile_rows",
    "index_q_s0",
    "index_q_s2",
    "index_q_s3",
    "q_work_s0",
    "q_work_s2",
    "q_work_s3",
    "weights_s0",
    "weights_s2",
    "weights_work_s0",
    "weights_work_s2",
)
_PREFILL_BATCHED_LOGITS_PACK_ALIGNMENT_DYNAMIC_ARGS = (
    "index_q",
    "weights",
    "tile_query_start_loc",
    "cu_seqlens_q",
    "q_work",
    "weights_work",
)


@triton.jit(
    do_not_specialize=_PREFILL_BATCHED_LOGITS_PACK_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_BATCHED_LOGITS_PACK_DYNAMIC_ARGS
        + _PREFILL_BATCHED_LOGITS_PACK_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_batched_logits_pack_kernel(
    index_q,
    weights,
    tile_query_start_loc,
    cu_seqlens_q,
    q_work,
    weights_work,
    num_reqs,
    tile_rows,
    index_q_s0,
    index_q_s2,
    index_q_s3,
    q_work_s0,
    q_work_s2,
    q_work_s3,
    weights_s0,
    weights_s2,
    weights_work_s0,
    weights_work_s2,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.program_id(2)
    q_start = tl.load(tile_query_start_loc + request, mask=request < num_reqs, other=0)
    q_end = tl.load(
        tile_query_start_loc + request + 1,
        mask=request < num_reqs,
        other=0,
    )
    q_dst = tl.load(cu_seqlens_q + request, mask=request < num_reqs, other=0)
    q_dst_end = tl.load(cu_seqlens_q + request + 1, mask=request < num_reqs, other=0)
    q_len = tl.maximum(q_end - q_start, 0)
    q_padded = tl.maximum(q_dst_end - q_dst, 0)
    valid = (request < num_reqs) & (row < q_len) & (q_start + row < tile_rows)
    padded = (request < num_reqs) & (row < q_padded)
    d = tl.arange(0, BLOCK_D)
    q_ptr = index_q + (q_start + row) * index_q_s0 + head * index_q_s2
    qw_ptr = q_work + (q_dst + row) * q_work_s0 + head * q_work_s2
    q = tl.load(q_ptr + d * index_q_s3, mask=valid & (d < HEAD_DIM), other=0)
    tl.store(qw_ptr + d * q_work_s3, q, mask=padded & (d < HEAD_DIM))
    w_ptr = weights + (q_start + row) * weights_s0 + head * weights_s2
    ww_ptr = weights_work + (q_dst + row) * weights_work_s0 + head * weights_work_s2
    w = tl.load(w_ptr, mask=valid, other=0.0)
    tl.store(ww_ptr, w, mask=padded)


_PREFILL_BATCHED_LOGITS_SCATTER_DYNAMIC_ARGS = (
    "tile_rows",
    "max_regions",
    "logits_work_s0",
    "logits_work_s1",
    "logits_s0",
    "logits_s1",
)
_PREFILL_BATCHED_LOGITS_SCATTER_ALIGNMENT_DYNAMIC_ARGS = (
    "logits_work",
    "row_owners",
    "tile_query_start_loc",
    "cu_seqlens_q",
    "logits",
)


@triton.jit(
    do_not_specialize=_PREFILL_BATCHED_LOGITS_SCATTER_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_BATCHED_LOGITS_SCATTER_DYNAMIC_ARGS
        + _PREFILL_BATCHED_LOGITS_SCATTER_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_batched_logits_scatter_kernel(
    logits_work,
    row_owners,
    tile_query_start_loc,
    cu_seqlens_q,
    logits,
    tile_rows,
    max_regions,
    logits_work_s0,
    logits_work_s1,
    logits_s0,
    logits_s1,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    region_block = tl.program_id(1)
    request = tl.load(row_owners + row)
    request_start = tl.load(tile_query_start_loc + request)
    request_q_dst = tl.load(cu_seqlens_q + request)
    src_row = request_q_dst + row - request_start
    regions = region_block * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = (row < tile_rows) & (regions < max_regions)
    values = tl.load(
        logits_work + src_row * logits_work_s0 + regions * logits_work_s1,
        mask=valid,
        other=0.0,
    )
    tl.store(
        logits + row * logits_s0 + regions * logits_s1,
        values,
        mask=valid,
    )


_PREFILL_UNION_GROUP_RANGES_DYNAMIC_ARGS = (
    "num_reqs",
    "total_groups",
)
_PREFILL_UNION_GROUP_RANGES_ALIGNMENT_DYNAMIC_ARGS = (
    "tile_query_start_loc",
    "tile_seq_lens",
    "work_q_global",
    "work_q_local",
    "work_q_len",
)


@triton.jit(
    do_not_specialize=_PREFILL_UNION_GROUP_RANGES_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_UNION_GROUP_RANGES_DYNAMIC_ARGS
        + _PREFILL_UNION_GROUP_RANGES_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_union_group_ranges_kernel(
    tile_query_start_loc,
    tile_seq_lens,
    work_q_global,
    work_q_local,
    work_q_len,
    num_reqs,
    total_groups,
    Q_GROUP: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    running = tl.full((), 0, tl.int32)
    found = tl.full((), False, tl.int1)
    out_global = tl.full((), 0, tl.int32)
    out_local = tl.full((), 0, tl.int32)
    out_len = tl.full((), 0, tl.int32)
    request = 0
    while request < num_reqs:
        q_start = tl.load(tile_query_start_loc + request)
        q_end = tl.load(tile_query_start_loc + request + 1)
        rows = tl.maximum(q_end - q_start, 0)
        req_groups = (rows + Q_GROUP - 1) // Q_GROUP
        in_request = (group >= running) & (group < running + req_groups)
        local_group = group - running
        local_base = q_start + local_group * Q_GROUP
        local_len = tl.minimum(local_base + Q_GROUP, q_end)
        global_base = tl.load(tile_seq_lens + request) - rows + local_group * Q_GROUP
        take = in_request & ~found
        out_local = tl.where(take, local_base, out_local)
        out_len = tl.where(take, local_len, out_len)
        out_global = tl.where(take, global_base, out_global)
        found = found | in_request
        running += req_groups
        request += 1
    valid_group = group < total_groups
    tl.store(work_q_global + group, out_global, mask=valid_group)
    tl.store(work_q_local + group, out_local, mask=valid_group)
    tl.store(work_q_len + group, out_len, mask=valid_group)


_FLATTEN_DECODE_Q_LE4_DYNAMIC_ARGS = (
    "num_pages",
    "block_table_stride0",
    "block_table_stride1",
    "flat_block_table_stride0",
    "flat_block_table_stride1",
)
_FLATTEN_DECODE_Q_LE4_ALIGNMENT_DYNAMIC_ARGS = (
    "query_start_loc",
    "seq_lens",
    "block_table",
    "token_seq_lens",
    "token_request_indices",
    "flat_block_table",
)


@triton.jit(
    do_not_specialize=_FLATTEN_DECODE_Q_LE4_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _FLATTEN_DECODE_Q_LE4_DYNAMIC_ARGS
        + _FLATTEN_DECODE_Q_LE4_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_flatten_decode_q_le4_metadata_kernel(
    query_start_loc,
    seq_lens,
    block_table,
    token_seq_lens,
    token_request_indices,
    flat_block_table,
    num_pages,
    block_table_stride0,
    block_table_stride1,
    flat_block_table_stride0,
    flat_block_table_stride1,
    BLOCK_PAGES: tl.constexpr,
    MAX_QUERY_LEN: tl.constexpr,
):
    # The caller only enters this path for max_query_len <= 4. The query is
    # already packed in request order, so expand each request's metadata to
    # its at-most-four token rows directly. There is no token->request inverse
    # lookup to perform here.
    request = tl.program_id(0)
    page_tile = tl.program_id(1)
    request_start = tl.load(query_start_loc + request)
    request_end = tl.load(query_start_loc + request + 1)
    request_q_len = request_end - request_start
    token_offsets = tl.arange(0, MAX_QUERY_LEN)
    tokens = request_start + token_offsets
    token_mask = token_offsets < request_q_len
    if page_tile == 0:
        request_seq_len = tl.load(seq_lens + request)
        tl.store(
            token_seq_lens + tokens,
            request_seq_len - request_q_len + token_offsets + 1,
            mask=token_mask,
        )
        tl.store(token_request_indices + tokens, request, mask=token_mask)

    page = tl.arange(0, BLOCK_PAGES)
    page_id = page_tile * BLOCK_PAGES + page
    page_mask = page_id < num_pages
    source = block_table + request * block_table_stride0 + page_id * block_table_stride1
    target = (
        flat_block_table
        + tokens[:, None] * flat_block_table_stride0
        + page_id[None, :] * flat_block_table_stride1
    )
    values = tl.load(source, mask=page_mask, other=-1)
    tl.store(
        target,
        values[None, :],
        mask=token_mask[:, None] & page_mask[None, :],
    )


_PREFILL_TILE_PACK_DYNAMIC_ARGS = (
    "topk",
    "block_table_pages",
    "block_table_stride",
)
_PREFILL_TILE_PACK_ALIGNMENT_DYNAMIC_ARGS = (
    "raw_topk",
    "row_owners",
    "tile_query_start_loc",
    "tile_seq_lens",
    "kernel_block_table",
    "region_counts",
    "packed_regions",
    "region_phys_indices",
    "region_indices",
)


@triton.jit(
    do_not_specialize=_PREFILL_TILE_PACK_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_TILE_PACK_DYNAMIC_ARGS + _PREFILL_TILE_PACK_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_tile_pack_kernel(
    raw_topk,
    row_owners,
    tile_query_start_loc,
    tile_seq_lens,
    kernel_block_table,
    region_counts,
    packed_regions,
    region_phys_indices,
    region_indices,
    topk,
    block_table_pages,
    block_table_stride,
    REGION_BLOCK_SIZE: tl.constexpr,
    REGIONS_PER_KERNEL_PAGE: tl.constexpr,
    VALID_SHIFT: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    request = tl.load(row_owners + row)
    request_start = tl.load(tile_query_start_loc + request)
    request_end = tl.load(tile_query_start_loc + request + 1)
    request_rows = request_end - request_start
    local_row = row - request_start
    tile_seq_len = tl.load(tile_seq_lens + request)
    query_position = tile_seq_len - request_rows + local_row
    current_region = tl.maximum(query_position // REGION_BLOCK_SIZE, 0)
    history_count = tl.minimum(current_region, topk)

    columns = tl.arange(0, BLOCK_K)
    in_topk = columns < topk
    valid_history = in_topk & (columns < history_count)
    raw = tl.load(
        raw_topk + row * topk + columns,
        mask=valid_history,
        other=-1,
    ).to(tl.int32)
    sentinel = tl.full((BLOCK_K,), 0x7FFFFFFF, tl.int32)
    sorted_regions = tl.sort(
        tl.where(valid_history & (raw >= 0), raw, sentinel),
        descending=False,
    )
    page_col = sorted_regions // REGIONS_PER_KERNEL_PAGE
    page_slot = sorted_regions - page_col * REGIONS_PER_KERNEL_PAGE
    safe_page_col = tl.maximum(tl.minimum(page_col, block_table_pages - 1), 0)
    physical_page = tl.load(
        kernel_block_table + request * block_table_stride + safe_page_col,
        mask=valid_history & (page_col < block_table_pages),
        other=-1,
    )
    valid_tokens = query_position + 1 - sorted_regions * REGION_BLOCK_SIZE
    valid_tokens = tl.maximum(tl.minimum(valid_tokens, REGION_BLOCK_SIZE), 0)
    packed = (physical_page * REGIONS_PER_KERNEL_PAGE + page_slot) | (
        valid_tokens << VALID_SHIFT
    )
    packed = tl.where(
        valid_history & (page_col < block_table_pages) & (physical_page >= 0),
        packed,
        -1,
    )
    tl.store(
        packed_regions + row * (topk + 1) + columns,
        packed,
        # The current region is written separately at history_count below.
        # Do not let the invalid tail of this history store overwrite that
        # slot with -1.
        mask=valid_history,
    )
    tl.store(
        region_phys_indices + row * (topk + 1) + columns,
        physical_page * REGIONS_PER_KERNEL_PAGE + page_slot,
        mask=valid_history,
    )
    tl.store(
        region_indices + row * (topk + 1) + columns,
        sorted_regions * REGION_BLOCK_SIZE,
        mask=valid_history,
    )

    current_page_col = current_region // REGIONS_PER_KERNEL_PAGE
    current_page_slot = current_region - current_page_col * REGIONS_PER_KERNEL_PAGE
    safe_current_page_col = tl.maximum(
        tl.minimum(current_page_col, block_table_pages - 1),
        0,
    )
    current_physical_page = tl.load(
        kernel_block_table + request * block_table_stride + safe_current_page_col,
        mask=current_page_col < block_table_pages,
        other=-1,
    )
    current_valid_tokens = query_position + 1 - current_region * REGION_BLOCK_SIZE
    current_valid_tokens = tl.maximum(
        tl.minimum(current_valid_tokens, REGION_BLOCK_SIZE),
        0,
    )
    current_packed = (
        current_physical_page * REGIONS_PER_KERNEL_PAGE + current_page_slot
    ) | (current_valid_tokens << VALID_SHIFT)
    current_packed = tl.where(
        (current_page_col < block_table_pages) & (current_physical_page >= 0),
        current_packed,
        -1,
    )
    tl.store(
        packed_regions + row * (topk + 1) + history_count,
        current_packed,
    )
    tl.store(
        region_phys_indices + row * (topk + 1) + history_count,
        current_physical_page * REGIONS_PER_KERNEL_PAGE + current_page_slot,
    )
    tl.store(
        region_indices + row * (topk + 1) + history_count,
        current_region * REGION_BLOCK_SIZE,
    )
    tl.store(region_counts + row, history_count + 1)


_PREFILL_UNION_WORK_LOCAL_OFFSET_DYNAMIC_ARGS = ("groups", "offset")
_PREFILL_UNION_WORK_LOCAL_OFFSET_ALIGNMENT_DYNAMIC_ARGS = (
    "work_q_local",
    "work_q_len",
)


@triton.jit(
    do_not_specialize=_PREFILL_UNION_WORK_LOCAL_OFFSET_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _PREFILL_UNION_WORK_LOCAL_OFFSET_DYNAMIC_ARGS
        + _PREFILL_UNION_WORK_LOCAL_OFFSET_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_prefill_union_work_local_offset_kernel(
    work_q_local,
    work_q_len,
    groups,
    offset,
    BLOCK: tl.constexpr,
) -> None:
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < groups
    local = tl.load(work_q_local + group, mask=mask, other=0)
    length = tl.load(work_q_len + group, mask=mask, other=0)
    tl.store(work_q_local + group, local + offset, mask=mask)
    tl.store(work_q_len + group, length + offset, mask=mask)


_DECODE_PREPARE_TOPK_DYNAMIC_ARGS = ("num_regions", "padded_regions")

_DECODE_ROW_RANGES_DYNAMIC_ARGS = ("num_reqs", "num_regions")
_DECODE_ROW_RANGES_ALIGNMENT_DYNAMIC_ARGS = (
    "seq_lens",
    "live_token_slots",
    "row_starts",
    "row_ends",
    "sanitized_seq_lens",
)


@triton.jit(
    do_not_specialize=_DECODE_ROW_RANGES_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _DECODE_ROW_RANGES_DYNAMIC_ARGS + _DECODE_ROW_RANGES_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_decode_row_ranges_kernel(
    seq_lens,
    live_token_slots,
    row_starts,
    row_ends,
    sanitized_seq_lens,
    num_reqs,
    num_regions,
    REGION_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < num_reqs

    safe_row = tl.minimum(row, tl.maximum(num_reqs - 1, 0))
    seq_len = tl.load(seq_lens + safe_row, mask=mask, other=0)
    live_slot = tl.load(live_token_slots + safe_row, mask=mask, other=-1)
    valid_seq_len = tl.where(live_slot >= 0, seq_len, 0)

    history_end = tl.minimum(
        tl.maximum(valid_seq_len - 1, 0) // REGION_BLOCK_SIZE,
        num_regions,
    )
    tl.store(row_starts + safe_row, 0, mask=mask)
    tl.store(row_ends + safe_row, history_end, mask=mask)
    tl.store(sanitized_seq_lens + safe_row, valid_seq_len, mask=mask)


@triton.jit(
    do_not_specialize=_DECODE_PREPARE_TOPK_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _DECODE_PREPARE_TOPK_DYNAMIC_ARGS
        + (
            "scores_in",
            "summary_valid",
            "row_starts",
            "row_ends",
            "scores_out",
        )
    ),
)
def _step5_decode_prepare_topk_scores_kernel(
    scores_in,
    summary_valid,
    row_starts,
    row_ends,
    scores_out,
    num_regions,
    padded_regions,
    HAS_SUMMARY_VALID: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < padded_regions
    in_bounds = cols < num_regions

    safe_cols = tl.minimum(cols, tl.maximum(num_regions - 1, 0))
    visible = in_bounds
    if HAS_SUMMARY_VALID:
        start = tl.load(row_starts + row)
        end = tl.load(row_ends + row)
        valid = tl.load(
            summary_valid + row * num_regions + safe_cols,
            mask=in_bounds,
            other=0,
        ).to(tl.int1)
        visible = visible & (cols >= start) & (cols < end) & valid
    values = tl.load(
        scores_in + row * num_regions + safe_cols,
        mask=in_bounds,
        other=-float("inf"),
    ).to(tl.float32)
    values = tl.where(visible, values, -float("inf"))
    tl.store(scores_out + row * padded_regions + cols, values, mask=col_mask)


@triton.jit
def _step5_decode_sort_logical_topk_kernel(
    logical_topk,
    topk,
    BLOCK_K: tl.constexpr,
) -> None:
    """Sort selected logical regions independently of physical KV pages."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    valid = cols < topk
    sentinel = tl.full((BLOCK_K,), 0x7FFFFFFF, tl.int32)
    regions = tl.load(
        logical_topk + row * topk + cols,
        mask=valid,
        other=sentinel,
    ).to(tl.int32)
    regions = tl.sort(regions, descending=False)
    tl.store(logical_topk + row * topk + cols, regions, mask=valid)


@triton.jit(
    do_not_specialize=("rows",),
    do_not_specialize_on_alignment=(
        "row_starts_in",
        "row_ends_in",
        "row_starts_out",
        "row_ends_out",
        "valid_rows_ptr",
        "rows",
    ),
)
def _step5_decode_mask_invalid_rows_kernel(
    row_starts_in,
    row_ends_in,
    row_starts_out,
    row_ends_out,
    valid_rows_ptr,
    rows,
):
    row = tl.program_id(0)
    valid = row < rows
    valid = valid & (row < tl.load(valid_rows_ptr).to(tl.int32))
    start = tl.load(row_starts_in + row, mask=valid, other=0).to(tl.int32)
    end = tl.load(row_ends_in + row, mask=valid, other=0).to(tl.int32)
    tl.store(row_starts_out + row, start, mask=row < rows)
    tl.store(row_ends_out + row, end, mask=row < rows)


@dataclass(frozen=True)
class Step5DSARuntimeLayout:
    token_flat_slot: torch.Tensor
    token_positions: torch.Tensor
    reset_slots: torch.Tensor
    token_valid: torch.Tensor


@dataclass
class Step5DSAStepMetadata:
    """Step-owned metadata shared by all layers in one attention group."""

    runtime_layout: Step5DSARuntimeLayout | None = None
    csa_task_prefix: torch.Tensor | None = None
    decode_flattened: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) = None
    decode_request_indices: torch.Tensor | None = None
    decode_table_indices: torch.Tensor | None = None
    decode_group_query_start_loc: torch.Tensor | None = None
    decode_group_row_starts: torch.Tensor | None = None
    decode_row_ranges: torch.Tensor | None = None
    decode_paged_summary: tuple[torch.Tensor, torch.Tensor] | None = None
    decode_kernel_block_table: torch.Tensor | None = None
    decode_split_plan: tuple[int, int, int] | None = None
    prefill_kernel_block_table: torch.Tensor | None = None
    prefill_tile_query_start_loc: torch.Tensor | None = None
    prefill_tile_seq_lens: torch.Tensor | None = None
    prefill_row_owners: torch.Tensor | None = None
    prefill_history_lengths: torch.Tensor | None = None
    prefill_tile_capacity: int = 0
    prefill_num_tiles: int = 0
    prefill_num_reqs: int = 0
    prefill_total_tokens: int = 0
    prefill_tile_union_groups: tuple[int, ...] | None = None


class Step5DSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """FlashAttention metadata builder for the Step5 DSA sparse path.

    This keeps the main-attention request boundaries and prefill history for
    the mixed path without adding device-to-host synchronization in forward.
    Cascade attention is disabled because DSA selects sparse regions from the
    normal block table and sequence lengths instead of using cascade metadata.
    """

    # Keep the default capability narrow. The config-aware override below only
    # enables multi-row FULL replay for the validated legacy static-MTP path,
    # without adding a Step5-specific branch to the generic model runner.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    _full_cudagraph_spec_tokens: ClassVar[int] = 3
    supports_update_block_table: bool = False
    reorder_batch_threshold: int = 1

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type[Step5DSAMetadataBuilder],
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        spec_config = getattr(vllm_config, "speculative_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        num_speculative_tokens = getattr(spec_config, "num_speculative_tokens", None)
        if (
            getattr(spec_config, "method", None) == "mtp"
            and num_speculative_tokens == cls._full_cudagraph_spec_tokens
            and getattr(spec_config, "num_speculative_tokens_per_batch_size", ())
            is None
            and not getattr(spec_config, "parallel_drafting", True)
            and not getattr(vllm_config, "use_v2_model_runner", True)
            and parallel_config is not None
            and not getattr(parallel_config, "use_ubatching", True)
            and getattr(parallel_config, "decode_context_parallel_size", 0) == 1
        ):
            return AttentionCGSupport.UNIFORM_BATCH
        return cls._cudagraph_support

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dsa_valid_requests = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self._dsa_valid_tokens = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self._mtp_enabled = (
            getattr(self.vllm_config.speculative_config, "method", None) == "mtp"
        )
        if self._mtp_enabled:
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        actual_num_reqs = int(common_attn_metadata.num_reqs)
        actual_num_tokens = int(common_attn_metadata.num_actual_tokens)
        self._dsa_valid_requests.fill_(actual_num_reqs)
        self._dsa_valid_tokens.fill_(actual_num_tokens)
        step5_metadata = cast(Any, metadata)
        step5_metadata.num_actual_reqs = actual_num_reqs
        step5_metadata.dsa_valid_requests = self._dsa_valid_requests
        step5_metadata.dsa_valid_tokens = self._dsa_valid_tokens
        step5_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        # Exact CPU sequence lengths are unavailable with async scheduling.
        step5_metadata.seq_lens_cpu = None
        step5_metadata.step5_kernel_plan = build_step5_kernel_plan(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
            mtp_enabled=self._mtp_enabled,
        )
        return metadata

    def use_cascade_attention(self, *args: Any, **kwargs: Any) -> bool:
        return False


# Main exposes [B, H, N, K+V]. Both SWA and DSA share this allocation;
# only the sparse kernel boundary converts it to a split, strided view.
class Step5SplitKVFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention using the canonical packed-content KV cache."""

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)


def _step5_sparse_kv_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """Expose [2, B, N, H, D] without copying the canonical [B, H, N, 2D].

    LBNHC keeps B/N adjacent in physical memory, so each scheduler block
    can be split into 16-token kernel pages while preserving physical IDs.
    K and V retain their actual strides through the CuTe kernel interface.
    """
    if kv_cache.ndim != 4 or kv_cache.shape[-1] % 2:
        raise ValueError("Step5 expects KV cache shape [B, H, N, 2D].")
    return kv_cache.unflatten(-1, (2, kv_cache.shape[-1] // 2)).permute(3, 0, 2, 1, 4)


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


class Step5DSAAttentionImpl(Step5SplitKVFlashAttentionImpl):
    """Step5 DSA backend entry.

    This backend is selected explicitly by the Step5 model when DSA is
    enabled. It executes the Step5 DSA kernel path directly.
    """

    can_return_lse_for_decode: ClassVar[bool] = False  # type: ignore[misc]
    supports_dcp: ClassVar[bool] = False  # type: ignore[misc]

    def __init__(
        self,
        *args: Any,
        sparse_config: Any | None = None,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        num_speculative_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
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
        self.sparse_config = sparse_config
        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Step5 DSA does not support decode context parallelism. "
                "Set --decode-context-parallel-size 1."
            )
        if (
            num_speculative_tokens > 0
            and vllm_config is not None
            and vllm_config.parallel_config.use_ubatching
        ):
            raise NotImplementedError(
                "Step5 DSA MTP does not support UBatch/DBO transaction isolation. "
                "Disable --enable-dbo and set --ubatch-size to 0."
            )
        if num_speculative_tokens > 3:
            raise ValueError("Step5 DSA supports at most three speculative tokens.")
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Step5 DSA does not support quantized KV cache dtype "
                f"{self.kv_cache_dtype!r}. Use auto, float16, or bfloat16."
            )
        self.sparse_topk = int(getattr(self.sparse_config, "topk", 0) or 0)
        self.sparse_region_block_size = int(
            getattr(self.sparse_config, "region_block_size", 1) or 1
        )
        cache_config = getattr(vllm_config, "cache_config", None)
        if (
            getattr(cache_config, "enable_prefix_caching", False)
            and not envs.VLLM_BATCH_INVARIANT
        ):
            logger.warning_once(
                "Step5 DSA prefix caching is enabled without "
                "VLLM_BATCH_INVARIANT=1. A cache hit recomputes a different "
                "token-batch shape from cold prefill, so default "
                "floating-point kernels may produce different logits (and "
                "therefore different greedy tokens). Set "
                "VLLM_BATCH_INVARIANT=1 when prefix-cache repeatability is "
                "required."
            )
        if not envs.VLLM_STEP5_DSA_FORCE_STABLE_TOPK:
            logger.warning_once(
                "Step5 DSA stable top-k is disabled. The SM90 streaming "
                "selector may choose different regions when scores tie, so "
                "greedy outputs are not guaranteed to be repeatable. Use "
                "VLLM_STEP5_DSA_FORCE_STABLE_TOPK=1 for correctness runs; "
                "disable it only for performance experiments."
            )
        prefix_match_unit = getattr(cache_config, "prefix_match_unit", None)
        if (
            getattr(cache_config, "enable_prefix_caching", False)
            and prefix_match_unit is not None
            and int(prefix_match_unit) % self.sparse_region_block_size != 0
        ):
            raise ValueError(
                "Step5 DSA prefix_match_unit must align to the sparse region "
                f"size ({self.sparse_region_block_size}), got "
                f"{prefix_match_unit}."
            )
        self.sparse_decode_split_max = int(
            getattr(self.sparse_config, "decode_split_max", 16) or 16
        )
        if self.sparse_decode_split_max not in (1, 2, 4, 16):
            raise ValueError(
                "Step5 DSA decode_split_max must be one of {1, 2, 4, 16}, got "
                f"{self.sparse_decode_split_max}."
            )
        self.max_model_len = int(
            max_model_len or getattr(self.sparse_config, "max_model_len", 0) or 0
        )
        self.max_num_seqs = max(
            1,
            int(max_num_seqs or getattr(self.sparse_config, "max_num_seqs", 0) or 0),
        )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self.max_num_batched_tokens = max(
            1,
            int(
                getattr(scheduler_config, "max_num_batched_tokens", 0)
                or self.max_num_seqs
            ),
        )
        self._decode_out_dtype: torch.dtype | None = None
        if (
            vllm_config is not None
            and getattr(vllm_config, "model_config", None) is not None
        ):
            self._decode_out_dtype = vllm_config.model_config.dtype
        self._sparse_gqa_selector_max_regions = (
            (self.max_model_len + self.sparse_region_block_size - 1)
            // self.sparse_region_block_size
            if self.max_model_len > 0
            else None
        )
        self.summary_cache_num_proxy_kv_heads = int(
            getattr(self.sparse_config, "summary_cache_num_proxy_kv_heads", 1) or 1
        )
        self._summary_cache: Step5SparseSummaryCache | None = None
        self._summary_cache_config: Step5SparseSummaryCacheConfig | None = None
        self._dsa_scratch_workspace = Step5DSAScratchWorkspace()
        self._dsa_scratch_bound = False
        self._prefill_q_tile_capacity = _PREFILL_Q_TILE_CAPACITY
        self._mtp = None
        if num_speculative_tokens > 0:
            if num_speculative_tokens + 1 > self.sparse_region_block_size:
                raise ValueError(
                    "Step5 DSA MTP verifier rows must fit one summary region: "
                    f"num_speculative_tokens={num_speculative_tokens}, "
                    f"region_block_size={self.sparse_region_block_size}."
                )
            from .sparse_attention_mtp import Step5DSAMTP

            self._mtp = Step5DSAMTP(self, num_speculative_tokens)

    def bind_scratch_workspace(
        self,
        workspace: Step5DSAScratchWorkspace,
    ) -> None:
        """Use one fixed scratch pool for every DSA layer in this model."""
        self._dsa_scratch_workspace = workspace
        self._dsa_scratch_bound = False

    def _dsa_virtual_engine(self) -> int:
        if is_forward_context_available():
            return int(getattr(get_forward_context(), "virtual_engine", 0) or 0)
        return 0

    def _get_dsa_tensor_buffer_at_least(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        virtual_engine = self._dsa_virtual_engine()
        # Allocate the ordering token alongside the profiled scratch pool.
        # Every real forward that can request a scratch buffer therefore has a
        # token ready before allocations are locked.
        self._dsa_scratch_workspace.get_order_token(
            virtual_engine,
            device=device,
        )
        buffers = self._dsa_scratch_workspace.tensor_buffers_by_engine.setdefault(
            virtual_engine, {}
        )
        required = 1
        for dim in shape:
            required *= int(dim)
        out = buffers.get(name)
        if (
            out is None
            or out.device != device
            or out.dtype != dtype
            or int(out.numel()) < int(required)
        ):
            if self._dsa_scratch_workspace.allocations_locked or getattr(
                self, "_dsa_scratch_bound", False
            ):
                raise RuntimeError(
                    "Step5 DSA scratch capacity was exceeded after bind: "
                    f"name={name!r}, requested_shape={shape}, "
                    f"required_elements={int(required)}, "
                    f"available_elements="
                    f"{0 if out is None else int(out.numel())}, "
                    f"virtual_engine={self._dsa_virtual_engine()}, "
                    f"device={device}, dtype={dtype}."
                )
            if _step5_dsa_is_cuda_graph_capturing():
                raise RuntimeError(
                    "Step5 DSA tensor scratch buffers must be initialized "
                    "before CUDA graph capture; "
                    f"name={name!r}, requested_shape={shape}, "
                    f"required_elements={int(required)}, "
                    f"available_elements="
                    f"{0 if out is None else int(out.numel())}, "
                    f"virtual_engine={self._dsa_virtual_engine()}, "
                    f"device={device}, dtype={dtype}."
                )
            out = torch.empty((int(required),), device=device, dtype=dtype)
            buffers[name] = out
        view = out[: int(required)].view(*shape)
        return view

    def _get_dsa_order_token(self, device: torch.device) -> torch.Tensor:
        """Return the model-wide per-virtual-engine ordering token."""
        return self._dsa_scratch_workspace.get_order_token(
            self._dsa_virtual_engine(),
            device=device,
        )

    def _get_step_metadata(
        self, attn_metadata: FlashAttentionMetadata
    ) -> Step5DSAStepMetadata:
        """Get the metadata payload shared by this forward pass's layers."""
        if not is_forward_context_available():
            return Step5DSAStepMetadata()
        forward_context = get_forward_context()
        cache = forward_context.additional_kwargs.setdefault("step5_dsa_metadata", {})
        key = id(attn_metadata)
        step_metadata = cache.get(key)
        if step_metadata is None:
            step_metadata = Step5DSAStepMetadata()
            cache[key] = step_metadata
        return step_metadata

    @staticmethod
    def _round_up(value: int, alignment: int) -> int:
        return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)

    @staticmethod
    def _prefill_tile_capacity(
        *, max_num_batched_tokens: int, padded_regions: int
    ) -> int:
        if max_num_batched_tokens <= 0 or padded_regions <= 0:
            raise ValueError("Prefill token and region capacities must be positive.")
        workspace_rows = _PREFILL_LOGITS_WORKSPACE_ELEMENTS // padded_regions
        if workspace_rows < 4:
            raise ValueError("Step5 prefill region bucket exceeds the logits budget.")
        rows = min(max_num_batched_tokens, _PREFILL_Q_TILE_CAPACITY, workspace_rows)
        return min(Step5DSAAttentionImpl._round_up(rows, 4), workspace_rows // 4 * 4)

    @staticmethod
    def _prefill_tile_union_group_counts(
        *,
        query_start_loc_cpu: torch.Tensor,
        num_decode_reqs: int,
        num_decode_tokens: int,
        num_prefill_reqs: int,
        total_prefill_tokens: int,
        tile_capacity: int,
        q_group: int,
    ) -> tuple[int, ...]:
        if query_start_loc_cpu.device.type != "cpu":
            raise RuntimeError("query_start_loc_cpu must be a CPU tensor")
        if num_prefill_reqs <= 0 or total_prefill_tokens <= 0:
            return ()
        if tile_capacity <= 0 or q_group <= 0:
            raise ValueError(
                f"tile_capacity and q_group must be positive, got "
                f"{tile_capacity=} {q_group=}"
            )

        num_reqs = num_decode_reqs + num_prefill_reqs
        if int(query_start_loc_cpu.numel()) < num_reqs + 1:
            raise RuntimeError(
                "query_start_loc_cpu is shorter than the active request prefix: "
                f"{int(query_start_loc_cpu.numel())} < {num_reqs + 1}"
            )
        query_starts = [
            int(value) for value in query_start_loc_cpu[: num_reqs + 1].tolist()
        ]
        query_base = int(num_decode_tokens)
        if query_starts[num_decode_reqs] != query_base:
            raise RuntimeError(
                "mixed sparse prefill decode token prefix does not match CPU "
                f"metadata: query_start_loc_cpu[{num_decode_reqs}]="
                f"{query_starts[num_decode_reqs]} != {query_base}"
            )
        if query_starts[num_reqs] - query_base != total_prefill_tokens:
            raise RuntimeError(
                "mixed sparse prefill token count does not match CPU request prefix: "
                f"{query_starts[num_reqs] - query_base} != {total_prefill_tokens}"
            )

        group_counts: list[int] = []
        for tile_start in range(0, total_prefill_tokens, tile_capacity):
            tile_end = min(tile_start + tile_capacity, total_prefill_tokens)
            tile_rows = tile_end - tile_start
            covered_rows = 0
            tile_groups = 0
            for request in range(num_prefill_reqs):
                source_start = query_starts[num_decode_reqs + request] - query_base
                source_end = query_starts[num_decode_reqs + request + 1] - query_base
                clipped_start = max(min(source_start, tile_end), tile_start)
                clipped_end = max(min(source_end, tile_end), tile_start)
                request_rows = max(clipped_end - clipped_start, 0)
                covered_rows += request_rows
                tile_groups += (request_rows + q_group - 1) // q_group
            if covered_rows != tile_rows or tile_groups <= 0:
                raise RuntimeError(
                    "mixed sparse prefill CPU request prefix does not cover tile: "
                    f"tile=[{tile_start}, {tile_end}) "
                    f"covered_rows={covered_rows} union_groups={tile_groups}"
                )
            group_counts.append(tile_groups)
        return tuple(group_counts)

    @staticmethod
    def _sparse_gqa_union_q_group(q_heads_per_kv: int) -> int:
        if int(q_heads_per_kv) == 4:
            return 32
        if int(q_heads_per_kv) == 8:
            return 16
        if int(q_heads_per_kv) == 16:
            return 8
        raise ValueError(
            "Step5 DSA grouped-union sparse_gqa prefill supports "
            f"q_heads_per_kv in (4, 8, 16), got {int(q_heads_per_kv)}."
        )

    def _sparse_gqa_region_bucket(self, num_regions: int) -> int:
        return _step5_sparse_gqa_region_capacity(
            num_regions,
            self.sparse_topk,
        )

    def _sparse_gqa_topk_capacity(self) -> int:
        topk = int(self.sparse_topk)
        if topk <= 0:
            raise ValueError(f"Step5 DSA sparse_topk must be positive, got {topk}")
        return topk

    def _csa_active_region_capacity(self) -> int:
        """Reserve overlapping tail generations until concurrent updates finish."""
        active_regions_per_generation = (
            int(self._mtp.num_speculative_tokens) + 1 if self._mtp is not None else 1
        )
        return 2 * int(self.max_num_seqs) * active_regions_per_generation

    @staticmethod
    def _begin_csa_allocation_check(
        summary_cache: Step5SparseSummaryCache,
    ) -> torch.Tensor:
        """Reset and return the graph-stable device allocation status."""
        success = getattr(
            summary_cache,
            "_step5_csa_allocation_success",
            None,
        )
        if (
            not isinstance(success, torch.Tensor)
            or success.dtype != torch.int32
            or tuple(success.shape) != (1,)
            or success.device != summary_cache.sum_cache.device
        ):
            raise RuntimeError(
                "Step5 CSA allocation-success state is not initialized on "
                "the summary-cache device."
            )
        success.fill_(1)
        return success

    @staticmethod
    def _assert_csa_allocation_success(success: torch.Tensor) -> None:
        """Fail closed after a device-side active-slot allocation attempt."""
        torch._assert_async(
            success,
            "Step5 DSA active-slot capacity is exhausted. Lower max_num_seqs "
            "or speculative token count, or increase the CSA capacity policy.",
        )

    def csa_fixed_runtime_state_size_bytes(self) -> int:
        """Return persistent CSA bytes that do not scale with KV blocks."""
        capacity = self._csa_active_region_capacity()
        num_kv_heads = int(self.summary_cache_num_proxy_kv_heads)
        proxy_dim = int(getattr(self.sparse_config, "proxy_dim", 0) or 0)
        region_block_size = int(self.sparse_region_block_size)
        stage_dtype = self._decode_out_dtype or torch.bfloat16
        stage_element_size = torch.tensor([], dtype=stage_dtype).element_size()

        state_elements = capacity * num_kv_heads * proxy_dim
        active_token_elements = state_elements * region_block_size
        fixed_bytes = (
            capacity * torch.tensor([], dtype=torch.long).element_size()
            + torch.tensor([], dtype=torch.int32).element_size()
            + 2 * state_elements * torch.tensor([], dtype=torch.float32).element_size()
            + capacity
            * num_kv_heads
            * torch.tensor([], dtype=torch.float32).element_size()
            + 2 * active_token_elements * stage_element_size
            + capacity
            * region_block_size
            * torch.tensor([], dtype=torch.uint8).element_size()
        )
        if self._mtp is not None:
            fixed_bytes += self._mtp.runtime_state_size_bytes(
                num_kv_heads=num_kv_heads,
                proxy_dim=proxy_dim,
                active_capacity=capacity,
            )
        return int(fixed_bytes)

    @staticmethod
    def _prewarm_csa_compact_prefill_update(
        summary_cache: Step5SparseSummaryCache,
        *,
        index_dtype: torch.dtype,
    ) -> None:
        prewarm_csa_compact_prefill_update_with_slots_sm90_gqa(
            device=summary_cache.sum_cache.device,
            index_dtype=index_dtype,
            head_dim=int(summary_cache.proxy_dim),
            summaries_per_page=int(summary_cache.summaries_per_page),
            sum_page_stride=int(summary_cache.sum_cache.stride(0)),
            count_page_stride=int(summary_cache.count_cache.stride(0)),
            region_block_size=int(summary_cache.region_block_size),
        )

    def bind_summary_cache(
        self,
        summary_cache: Step5SparseSummaryCache,
        *,
        initialize_runtime_state: bool = True,
    ) -> None:
        self._dsa_scratch_bound = False
        if initialize_runtime_state:
            self._summary_cache = summary_cache
            self._summary_cache_config = summary_cache.config
        active_capacity = self._csa_active_region_capacity()
        if int(summary_cache.sum_cache.shape[0]) < active_capacity:
            raise RuntimeError(
                "Step5 CSA active scratch capacity is smaller than the "
                "maximum concurrent active-region ownership: "
                f"required={active_capacity}, "
                f"sum_cache_slots={int(summary_cache.sum_cache.shape[0])}."
            )
        if initialize_runtime_state:
            self._initialize_csa_summary_state(
                summary_cache,
                capacity=active_capacity,
            )
            if self._mtp is not None:
                self._mtp.initialize(summary_cache)
        elif self._mtp is not None:
            self._mtp.prepare_scratch(summary_cache)
        max_regions = self._sparse_gqa_selector_max_regions
        if max_regions is None:
            max_regions = (
                int(summary_cache.num_pages) * int(summary_cache.page_size)
                + int(summary_cache.region_block_size)
                - 1
            ) // int(summary_cache.region_block_size)
        padded_regions = self._sparse_gqa_region_bucket(int(max_regions))
        max_decode_batch = int(self.max_num_seqs)
        max_decode_q_len = (
            int(self._mtp.num_speculative_tokens) + 1 if self._mtp is not None else 1
        )
        max_decode_rows = max(
            max_decode_batch,
            min(
                int(self.max_num_batched_tokens),
                max_decode_batch * max_decode_q_len,
            ),
        )
        device = summary_cache.sum_cache.device
        max_mtp_rows = 0
        if self._mtp is not None:
            max_mtp_rows = max_decode_batch * (
                int(self._mtp.num_speculative_tokens) + 1
            )
        max_layout_rows = self._round_up(
            max(int(self.max_num_batched_tokens), max_mtp_rows), 64
        )
        layout_dtypes = (
            ("csa_layout_flat_slot", torch.int64),
            ("csa_layout_token_slots", torch.int64),
            ("csa_layout_token_positions", torch.int64),
            ("csa_layout_reset_slots", torch.int64),
            ("csa_layout_token_valid", torch.bool),
        )
        for name, dtype in layout_dtypes:
            self._get_dsa_tensor_buffer_at_least(
                name,
                (max_layout_rows,),
                device=device,
                dtype=dtype,
            )
        self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_task_prefix",
            (max_decode_batch + 1,),
            device=device,
            dtype=torch.int32,
        )
        index_dtype = (
            getattr(self, "_summary_transaction_dtype", None)
            or self._decode_out_dtype
            or torch.bfloat16
        )
        # Dummy mixed-prefill warmup may classify q_len=1 rows as speculative
        # verifier rows and therefore never exercise the ordinary grouped
        # prefill update. Compile its exact cache geometry at bind time instead
        # of allowing the first user prompt to trigger a post-startup JIT.
        self._prewarm_csa_compact_prefill_update(
            summary_cache,
            index_dtype=index_dtype,
        )
        for name in ("csa_padded_index_k", "csa_padded_index_z"):
            self._get_dsa_tensor_buffer_at_least(
                name,
                (
                    max_layout_rows,
                    int(summary_cache.num_kv_heads),
                    int(summary_cache.proxy_dim),
                ),
                device=device,
                dtype=index_dtype,
            )
        num_regions = int(padded_regions)
        topk = self._sparse_gqa_topk_capacity()
        # The base multi-CTA top-k selector (used by prefill and by the stable
        # decode-meta path) forbids JIT compilation during CUDA graph capture.
        # Prewarm the selected stable/unstable variant so capture never
        # cache-misses.
        prewarm_cutedsl_topk_selector_sm90_compilation(
            device=device,
            dtype=torch.float32,
            topk=int(topk),
            max_seq_len=int(padded_regions),
            stable_sort=envs.VLLM_STEP5_DSA_FORCE_STABLE_TOPK,
        )
        if not envs.VLLM_STEP5_DSA_FORCE_STABLE_TOPK:
            prewarm_topk_selector_decode_meta_sm90_gqa(
                device=device,
                dtype=torch.float32,
                k=int(topk),
                max_score_capacity=int(padded_regions),
                score_capacity_step=self._sparse_gqa_region_bucket(1),
                score_nonneg=False,
            )
        q_heads_per_kv = int(self.num_heads) // int(summary_cache.num_kv_heads)
        block_q = 128 // q_heads_per_kv
        max_prefill_rows = self._prefill_tile_capacity(
            max_num_batched_tokens=int(self.max_num_batched_tokens),
            padded_regions=int(padded_regions),
        )
        self._prefill_q_tile_capacity = max_prefill_rows
        max_sparse_rows = max(max_decode_rows, max_prefill_rows)
        sparse_logits_elements = max_prefill_rows * int(padded_regions)
        prefill_logits_rows = self._round_up(
            max_prefill_rows + self.max_num_seqs * (int(block_q) - 1),
            int(block_q),
        )
        prefill_logits_work_elements = prefill_logits_rows * int(padded_regions)
        logits_elements = max(
            max_decode_rows * num_regions,
            sparse_logits_elements,
        )
        self._get_dsa_tensor_buffer_at_least(
            "sparse_logits_workspace",
            (logits_elements,),
            device=device,
            dtype=torch.float32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_meta_region_scores_prepared",
            (max_decode_rows, padded_regions),
            device=device,
            dtype=torch.float32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "sparse_topk_indices",
            (max_sparse_rows, topk),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_region_workspace",
            (max_prefill_rows, topk + 1),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_region_phys_indices_workspace",
            (max_prefill_rows, topk + 1),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_region_indices_workspace",
            (max_prefill_rows, topk + 1),
            device=device,
            dtype=torch.int32,
        )
        union_q_group = self._sparse_gqa_union_q_group(q_heads_per_kv)
        max_union_groups = triton.cdiv(max_prefill_rows, union_q_group) + int(
            self.max_num_seqs
        )
        max_union_windows = min(
            union_q_group * (topk + 1),
            1 << (max(num_regions, 1) - 1).bit_length(),
        )
        max_union_bit_words = ((1 << (max(num_regions, 1) - 1).bit_length()) + 31) // 32
        self._get_dsa_tensor_buffer_at_least(
            "prefill_union_counts",
            (max_union_groups,),
            device=device,
            dtype=torch.int32,
        )
        for name in (
            "prefill_union_phys",
            "prefill_union_logical",
            "prefill_union_exact_mask",
        ):
            self._get_dsa_tensor_buffer_at_least(
                name,
                (max_union_groups, max_union_windows),
                device=device,
                dtype=torch.int32,
            )
        for name in (
            "prefill_union_out_req_idx",
            "prefill_union_work_q_global",
            "prefill_union_work_q_local",
            "prefill_union_work_q_len",
        ):
            self._get_dsa_tensor_buffer_at_least(
                name, (max_union_groups,), device=device, dtype=torch.int32
            )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_union_causal_limits",
            (max_union_groups, union_q_group),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_union_bitset",
            (max_union_groups, max_union_bit_words),
            device=device,
            dtype=torch.int32,
        )
        prefill_region_starts = self._get_dsa_tensor_buffer_at_least(
            "prefill_tile_region_starts",
            (max_prefill_rows,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_tile_region_counts",
            (max_prefill_rows,),
            device=device,
            dtype=torch.int32,
        )
        # These buffers are allocated while the model is constructed under
        # inference mode, while KV side storage is bound outside that context.
        # Initialize their bind-time constants once under inference mode too.
        with torch.inference_mode():
            prefill_region_starts.zero_()
        max_tile_capacity = self._prefill_q_tile_capacity
        max_prefill_tiles = triton.cdiv(
            int(self.max_num_batched_tokens), max_tile_capacity
        )
        max_tile_reqs = int(self.max_num_seqs)
        # A smaller actual region bucket yields a larger runtime tile, so the
        # tile-count backing uses the largest possible number of tiles.
        self._get_dsa_tensor_buffer_at_least(
            "prefill_step_tile_query_start_loc",
            (max_prefill_tiles, max_tile_reqs + 1),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_step_tile_seq_lens",
            (max_prefill_tiles, max_tile_reqs),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_step_row_owners",
            (int(self.max_num_batched_tokens),),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_step_history_lengths",
            (int(self.max_num_batched_tokens),),
            device=device,
            dtype=torch.int32,
        )
        q_capacity = prefill_logits_rows
        q_dtype = self._decode_out_dtype or torch.bfloat16
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_q_work",
            (q_capacity, 1, q_heads_per_kv, 256),
            device=device,
            dtype=q_dtype,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_weights_work",
            (q_capacity, 1, q_heads_per_kv),
            device=device,
            dtype=torch.float32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_q_runtime",
            (q_capacity * q_heads_per_kv, 256),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_kernel_weights",
            (q_capacity, q_heads_per_kv),
            device=device,
            dtype=torch.float32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_out_work",
            (prefill_logits_work_elements,),
            device=device,
            dtype=torch.float32,
        )
        # cu_q/cu_k are the batched prefix sums over prefill requests, so they
        # need num_prefill_reqs + 1 entries.  Reserve for the worst case; the
        # old (2,) reservation only covered the single-request layout that the
        # batched logits path replaced.
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_cu_q",
            (int(self.max_num_seqs) + 1,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_paged_logits_cu_k",
            (int(self.max_num_seqs) + 1,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_meta_region_counts",
            (max_decode_rows, 1),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_meta_request_indices",
            (max_decode_rows,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_meta_region_packed_indices",
            (max_decode_rows, 1, topk + 1),
            device=device,
            dtype=torch.int64,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_meta_row_ranges",
            (3, self._round_up(int(max_decode_rows), 4)),
            device=device,
            dtype=torch.int32,
        )
        for name in ("decode_meta_topk_row_starts", "decode_meta_topk_row_ends"):
            self._get_dsa_tensor_buffer_at_least(
                name,
                (max_decode_rows,),
                device=device,
                dtype=torch.int32,
            )
        decode_batch_row_req_idx = self._get_dsa_tensor_buffer_at_least(
            "decode_batch_row_req_idx",
            (max_decode_rows,),
            device=device,
            dtype=torch.int32,
        )
        decode_batch_row_starts = self._get_dsa_tensor_buffer_at_least(
            "decode_batch_row_starts",
            (max_decode_batch,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_batch_row_ends",
            (max_decode_rows,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_batch_q_fp8",
            (max_decode_rows, 1, q_heads_per_kv, 256),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_batch_q_runtime",
            (max_decode_rows * _BATCH_DECODE_Q_RUNTIME_ROWS, 256),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_batch_kernel_weights",
            (max_decode_rows, _BATCH_DECODE_BLOCK_Q, q_heads_per_kv),
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            torch.arange(max_decode_rows, out=decode_batch_row_req_idx)
            decode_batch_row_starts.zero_()
        padded_topk = topk + 1
        if 1 < self.sparse_decode_split_max < 16:
            padded_topk = self._round_up(padded_topk, self.sparse_decode_split_max)
        self._get_dsa_tensor_buffer_at_least(
            "decode_splitkv_region_packed_indices",
            (max_decode_rows, padded_topk),
            device=device,
            dtype=torch.int64,
        )
        if self.sparse_decode_split_max > 1:
            work_items_max = self.sparse_decode_split_max * max_decode_rows
            self._get_dsa_tensor_buffer_at_least(
                "decode_splitkv_partial_out",
                (work_items_max, int(self.num_heads), int(self.head_size)),
                device=device,
                dtype=self._decode_out_dtype or torch.bfloat16,
            )
            self._get_dsa_tensor_buffer_at_least(
                "decode_splitkv_partial_lse",
                (work_items_max, int(self.num_heads)),
                device=device,
                dtype=torch.float32,
            )
            self._get_dsa_tensor_buffer_at_least(
                "decode_splitkv_merged_lse",
                (max_decode_rows, int(self.num_heads)),
                device=device,
                dtype=torch.float32,
            )
        if self._supports_mtp_union_exact_mask():
            mtp_union_rows = max_decode_batch * _MTP_UNION_Q_GROUP
            mtp_union_work = max_decode_batch * _MTP_UNION_SPLITS
            mtp_union_full_width = _MTP_UNION_Q_GROUP * (topk + 1)
            mtp_union_split_width = triton.cdiv(mtp_union_full_width, _MTP_UNION_SPLITS)
            for name in (
                "decode_mtp_union_unpacked_logical",
                "decode_mtp_union_unpacked_phys",
            ):
                self._get_dsa_tensor_buffer_at_least(
                    name,
                    (mtp_union_rows, topk + 1),
                    device=device,
                    dtype=torch.int32,
                )
            self._get_dsa_tensor_buffer_at_least(
                "decode_mtp_union_bitset",
                (max_decode_batch, triton.cdiv(padded_regions, 32)),
                device=device,
                dtype=torch.int32,
            )
            for name in (
                "decode_mtp_union_full_out_req_idx",
                "decode_mtp_union_full_work_q_global",
                "decode_mtp_union_full_work_q_local",
                "decode_mtp_union_full_work_q_len",
                "decode_mtp_union_full_counts",
            ):
                self._get_dsa_tensor_buffer_at_least(
                    name,
                    (max_decode_batch,),
                    device=device,
                    dtype=torch.int32,
                )
            for name in (
                "decode_mtp_union_full_phys",
                "decode_mtp_union_full_logical",
                "decode_mtp_union_full_exact_mask",
            ):
                self._get_dsa_tensor_buffer_at_least(
                    name,
                    (max_decode_batch, mtp_union_full_width),
                    device=device,
                    dtype=torch.int32,
                )
            self._get_dsa_tensor_buffer_at_least(
                "decode_mtp_union_full_causal_limits",
                (max_decode_batch, _MTP_UNION_Q_GROUP),
                device=device,
                dtype=torch.int32,
            )
            for name in (
                "decode_mtp_union_split_counts",
                "decode_mtp_union_work_q_global",
                "decode_mtp_union_work_q_input_local",
                "decode_mtp_union_work_q_output_base",
                "decode_mtp_union_work_q_len",
            ):
                self._get_dsa_tensor_buffer_at_least(
                    name,
                    (mtp_union_work,),
                    device=device,
                    dtype=torch.int32,
                )
            for name in (
                "decode_mtp_union_split_phys",
                "decode_mtp_union_split_logical",
                "decode_mtp_union_split_exact_mask",
            ):
                self._get_dsa_tensor_buffer_at_least(
                    name,
                    (mtp_union_work, mtp_union_split_width),
                    device=device,
                    dtype=torch.int32,
                )
            self._get_dsa_tensor_buffer_at_least(
                "decode_mtp_union_causal_limits",
                (mtp_union_work, _MTP_UNION_Q_GROUP),
                device=device,
                dtype=torch.int32,
            )
            self._get_dsa_tensor_buffer_at_least(
                "decode_mtp_union_partial_out",
                (
                    mtp_union_rows * _MTP_UNION_SPLITS,
                    self.num_heads,
                    self.head_size,
                ),
                device=device,
                dtype=self._decode_out_dtype or torch.bfloat16,
            )
            self._get_dsa_tensor_buffer_at_least(
                "decode_mtp_union_partial_lse",
                (mtp_union_rows * _MTP_UNION_SPLITS, self.num_heads),
                device=device,
                dtype=torch.float32,
            )
            self._get_dsa_tensor_buffer_at_least(
                "decode_mtp_union_merged_lse",
                (mtp_union_rows, self.num_heads),
                device=device,
                dtype=torch.float32,
            )
        self._get_dsa_tensor_buffer_at_least(
            "decode_kernel_block_table",
            (max_decode_rows, (num_regions + 1) // 2),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "prefill_kernel_block_table",
            (int(self.max_num_seqs), (num_regions + 1) // 2),
            device=device,
            dtype=torch.int32,
        )
        logical_max_model_len = int(self.max_model_len)
        if logical_max_model_len <= 0:
            logical_max_model_len = int(summary_cache.num_pages) * int(
                summary_cache.page_size
            )
        block_table_pages = max(
            1,
            (logical_max_model_len + int(summary_cache.page_size) - 1)
            // int(summary_cache.page_size),
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_query_start_loc",
            (max_decode_rows + 1,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_seq_lens",
            (max_decode_rows,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_request_indices",
            (max_decode_rows,),
            device=device,
            dtype=torch.int32,
        )
        self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_block_table",
            (max_decode_rows, block_table_pages),
            device=device,
            dtype=torch.int32,
        )
        self._dsa_scratch_bound = True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        dsa_proxy_query: torch.Tensor | None = None,
        dsa_proxy_key: torch.Tensor | None = None,
        dsa_proxy_weights: torch.Tensor | None = None,
        dsa_proxy_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Step5 DSA attention does not support fused output quantization."
            )
        if output is None:
            output = torch.empty_like(query)
        if attn_metadata is None:
            return output.zero_()
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        step_metadata = self._get_step_metadata(attn_metadata)
        with torch.profiler.record_function("step5_dsa.summary_update"):
            summary_cache = self._maybe_update_summary_cache(
                proxy_key=dsa_proxy_key,
                attn_metadata=attn_metadata,
                num_actual_tokens=num_actual_tokens,
                proxy_z=dsa_proxy_z,
                step_metadata=step_metadata,
            )
        with torch.profiler.record_function("step5_dsa.sparse_gqa"):
            return self._forward_sparse_gqa_cutedsl(
                query=query,
                kv_cache=_step5_sparse_kv_view(kv_cache),
                attn_metadata=attn_metadata,
                output=output,
                summary_cache=summary_cache,
                proxy_query=dsa_proxy_query,
                proxy_weights=dsa_proxy_weights,
                step_metadata=step_metadata,
            )

    def _get_or_build_runtime_layout(
        self,
        *,
        summary_cache: Step5SparseSummaryCache,
        attn_metadata: FlashAttentionMetadata,
        num_actual_tokens: int,
        padded_rows: int | None = None,
    ) -> Step5DSARuntimeLayout:
        live_tokens = int(num_actual_tokens)
        if padded_rows is None:
            padded_rows = self._round_up(live_tokens, 64)
        else:
            padded_rows = int(padded_rows)
            if padded_rows < live_tokens:
                raise ValueError(
                    "padded_rows must be greater than or equal to "
                    f"num_actual_tokens ({padded_rows} < {live_tokens})"
                )
        # Runtime layout is owned by Step5DSAStepMetadata, which is scoped to
        # one forward pass. Do not attach a shape-only cache to
        # ``attn_metadata``: CUDA-graph and piecewise paths may reuse the same
        # metadata object while updating slot_mapping/seq_lens in place, and a
        # cached layout would then retain stale physical slots or reset rows.
        device = attn_metadata.slot_mapping.device
        token_flat_slot = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_flat_slot",
            (padded_rows,),
            device=device,
            dtype=torch.int64,
        )
        token_slots = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_token_slots",
            (padded_rows,),
            device=device,
            dtype=torch.int64,
        )
        token_positions = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_token_positions",
            (padded_rows,),
            device=device,
            dtype=torch.int64,
        )
        reset_slots = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_reset_slots",
            (padded_rows,),
            device=device,
            dtype=torch.int64,
        )
        token_valid = self._get_dsa_tensor_buffer_at_least(
            "csa_layout_token_valid",
            (padded_rows,),
            device=device,
            dtype=torch.bool,
        )
        (
            token_flat_slot,
            token_slots,
            token_positions,
            reset_slots,
            token_valid,
        ) = build_decode_summary_layout_step3p5(
            attn_metadata.slot_mapping,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            num_actual_tokens=live_tokens,
            num_pages=int(summary_cache.num_pages),
            page_size=int(summary_cache.page_size),
            region_block_size=int(summary_cache.region_block_size),
            summaries_per_page=int(summary_cache.summaries_per_page),
            padded_rows=padded_rows,
            out_flat_slot=token_flat_slot,
            out_token_slots=token_slots,
            out_token_positions=token_positions,
            out_reset_slots=reset_slots,
            out_token_valid=token_valid,
        )
        layout = Step5DSARuntimeLayout(
            token_flat_slot=token_flat_slot,
            token_positions=token_positions,
            reset_slots=reset_slots,
            token_valid=token_valid,
        )
        return layout

    def _update_summary_cache_with_padded_layout(
        self,
        *,
        summary_cache: Step5SparseSummaryCache,
        layout: Step5DSARuntimeLayout,
        index_k: torch.Tensor,
        num_actual_tokens: int,
        proxy_dim: int,
        use_decode_update: bool = False,
        index_z: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        step_metadata: Step5DSAStepMetadata | None = None,
    ) -> None:
        live_tokens = int(num_actual_tokens)
        if live_tokens <= 0:
            return
        # The model-side sparse indexer already produces contiguous
        # [token, kv_head, proxy_dim] tensors. Keep the producer layout and
        # only create the fixed padded view below when the consumer requires
        # the 64-row layout.
        csa_index_k = index_k[:live_tokens, :, :proxy_dim]
        csa_index_z = index_z[:live_tokens, :, :proxy_dim]
        with torch.profiler.record_function("step5_dsa.summary.csa_update"):
            self._update_csa_summary_cache_with_padded_layout(
                summary_cache=summary_cache,
                layout=layout,
                index_k=csa_index_k,
                index_z=csa_index_z,
                live_tokens=live_tokens,
                use_decode_update=use_decode_update,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                step_metadata=step_metadata,
            )

    def _initialize_csa_summary_state(
        self,
        summary_cache: Step5SparseSummaryCache,
        *,
        capacity: int | None = None,
    ) -> None:
        capacity = int(capacity or self.max_num_seqs)
        device = summary_cache.sum_cache.device
        state_shape = (
            capacity,
            int(summary_cache.num_kv_heads),
            int(summary_cache.proxy_dim),
        )
        total_regions = int(summary_cache.num_pages) * int(
            summary_cache.summaries_per_page
        )
        summary_cache._step5_csa_active_region_ids = torch.full(
            (capacity,), -1, device=device, dtype=torch.long
        )
        summary_cache._step5_csa_active_slot_by_region = torch.full(
            (total_regions,), -1, device=device, dtype=torch.int32
        )
        summary_cache._step5_csa_allocation_success = torch.ones(
            (1,), device=device, dtype=torch.int32
        )
        summary_cache._step5_csa_numerator_cache = torch.zeros(
            state_shape, device=device, dtype=torch.float32
        )
        summary_cache._step5_csa_denominator_cache = torch.zeros(
            state_shape, device=device, dtype=torch.float32
        )
        summary_cache._step5_csa_max_cache = torch.full(
            state_shape[:2], float("-inf"), device=device, dtype=torch.float32
        )
        stage_dtype = self._decode_out_dtype or torch.bfloat16
        dynamic_cache = cast(Any, summary_cache)
        dynamic_cache._step5_csa_active_token_k = torch.zeros(
            (
                capacity,
                int(summary_cache.region_block_size),
                int(summary_cache.num_kv_heads),
                int(summary_cache.proxy_dim),
            ),
            device=device,
            dtype=stage_dtype,
        )
        dynamic_cache._step5_csa_active_token_z = torch.zeros_like(
            dynamic_cache._step5_csa_active_token_k
        )
        dynamic_cache._step5_csa_active_token_valid = torch.zeros(
            (capacity, int(summary_cache.region_block_size)),
            device=device,
            dtype=torch.uint8,
        )

    @staticmethod
    def _csa_summary_state(
        summary_cache: Step5SparseSummaryCache,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            summary_cache._step5_csa_active_region_ids,
            summary_cache._step5_csa_active_slot_by_region,
            summary_cache._step5_csa_numerator_cache,
            summary_cache._step5_csa_denominator_cache,
            summary_cache._step5_csa_max_cache,
        )

    def _update_csa_summary_cache_with_padded_layout(
        self,
        *,
        summary_cache: Step5SparseSummaryCache,
        layout: Step5DSARuntimeLayout,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        live_tokens: int,
        use_decode_update: bool,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        step_metadata: Step5DSAStepMetadata | None,
    ) -> None:
        if live_tokens <= 0:
            return
        (
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            denominator,
            max_logits,
        ) = self._csa_summary_state(summary_cache)
        allocation_success = self._begin_csa_allocation_check(summary_cache)
        padded_rows = int(layout.token_flat_slot.shape[0])
        if padded_rows > live_tokens:
            padded_index_k = self._get_dsa_tensor_buffer_at_least(
                "csa_padded_index_k",
                (padded_rows, *tuple(index_k.shape[1:])),
                device=index_k.device,
                dtype=index_k.dtype,
            )
            padded_index_z = self._get_dsa_tensor_buffer_at_least(
                "csa_padded_index_z",
                (padded_rows, *tuple(index_z.shape[1:])),
                device=index_z.device,
                dtype=index_z.dtype,
            )
            padded_index_k.zero_()
            padded_index_z.zero_()
            padded_index_k[:live_tokens].copy_(index_k[:live_tokens])
            padded_index_z[:live_tokens].copy_(index_z[:live_tokens])
            index_k = padded_index_k
            index_z = padded_index_z
        if use_decode_update:
            dynamic_cache = cast(Any, summary_cache)
            with torch.profiler.record_function(
                "step5_dsa.csa.compact_decode_update_op"
            ):
                csa_compact_decode_stage_flush_with_slots_prevalidated_sm90_gqa(
                    summary_cache.sum_cache,
                    summary_cache.count_cache,
                    summary_cache.mean_cache,
                    active_region_ids,
                    active_slot_by_region,
                    active_numerator,
                    denominator,
                    max_logits,
                    dynamic_cache._step5_csa_active_token_k,
                    dynamic_cache._step5_csa_active_token_z,
                    dynamic_cache._step5_csa_active_token_valid,
                    layout.token_flat_slot,
                    layout.reset_slots,
                    layout.token_valid,
                    layout.token_positions,
                    index_k,
                    index_z,
                    int(summary_cache.region_block_size),
                    allocation_success,
                )
            self._assert_csa_allocation_success(allocation_success)
            return

        if step_metadata is None:
            step_metadata = Step5DSAStepMetadata()
        task_prefix = self._get_or_build_csa_task_prefix(
            summary_cache=summary_cache,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            device=index_k.device,
            step_metadata=step_metadata,
        )
        with torch.profiler.record_function("step5_dsa.csa.compact_prefill_update_op"):
            csa_compact_prefill_update_with_slots_prevalidated_sm90_gqa(
                summary_cache.sum_cache,
                summary_cache.count_cache,
                summary_cache.mean_cache,
                active_region_ids,
                active_slot_by_region,
                active_numerator,
                denominator,
                max_logits,
                layout.token_flat_slot,
                layout.reset_slots,
                layout.token_valid,
                layout.token_positions,
                query_start_loc,
                seq_lens,
                task_prefix,
                index_k,
                index_z,
                int(summary_cache.region_block_size),
                allocation_success,
            )
        self._assert_csa_allocation_success(allocation_success)

    def _get_or_build_csa_task_prefix(
        self,
        *,
        summary_cache: Step5SparseSummaryCache,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
        step_metadata: Step5DSAStepMetadata,
    ) -> torch.Tensor:
        # ``step_metadata`` may be retained by a CUDA-graph replay.  The
        # request query lengths and sequence lengths are dynamic contents, so
        # retaining the first replay's prefix scan silently assigns later
        # tokens to the wrong CSA update task.  Reuse the fixed buffer but
        # refresh its contents for every invocation.
        task_prefix = self._get_dsa_tensor_buffer_at_least(
            "csa_prefill_task_prefix",
            (int(query_start_loc.shape[0]),),
            device=device,
            dtype=torch.int32,
        )
        q_lens = query_start_loc[1:] - query_start_loc[:-1]
        region_size = int(summary_cache.region_block_size)
        task_counts = (seq_lens[: q_lens.shape[0]] - q_lens) % region_size
        task_counts = (task_counts + q_lens + region_size - 1) // region_size
        task_counts = torch.where(q_lens > 0, task_counts, 0)
        task_prefix.zero_()
        torch.cumsum(task_counts, dim=0, out=task_prefix[1:])
        step_metadata.csa_task_prefix = task_prefix
        return task_prefix

    def _maybe_update_summary_cache(
        self,
        *,
        proxy_key: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        num_actual_tokens: int,
        proxy_z: torch.Tensor,
        step_metadata: Step5DSAStepMetadata,
    ) -> Step5SparseSummaryCache:
        index_k = proxy_key
        proxy_dim = int(index_k.shape[-1])
        plan: Step5KernelPlan = cast(Any, attn_metadata).step5_kernel_plan
        use_decode_update = plan.use_flattened_decode
        summary_cache = self._summary_cache
        assert summary_cache is not None, (
            "Step5 sparse summary cache must be bound during KV cache "
            "initialization before attention forward."
        )
        if step_metadata.runtime_layout is None:
            with torch.profiler.record_function("step5_dsa.summary.build_layout"):
                step_metadata.runtime_layout = self._get_or_build_runtime_layout(
                    summary_cache=summary_cache,
                    attn_metadata=attn_metadata,
                    num_actual_tokens=num_actual_tokens,
                )
        layout = step_metadata.runtime_layout
        if self._mtp is None:
            with torch.profiler.record_function("step5_dsa.summary.update_kernel"):
                self._update_summary_cache_with_padded_layout(
                    summary_cache=summary_cache,
                    layout=layout,
                    index_k=index_k,
                    num_actual_tokens=num_actual_tokens,
                    proxy_dim=proxy_dim,
                    use_decode_update=use_decode_update,
                    index_z=proxy_z,
                    query_start_loc=attn_metadata.query_start_loc,
                    seq_lens=attn_metadata.seq_lens,
                    step_metadata=step_metadata,
                )
        else:
            self._mtp.update(
                summary_cache=summary_cache,
                attn_metadata=attn_metadata,
                layout=layout,
                index_k=index_k,
                index_z=proxy_z,
                num_actual_tokens=num_actual_tokens,
                use_decode_update=use_decode_update,
                step_metadata=step_metadata,
            )
        return summary_cache

    def _sparse_gqa_prefill_kernel_block_table(
        self,
        *,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        padded_regions: int,
        device: torch.device,
        scratch_name: str = "kernel_block_table",
    ) -> torch.Tensor:
        kernel_page_size = 16
        vllm_block_size = int(kv_cache.shape[2])
        if vllm_block_size <= 0 or vllm_block_size % kernel_page_size != 0:
            raise RuntimeError(
                "Step5 DSA sparse_gqa requires KV block size to be a "
                f"positive multiple of {kernel_page_size}, got "
                f"{vllm_block_size}."
            )
        kernel_pages_per_vllm_block = vllm_block_size // kernel_page_size
        pages_needed = (int(padded_regions) + 1) // 2
        kernel_block_table = self._get_dsa_tensor_buffer_at_least(
            scratch_name,
            (int(block_table.shape[0]), int(pages_needed)),
            device=device,
            dtype=torch.int32,
        )
        total = int(kernel_block_table.numel())
        _step5_expand_block_table_kernel[(triton.cdiv(total, 1024),)](
            block_table,
            kernel_block_table,
            int(block_table.shape[0]),
            int(block_table.shape[1]),
            int(kernel_pages_per_vllm_block),
            int(pages_needed),
            int(kv_cache.shape[1]),
            int(block_table.stride(0)),
            int(block_table.stride(1)),
            int(kernel_block_table.stride(0)),
            int(kernel_block_table.stride(1)),
            VALIDATE_PHYSICAL_PAGE=True,
            BLOCK=1024,
        )
        return kernel_block_table

    def _sparse_gqa_kv_cache_for_kernel(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_page_size = 16
        vllm_block_size = int(kv_cache.shape[2])
        if vllm_block_size <= 0 or vllm_block_size % kernel_page_size != 0:
            raise RuntimeError(
                "Step5 DSA sparse_gqa requires KV block size to be a "
                "positive multiple of 16, got "
                f"{vllm_block_size}."
            )
        key_cache, value_cache = kv_cache.unbind(0)
        if vllm_block_size == kernel_page_size:
            return key_cache, value_cache
        kernel_pages_per_vllm_block = vllm_block_size // kernel_page_size
        kernel_cache_shape = (
            int(kv_cache.shape[1]) * kernel_pages_per_vllm_block,
            kernel_page_size,
            int(kv_cache.shape[3]),
            int(kv_cache.shape[4]),
        )
        return key_cache.view(kernel_cache_shape), value_cache.view(kernel_cache_shape)

    @staticmethod
    def _validate_sparse_gqa_block_table_capacity(
        *, max_seq_len: int, block_table: torch.Tensor, page_size: int
    ) -> None:
        capacity = int(block_table.shape[1]) * int(page_size)
        if max_seq_len > capacity:
            raise RuntimeError(
                "Step5 DSA logical sequence length exceeds block-table capacity: "
                f"max_seq_len={max_seq_len}, capacity={capacity}."
            )

    def _supports_mtp_union_exact_mask(self) -> bool:
        mtp = getattr(self, "_mtp", None)
        return bool(
            mtp is not None
            and int(mtp.num_speculative_tokens) == 3
            and int(getattr(self, "num_heads", 0)) == 16
            and int(getattr(self, "num_kv_heads", 0)) == 1
        )

    def _uses_mtp_union_exact_mask(self, plan: Step5KernelPlan) -> bool:
        """Return whether this decode prefix is uniformly grouped as q4."""
        return bool(
            self._supports_mtp_union_exact_mask()
            and plan.num_decode_requests > 0
            and plan.has_verifier
            and plan.all_decode_q4_or_padding
        )

    @staticmethod
    def _decode_batch_logits_layout(
        num_reqs: int,
        *,
        use_mtp_union_exact_mask: bool,
    ) -> tuple[int, int]:
        q_len = _MTP_UNION_Q_GROUP if use_mtp_union_exact_mask else 1
        if num_reqs % q_len != 0:
            raise RuntimeError(
                "Step5 grouped decode logits received incomplete query "
                f"groups: token_rows={num_reqs}, q_len={q_len}."
            )
        return num_reqs // q_len, q_len

    @staticmethod
    def _decode_batch_logits_group_table_indices(
        group_query_start_loc: torch.Tensor | None,
        *,
        group_reqs: int,
    ) -> torch.Tensor:
        if (
            group_query_start_loc is None
            or int(group_query_start_loc.shape[0]) != group_reqs + 1
        ):
            raise RuntimeError(
                "Step5 grouped decode logits received inconsistent "
                f"request metadata: group_reqs={group_reqs}."
            )
        table_indices = group_query_start_loc[:-1]
        if not table_indices.is_contiguous():
            raise RuntimeError(
                "Step5 grouped decode logits requires contiguous request "
                "start metadata."
            )
        return table_indices

    def _forward_sparse_gqa_cutedsl(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        summary_cache: Step5SparseSummaryCache,
        proxy_query: torch.Tensor,
        proxy_weights: torch.Tensor,
        step_metadata: Step5DSAStepMetadata,
    ) -> torch.Tensor:
        query_start_loc = attn_metadata.query_start_loc
        metadata_num_reqs = int(query_start_loc.shape[0]) - 1
        num_reqs = int(getattr(attn_metadata, "num_actual_reqs", metadata_num_reqs))
        num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", 0))
        plan: Step5KernelPlan = cast(Any, attn_metadata).step5_kernel_plan
        use_mtp_union_exact_mask = self._uses_mtp_union_exact_mask(plan)
        valid_requests = getattr(attn_metadata, "dsa_valid_requests", None)
        valid_tokens = getattr(attn_metadata, "dsa_valid_tokens", None)
        if _step5_dsa_is_cuda_graph_capturing() and (
            valid_requests is None or valid_tokens is None
        ):
            raise RuntimeError(
                "Step5 sparse decode CUDA Graph requires device-resident "
                "valid_requests and valid_tokens metadata"
            )

        with torch.profiler.record_function("step5_dsa.sparse_gqa.kv_cache_for_kernel"):
            key_cache, value_cache = self._sparse_gqa_kv_cache_for_kernel(kv_cache)

        region_block_size = int(self.sparse_region_block_size)
        max_seq_len = int(attn_metadata.max_seq_len)
        if max_seq_len <= 0:
            raise RuntimeError(
                "Step5 DSA requires a positive logical max_seq_len in "
                "attention metadata; refusing to derive it from physical KV "
                f"cache capacity {tuple(kv_cache.shape)}."
            )
        self._validate_sparse_gqa_block_table_capacity(
            max_seq_len=max_seq_len,
            block_table=attn_metadata.block_table,
            page_size=int(kv_cache.shape[2]),
        )
        actual_num_regions = (max_seq_len + region_block_size - 1) // region_block_size

        output.zero_()

        if plan.use_flattened_decode:
            with torch.profiler.record_function("step5_dsa.sparse_gqa.decode_q_le4"):
                if use_mtp_union_exact_mask:
                    step_metadata.decode_group_query_start_loc = query_start_loc[
                        : num_reqs + 1
                    ]
                # The step metadata object can survive CUDA-graph replay.
                # Request lengths and block-table contents change even when
                # the flattened shape stays constant, so refresh the
                # graph-stable buffers on every execution.
                step_metadata.decode_flattened = self._flatten_decode_q_le4_metadata(
                    query_start_loc=query_start_loc,
                    seq_lens=attn_metadata.seq_lens[:metadata_num_reqs],
                    block_table=attn_metadata.block_table[:metadata_num_reqs],
                    num_tokens=num_actual_tokens,
                )
                (
                    flat_query_start_loc,
                    flat_seq_lens,
                    flat_block_table,
                    flat_request_indices,
                ) = step_metadata.decode_flattened
                step_metadata.decode_request_indices = flat_request_indices
                step_metadata.decode_table_indices = flat_query_start_loc[:-1]
                return self._forward_sparse_gqa_cutedsl_decode(
                    query=query[:num_actual_tokens],
                    kv_cache=kv_cache,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    seq_lens=flat_seq_lens,
                    block_table=flat_block_table,
                    query_start_loc=flat_query_start_loc,
                    slot_mapping=attn_metadata.slot_mapping[:num_actual_tokens],
                    output=output,
                    summary_cache=summary_cache,
                    proxy_query=proxy_query[:num_actual_tokens],
                    proxy_weights=proxy_weights[:num_actual_tokens],
                    actual_num_regions=actual_num_regions,
                    step_metadata=step_metadata,
                    valid_requests=valid_requests,
                    valid_tokens=valid_tokens,
                    use_mtp_union_exact_mask=use_mtp_union_exact_mask,
                )

        num_decode_reqs = plan.num_decode_requests
        num_decode_tokens = plan.num_decode_tokens
        if num_decode_reqs > 0:
            with torch.profiler.record_function("step5_dsa.sparse_gqa.mixed_decode"):
                if plan.has_verifier:
                    if use_mtp_union_exact_mask:
                        step_metadata.decode_group_query_start_loc = query_start_loc[
                            : num_decode_reqs + 1
                        ]
                    step_metadata.decode_flattened = (
                        self._flatten_decode_q_le4_metadata(
                            query_start_loc=query_start_loc[: num_decode_reqs + 1],
                            seq_lens=attn_metadata.seq_lens[:num_decode_reqs],
                            block_table=attn_metadata.block_table[:num_decode_reqs],
                            num_tokens=num_decode_tokens,
                        )
                    )
                    (
                        decode_query_start_loc,
                        decode_seq_lens,
                        decode_block_table,
                        flat_request_indices,
                    ) = step_metadata.decode_flattened
                    step_metadata.decode_request_indices = flat_request_indices
                    step_metadata.decode_table_indices = decode_query_start_loc[:-1]
                else:
                    decode_query_start_loc = query_start_loc[: num_decode_reqs + 1]
                    decode_seq_lens = attn_metadata.seq_lens[:num_decode_reqs]
                    decode_block_table = attn_metadata.block_table[:num_decode_reqs]
                self._forward_sparse_gqa_cutedsl_decode(
                    query=query[:num_decode_tokens],
                    kv_cache=kv_cache,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    seq_lens=decode_seq_lens,
                    block_table=decode_block_table,
                    query_start_loc=decode_query_start_loc,
                    slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                    output=output,
                    summary_cache=summary_cache,
                    proxy_query=proxy_query[:num_decode_tokens],
                    proxy_weights=proxy_weights[:num_decode_tokens],
                    actual_num_regions=actual_num_regions,
                    step_metadata=step_metadata,
                    valid_requests=valid_requests,
                    valid_tokens=valid_tokens,
                    use_mtp_union_exact_mask=use_mtp_union_exact_mask,
                )

        num_prefill_reqs = num_reqs - num_decode_reqs
        token_start = num_decode_tokens
        total_prefill_tokens = num_actual_tokens - token_start
        if num_prefill_reqs <= 0 or total_prefill_tokens <= 0:
            return output

        # The GPU attention metadata already exposes the step's logical
        # max_seq_len. Keep prefill sizing on that path; consulting
        # seq_lens_cpu here would reintroduce a host metadata side channel in
        # every layer of a mixed step.
        prefill_num_regions = actual_num_regions

        with torch.profiler.record_function("step5_dsa.sparse_gqa.prefill_mixed"):
            self._forward_sparse_gqa_prefill_tiles(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                kv_cache=kv_cache,
                output=output,
                summary_cache=summary_cache,
                proxy_query=proxy_query,
                proxy_weights=proxy_weights,
                attn_metadata=attn_metadata,
                num_decode_reqs=num_decode_reqs,
                num_decode_tokens=num_decode_tokens,
                num_prefill_reqs=num_prefill_reqs,
                total_prefill_tokens=total_prefill_tokens,
                actual_num_regions=prefill_num_regions,
                step_metadata=step_metadata,
            )
        return output

    def _sparse_gqa_decode_meta_cutedsl(
        self,
        *,
        query: torch.Tensor,
        proxy_query: torch.Tensor,
        proxy_weights: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        summary_cache: Step5SparseSummaryCache,
        summary_num_regions: int,
        live_token_slots: torch.Tensor,
        step_metadata: Step5DSAStepMetadata,
        valid_requests: torch.Tensor | None = None,
        valid_tokens: torch.Tensor | None = None,
        use_mtp_union_exact_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_reqs = int(query.shape[0])
        region_block_size = int(self.sparse_region_block_size)
        index_q = proxy_query
        num_regions = int(summary_num_regions)
        # Keep selector/sort topology structural. Short contexts constrain the
        # live candidate interval through row_starts/row_ends; changing topk
        # here would specialize a new CuTeDSL artifact for each short length.
        topk = self._sparse_gqa_topk_capacity()
        seq_lens_i32 = seq_lens
        query_start_loc_i32 = query_start_loc[: num_reqs + 1]
        if step_metadata.decode_row_ranges is None:
            row_ranges = self._get_dsa_tensor_buffer_at_least(
                "decode_meta_row_ranges",
                (3, self._round_up(int(num_reqs), 4)),
                device=query.device,
                dtype=torch.int32,
            )
            row_ranges_grid = (triton.cdiv(num_reqs, _DECODE_ROW_RANGES_BLOCK_SIZE),)
            _step5_decode_row_ranges_kernel[row_ranges_grid](
                seq_lens_i32,
                live_token_slots,
                row_ranges[0, :num_reqs],
                row_ranges[1, :num_reqs],
                row_ranges[2, :num_reqs],
                num_reqs,
                num_regions,
                REGION_BLOCK_SIZE=region_block_size,
                BLOCK=_DECODE_ROW_RANGES_BLOCK_SIZE,
                num_warps=_DECODE_ROW_RANGES_NUM_WARPS,
            )
            step_metadata.decode_row_ranges = row_ranges
        row_ranges = step_metadata.decode_row_ranges
        row_starts = row_ranges[0, :num_reqs]
        row_ends = row_ranges[1, :num_reqs]
        sanitized_seq_lens = row_ranges[2, :num_reqs]
        topk_row_starts = row_starts
        topk_row_ends = row_ends
        if valid_tokens is not None:
            topk_row_starts = self._get_dsa_tensor_buffer_at_least(
                "decode_meta_topk_row_starts",
                (num_reqs,),
                device=query.device,
                dtype=torch.int32,
            )
            topk_row_ends = self._get_dsa_tensor_buffer_at_least(
                "decode_meta_topk_row_ends",
                (num_reqs,),
                device=query.device,
                dtype=torch.int32,
            )
            _step5_decode_mask_invalid_rows_kernel[(num_reqs,)](
                row_starts,
                row_ends,
                topk_row_starts,
                topk_row_ends,
                valid_tokens,
                num_reqs,
            )
        if step_metadata.decode_request_indices is None:
            request_indices = self._get_dsa_tensor_buffer_at_least(
                "decode_meta_request_indices",
                (num_reqs,),
                device=query.device,
                dtype=torch.int32,
            )
            torch.arange(int(num_reqs), out=request_indices)
            step_metadata.decode_request_indices = request_indices
        request_indices = step_metadata.decode_request_indices
        table_indices = step_metadata.decode_table_indices
        if table_indices is None:
            table_indices = request_indices
            step_metadata.decode_table_indices = table_indices
        q_heads_per_kv = int(index_q.shape[2])
        if step_metadata.decode_paged_summary is None:
            with torch.profiler.record_function("step5_dsa.decode_meta.paged_summary"):
                step_metadata.decode_paged_summary = (
                    build_decode_paged_summary_block_table_and_valid_step3p5(
                        summary_cache=summary_cache,
                        block_table=block_table,
                        seq_lens=seq_lens_i32,
                        live_token_slots=live_token_slots,
                        num_regions=int(num_regions),
                        device=query.device,
                    )
                )
        paged_summary_block_table, decode_summary_valid = (
            step_metadata.decode_paged_summary
        )
        with torch.profiler.record_function("step5_dsa.decode_meta.weighted_logits"):
            region_scores_out = self._get_dsa_tensor_buffer_at_least(
                "sparse_logits_workspace",
                (int(num_reqs), int(num_regions)),
                device=query.device,
                dtype=torch.float32,
            )
            if q_heads_per_kv == _BATCH_DECODE_Q_HEADS_PER_KV:
                # The batch WGMMA entrypoint carries a query-length dimension
                # for multi-row speculative verification, so index_q, weights,
                # row_ends, and out are all [B, Q, ...]. The union exact-mask
                # path has already proved that every live verifier request is
                # q4; all other decode shapes retain the q1 layout.
                group_reqs, q_len = self._decode_batch_logits_layout(
                    num_reqs,
                    use_mtp_union_exact_mask=use_mtp_union_exact_mask,
                )
                wgmma_n = batch_logits_wgmma_n(q_len)
                batch_q_fp8 = self._get_dsa_tensor_buffer_at_least(
                    "decode_batch_q_fp8",
                    (group_reqs, q_len, 1, q_heads_per_kv, 256),
                    device=query.device,
                    dtype=torch.float8_e4m3fn,
                )
                # The batch WGMMA entrypoint consumes pre-quantized FP8 q.
                batch_q_fp8.copy_(
                    index_q.view(group_reqs, q_len, 1, q_heads_per_kv, 256)
                )
                if use_mtp_union_exact_mask:
                    batch_row_req_idx = self._get_dsa_tensor_buffer_at_least(
                        "decode_batch_row_req_idx",
                        (group_reqs,),
                        device=query.device,
                        dtype=torch.int32,
                    )
                    if step_metadata.decode_group_row_starts is None:
                        step_metadata.decode_group_row_starts = (
                            self._get_dsa_tensor_buffer_at_least(
                                "decode_batch_row_starts",
                                (group_reqs,),
                                device=query.device,
                                dtype=torch.int32,
                            )
                        )
                    group_row_starts = step_metadata.decode_group_row_starts
                    batch_row_table_idx = self._decode_batch_logits_group_table_indices(
                        step_metadata.decode_group_query_start_loc,
                        group_reqs=group_reqs,
                    )
                else:
                    batch_row_req_idx = request_indices
                    group_row_starts = row_starts
                    batch_row_table_idx = table_indices
                batch_row_ends = self._get_dsa_tensor_buffer_at_least(
                    "decode_batch_row_ends",
                    (group_reqs, q_len),
                    device=query.device,
                    dtype=torch.int32,
                )
                # Selector row_ends is inclusive; WGMMA consumes half-open
                # region intervals.
                torch.add(
                    sanitized_seq_lens.view(group_reqs, q_len),
                    region_block_size - 1,
                    out=batch_row_ends,
                )
                torch.div(
                    batch_row_ends,
                    region_block_size,
                    rounding_mode="floor",
                    out=batch_row_ends,
                )
                q_runtime = self._get_dsa_tensor_buffer_at_least(
                    "decode_batch_q_runtime",
                    (group_reqs * wgmma_n, 256),
                    device=query.device,
                    dtype=torch.float8_e4m3fn,
                )
                kernel_weights = self._get_dsa_tensor_buffer_at_least(
                    "decode_batch_kernel_weights",
                    (group_reqs, wgmma_n),
                    device=query.device,
                    dtype=torch.float32,
                )
                region_scores = batch_logits(
                    batch_q_fp8,
                    proxy_weights.view(group_reqs, q_len, 1, q_heads_per_kv),
                    summary_cache.mean_cache,
                    paged_summary_block_table,
                    batch_row_req_idx,
                    group_row_starts,
                    batch_row_ends,
                    row_table_idx=batch_row_table_idx,
                    q_runtime=q_runtime,
                    kernel_weights=kernel_weights,
                    out=region_scores_out.view(group_reqs, q_len, num_regions),
                    valid_requests=valid_requests,
                    valid_tokens=valid_tokens,
                )
            else:
                # The model's q_heads_per_kv=8 path is still served by the
                # established mean-warp operator; the new WGMMA entrypoint
                # only accepts four query heads per KV head.
                region_scores = (
                    decode_weighted_relu_logits_sum_paged_mean_warp_sm90_steptron_gqa(
                        index_q,
                        proxy_weights,
                        summary_cache.mean_cache,
                        paged_summary_block_table,
                        out=region_scores_out,
                        row_req_idx=request_indices,
                        row_table_idx=table_indices,
                        valid_requests=valid_requests,
                        valid_tokens=valid_tokens,
                    )
                )
            region_scores = region_scores.view(num_reqs, num_regions)

        padded_regions = self._sparse_gqa_region_bucket(num_regions)
        with torch.profiler.record_function("step5_dsa.decode_meta.prepare_topk"):
            prepared_region_scores = self._get_dsa_tensor_buffer_at_least(
                "decode_meta_region_scores_prepared",
                (int(num_reqs), int(padded_regions)),
                device=query.device,
                dtype=region_scores.dtype,
            )
            block_n = 256
            _step5_decode_prepare_topk_scores_kernel[
                (int(num_reqs), triton.cdiv(int(padded_regions), block_n))
            ](
                region_scores,
                decode_summary_valid,
                row_starts,
                row_ends,
                prepared_region_scores,
                int(num_regions),
                int(padded_regions),
                HAS_SUMMARY_VALID=True,
                BLOCK_N=block_n,
            )
            region_scores = prepared_region_scores
        if step_metadata.decode_kernel_block_table is None:
            with torch.profiler.record_function(
                "step5_dsa.decode_meta.kernel_block_table"
            ):
                step_metadata.decode_kernel_block_table = (
                    self._sparse_gqa_prefill_kernel_block_table(
                        block_table=block_table,
                        kv_cache=kv_cache,
                        padded_regions=padded_regions,
                        device=query.device,
                        scratch_name="decode_kernel_block_table",
                    )
                )
        kernel_block_table = step_metadata.decode_kernel_block_table
        region_counts = self._get_dsa_tensor_buffer_at_least(
            "decode_meta_region_counts",
            (int(num_reqs),),
            device=query.device,
            dtype=torch.int32,
        )
        # The metadata converter intentionally leaves invalid token rows
        # untouched during Graph replay. Clear counts first so a shorter
        # replay cannot reuse a previous row's split-KV work description.
        region_counts.zero_()
        region_packed_indices = self._get_dsa_tensor_buffer_at_least(
            "decode_meta_region_packed_indices",
            (int(num_reqs), int(topk) + 1),
            device=query.device,
            dtype=torch.int64,
        )
        with torch.profiler.record_function("step5_dsa.decode_meta.topk_selector"):
            if envs.VLLM_STEP5_DSA_FORCE_STABLE_TOPK:
                raw_topk = cutedsl_topk_selector_sm90_multi_cta(
                    region_scores,
                    topk_row_starts,
                    topk_row_ends,
                    topk=int(topk),
                    stable_sort=True,
                )
                # The selector returns logical region ids, but the converter
                # can otherwise order packed entries by their physical page.
                # Prefix-cache reuse is allowed to remap pages, so preserve a
                # logical ordering before packing the attention metadata.
                block_k = 1 << (int(topk) - 1).bit_length()
                _step5_decode_sort_logical_topk_kernel[(int(num_reqs),)](
                    raw_topk,
                    int(topk),
                    BLOCK_K=block_k,
                    num_warps=8 if block_k >= 256 else 4,
                )
                convert_region_block_topk_to_sparse_meta_step3p5(
                    raw_topk,
                    query_start_loc_i32,
                    sanitized_seq_lens,
                    topk_row_starts,
                    kernel_block_table,
                    block_size=8,
                    window=0,
                    block_counts_out=region_counts,
                    block_packed_indices_out=region_packed_indices,
                    request_indices=table_indices,
                    sort_output=False,
                    valid_rows=valid_tokens,
                )
            else:
                cutedsl_topk_selector_decode_meta_sm90_gqa(
                    region_scores,
                    topk_row_ends,
                    topk_row_starts,
                    query_start_loc_i32,
                    sanitized_seq_lens,
                    topk_row_starts,
                    kernel_block_table,
                    k=int(topk),
                    active_rows=num_reqs,
                    score_nonneg=False,
                    block_counts_out=region_counts,
                    block_packed_indices_out=region_packed_indices,
                    logical_num_regions=int(num_regions),
                    request_indices=table_indices,
                    valid_rows=valid_tokens,
                )
        return (
            region_counts,
            region_packed_indices,
            sanitized_seq_lens,
        )

    def _flatten_decode_q_le4_metadata(
        self,
        *,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand request-row metadata to one row per query token on GPU."""
        num_reqs = int(block_table.shape[0])
        num_pages = int(block_table.shape[1])
        flat_query_start_loc = self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_query_start_loc",
            (int(num_tokens) + 1,),
            device=query_start_loc.device,
            dtype=torch.int32,
        )
        torch.arange(
            int(num_tokens) + 1,
            out=flat_query_start_loc,
        )
        flat_seq_lens = self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_seq_lens",
            (int(num_tokens),),
            device=query_start_loc.device,
            dtype=torch.int32,
        )
        flat_request_indices = self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_request_indices",
            (int(num_tokens),),
            device=query_start_loc.device,
            dtype=torch.int32,
        )
        flat_block_table = self._get_dsa_tensor_buffer_at_least(
            "decode_q_le4_block_table",
            (int(num_tokens), num_pages),
            device=block_table.device,
            dtype=block_table.dtype,
        )
        num_page_tiles = triton.cdiv(num_pages, 128)
        _step5_flatten_decode_q_le4_metadata_kernel[(num_reqs, num_page_tiles)](
            query_start_loc,
            seq_lens,
            block_table,
            flat_seq_lens,
            flat_request_indices,
            flat_block_table,
            num_pages,
            int(block_table.stride(0)),
            int(block_table.stride(1)),
            int(flat_block_table.stride(0)),
            int(flat_block_table.stride(1)),
            BLOCK_PAGES=128,
            MAX_QUERY_LEN=4,
            num_warps=4,
        )
        return (
            flat_query_start_loc,
            flat_seq_lens,
            flat_block_table,
            flat_request_indices,
        )

    def _forward_sparse_gqa_cutedsl_decode(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        slot_mapping: torch.Tensor,
        output: torch.Tensor,
        summary_cache: Step5SparseSummaryCache,
        proxy_query: torch.Tensor,
        proxy_weights: torch.Tensor,
        actual_num_regions: int,
        step_metadata: Step5DSAStepMetadata,
        valid_requests: torch.Tensor | None = None,
        valid_tokens: torch.Tensor | None = None,
        use_mtp_union_exact_mask: bool = False,
    ) -> torch.Tensor:
        num_reqs = int(query.shape[0])
        if num_reqs <= 0:
            return output

        with torch.profiler.record_function("step5_dsa.decode.meta"):
            region_counts, region_packed_indices, seq_lens = (
                self._sparse_gqa_decode_meta_cutedsl(
                    query=query,
                    proxy_query=proxy_query,
                    proxy_weights=proxy_weights,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc[: num_reqs + 1],
                    block_table=block_table,
                    kv_cache=kv_cache,
                    summary_cache=summary_cache,
                    summary_num_regions=actual_num_regions,
                    live_token_slots=slot_mapping,
                    step_metadata=step_metadata,
                    valid_requests=valid_requests,
                    valid_tokens=valid_tokens,
                    use_mtp_union_exact_mask=use_mtp_union_exact_mask,
                )
            )
        output_view = output.view(-1, self.num_heads, self.head_size)
        attn_out = output_view[:num_reqs]
        with torch.profiler.record_function("step5_dsa.decode.attn_kernel"):
            if use_mtp_union_exact_mask and num_reqs % _MTP_UNION_Q_GROUP == 0:
                self._sparse_gqa_decode_attn_mtp_union_exact_mask(
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    region_counts=region_counts,
                    region_packed_indices=region_packed_indices,
                    seq_lens=seq_lens,
                    attn_out=attn_out,
                    num_reqs=num_reqs,
                    num_region_bins=self._sparse_gqa_region_bucket(
                        int(actual_num_regions)
                    ),
                )
            elif self.sparse_decode_split_max == 1:
                token_wise_flash_attn_decode_sm90_gqa_func(
                    query,
                    key_cache,
                    value_cache,
                    region_counts=region_counts,
                    region_packed_indices=region_packed_indices,
                    kv_seqlens=seq_lens,
                    out=attn_out,
                    lse=None,
                    softmax_scale=float(self.scale),
                    variable_split_max=1,
                    valid_rows=valid_tokens,
                )
            else:
                self._sparse_gqa_decode_attn_split_kv(
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    region_counts=region_counts,
                    region_packed_indices=region_packed_indices,
                    seq_lens=seq_lens,
                    attn_out=attn_out,
                    num_reqs=num_reqs,
                    step_metadata=step_metadata,
                    valid_tokens=valid_tokens,
                )
        return output

    def _sparse_gqa_decode_attn_mtp_union_exact_mask(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        region_counts: torch.Tensor,
        region_packed_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_out: torch.Tensor,
        num_reqs: int,
        num_region_bins: int,
    ) -> None:
        device = query.device
        requests = int(num_reqs) // _MTP_UNION_Q_GROUP
        work_count = requests * _MTP_UNION_SPLITS
        topk_windows = int(region_packed_indices.shape[1])
        full_union_width = _MTP_UNION_Q_GROUP * topk_windows
        split_union_width = triton.cdiv(full_union_width, _MTP_UNION_SPLITS)

        def int32_buffer(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            return self._get_dsa_tensor_buffer_at_least(
                name, shape, device=device, dtype=torch.int32
            )

        unpacked_logical = int32_buffer(
            "decode_mtp_union_unpacked_logical", (num_reqs, topk_windows)
        )
        unpacked_phys = int32_buffer(
            "decode_mtp_union_unpacked_phys", (num_reqs, topk_windows)
        )
        bitset = int32_buffer(
            "decode_mtp_union_bitset",
            (requests, triton.cdiv(num_region_bins, 32)),
        )
        full_out_req_idx = int32_buffer(
            "decode_mtp_union_full_out_req_idx", (requests,)
        )
        full_work_q_global = int32_buffer(
            "decode_mtp_union_full_work_q_global", (requests,)
        )
        full_work_q_local = int32_buffer(
            "decode_mtp_union_full_work_q_local", (requests,)
        )
        full_work_q_len = int32_buffer("decode_mtp_union_full_work_q_len", (requests,))
        full_union_counts = int32_buffer("decode_mtp_union_full_counts", (requests,))
        full_union_phys = int32_buffer(
            "decode_mtp_union_full_phys", (requests, full_union_width)
        )
        full_union_logical = int32_buffer(
            "decode_mtp_union_full_logical", (requests, full_union_width)
        )
        full_causal_limits = int32_buffer(
            "decode_mtp_union_full_causal_limits",
            (requests, _MTP_UNION_Q_GROUP),
        )
        full_exact_mask = int32_buffer(
            "decode_mtp_union_full_exact_mask", (requests, full_union_width)
        )
        split_union_counts = int32_buffer(
            "decode_mtp_union_split_counts", (work_count,)
        )
        split_union_phys = int32_buffer(
            "decode_mtp_union_split_phys", (work_count, split_union_width)
        )
        split_union_logical = int32_buffer(
            "decode_mtp_union_split_logical", (work_count, split_union_width)
        )
        split_exact_mask = int32_buffer(
            "decode_mtp_union_split_exact_mask", (work_count, split_union_width)
        )
        work_q_global = int32_buffer("decode_mtp_union_work_q_global", (work_count,))
        work_q_input_local = int32_buffer(
            "decode_mtp_union_work_q_input_local", (work_count,)
        )
        work_q_output_base = int32_buffer(
            "decode_mtp_union_work_q_output_base", (work_count,)
        )
        work_q_len = int32_buffer("decode_mtp_union_work_q_len", (work_count,))
        causal_limits = int32_buffer(
            "decode_mtp_union_causal_limits",
            (work_count, _MTP_UNION_Q_GROUP),
        )
        partial_out = self._get_dsa_tensor_buffer_at_least(
            "decode_mtp_union_partial_out",
            (num_reqs * _MTP_UNION_SPLITS, self.num_heads, self.head_size),
            device=device,
            dtype=query.dtype,
        )
        partial_lse = self._get_dsa_tensor_buffer_at_least(
            "decode_mtp_union_partial_lse",
            (num_reqs * _MTP_UNION_SPLITS, self.num_heads),
            device=device,
            dtype=torch.float32,
        )
        merged_lse = self._get_dsa_tensor_buffer_at_least(
            "decode_mtp_union_merged_lse",
            (num_reqs, self.num_heads),
            device=device,
            dtype=torch.float32,
        )

        with torch.profiler.record_function(
            "step5_dsa.decode.mtp_union_exact_mask_meta"
        ):
            build_decode_mtp_union_exact_mask_meta_sm90_gqa(
                region_counts=region_counts,
                region_packed_indices=region_packed_indices,
                kv_seq_lens=seq_lens,
                unpacked_logical=unpacked_logical,
                unpacked_phys=unpacked_phys,
                bitset=bitset,
                full_out_req_idx=full_out_req_idx,
                full_work_q_global=full_work_q_global,
                full_work_q_local=full_work_q_local,
                full_work_q_len=full_work_q_len,
                full_union_counts=full_union_counts,
                full_union_phys=full_union_phys,
                full_union_logical=full_union_logical,
                full_causal_limits=full_causal_limits,
                full_exact_mask=full_exact_mask,
                split_union_counts=split_union_counts,
                split_union_phys=split_union_phys,
                split_union_logical=split_union_logical,
                split_exact_mask=split_exact_mask,
                work_q_global=work_q_global,
                work_q_input_local=work_q_input_local,
                work_q_output_base=work_q_output_base,
                work_q_len=work_q_len,
                causal_limits=causal_limits,
                num_region_bins=int(num_region_bins),
            )
        with torch.profiler.record_function(
            "step5_dsa.decode.mtp_union_exact_mask_attn"
        ):
            token_wise_flash_attn_decode_mtp_union_exact_mask_sm90_gqa_func(
                query,
                key_cache,
                value_cache,
                union_counts=split_union_counts,
                union_phys_indices=split_union_phys,
                union_logical_indices=split_union_logical,
                exact_mask_bits=split_exact_mask,
                work_q_global=work_q_global,
                work_q_input_local=work_q_input_local,
                work_q_output_base=work_q_output_base,
                work_q_len=work_q_len,
                causal_limits=causal_limits,
                partial_out=partial_out,
                partial_lse=partial_lse,
                softmax_scale=float(self.scale),
            )
        with torch.profiler.record_function(
            "step5_dsa.decode.mtp_union_exact_mask_merge"
        ):
            merge_mtp_union_exact_mask_nat_lse_states_sm90_gqa(
                partial_out,
                partial_lse,
                attn_out,
                merged_lse,
            )

    def _sparse_gqa_decode_attn_split_kv(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        region_counts: torch.Tensor,
        region_packed_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_out: torch.Tensor,
        num_reqs: int,
        step_metadata: Step5DSAStepMetadata,
        valid_tokens: torch.Tensor | None = None,
    ) -> None:
        """Split-KV decode: kernel emits per-split partial states, then merge.

        Graph-safe: partial/merge scratch use the persistent at-least buffers so
        no allocation happens during CUDA graph capture. work_items is a
        deterministic function of num_reqs, so it is fixed per captured batch.
        """
        device = query.device
        q_heads = int(query.shape[1])
        head_dim = int(query.shape[2])
        head_dim_v = int(value_cache.shape[3])
        split_max = self.sparse_decode_split_max
        batch_invariant = bool(envs.VLLM_BATCH_INVARIANT)
        topk_raw = int(region_packed_indices.shape[-1])
        padded_topk = topk_raw
        if 1 < split_max < 16:
            # Variable split {2,4} evenly partitions the metadata window loop.
            # Dynamic16 uses proportional runtime boundaries and keeps odd
            # topk+1 metadata without padding.
            padded_topk = ((topk_raw + split_max - 1) // split_max) * split_max
        if padded_topk != topk_raw:
            padded_rpi = self._get_dsa_tensor_buffer_at_least(
                "decode_splitkv_region_packed_indices",
                (int(num_reqs), padded_topk),
                device=device,
                dtype=region_packed_indices.dtype,
            )
            padded_rpi.fill_(-1)
            padded_rpi[:, :topk_raw].copy_(region_packed_indices)
            region_packed_indices = padded_rpi
        topk_windows = padded_topk
        sm_count = getattr(self, "_sparse_gqa_decode_sm_count", None)
        if sm_count is None:
            sm_count = int(
                torch.cuda.get_device_properties(device).multi_processor_count
            )
            self._sparse_gqa_decode_sm_count = sm_count
        if step_metadata.decode_split_plan is None:
            step_metadata.decode_split_plan = (
                token_wise_flash_attn_decode_sm90_gqa_plan(
                    int(num_reqs),
                    dtype=query.dtype,
                    q_heads=q_heads,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    topk_windows=topk_windows,
                    variable_split_max=self.sparse_decode_split_max,
                    batch_invariant=batch_invariant,
                    sm_count=sm_count,
                )
            )
        n_split4, n_split2, work_items = step_metadata.decode_split_plan
        if int(work_items) == int(num_reqs):
            # Plan chose no split (batch already saturates the SMs): write the
            # final output directly, no partial states / merge needed.
            token_wise_flash_attn_decode_sm90_gqa_func(
                query,
                key_cache,
                value_cache,
                region_counts=region_counts,
                region_packed_indices=region_packed_indices,
                kv_seqlens=seq_lens,
                out=attn_out,
                lse=None,
                softmax_scale=float(self.scale),
                variable_split_max=self.sparse_decode_split_max,
                batch_invariant=batch_invariant,
                valid_rows=valid_tokens,
            )
            return
        partial_out = self._get_dsa_tensor_buffer_at_least(
            "decode_splitkv_partial_out",
            (int(work_items), q_heads, head_dim),
            device=device,
            dtype=query.dtype,
        )
        partial_lse = self._get_dsa_tensor_buffer_at_least(
            "decode_splitkv_partial_lse",
            (int(work_items), q_heads),
            device=device,
            dtype=torch.float32,
        )
        merged_lse = self._get_dsa_tensor_buffer_at_least(
            "decode_splitkv_merged_lse",
            (int(num_reqs), q_heads),
            device=device,
            dtype=torch.float32,
        )
        with torch.profiler.record_function("step5_dsa.decode.attn_split_decode"):
            token_wise_flash_attn_decode_sm90_gqa_func(
                query,
                key_cache,
                value_cache,
                region_counts=region_counts,
                region_packed_indices=region_packed_indices,
                kv_seqlens=seq_lens,
                out=partial_out,
                lse=partial_lse,
                softmax_scale=float(self.scale),
                variable_split_max=self.sparse_decode_split_max,
                batch_invariant=batch_invariant,
                valid_rows=valid_tokens,
            )
        with torch.profiler.record_function("step5_dsa.decode.attn_split_merge"):
            if split_max == 16:
                merge_dynamic_split_nat_lse_states_sm90_gqa(
                    partial_out,
                    partial_lse,
                    region_counts,
                    attn_out,
                    merged_lse,
                    batch_invariant=batch_invariant,
                )
            else:
                merge_variable_split_nat_lse_states_sm90_gqa(
                    partial_out,
                    partial_lse,
                    attn_out,
                    merged_lse,
                    n_split4=int(n_split4),
                    n_split2=int(n_split2),
                )
        return None

    def _prepare_prefill_step_metadata(
        self,
        *,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        step_metadata: Step5DSAStepMetadata,
        num_decode_reqs: int,
        num_decode_tokens: int,
        num_prefill_reqs: int,
        total_prefill_tokens: int,
        max_regions: int,
        tile_capacity: int,
        union_q_group: int,
    ) -> None:
        if step_metadata.prefill_tile_query_start_loc is not None:
            return
        if total_prefill_tokens <= 0 or num_prefill_reqs <= 0:
            return

        prefill_block_table = attn_metadata.block_table[
            num_decode_reqs : num_decode_reqs + num_prefill_reqs
        ]
        if prefill_block_table.data_ptr() % 16 != 0:
            raise RuntimeError(
                "Step5 DSA mixed prefill requires 16-byte aligned block_table rows."
            )
        step_metadata.prefill_kernel_block_table = (
            self._sparse_gqa_prefill_kernel_block_table(
                block_table=prefill_block_table,
                kv_cache=kv_cache,
                padded_regions=max_regions,
                device=attn_metadata.query_start_loc.device,
                scratch_name="prefill_kernel_block_table",
            )
        )
        num_tiles = triton.cdiv(total_prefill_tokens, tile_capacity)
        tile_union_groups = self._prefill_tile_union_group_counts(
            query_start_loc_cpu=cast(Any, attn_metadata).query_start_loc_cpu,
            num_decode_reqs=num_decode_reqs,
            num_decode_tokens=num_decode_tokens,
            num_prefill_reqs=num_prefill_reqs,
            total_prefill_tokens=total_prefill_tokens,
            tile_capacity=tile_capacity,
            q_group=union_q_group,
        )
        if len(tile_union_groups) != num_tiles:
            raise RuntimeError(
                "mixed sparse prefill union group metadata has the wrong tile count: "
                f"{len(tile_union_groups)} != {num_tiles}"
            )
        tile_query_start_loc = self._get_dsa_tensor_buffer_at_least(
            "prefill_step_tile_query_start_loc",
            (num_tiles, num_prefill_reqs + 1),
            device=attn_metadata.query_start_loc.device,
            dtype=torch.int32,
        )
        tile_seq_lens = self._get_dsa_tensor_buffer_at_least(
            "prefill_step_tile_seq_lens",
            (num_tiles, num_prefill_reqs),
            device=attn_metadata.query_start_loc.device,
            dtype=torch.int32,
        )
        row_owners = self._get_dsa_tensor_buffer_at_least(
            "prefill_step_row_owners",
            (total_prefill_tokens,),
            device=attn_metadata.query_start_loc.device,
            dtype=torch.int32,
        )
        history_lengths = self._get_dsa_tensor_buffer_at_least(
            "prefill_step_history_lengths",
            (total_prefill_tokens,),
            device=attn_metadata.query_start_loc.device,
            dtype=torch.int32,
        )
        region_block_size = int(self.sparse_region_block_size)
        for tile_idx in range(num_tiles):
            tile_start = tile_idx * tile_capacity
            tile_rows = min(tile_capacity, total_prefill_tokens - tile_start)
            _step5_prefill_tile_request_meta_kernel[(num_prefill_reqs,)](
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                tile_query_start_loc[tile_idx],
                tile_seq_lens[tile_idx],
                row_owners[tile_start : tile_start + tile_rows],
                history_lengths[tile_start : tile_start + tile_rows],
                num_prefill_reqs,
                num_decode_reqs,
                num_decode_tokens,
                tile_start,
                tile_rows,
                max_regions,
                REGION_BLOCK_SIZE=region_block_size,
                BLOCK=256,
                num_warps=4,
            )

        step_metadata.prefill_tile_query_start_loc = tile_query_start_loc
        step_metadata.prefill_tile_seq_lens = tile_seq_lens
        step_metadata.prefill_row_owners = row_owners
        step_metadata.prefill_history_lengths = history_lengths
        step_metadata.prefill_tile_capacity = int(tile_capacity)
        step_metadata.prefill_num_tiles = int(num_tiles)
        step_metadata.prefill_num_reqs = int(num_prefill_reqs)
        step_metadata.prefill_total_tokens = int(total_prefill_tokens)
        step_metadata.prefill_tile_union_groups = tile_union_groups
        self._log_csa_occupancy(num_prefill_reqs=num_prefill_reqs)

    def _log_csa_occupancy(self, *, num_prefill_reqs: int) -> None:
        """Sample CSA active-slot occupancy so shortfall can be told from leak.

        A trace that rises then flattens means the plateau is the real peak
        demand and _csa_active_region_capacity can be sized from it; monotonic
        growth means slots are never returned, in which case no multiplier
        helps.  Called from _prepare_prefill_step_metadata, which is memoized
        per step, so this samples per step rather than per layer.
        """
        global _dsa_occ_peak, _dsa_occ_step
        if _DSA_OCC_EVERY <= 0:
            return
        cache = self._summary_cache
        ids = getattr(cache, "_step5_csa_active_region_ids", None)
        if ids is None:
            return
        _dsa_occ_step += 1
        if _dsa_occ_step % _DSA_OCC_EVERY:
            return
        occ = int((ids >= 0).sum().item())
        cap = int(ids.numel())
        _dsa_occ_peak = max(_dsa_occ_peak, occ)
        print(
            f"[dsa-csa-occupancy] step={_dsa_occ_step} occ={occ}/{cap} "
            f"peak={_dsa_occ_peak} util={100.0 * occ / max(cap, 1):.1f}% "
            f"prefill_reqs={num_prefill_reqs}",
            flush=True,
        )

    def _forward_sparse_gqa_prefill_tiles(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        summary_cache: Step5SparseSummaryCache,
        proxy_query: torch.Tensor,
        proxy_weights: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        num_decode_reqs: int,
        num_decode_tokens: int,
        num_prefill_reqs: int,
        total_prefill_tokens: int,
        actual_num_regions: int,
        step_metadata: Step5DSAStepMetadata,
    ) -> None:
        region_block_size = int(self.sparse_region_block_size)
        # Keep the compile-time capacity independent of the live prompt shape.
        # Runtime seq_lens/cu_seqlens still bound logits, top-k, union metadata,
        # and attention payload work inside this declared 32K-token chunk.
        max_regions = self._sparse_gqa_region_bucket(
            max(int(self.sparse_topk), actual_num_regions)
        )
        topk = self._sparse_gqa_topk_capacity()
        tile_capacity = min(
            total_prefill_tokens,
            int(self.max_num_batched_tokens),
            self._prefill_q_tile_capacity,
        )
        # Every tile starts at an int32 offset divisible by four. This keeps
        # the starts/ends slices passed to the CuTeDSL selector 16-byte
        # aligned without a per-tile copy.
        tile_capacity = self._round_up(tile_capacity, 4)
        num_reqs = num_decode_reqs + num_prefill_reqs
        token_start = num_decode_tokens
        output_view = output.view(-1, self.num_heads, self.head_size)
        q_heads_per_kv = int(query.shape[1]) // int(key_cache.shape[2])
        union_q_group = self._sparse_gqa_union_q_group(q_heads_per_kv)

        prefill_block_table = attn_metadata.block_table[num_decode_reqs:num_reqs]
        self._prepare_prefill_step_metadata(
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            step_metadata=step_metadata,
            num_decode_reqs=num_decode_reqs,
            num_decode_tokens=num_decode_tokens,
            num_prefill_reqs=num_prefill_reqs,
            total_prefill_tokens=total_prefill_tokens,
            max_regions=max_regions,
            tile_capacity=tile_capacity,
            union_q_group=union_q_group,
        )
        kernel_block_table = step_metadata.prefill_kernel_block_table
        tile_query_start_loc = step_metadata.prefill_tile_query_start_loc
        tile_seq_lens = step_metadata.prefill_tile_seq_lens
        row_owners = step_metadata.prefill_row_owners
        history_lengths = step_metadata.prefill_history_lengths
        tile_union_groups = step_metadata.prefill_tile_union_groups
        assert (
            kernel_block_table is not None
            and tile_query_start_loc is not None
            and tile_seq_lens is not None
            and row_owners is not None
            and history_lengths is not None
            and tile_union_groups is not None
        )
        region_starts = self._get_dsa_tensor_buffer_at_least(
            "prefill_tile_region_starts",
            (tile_capacity,),
            device=query.device,
            dtype=torch.int32,
        )
        _dsa_tile_seq_lens_redzone(
            tile_seq_lens,
            where="layer-entry",
            layer=self,
            num_prefill_reqs=num_prefill_reqs,
        )
        for tile_idx in range(step_metadata.prefill_num_tiles):
            tile_start = tile_idx * tile_capacity
            tile_rows = min(tile_capacity, total_prefill_tokens - tile_start)
            tile_query_start_loc_view = tile_query_start_loc[tile_idx]
            tile_seq_lens_view = tile_seq_lens[tile_idx]
            row_owners_view = row_owners[tile_start : tile_start + tile_rows]
            history_lengths_view = history_lengths[tile_start : tile_start + tile_rows]
            # These tile buffers live in scratch shared by every DSA layer, but
            # _prepare_prefill_step_metadata only fills them on the first layer
            # of the step - the other ~91 layers just read them, so the window
            # between write and use spans the whole step.  A stray write into
            # that window used to turn tile_seq_lens into float32 bit patterns
            # (~1e9), which inflated cu_k by four orders of magnitude and sent
            # the paged-logits kernel through ~484k KV blocks, and also fed the
            # wrong query_position/history_count into the tile pack kernel (a
            # silent wrong-region bug, not just a slow one).  Recomputing from
            # the trusted sources right before use shrinks that window to zero;
            # the grid is only num_prefill_reqs so the cost is negligible.
            _step5_prefill_tile_request_meta_kernel[(num_prefill_reqs,)](
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                tile_query_start_loc_view,
                tile_seq_lens_view,
                row_owners_view,
                history_lengths_view,
                num_prefill_reqs,
                num_decode_reqs,
                num_decode_tokens,
                tile_start,
                tile_rows,
                max_regions,
                REGION_BLOCK_SIZE=region_block_size,
                BLOCK=256,
                num_warps=4,
            )

            tile_token_start = token_start + tile_start
            tile_token_end = tile_token_start + tile_rows
            index_q = proxy_query[tile_token_start:tile_token_end]
            weights = proxy_weights[tile_token_start:tile_token_end]
            logits = self._get_dsa_tensor_buffer_at_least(
                "sparse_logits_workspace",
                (tile_rows, max_regions),
                device=query.device,
                dtype=torch.float32,
            )
            with torch.profiler.record_function("step5_dsa.prefill_mixed.mean_logits"):
                q_heads_per_kv = int(index_q.shape[2])
                block_q = 128 // q_heads_per_kv
                q_capacity = self._round_up(
                    tile_rows + num_prefill_reqs * (block_q - 1),
                    block_q,
                )
                max_request_q_capacity = self._round_up(tile_rows, block_q)
                q_work = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_q_work",
                    (q_capacity, 1, q_heads_per_kv, 256),
                    device=query.device,
                    dtype=index_q.dtype,
                )
                weights_work = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_weights_work",
                    (q_capacity, 1, q_heads_per_kv),
                    device=query.device,
                    dtype=weights.dtype,
                )
                q_runtime = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_q_runtime",
                    (q_capacity * q_heads_per_kv, 256),
                    device=query.device,
                    dtype=torch.float8_e4m3fn,
                )
                kernel_weights = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_kernel_weights",
                    (q_capacity, q_heads_per_kv),
                    device=query.device,
                    dtype=torch.float32,
                )
                logits_work = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_out_work",
                    (q_capacity, max_regions),
                    device=query.device,
                    dtype=torch.float32,
                )
                cu_q = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_cu_q",
                    (num_prefill_reqs + 1,),
                    device=query.device,
                    dtype=torch.int32,
                )
                cu_k = self._get_dsa_tensor_buffer_at_least(
                    "prefill_paged_logits_cu_k",
                    (num_prefill_reqs + 1,),
                    device=query.device,
                    dtype=torch.int32,
                )
                _step5_prefill_batched_logits_cu_kernel[(1,)](
                    tile_query_start_loc_view,
                    tile_seq_lens_view,
                    cu_q,
                    cu_k,
                    num_prefill_reqs,
                    region_block_size,
                    block_q,
                    num_warps=1,
                )
                _step5_prefill_batched_logits_pack_kernel[
                    (num_prefill_reqs, q_heads_per_kv, max_request_q_capacity)
                ](
                    index_q,
                    weights,
                    tile_query_start_loc_view,
                    cu_q,
                    q_work,
                    weights_work,
                    num_prefill_reqs,
                    tile_rows,
                    int(index_q.stride(0)),
                    int(index_q.stride(2)),
                    int(index_q.stride(3)),
                    int(q_work.stride(0)),
                    int(q_work.stride(2)),
                    int(q_work.stride(3)),
                    int(weights.stride(0)),
                    int(weights.stride(2)),
                    int(weights_work.stride(0)),
                    int(weights_work.stride(2)),
                    Q_HEADS=q_heads_per_kv,
                    HEAD_DIM=256,
                    BLOCK_D=256,
                    num_warps=4,
                )
                prefill_paged_weighted_relu_logits_sm90_steptron_gqa(
                    q_work,
                    weights_work,
                    summary_cache.mean_cache,
                    prefill_block_table,
                    cu_q,
                    cu_k,
                    q_runtime=q_runtime,
                    kernel_weights=kernel_weights,
                    out=logits_work,
                )
                _step5_prefill_batched_logits_scatter_kernel[
                    (tile_rows, triton.cdiv(max_regions, 256))
                ](
                    logits_work,
                    row_owners_view,
                    tile_query_start_loc_view,
                    cu_q,
                    logits,
                    tile_rows,
                    max_regions,
                    int(logits_work.stride(0)),
                    int(logits_work.stride(1)),
                    int(logits.stride(0)),
                    int(logits.stride(1)),
                    BLOCK_R=256,
                    num_warps=4,
                )

            raw_topk = self._get_dsa_tensor_buffer_at_least(
                "sparse_topk_indices",
                (tile_rows, topk),
                device=query.device,
                dtype=torch.int32,
            )
            with torch.profiler.record_function("step5_dsa.prefill_mixed.topk"):
                cutedsl_topk_selector_sm90_multi_cta(
                    logits,
                    region_starts[:tile_rows],
                    history_lengths_view,
                    topk=topk,
                    out_idx=raw_topk,
                    stable_sort=envs.VLLM_STEP5_DSA_FORCE_STABLE_TOPK,
                )

            region_counts = self._get_dsa_tensor_buffer_at_least(
                "prefill_tile_region_counts",
                (tile_rows,),
                device=query.device,
                dtype=torch.int32,
            )
            packed_regions = self._get_dsa_tensor_buffer_at_least(
                "prefill_region_workspace",
                (tile_rows, topk + 1),
                device=query.device,
                dtype=torch.int32,
            )
            region_phys_indices = self._get_dsa_tensor_buffer_at_least(
                "prefill_region_phys_indices_workspace",
                (tile_rows, topk + 1),
                device=query.device,
                dtype=torch.int32,
            )
            region_indices = self._get_dsa_tensor_buffer_at_least(
                "prefill_region_indices_workspace",
                (tile_rows, topk + 1),
                device=query.device,
                dtype=torch.int32,
            )
            block_k = 1 << (topk - 1).bit_length()
            _step5_prefill_tile_pack_kernel[(tile_rows,)](
                raw_topk,
                row_owners_view,
                tile_query_start_loc_view,
                tile_seq_lens_view,
                kernel_block_table,
                region_counts,
                packed_regions,
                region_phys_indices,
                region_indices,
                topk,
                int(kernel_block_table.shape[1]),
                int(kernel_block_table.stride(0)),
                REGION_BLOCK_SIZE=region_block_size,
                REGIONS_PER_KERNEL_PAGE=16 // region_block_size,
                VALID_SHIFT=_PREFILL_REGION_VALID_SHIFT,
                BLOCK_K=block_k,
                num_warps=8 if block_k >= 256 else 4,
            )

            with torch.profiler.record_function("step5_dsa.prefill_mixed.attn"):
                num_region_bins = 1 << (max(max_regions, 1) - 1).bit_length()
                max_union_windows = min(
                    union_q_group * (topk + 1),
                    num_region_bins,
                )
                bit_words = (num_region_bins + 31) // 32
                union_groups = tile_union_groups[tile_idx]
                union_out_req_idx = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_out_req_idx",
                    (union_groups,),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_work_q_global = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_work_q_global",
                    (union_groups,),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_work_q_local = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_work_q_local",
                    (union_groups,),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_work_q_len = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_work_q_len",
                    (union_groups,),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_counts = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_counts",
                    (union_groups,),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_phys = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_phys",
                    (union_groups, max_union_windows),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_logical = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_logical",
                    (union_groups, max_union_windows),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_exact_mask = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_exact_mask",
                    (union_groups, max_union_windows),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_causal_limits = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_causal_limits",
                    (union_groups, union_q_group),
                    device=query.device,
                    dtype=torch.int32,
                )
                union_bitset = self._get_dsa_tensor_buffer_at_least(
                    "prefill_union_bitset",
                    (union_groups, bit_words),
                    device=query.device,
                    dtype=torch.int32,
                )
                _step5_prefill_union_group_ranges_kernel[(union_groups,)](
                    tile_query_start_loc_view,
                    tile_seq_lens_view,
                    union_work_q_global,
                    union_work_q_local,
                    union_work_q_len,
                    num_prefill_reqs,
                    union_groups,
                    Q_GROUP=union_q_group,
                    num_warps=1,
                )
                (
                    _,
                    union_work_q_global,
                    union_work_q_local,
                    union_work_q_len,
                    union_counts,
                    union_phys,
                    union_causal_limits,
                    union_logical,
                    union_exact_mask,
                ) = build_grouped_union_sparse_work_queue_gqa(
                    total_q=tile_rows,
                    total_groups=union_groups,
                    q_group=union_q_group,
                    region_counts=region_counts,
                    region_phys_indices=region_phys_indices,
                    region_indices=region_indices,
                    max_union_windows=max_union_windows,
                    num_region_bins=max_regions,
                    return_exact_mask=True,
                    out_req_idx=union_out_req_idx,
                    work_q_global=union_work_q_global,
                    work_q_local=union_work_q_local,
                    work_q_len=union_work_q_len,
                    union_counts=union_counts,
                    union_phys=union_phys,
                    union_logical=union_logical,
                    causal_limits=union_causal_limits,
                    exact_mask_bits=union_exact_mask,
                    bitset=union_bitset,
                )
                token_wise_flash_attn_prefill_union_sm90_gqa_func(
                    query[tile_token_start:tile_token_end],
                    key_cache,
                    value_cache,
                    union_counts=union_counts,
                    union_phys_indices=union_phys,
                    union_logical_indices=union_logical,
                    exact_mask_bits=union_exact_mask,
                    work_q_global=union_work_q_global,
                    work_q_local=union_work_q_local,
                    work_q_len=union_work_q_len,
                    causal_limits=union_causal_limits,
                    out=output_view[tile_token_start:tile_token_end],
                    lse=None,
                    softmax_scale=float(self.scale),
                )
            _dsa_tile_seq_lens_redzone(
                tile_seq_lens,
                where=f"after-tile-{tile_idx}",
                layer=self,
                num_prefill_reqs=num_prefill_reqs,
            )


# Both attention types require token-major pages. Layer-major storage lets
# DSA subdivide a scheduler block into 16-token pages as a zero-copy view.
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
        return True

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
        # All DSA kernels in this backend are SM90 CuTeDSL kernels.
        return capability == DeviceCapability(9, 0)

    @staticmethod
    def get_attention_warmup_num_tokens() -> int:
        # A query length above four selects the sparse prefill path instead of
        # the q_len<=4 decode path already covered by CUDA graph capture.
        return 16

    @staticmethod
    def get_attention_warmup_decode_query_len() -> int:
        # The decode path uses persistent DSA state and must be eagerly
        # exercised before the first real request.  The runner raises this to
        # its configured MTP query length when speculative decoding is active.
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
