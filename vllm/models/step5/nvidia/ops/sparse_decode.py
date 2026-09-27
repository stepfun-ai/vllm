# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sparse decode addresses, packed masks and optional KV gathering."""

import torch

from vllm.triton_utils import tl, triton

from .sparse_prefill_pages import view_sparse_kv_pages


@triton.jit(do_not_specialize=["BATCH"])
def _prepare_sparse_decode(
    Cache,
    Regions,
    Table,
    Positions,
    Active,
    Mask,
    Indices,
    K,
    V,
    KVIndptr,
    KScale,
    VScale,
    Query,
    SplitQuery,
    QUERY_ROW: tl.constexpr,
    QUERY_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    FP8_CACHE: tl.constexpr,
    COMPACT: tl.constexpr,
    SPLITS: tl.constexpr,
    BATCH,
    BLOCK_B: tl.constexpr,
    REGION_COUNT: tl.constexpr,
    REGION_STRIDE: tl.constexpr,
    REGION_COL_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    ACTIVE_STRIDE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_DIM_STRIDE: tl.constexpr,
    CACHE_GROUP_STRIDE: tl.constexpr,
    GROUPS: tl.constexpr,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    PAGED: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(1)
    # Replicate the query for each split request, removing a separate copy.
    if SPLITS > 1:  # noqa: SIM102 - constexpr guard
        if tl.program_id(0) < SPLITS:
            offset = tl.arange(0, BLOCK_Q)
            value = tl.load(
                Query
                + row.to(tl.int64) * QUERY_ROW
                + (offset // DIM) * QUERY_HEAD
                + offset % DIM,
                offset < HEADS * DIM,
            )
            tl.store(
                SplitQuery
                + (row.to(tl.int64) * SPLITS + tl.program_id(0)) * HEADS * DIM
                + offset,
                value,
                offset < HEADS * DIM,
            )
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    output_region = row.to(tl.int64) * REGION_COUNT + index
    compact_count = REGION_COUNT
    if COMPACT:
        # Internal decode contract: sorted selection is left-padded with -1;
        # it includes all history when fewer than K completed regions exist.
        # Drop multiples of 256 tokens, preserving FA2 tile alignment.
        rows = tl.arange(0, BLOCK_B)
        pos = tl.load(Positions + rows * POSITION_STRIDE, rows < BATCH, other=0)
        historical = tl.minimum(tl.maximum(pos >> 3, 0), REGION_COUNT - 1)
        drops = ((REGION_COUNT - 1 - historical) // 32) * 32
        counts = REGION_COUNT - drops
        prefix = tl.sum(tl.where(rows < row, counts, 0), 0)
        compact_count = tl.sum(tl.where(rows == row, counts, 0), 0)
        drop = REGION_COUNT - compact_count
        output_region = prefix.to(tl.int64) + index
        if tl.program_id(0) == 0:
            # Each split is a fixed region range of the compacted row; splits
            # past the kept regions become empty FA2 requests.
            for split in tl.static_range(SPLITS):
                kept = tl.minimum(split * (REGION_COUNT // SPLITS), compact_count)
                tl.store(KVIndptr + row * SPLITS + split, (prefix + kept) * 8)
            if row == BATCH - 1:
                tl.store(KVIndptr + BATCH * SPLITS, (prefix + compact_count) * 8)
        index += drop
    if not COMPACT or tl.program_id(0) < compact_count:
        region = tl.load(
            Regions + row * REGION_STRIDE + index * REGION_COL_STRIDE,
            index < REGION_COUNT,
            other=-1,
        )
        active = tl.load(Active + row * ACTIVE_STRIDE)
        position = tl.load(Positions + row * POSITION_STRIDE)
        # Each region contributes one little-endian byte to FlashInfer's mask.
        length = tl.minimum(tl.maximum(position - region * 8 + 1, 0), 8)
        bits = tl.where((region >= 0) & active, (1 << length) - 1, 0)
        output = (
            row.to(tl.int64) * REGION_COUNT
            + tl.program_id(0) * TILE
            + tl.arange(0, TILE)
        )
        tl.store(Mask + output, bits, index < REGION_COUNT)

        if PAGED:
            selected = tl.maximum(region, 0)
            physical = tl.load(
                Table
                + row // ROWS_PER_REQUEST * TABLE_STRIDE
                + selected // (BLOCK_SIZE // 8) * TABLE_COL_STRIDE,
                index < REGION_COUNT,
                other=0,
            )
            page = physical * (
                CACHE_PAGE_STRIDE // (8 * CACHE_TOKEN_STRIDE)
            ) + selected % (BLOCK_SIZE // 8)
            tl.store(Indices + output, page, index < REGION_COUNT)
        else:
            # One CTA gathers both halves of one region, with no arithmetic on KV.
            region_start = tl.sum(region, 0) * 8
            token = tl.maximum(region_start + tl.arange(0, 8), 0)
            physical = tl.load(
                Table
                + row // ROWS_PER_REQUEST * TABLE_STRIDE
                + token // BLOCK_SIZE * TABLE_COL_STRIDE,
            ).to(tl.int64)
            dim = tl.arange(0, BLOCK_D)
            # Grouped rows are (request, KV group) pairs in request-major order.
            address = (
                physical[:, None] * CACHE_PAGE_STRIDE
                + (token % BLOCK_SIZE)[:, None] * CACHE_TOKEN_STRIDE
                + dim[None, :] * CACHE_DIM_STRIDE
                + (row % GROUPS) * CACHE_GROUP_STRIDE
            )
            key = tl.load(Cache + address, dim[None, :] < DIM, other=0.0)
            value = tl.load(
                Cache + address + DIM * CACHE_DIM_STRIDE,
                dim[None, :] < DIM,
                other=0.0,
            )
            if FP8_CACHE:
                key = key.to(tl.float32) * tl.load(KScale)
                value = value.to(tl.float32) * tl.load(VScale)
            destination = (output_region * 8 + tl.arange(0, 8))[:, None] * DIM + dim[
                None, :
            ]
            tl.store(K + destination, key, dim[None, :] < DIM)
            tl.store(V + destination, value, dim[None, :] < DIM)


def prepare_sparse_decode(
    cache: torch.Tensor,
    regions: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    active: torch.Tensor,
    mask: torch.Tensor,
    indices: torch.Tensor | None,
    kv_indptr=None,
    *,
    output_dtype=None,
    k_scale=None,
    v_scale=None,
    splits=1,
    query=None,
    split_query=None,
    query_len=1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare fixed masks and KV, optionally trimming whole padding tiles.

    With ``splits > 1``, also copies ``query`` into ``split_query`` once per
    split request. A cache with ``G > 1`` KV groups treats row ``r`` as KV group
    ``r % G``; rows ``r // (G * query_len)`` share a ``block_table`` row.
    ``positions`` and ``active`` are per row.
    """
    groups = cache.shape[1]
    rows_per_request = groups * query_len
    assert rows_per_request == 1 or (
        indices is None and regions.shape[0] % rows_per_request == 0
    )
    assert splits == 1 or (
        query is not None
        and split_query is not None
        and query.stride(2) == 1
        and regions.shape[1] >= splits
    )
    fp8_cache = cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    if fp8_cache:
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FP8 gather requires a model output dtype")
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 gather requires K/V scales")
        cache = cache.view(torch.float8_e4m3fn)
    batch, count = regions.shape
    dim = cache.shape[-1] // 2
    paged = indices is not None
    if paged:
        key, value = view_sparse_kv_pages(cache)
    else:
        key = torch.empty(
            (batch * count * 8, 1, dim),
            dtype=output_dtype or cache.dtype,
            device=cache.device,
        )
        value = torch.empty_like(key)
    tile = 128 if paged else 1
    _prepare_sparse_decode[(triton.cdiv(count, tile), batch)](
        cache,
        regions,
        block_table,
        positions,
        active,
        mask,
        indices,
        key,
        value,
        kv_indptr,
        k_scale,
        v_scale,
        query,
        split_query,
        query.stride(0) if splits > 1 else 0,
        query.stride(1) if splits > 1 else 0,
        query.shape[1] if splits > 1 else 1,
        triton.next_power_of_2(query.shape[1] * query.shape[2]) if splits > 1 else 1,
        fp8_cache,
        kv_indptr is not None,
        splits,
        batch,
        triton.next_power_of_2(batch),
        count,
        regions.stride(0),
        regions.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        positions.stride(0),
        active.stride(0),
        cache.stride(0),
        cache.stride(2),
        cache.stride(3),
        cache.stride(1),
        groups,
        rows_per_request,
        cache.shape[2],
        dim,
        paged,
        tile,
        triton.next_power_of_2(dim),
    )
    return key, value


@triton.jit
def _zero_inactive_decode(
    Output,
    Active,
    ROW_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr,
    ACTIVE_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if not tl.load(Active + row * ACTIVE_STRIDE):
        offset = tl.arange(0, BLOCK)
        tl.store(
            Output
            + row.to(tl.int64) * ROW_STRIDE
            + (offset // DIM) * HEAD_STRIDE
            + (offset % DIM) * DIM_STRIDE,
            0,
            offset < HEADS * DIM,
        )


def zero_inactive_decode(output: torch.Tensor, active: torch.Tensor) -> None:
    """All-masked FA2 output can still be NaN when padding KV contains NaNs."""
    _zero_inactive_decode[(output.shape[0],)](
        output,
        active,
        *output.stride(),
        active.stride(0),
        output.shape[1],
        output.shape[2],
        triton.next_power_of_2(output.shape[1] * output.shape[2]),
    )


@triton.jit
def _merge_split_decode(
    Partial,
    Lse,
    Output,
    Active,
    CLEAR_INACTIVE: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    OUTPUT_HEAD: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.arange(0, BLOCK_S)
    dim = tl.arange(0, BLOCK_D)
    partial_row = row * SPLITS + split
    # FlashInfer reports base-2 LSE; empty or fully masked splits are -inf.
    lse = tl.load(Lse + partial_row * HEADS + head, split < SPLITS, other=-float("inf"))
    maximum = tl.max(lse, 0)
    weight = tl.where(lse > -float("inf"), tl.exp2(lse - maximum), 0.0)
    weight = weight / tl.maximum(tl.sum(weight, 0), 1e-30)
    partial = tl.load(
        Partial + (partial_row[:, None] * HEADS + head) * DIM + dim[None, :],
        (split[:, None] < SPLITS) & (dim[None, :] < DIM),
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(tl.where(weight[:, None] > 0, partial * weight[:, None], 0.0), 0)
    if CLEAR_INACTIVE:
        merged = tl.where(tl.load(Active + row) != 0, merged, 0.0)
    tl.store(
        Output + row.to(tl.int64) * OUTPUT_ROW + head * OUTPUT_HEAD + dim,
        merged.to(Output.dtype.element_ty),
        dim < DIM,
    )


def merge_split_decode(
    partial: torch.Tensor,
    lse: torch.Tensor,
    output: torch.Tensor,
    splits: int,
    active: torch.Tensor | None = None,
) -> None:
    """Merge split-KV decode partials; FlashInfer's merge lacks head_dim 192.

    Rows with ``active`` false are written as zeros when ``active`` is given.
    """
    rows, heads, dim = output.shape
    assert output.stride(2) == 1
    assert active is None or active.stride(0) == 1
    _merge_split_decode[(rows, heads)](
        partial,
        lse,
        output,
        active,
        active is not None,
        output.stride(0),
        output.stride(1),
        splits,
        triton.next_power_of_2(splits),
        heads,
        dim,
        triton.next_power_of_2(dim),
    )
