# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused sparse decode addresses, packed masks and optional KV gathering."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
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
    COMPACT: tl.constexpr,
    BATCH: tl.constexpr,
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
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    PAGED: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(1)
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
            tl.store(KVIndptr + row, prefix * 8)
            if row == BATCH - 1:
                tl.store(KVIndptr + BATCH, (prefix + compact_count) * 8)
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
                + row * TABLE_STRIDE
                + selected // (BLOCK_SIZE // 8) * TABLE_COL_STRIDE,
                index < REGION_COUNT,
                other=0,
            )
            page = physical * (BLOCK_SIZE // 8) + selected % (BLOCK_SIZE // 8)
            tl.store(Indices + output, page, index < REGION_COUNT)
        else:
            # One CTA gathers both halves of one region, with no arithmetic on KV.
            region_start = tl.sum(region, 0) * 8
            token = tl.maximum(region_start + tl.arange(0, 8), 0)
            physical = tl.load(
                Table + row * TABLE_STRIDE + token // BLOCK_SIZE * TABLE_COL_STRIDE,
            ).to(tl.int64)
            dim = tl.arange(0, BLOCK_D)
            address = (
                physical[:, None] * CACHE_PAGE_STRIDE
                + (token % BLOCK_SIZE)[:, None] * CACHE_TOKEN_STRIDE
                + dim[None, :] * CACHE_DIM_STRIDE
            )
            key = tl.load(Cache + address, dim[None, :] < DIM, other=0)
            value = tl.load(
                Cache + address + DIM * CACHE_DIM_STRIDE,
                dim[None, :] < DIM,
                other=0,
            )
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare fixed masks and KV, optionally trimming whole padding tiles."""
    batch, count = regions.shape
    dim = cache.shape[-1] // 2
    paged = indices is not None
    if paged:
        key, value = cache.view(-1, 8, 1, 2 * dim).split(dim, -1)
    else:
        key = torch.empty(
            (batch * count * 8, 1, dim), dtype=cache.dtype, device=cache.device
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
        kv_indptr is not None,
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
