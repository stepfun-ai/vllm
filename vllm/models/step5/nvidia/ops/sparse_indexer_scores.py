# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 indexer scores directly from paged summary keys."""

import torch

from vllm.triton_utils import tl, triton

from .sparse_indexer import DecodeIndexerMetadata


@triton.jit
def _direct_scores(
    Q,
    W,
    Mean,
    Table,
    Slots,
    SeqLens,
    Scores,
    Active,
    Positions,
    Current,
    Lengths,
    TopKLengths,
    Q_ROW: tl.constexpr,
    W_ROW: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    REGIONS_PER_BLOCK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCORE_ROW: tl.constexpr,
    HEADS: tl.constexpr,
    TILE: tl.constexpr,
    WORKERS: tl.constexpr,
):
    worker = tl.program_id(0)
    row = tl.program_id(1)
    active = tl.load(Slots + row * SLOT_STRIDE) >= 0
    seq = tl.load(SeqLens + row * SEQ_STRIDE).to(tl.int64)
    position = tl.where(active, seq - 1, 0)
    current = position >> 3
    length = current + 1
    if worker == 0:
        tl.store(Active + row, active)
        tl.store(Positions + row, position)
        tl.store(Current + row, current)
        tl.store(Lengths + row, length)
        tl.store(TopKLengths + row, current)
    live_tiles = tl.cdiv(tl.minimum(tl.maximum(length, 0), CAPACITY).to(tl.int32), TILE)
    for tile in range(worker, live_tiles, WORKERS):
        region = tile * TILE + tl.arange(0, TILE)
        col = region // REGIONS_PER_BLOCK
        physical = tl.load(Table + row * TABLE_ROW + col, col < TABLE_COLS, other=-1)
        valid = (region < CAPACITY) & (physical >= 0) & (physical < CACHE_PAGES)
        d = tl.arange(0, 256)
        h = tl.arange(0, 16)
        raw = tl.load(
            Mean
            + (physical.to(tl.int64) * REGIONS_PER_BLOCK + region % REGIONS_PER_BLOCK)[
                :, None
            ]
            * 256
            + d[None, :],
            valid[:, None],
            other=0,
        )
        key = raw.to(tl.float8e4nv, bitcast=True)
        query = tl.load(
            Q.to(tl.pointer_type(tl.uint8))
            + row * Q_ROW
            + h[None, :] * 256
            + d[:, None],
            h[None, :] < HEADS,
            other=0,
        )
        query = query.to(tl.float8e4nv, bitcast=True)
        dot = tl.dot(key, query, out_dtype=tl.float32, max_num_imprecise_acc=0)
        dot = tl.maximum(dot, 0.0)
        w = tl.load(W + row * W_ROW + h, h < HEADS, other=0)
        # FI's FP32 epilogue has four independent FMA chains followed by
        # left-associated addition. Keep this order, not a tree reduction.
        sums = tl.full((TILE, 4), 0, tl.float32)
        lane = tl.arange(0, 4)
        for group in tl.static_range(HEADS // 4):
            index = group * 4 + lane
            a = tl.gather(dot, tl.broadcast_to(index[None, :], (TILE, 4)), 1)
            weights = tl.gather(w, index, 0)
            sums = tl.fma(a, weights[None, :], sums)
        a = tl.gather(sums, tl.full((TILE, 1), 0, tl.int32), 1).reshape((TILE,))
        b = tl.gather(sums, tl.full((TILE, 1), 1, tl.int32), 1).reshape((TILE,))
        c = tl.gather(sums, tl.full((TILE, 1), 2, tl.int32), 1).reshape((TILE,))
        e = tl.gather(sums, tl.full((TILE, 1), 3, tl.int32), 1).reshape((TILE,))
        result = ((a + b) + c) + e
        tl.store(Scores + row * SCORE_ROW + region, result, region < CAPACITY)


def decode_summary_scores(
    query: torch.Tensor,
    weights: torch.Tensor,
    mean: torch.Tensor,
    block_table: torch.Tensor,
    slots: torch.Tensor,
    seq_lens: torch.Tensor,
    capacity: int,
    *,
    num_sms: int,
) -> tuple[torch.Tensor, DecodeIndexerMetadata]:
    """Read raw FP8 summaries without repacking keys or allocating unit scales.

    Only the 128-region rounded live prefix is written. Consumers must use
    lengths/topk_lengths to ignore the unspecified score tail.
    """
    batch = query.shape[0]
    scores = torch.empty(
        (batch, triton.cdiv(capacity, 256) * 256),
        device=query.device,
        dtype=torch.float32,
    )
    active = torch.empty(batch, device=query.device, dtype=torch.bool)
    positions = torch.empty(batch, device=query.device, dtype=torch.int64)
    current = torch.empty_like(positions)
    lengths = torch.empty(batch, device=query.device, dtype=torch.int32)
    topk_lengths = torch.empty_like(lengths)
    # Bound idle CTAs for short contexts; each worker walks its live tiles.
    workers = min(triton.cdiv(capacity, 128), triton.cdiv(2 * num_sms, batch))
    _direct_scores[(workers, batch)](
        query,
        weights,
        mean,
        block_table,
        slots,
        seq_lens,
        scores,
        active,
        positions,
        current,
        lengths,
        topk_lengths,
        query.stride(0),
        weights.stride(0),
        block_table.shape[1],
        block_table.stride(0),
        slots.stride(0),
        seq_lens.stride(0),
        mean.shape[0],
        mean.shape[1],
        capacity,
        scores.stride(0),
        8,
        128,
        workers,
        num_warps=4,
        enable_fp_fusion=False,
    )
    metadata = DecodeIndexerMetadata(
        active, positions, current, lengths, block_table, topk_lengths
    )
    return scores[:, :capacity], metadata
