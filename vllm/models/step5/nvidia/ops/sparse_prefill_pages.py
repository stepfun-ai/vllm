# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact eight-token pages for sparse prefill attention."""

import torch

from vllm.triton_utils import tl, triton

from .sparse_indexer import PagedPrefillMetadata


@triton.jit
def _compact_prefill_pages(
    Regions,
    Table,
    Owners,
    Indptr,
    Pages,
    Query,
    ContiguousQuery,
    QUERY_ROW: tl.constexpr,
    QUERY_HEAD: tl.constexpr,
    QUERY_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    COPY_QUERY: tl.constexpr,
    REGION_COUNT: tl.constexpr,
    REGION_ROW: tl.constexpr,
    REGION_COL: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    OWNER_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(1)
    col = tl.program_id(0) * TILE + tl.arange(0, TILE)
    start = tl.load(Indptr + row)
    end = tl.load(Indptr + row + 1)
    count = end - start
    # The prefill reference removes all left padding before attention.
    region = tl.load(
        Regions + row * REGION_ROW + (REGION_COUNT - count + col) * REGION_COL,
        col < count,
        other=0,
    )
    owner = tl.load(Owners + row * OWNER_STRIDE)
    physical = tl.load(
        Table + owner * TABLE_ROW + region // (BLOCK_SIZE // 8) * TABLE_COL,
        col < count,
        other=0,
    )
    page = physical * (BLOCK_SIZE // 8) + region % (BLOCK_SIZE // 8)
    tl.store(Pages + start + col, page, col < count)
    if COPY_QUERY:
        q_col = tl.program_id(0) * 256 + tl.arange(0, 256)
        value = tl.load(
            Query
            + row * QUERY_ROW
            + (q_col // HEAD_DIM) * QUERY_HEAD
            + (q_col % HEAD_DIM) * QUERY_DIM,
            q_col < QUERY_WIDTH,
            other=0,
        )
        tl.store(
            ContiguousQuery + row * QUERY_WIDTH + q_col,
            value,
            q_col < QUERY_WIDTH,
        )


def fill_prefill_pages(
    regions, table, owners, indptr, pages, block_size, query, *, contiguous_query=None
):
    rows, count = regions.shape
    copy_query = contiguous_query is not None or not query.is_contiguous()
    if contiguous_query is None:
        contiguous_query = (
            torch.empty_like(query, memory_format=torch.contiguous_format)
            if copy_query
            else query
        )
    width = query.shape[1] * query.shape[2]
    tiles = max(triton.cdiv(count, 128), triton.cdiv(width, 256) if copy_query else 1)
    _compact_prefill_pages[(tiles, rows)](
        regions,
        table,
        owners,
        indptr,
        pages,
        query.view(torch.int16),
        contiguous_query.view(torch.int16),
        *query.stride(),
        query.shape[2],
        width,
        copy_query,
        count,
        *regions.stride(),
        *table.stride(),
        owners.stride(0),
        block_size,
        128,
        num_warps=4,
    )
    return contiguous_query


def compact_page_metadata(positions, region_count):
    if region_count < 1 or any(position < 0 for position in positions):
        raise ValueError("Prefill requires nonnegative positions and a current region")
    counts = [min(position // 8 + 1, region_count) for position in positions]
    indptr = [0]
    for count in counts:
        indptr.append(indptr[-1] + count)
    return (
        torch.tensor(indptr, dtype=torch.int32, device="cpu"),
        torch.tensor([p % 8 + 1 for p in positions], dtype=torch.int32, device="cpu"),
    )


class CompactPagedPrefill:
    """One native attention plan for a full prefill step, shared by DSA layers."""

    def __init__(self, workspace):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        self._plan_event = torch.cuda.Event()

    def plan(self, query, positions, region_count, scale, block_size):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 prefill planning requires a breakable graph")
        rows, heads, dim = query.shape
        assert rows == len(positions) and rows > 0
        indptr, last_page_len = compact_page_metadata(positions, region_count)
        self.pages = torch.empty(
            int(indptr[-1]), device=query.device, dtype=torch.int32
        )
        self.indptr = indptr.to(query.device)
        self.region_count = region_count
        self.block_size = block_size
        # The native planner reuses pinned memory for asynchronous HtoD copies.
        self._plan_event.synchronize()
        self.wrapper.plan(
            torch.arange(rows + 1, dtype=torch.int32, device="cpu"),
            indptr,
            self.pages,
            last_page_len,
            heads,
            1,
            dim,
            8,
            head_dim_vo=dim,
            causal=False,
            sm_scale=scale,
            q_data_type=query.dtype,
            kv_data_type=query.dtype,
            disable_split_kv=True,
        )
        self._plan_event.record(torch.cuda.current_stream(query.device))

        self.query_shape = query.shape
        self.query_buffer = (
            torch.empty_like(query, memory_format=torch.contiguous_format)
            if not query.is_contiguous()
            else None
        )

    def page_metadata(self, table, owners, start=0, end=None):
        if self.query_buffer is not None:
            return None
        end = self.query_shape[0] if end is None else end
        return PagedPrefillMetadata(
            owners[start:end],
            table,
            self.pages,
            self.indptr[start : end + 1],
            self.block_size,
        )

    def prepare_chunk(self, start, query, cache, regions, table, owners):
        end = start + query.shape[0]
        assert 0 <= start < end <= self.query_shape[0]
        assert query.shape[1:] == self.query_shape[1:]
        assert regions.shape == (query.shape[0], self.region_count)
        destination = (
            self.query_buffer[start:end] if self.query_buffer is not None else None
        )
        fill_prefill_pages(
            regions,
            table,
            owners,
            self.indptr[start : end + 1],
            self.pages,
            cache.shape[2],
            query,
            contiguous_query=destination,
        )

    def finish(self, query, cache, out):
        assert query.shape == self.query_shape
        assert (self.query_buffer is not None) == (not query.is_contiguous())
        dim = query.shape[-1]
        key, value = cache.view(-1, 8, 1, 2 * dim).split(dim, -1)
        q = self.query_buffer if self.query_buffer is not None else query
        self.wrapper.run(q, (key, value), out=out)
