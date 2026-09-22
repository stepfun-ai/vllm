# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 sparse summary cache side storage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

logger = init_logger(__name__)


@dataclass
class Step5DSAScratchWorkspace:
    """Scratch storage shared by all DSA attention layers in one model."""

    tensor_buffers_by_engine: dict[int, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    # A single dependency token per virtual engine is shared by every DSA
    # attention call.  The token is intentionally kept separate from the
    # scratch-buffer accounting: it is a graph-ordering primitive, not a
    # producer workspace, and counting it as a scratch buffer would make the
    # public reset count depend on this implementation detail.
    order_tokens_by_engine: dict[int, torch.Tensor] = field(default_factory=dict)
    memory_profiled: bool = False
    allocations_locked: bool = False

    def __post_init__(self) -> None:
        # Tests and a few compatibility paths construct a profiled workspace
        # from already allocated buffers.  Materialize the matching ordering
        # tokens at construction time so a locked workspace never grows on its
        # first real forward.
        for virtual_engine, buffers in self.tensor_buffers_by_engine.items():
            if virtual_engine in self.order_tokens_by_engine:
                continue
            device = next(
                (buffer.device for buffer in buffers.values()),
                None,
            )
            if device is not None:
                self.order_tokens_by_engine[int(virtual_engine)] = torch.zeros(
                    (1,), device=device, dtype=torch.int64
                )

    def get_order_token(
        self,
        virtual_engine: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the preallocated cross-layer ordering token.

        All DSA layers use a common scratch pool.  TorchDynamo cannot
        reliably infer that distinct shaped views returned by
        ``_get_dsa_tensor_buffer_at_least`` alias the same backing storage,
        so the outer DSA custom op carries this explicit token and mutates it.
        Do not silently allocate it on the first real forward after profiling;
        that would invalidate compiled/CUDA-graph addresses.
        """
        virtual_engine = int(virtual_engine)
        token = self.order_tokens_by_engine.get(virtual_engine)
        if (
            token is None
            or token.device != device
            or token.dtype is not torch.int64
            or tuple(token.shape) != (1,)
        ):
            if self.allocations_locked:
                raise RuntimeError(
                    "Step5 DSA ordering-token capacity was exceeded after bind: "
                    f"virtual_engine={virtual_engine}, device={device}."
                )
            token = torch.zeros((1,), device=device, dtype=torch.int64)
            self.order_tokens_by_engine[virtual_engine] = token
        return token

    @torch.inference_mode()
    def reset_runtime_state(self) -> int:
        """Restore shared DSA scratch without changing captured addresses."""
        sentinel_minus_one = {
            "csa_prefill_scratch_region_ids",
            "csa_prefill_scratch_row_map",
        }
        num_buffers = 0
        for buffers in self.tensor_buffers_by_engine.values():
            for name, buffer in buffers.items():
                if name in sentinel_minus_one:
                    buffer.fill_(-1)
                else:
                    buffer.zero_()
                num_buffers += 1
        for token in self.order_tokens_by_engine.values():
            token.zero_()
        return num_buffers


def step5_sparse_summary_cache_bytes_per_block(
    *,
    block_size: int,
    region_block_size: int,
    proxy_dim: int,
    num_kv_heads: int = 1,
    sum_dtype: torch.dtype | None = None,
    count_dtype: torch.dtype | None = None,
) -> int:
    proxy_dim = int(proxy_dim)
    num_kv_heads = int(num_kv_heads)
    if proxy_dim <= 0 or num_kv_heads <= 0:
        raise ValueError(
            "Step5 sparse summary cache requires positive proxy_dim and "
            f"num_kv_heads, got proxy_dim={proxy_dim}, "
            f"num_kv_heads={num_kv_heads}."
        )
    fragments = Step5SparseSummaryCacheConfig.max_region_fragments_per_block(
        block_size=block_size, region_block_size=region_block_size
    )
    # Only FP8 mean values belong to the prefix-cache backing. FP32 sum/count
    # tensors are producer workspaces allocated separately.
    del sum_dtype, count_dtype
    return int(fragments * num_kv_heads * proxy_dim)


_RESET_BLOCKS_DYNAMIC_ARGS = (
    "num_block_ids",
    "num_pages",
    "BLOCKS_PER_SCHEDULER_BLOCK",
)
_RESET_BLOCKS_ALIGNMENT_DYNAMIC_ARGS = (
    "block_ids",
    "mean_cache",
    "active_region_ids",
    "active_slot_by_region",
    "active_numerator",
    "denominator",
    "max_logits",
)


@triton.jit(
    do_not_specialize=_RESET_BLOCKS_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _RESET_BLOCKS_DYNAMIC_ARGS + _RESET_BLOCKS_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_summary_reset_blocks_kernel(
    block_ids,
    mean_cache,
    active_region_ids,
    active_slot_by_region,
    active_numerator,
    denominator,
    max_logits,
    num_block_ids,
    num_pages,
    MEAN_STRIDE_PAGE: tl.constexpr,
    MEAN_STRIDE_REGION: tl.constexpr,
    MEAN_STRIDE_HEAD: tl.constexpr,
    NUMERATOR_STRIDE_SLOT: tl.constexpr,
    NUMERATOR_STRIDE_HEAD: tl.constexpr,
    DENOMINATOR_STRIDE_SLOT: tl.constexpr,
    DENOMINATOR_STRIDE_HEAD: tl.constexpr,
    MAX_STRIDE_SLOT: tl.constexpr,
    BLOCKS_PER_SCHEDULER_BLOCK,
    SUMMARIES_PER_PAGE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    ACTIVE_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CLEAR_MEAN: tl.constexpr = True,
) -> None:
    block_index = tl.program_id(0)
    sub_block = tl.program_id(1)
    fragment = tl.program_id(2)
    scheduler_block = tl.load(
        block_ids + block_index,
        mask=block_index < num_block_ids,
        other=-1,
    ).to(tl.int64)
    page = scheduler_block * BLOCKS_PER_SCHEDULER_BLOCK + sub_block
    valid_page = (
        (block_index < num_block_ids)
        & (scheduler_block >= 0)
        & (page >= 0)
        & (page < num_pages)
    )

    offsets = tl.arange(0, BLOCK_D)
    head = offsets // PROXY_DIM
    dim = offsets - head * PROXY_DIM
    state_mask = valid_page & (offsets < NUM_KV_HEADS * PROXY_DIM)
    # Invalid scheduler block ids make page/region invalid. Masked memory ops
    # still compute addresses, so clamp physical page/region before pointer
    # arithmetic; valid_page still controls whether values are used.
    safe_page = tl.where(valid_page, page, 0)

    # The only persistent summary payload is the FP8 mean backing. Reset it
    # here with the same physical-page mask; active sum/count are scratch and
    # are owned by active slots, so they are not touched by page reset.
    mean_offsets = (
        safe_page * MEAN_STRIDE_PAGE
        + fragment * MEAN_STRIDE_REGION
        + head * MEAN_STRIDE_HEAD
        + dim
    )
    if CLEAR_MEAN:
        tl.store(
            mean_cache + mean_offsets,
            0,
            mask=state_mask,
        )

    region = page * SUMMARIES_PER_PAGE + fragment
    safe_region = tl.where(valid_page, region, 0)
    slot = tl.load(
        active_slot_by_region + safe_region,
        mask=valid_page,
        other=-1,
    ).to(tl.int64)
    slot_in_range = (slot >= 0) & (slot < ACTIVE_CAPACITY)
    # active_slot_by_region can be stale. Masked loads/stores still compute
    # addresses, so clamp slot before pointer arithmetic.
    safe_slot = tl.where(slot_in_range, slot, 0)
    owner_region = tl.load(
        active_region_ids + safe_slot,
        mask=valid_page & slot_in_range,
        other=-1,
    )
    owns_slot = valid_page & slot_in_range & (owner_region == region)
    tl.store(active_slot_by_region + safe_region, -1, mask=valid_page)
    tl.store(active_region_ids + safe_slot, -1, mask=owns_slot)

    numerator_offsets = (
        safe_slot * NUMERATOR_STRIDE_SLOT + head * NUMERATOR_STRIDE_HEAD + dim
    )
    denominator_offsets = (
        safe_slot * DENOMINATOR_STRIDE_SLOT + head * DENOMINATOR_STRIDE_HEAD + dim
    )
    tl.store(
        active_numerator + numerator_offsets,
        0.0,
        mask=owns_slot & state_mask,
    )
    tl.store(
        denominator + denominator_offsets,
        0.0,
        mask=owns_slot & state_mask,
    )
    tl.store(
        max_logits + safe_slot * MAX_STRIDE_SLOT + offsets,
        -float("inf"),
        mask=owns_slot & (offsets < NUM_KV_HEADS),
    )


_DECODE_UPDATE_DYNAMIC_ARGS = ("source_rows", "live_rows")
_DECODE_UPDATE_ALIGNMENT_DYNAMIC_ARGS = (
    "sum_cache",
    "count_cache",
    "active_region_ids",
    "active_slot_by_region",
    "allocation_success",
    "denominator",
    "max_logits",
    "flat_slot",
    "reset_slots",
    "token_valid",
    "token_positions",
    "index_k",
    "index_z",
)


@triton.jit(
    do_not_specialize=_DECODE_UPDATE_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _DECODE_UPDATE_DYNAMIC_ARGS + _DECODE_UPDATE_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_summary_cache_csa_compact_decode_update_kernel(
    sum_cache,
    count_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    denominator,
    max_logits,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    index_k,
    index_z,
    source_rows,
    live_rows,
    total_regions: tl.constexpr,
    summaries_per_page: tl.constexpr,
    num_kv_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    sum_stride_page: tl.constexpr,
    sum_stride_region: tl.constexpr,
    sum_stride_head: tl.constexpr,
    count_stride_page: tl.constexpr,
    count_stride_region: tl.constexpr,
    count_stride_head: tl.constexpr,
    denominator_stride_slot: tl.constexpr,
    denominator_stride_head: tl.constexpr,
    max_stride_slot: tl.constexpr,
    MAINTAIN_SLOT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    if row >= live_rows:
        return
    region = tl.load(flat_slot + row)
    valid = tl.load(token_valid + row).to(tl.int1)
    region_ok = (region >= 0) & (region < total_regions)
    if ~(valid & region_ok):
        return

    # Multiple decode rows may land in the same physical summary region. Only
    # the first row for a region performs the update; it folds all rows for the
    # region in token-offset order so compact CSA state is written once.
    duplicate_region = False
    for prev_row in tl.range(0, live_rows):
        if prev_row < row:
            prev_region = tl.load(flat_slot + prev_row)
            prev_valid = tl.load(token_valid + prev_row).to(tl.int1)
            duplicate_region = duplicate_region | (prev_valid & (prev_region == region))
    if duplicate_region:
        return

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < proxy_dim

    found_slot = tl.full((), active_capacity, dtype=tl.int64)
    for slot in tl.range(0, active_capacity):
        active_region = tl.load(active_region_ids + slot)
        if (active_region == region) & (found_slot == active_capacity):
            found_slot = slot.to(tl.int64)

    page = region // summaries_per_page
    fragment = region - page * summaries_per_page
    sum_offsets = (
        page * sum_stride_page
        + fragment * sum_stride_region
        + head * sum_stride_head
        + dim_offsets
    )
    count_offset = (
        page * count_stride_page
        + fragment * count_stride_region
        + head * count_stride_head
    )

    materialized = tl.load(sum_cache + sum_offsets, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    old_denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    old_max = -float("inf")
    if found_slot != active_capacity:
        denominator_offsets = (
            found_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        max_offset = found_slot * max_stride_slot + head
        old_denominator = tl.load(
            denominator + denominator_offsets, mask=dim_mask, other=0.0
        ).to(tl.float32)
        old_max = tl.load(max_logits + max_offset).to(tl.float32)

    reset = False
    complete = False
    for candidate in tl.range(0, live_rows):
        candidate_region = tl.load(flat_slot + candidate)
        candidate_valid = tl.load(token_valid + candidate).to(tl.int1)
        same_region = candidate_valid & (candidate_region == region)
        candidate_reset = tl.load(reset_slots + candidate) == region
        candidate_position = tl.load(token_positions + candidate)
        candidate_offset = (
            candidate_position
            - (candidate_position // region_block_size) * region_block_size
        )
        reset = reset | (same_region & candidate_reset)
        complete = complete | (
            same_region & (candidate_offset == region_block_size - 1)
        )

    old_denominator = tl.where(
        reset, tl.zeros((BLOCK_D,), dtype=tl.float32), old_denominator
    )
    old_max = tl.where(reset, -float("inf"), old_max)
    numerator = materialized * old_denominator
    denominator_acc = old_denominator
    max_acc = old_max

    for offset in tl.range(0, region_block_size):
        matched_row = live_rows
        for candidate in tl.range(0, live_rows):
            candidate_region = tl.load(flat_slot + candidate)
            candidate_valid = tl.load(token_valid + candidate).to(tl.int1)
            candidate_position = tl.load(token_positions + candidate)
            candidate_offset = (
                candidate_position
                - (candidate_position // region_block_size) * region_block_size
            )
            same_slot = (
                (matched_row == source_rows)
                & candidate_valid
                & (candidate_region == region)
                & (candidate_offset == offset)
            )
            matched_row = tl.where(same_slot, candidate, matched_row)
        valid_row = matched_row < live_rows
        # matched_row is live_rows when no row exists for this offset. Masked
        # loads still compute addresses, so clamp before pointer arithmetic.
        safe_matched_row = tl.where(valid_row, matched_row, 0)
        value_offsets = (
            safe_matched_row * num_kv_heads + head
        ) * proxy_dim + dim_offsets
        values = tl.load(
            index_k + value_offsets, mask=dim_mask & valid_row, other=0.0
        ).to(tl.float32)
        logits = tl.load(
            index_z + value_offsets, mask=dim_mask & valid_row, other=-float("inf")
        ).to(tl.float32)
        row_max = tl.max(logits, axis=0)
        new_max = tl.maximum(max_acc, row_max)
        old_scale = tl.where(denominator_acc > 0.0, tl.exp(max_acc - new_max), 0.0)
        weights = tl.where(valid_row, tl.exp(logits - new_max), 0.0)
        numerator = numerator * old_scale + weights * values
        denominator_acc = denominator_acc * old_scale + weights
        max_acc = new_max

    materialized_out = numerator / tl.maximum(denominator_acc, 1.0e-20)
    materialized_out = tl.where(denominator_acc > 0.0, materialized_out, 0.0)

    tl.store(sum_cache + sum_offsets, materialized_out, mask=dim_mask)
    tl.store(count_cache + count_offset, 1.0)

    if complete:
        if MAINTAIN_SLOT_MAP:
            # The reverse map is only a hint and can outlive slot reuse. Clear
            # the completed region even if the forward ownership check below
            # shows that its old slot now belongs to another region.
            tl.store(active_slot_by_region + region, -1)
        if found_slot != active_capacity:
            denominator_offsets = (
                found_slot * denominator_stride_slot
                + head * denominator_stride_head
                + dim_offsets
            )
            max_offset = found_slot * max_stride_slot + head
            tl.store(
                denominator + denominator_offsets,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=dim_mask,
            )
            tl.store(max_logits + max_offset, -float("inf"))
            tl.store(active_region_ids + found_slot, -1)
        return

    slot_to_store = found_slot.to(tl.int64)
    if slot_to_store == active_capacity:
        empty_region = tl.full((), -1, dtype=tl.int64)
        for slot in tl.range(0, active_capacity):
            if slot_to_store == active_capacity:
                old = tl.atomic_cas(active_region_ids + slot, empty_region, region)
                if (old == empty_region) | (old == region):
                    slot_to_store = slot.to(tl.int64)
        # A full active-slot table used to silently drop this region.  Keep
        # the fallback path fail-closed, matching the CuTeDSL allocator:
        # publish a device-side status that the caller checks with
        # ``torch._assert_async`` after the launch.
        if slot_to_store == active_capacity:
            tl.store(allocation_success, 0)
    # active_slot_by_region is a reverse map that can go stale (the reset/clear
    # read paths already clamp for this reason).  Testing only against the
    # sentinel is not enough: any out-of-range slot value lands past the end of
    # denominator, where the next scratch buffer lives.  Use the same range
    # check + clamp + mask idiom as the read paths, so that even if the
    # predicate were lowered wrong the address stays in bounds and the mask
    # keeps a valid slot from being polluted.
    slot_ok = (slot_to_store >= 0) & (slot_to_store < active_capacity)
    safe_slot = tl.where(slot_ok, slot_to_store, 0)
    if slot_ok:
        if MAINTAIN_SLOT_MAP:
            tl.store(active_slot_by_region + region, safe_slot.to(tl.int32))
        denominator_offsets = (
            safe_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        max_offset = safe_slot * max_stride_slot + head
        tl.store(
            denominator + denominator_offsets, denominator_acc, mask=dim_mask & slot_ok
        )
        tl.store(max_logits + max_offset, max_acc, mask=slot_ok)


_DECODE_UPDATE_FAST_ALIGNMENT_DYNAMIC_ARGS = _DECODE_UPDATE_ALIGNMENT_DYNAMIC_ARGS + (
    "active_numerator",
)


@triton.jit(
    do_not_specialize=_DECODE_UPDATE_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _DECODE_UPDATE_DYNAMIC_ARGS + _DECODE_UPDATE_FAST_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_summary_cache_csa_compact_decode_update_fast_kernel(
    sum_cache,
    count_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_numerator,
    denominator,
    max_logits,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    index_k,
    index_z,
    source_rows,
    live_rows,
    total_regions: tl.constexpr,
    summaries_per_page: tl.constexpr,
    num_kv_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    sum_stride_page: tl.constexpr,
    sum_stride_region: tl.constexpr,
    sum_stride_head: tl.constexpr,
    count_stride_page: tl.constexpr,
    count_stride_region: tl.constexpr,
    count_stride_head: tl.constexpr,
    denominator_stride_slot: tl.constexpr,
    denominator_stride_head: tl.constexpr,
    numerator_stride_slot: tl.constexpr,
    numerator_stride_head: tl.constexpr,
    max_stride_slot: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOC_PROBE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    if row >= live_rows:
        return
    region = tl.load(flat_slot + row)
    valid = tl.load(token_valid + row).to(tl.int1)
    region_ok = (region >= 0) & (region < total_regions)
    if ~(valid & region_ok):
        return

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < proxy_dim

    found_slot_i32 = tl.load(active_slot_by_region + region)
    found_slot = found_slot_i32.to(tl.int64)
    found_slot_valid = (found_slot >= 0) & (found_slot < active_capacity)
    safe_found_slot = tl.where(found_slot_valid, found_slot, 0)
    found_region = tl.load(
        active_region_ids + safe_found_slot,
        mask=found_slot_valid,
        other=-1,
    )
    found_slot = tl.where(found_region == region, found_slot, active_capacity)

    reset = tl.load(reset_slots + row) == region

    old_denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    numerator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    old_max = -float("inf")
    if found_slot != active_capacity:
        denominator_offsets = (
            found_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        numerator_offsets = (
            found_slot * numerator_stride_slot
            + head * numerator_stride_head
            + dim_offsets
        )
        max_offset = found_slot * max_stride_slot + head
        old_denominator = tl.load(
            denominator + denominator_offsets, mask=dim_mask, other=0.0
        ).to(tl.float32)
        numerator = tl.load(
            active_numerator + numerator_offsets, mask=dim_mask, other=0.0
        ).to(tl.float32)
        old_max = tl.load(max_logits + max_offset).to(tl.float32)

    old_denominator = tl.where(
        reset, tl.zeros((BLOCK_D,), dtype=tl.float32), old_denominator
    )
    numerator = tl.where(reset, tl.zeros((BLOCK_D,), dtype=tl.float32), numerator)
    old_max = tl.where(reset, -float("inf"), old_max)
    denominator_acc = old_denominator
    max_acc = old_max

    value_offsets = (row * num_kv_heads + head) * proxy_dim + dim_offsets
    values = tl.load(index_k + value_offsets, mask=dim_mask, other=0.0).to(tl.float32)
    logits = tl.load(index_z + value_offsets, mask=dim_mask, other=-float("inf")).to(
        tl.float32
    )
    row_max = tl.max(logits, axis=0)
    new_max = tl.maximum(max_acc, row_max)
    old_scale = tl.where(denominator_acc > 0.0, tl.exp(max_acc - new_max), 0.0)
    weights = tl.exp(logits - new_max)
    numerator = numerator * old_scale + weights * values
    denominator_acc = denominator_acc * old_scale + weights
    max_acc = new_max

    token_position = tl.load(token_positions + row)
    offset = token_position - (token_position // region_block_size) * region_block_size
    complete = offset == region_block_size - 1
    if complete:
        tl.store(active_slot_by_region + region, -1)
        if found_slot != active_capacity:
            denominator_offsets = (
                found_slot * denominator_stride_slot
                + head * denominator_stride_head
                + dim_offsets
            )
            numerator_offsets = (
                found_slot * numerator_stride_slot
                + head * numerator_stride_head
                + dim_offsets
            )
            max_offset = found_slot * max_stride_slot + head
            tl.store(
                active_numerator + numerator_offsets,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=dim_mask,
            )
            tl.store(
                denominator + denominator_offsets,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=dim_mask,
            )
            tl.store(max_logits + max_offset, -float("inf"))
            tl.store(active_region_ids + found_slot, -1)
        return

    slot_to_store = found_slot.to(tl.int64)
    if slot_to_store == active_capacity:
        empty_region = tl.full((), -1, dtype=tl.int64)
        # The capacity is fixed from max_num_seqs at bind time (doubled for
        # MTP). Probe that complete capacity: stopping at an unrelated fixed
        # bound can silently drop a live region even when a later slot is free.
        start = region - (region // active_capacity) * active_capacity
        for p in tl.range(0, ALLOC_PROBE):
            if slot_to_store == active_capacity:
                cand = start + p
                cand = cand - (cand // active_capacity) * active_capacity
                old = tl.atomic_cas(active_region_ids + cand, empty_region, region)
                if (old == empty_region) | (old == region):
                    slot_to_store = cand.to(tl.int64)
                    tl.store(active_slot_by_region + region, cand.to(tl.int32))
        if slot_to_store == active_capacity:
            tl.store(allocation_success, 0)
    # Same range check + clamp + mask as the non-fast kernel above.
    slot_ok = (slot_to_store >= 0) & (slot_to_store < active_capacity)
    safe_slot = tl.where(slot_ok, slot_to_store, 0)
    if slot_ok:
        denominator_offsets = (
            safe_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        numerator_offsets = (
            safe_slot * numerator_stride_slot
            + head * numerator_stride_head
            + dim_offsets
        )
        max_offset = safe_slot * max_stride_slot + head
        tl.store(
            active_numerator + numerator_offsets, numerator, mask=dim_mask & slot_ok
        )
        tl.store(
            denominator + denominator_offsets, denominator_acc, mask=dim_mask & slot_ok
        )
        tl.store(max_logits + max_offset, max_acc, mask=slot_ok)


_DECODE_CLEAR_COMPLETED_ALIGNMENT_DYNAMIC_ARGS = (
    "active_region_ids",
    "active_slot_by_region",
    "flat_slot",
    "token_valid",
    "token_positions",
)


@triton.jit(
    do_not_specialize=_DECODE_UPDATE_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _DECODE_UPDATE_DYNAMIC_ARGS + _DECODE_CLEAR_COMPLETED_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_csa_compact_decode_clear_completed_kernel(
    active_region_ids,
    active_slot_by_region,
    flat_slot,
    token_valid,
    token_positions,
    source_rows,
    live_rows,
    total_regions: tl.constexpr,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    MAINTAIN_SLOT_MAP: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    if row >= live_rows:
        return
    region = tl.load(flat_slot + row)
    valid = tl.load(token_valid + row).to(tl.int1)
    region_ok = (region >= 0) & (region < total_regions)
    if ~(valid & region_ok):
        return
    token_position = tl.load(token_positions + row)
    offset = token_position - (token_position // region_block_size) * region_block_size
    if offset != region_block_size - 1:
        return
    if MAINTAIN_SLOT_MAP:
        tl.store(active_slot_by_region + region, -1)
    for slot in tl.range(0, active_capacity):
        active_region = tl.load(active_region_ids + slot)
        if active_region == region:
            tl.store(active_region_ids + slot, -1)


# These helpers are shared by MTP and can receive offset views from an ubatch.
# Pointer alignment is therefore runtime data, not a compile-key dimension.
_CLEAR_SCRATCH_DYNAMIC_ARGS = ("scratch_rows", "total_items")
_CLEAR_SCRATCH_ALIGNMENT_DYNAMIC_ARGS = (
    "scratch_region_ids",
    "scratch_row_map",
    "scratch_reset_map",
)


@triton.jit(
    do_not_specialize=_CLEAR_SCRATCH_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _CLEAR_SCRATCH_DYNAMIC_ARGS + _CLEAR_SCRATCH_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_csa_compact_clear_scratch_kernel(
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    scratch_rows,
    region_block_size: tl.constexpr,
    total_items,
    BLOCK_N: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total_items
    tl.store(scratch_row_map + offsets, -1, mask=mask)
    row_mask = offsets < scratch_rows
    tl.store(scratch_region_ids + offsets, -1, mask=row_mask)
    tl.store(scratch_reset_map + offsets, 0, mask=row_mask)


_FILL_SCRATCH_DYNAMIC_ARGS = ("live_rows", "scratch_rows")
_FILL_SCRATCH_ALIGNMENT_DYNAMIC_ARGS = (
    "scratch_region_ids",
    "scratch_row_map",
    "scratch_reset_map",
    "flat_slot",
    "reset_slots",
    "token_valid",
    "token_positions",
)


@triton.jit(
    do_not_specialize=_FILL_SCRATCH_DYNAMIC_ARGS,
    do_not_specialize_on_alignment=(
        _FILL_SCRATCH_DYNAMIC_ARGS + _FILL_SCRATCH_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_csa_compact_fill_scratch_kernel(
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    flat_slot,
    reset_slots,
    token_valid,
    token_positions,
    live_rows,
    total_regions: tl.constexpr,
    scratch_rows,
    region_block_size: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    if row >= live_rows:
        return
    region = tl.load(flat_slot + row)
    valid = tl.load(token_valid + row).to(tl.int1)
    region_ok = (region >= 0) & (region < total_regions)
    if ~(valid & region_ok):
        return

    region_i32 = region.to(tl.int32)
    found_slot = scratch_rows
    start_slot = region_i32 % scratch_rows
    probe = 0
    while (probe < scratch_rows) & (found_slot == scratch_rows):
        slot = (start_slot + probe) % scratch_rows
        old = tl.atomic_cas(
            scratch_region_ids + slot,
            -1,
            region_i32,
            sem="relaxed",
        )
        found_slot = tl.where(
            (old == -1) | (old == region_i32),
            slot,
            found_slot,
        )
        probe += 1

    if found_slot == scratch_rows:
        return

    token_position = tl.load(token_positions + row)
    offset = token_position - (token_position // region_block_size) * region_block_size
    offset_ok = (offset >= 0) & (offset < region_block_size)
    if offset_ok:
        tl.store(scratch_row_map + found_slot * region_block_size + offset, row)
    reset = tl.load(reset_slots + row)
    if reset == region:
        tl.store(scratch_reset_map + found_slot, 1)


_ORDERED_UPDATE_ALIGNMENT_DYNAMIC_ARGS = (
    "sum_cache",
    "count_cache",
    "active_region_ids",
    "active_slot_by_region",
    "allocation_success",
    "active_numerator",
    "denominator",
    "max_logits",
    "scratch_region_ids",
    "scratch_row_map",
    "scratch_reset_map",
    "index_k",
    "index_z",
)


@triton.jit(
    do_not_specialize=("live_rows",),
    do_not_specialize_on_alignment=(
        ("live_rows",) + _ORDERED_UPDATE_ALIGNMENT_DYNAMIC_ARGS
    ),
)
def _step5_sparse_summary_cache_csa_compact_update_ordered_kernel(
    sum_cache,
    count_cache,
    active_region_ids,
    active_slot_by_region,
    allocation_success,
    active_numerator,
    denominator,
    max_logits,
    scratch_region_ids,
    scratch_row_map,
    scratch_reset_map,
    index_k,
    index_z,
    live_rows,
    total_regions: tl.constexpr,
    summaries_per_page: tl.constexpr,
    num_kv_heads: tl.constexpr,
    proxy_dim: tl.constexpr,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    sum_stride_page: tl.constexpr,
    sum_stride_region: tl.constexpr,
    sum_stride_head: tl.constexpr,
    count_stride_page: tl.constexpr,
    count_stride_region: tl.constexpr,
    count_stride_head: tl.constexpr,
    denominator_stride_slot: tl.constexpr,
    denominator_stride_head: tl.constexpr,
    numerator_stride_slot: tl.constexpr,
    numerator_stride_head: tl.constexpr,
    max_stride_slot: tl.constexpr,
    use_active_slot_map: tl.constexpr,
    maintain_slot_map: tl.constexpr,
    use_active_numerator: tl.constexpr,
    PROCESS_COMPLETE: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    group_slot = tl.program_id(0)
    head = tl.program_id(1)
    region_i32 = tl.load(scratch_region_ids + group_slot)
    if region_i32 < 0:
        return
    region = region_i32.to(tl.int64)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < proxy_dim

    has_row = False
    complete = False
    for offset in tl.range(0, region_block_size):
        row = tl.load(scratch_row_map + group_slot * region_block_size + offset)
        has_row = has_row | (row >= 0)
        if offset == region_block_size - 1:
            complete = row >= 0
    if ~has_row:
        return
    if complete != PROCESS_COMPLETE:
        return

    reset = tl.load(scratch_reset_map + group_slot) != 0
    found_slot = tl.full((), active_capacity, dtype=tl.int64)
    if use_active_slot_map:
        mapped_slot_i32 = tl.load(active_slot_by_region + region)
        mapped_slot = mapped_slot_i32.to(tl.int64)
        mapped_ok = (mapped_slot >= 0) & (mapped_slot < active_capacity)
        # Triton masked load computes the pointer regardless of the mask.
        # Clamp stale reverse-map slots before pointer arithmetic; mapped_ok
        # still discards the loaded value for invalid slots.
        safe_mapped_slot = tl.where(mapped_ok, mapped_slot, 0)
        mapped_region = tl.load(
            active_region_ids + safe_mapped_slot, mask=mapped_ok, other=-1
        )
        found_slot = tl.where(mapped_region == region, mapped_slot, active_capacity)
    else:
        for slot in tl.range(0, active_capacity):
            active_region = tl.load(active_region_ids + slot)
            if (active_region == region) & (found_slot == active_capacity):
                found_slot = slot.to(tl.int64)

    page = region // summaries_per_page
    fragment = region - page * summaries_per_page
    sum_offsets = (
        page * sum_stride_page
        + fragment * sum_stride_region
        + head * sum_stride_head
        + dim_offsets
    )
    count_offset = (
        page * count_stride_page
        + fragment * count_stride_region
        + head * count_stride_head
    )

    materialized = tl.load(sum_cache + sum_offsets, mask=dim_mask, other=0.0).to(
        tl.float32
    )
    old_denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    numerator = materialized * old_denominator
    old_max = -float("inf")
    if found_slot != active_capacity:
        denominator_offsets = (
            found_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        numerator_offsets = (
            found_slot * numerator_stride_slot
            + head * numerator_stride_head
            + dim_offsets
        )
        max_offset = found_slot * max_stride_slot + head
        old_denominator = tl.load(
            denominator + denominator_offsets, mask=dim_mask, other=0.0
        ).to(tl.float32)
        if use_active_numerator:
            numerator = tl.load(
                active_numerator + numerator_offsets, mask=dim_mask, other=0.0
            ).to(tl.float32)
        old_max = tl.load(max_logits + max_offset).to(tl.float32)

    old_denominator = tl.where(
        reset, tl.zeros((BLOCK_D,), dtype=tl.float32), old_denominator
    )
    numerator = tl.where(reset, tl.zeros((BLOCK_D,), dtype=tl.float32), numerator)
    old_max = tl.where(reset, -float("inf"), old_max)
    denominator_acc = old_denominator
    max_acc = old_max

    for offset in tl.range(0, region_block_size):
        row = tl.load(scratch_row_map + group_slot * region_block_size + offset)
        valid_row = (row >= 0) & (row < live_rows)
        # Empty scratch entries are -1. Masked loads still compute addresses,
        # so clamp before pointer arithmetic; valid_row still gates the value.
        safe_row = tl.where(valid_row, row, 0)
        value_offsets = (safe_row * num_kv_heads + head) * proxy_dim + dim_offsets
        values = tl.load(
            index_k + value_offsets, mask=dim_mask & valid_row, other=0.0
        ).to(tl.float32)
        logits = tl.load(
            index_z + value_offsets, mask=dim_mask & valid_row, other=-float("inf")
        ).to(tl.float32)
        row_max = tl.max(logits, axis=0)
        new_max = tl.maximum(max_acc, row_max)
        old_scale = tl.where(denominator_acc > 0.0, tl.exp(max_acc - new_max), 0.0)
        weights = tl.where(valid_row, tl.exp(logits - new_max), 0.0)
        numerator = numerator * old_scale + weights * values
        denominator_acc = denominator_acc * old_scale + weights
        max_acc = new_max

    materialized_out = numerator / tl.maximum(denominator_acc, 1.0e-20)
    materialized_out = tl.where(denominator_acc > 0.0, materialized_out, 0.0)
    tl.store(sum_cache + sum_offsets, materialized_out, mask=dim_mask)
    tl.store(count_cache + count_offset, 1.0)

    if complete:
        if maintain_slot_map:
            # The reverse map is only a hint and can outlive slot reuse. Clear
            # the completed region even if the forward ownership check below
            # shows that its old slot now belongs to another region.
            tl.store(active_slot_by_region + region, -1)
        if found_slot != active_capacity:
            denominator_offsets = (
                found_slot * denominator_stride_slot
                + head * denominator_stride_head
                + dim_offsets
            )
            numerator_offsets = (
                found_slot * numerator_stride_slot
                + head * numerator_stride_head
                + dim_offsets
            )
            max_offset = found_slot * max_stride_slot + head
            if use_active_numerator:
                tl.store(
                    active_numerator + numerator_offsets,
                    tl.zeros((BLOCK_D,), dtype=tl.float32),
                    mask=dim_mask,
                )
            tl.store(
                denominator + denominator_offsets,
                tl.zeros((BLOCK_D,), dtype=tl.float32),
                mask=dim_mask,
            )
            tl.store(max_logits + max_offset, -float("inf"))
            tl.store(active_region_ids + found_slot, -1)
        return

    slot_to_store = found_slot.to(tl.int64)
    if slot_to_store == active_capacity:
        empty_region = tl.full((), -1, dtype=tl.int64)
        for slot in tl.range(0, active_capacity):
            if slot_to_store == active_capacity:
                old = tl.atomic_cas(active_region_ids + slot, empty_region, region)
                if (old == empty_region) | (old == region):
                    slot_to_store = slot.to(tl.int64)
                    if maintain_slot_map:
                        tl.store(active_slot_by_region + region, slot)
        if slot_to_store == active_capacity:
            tl.store(allocation_success, 0)
    # Same range check + clamp + mask as the decode kernels above.
    slot_ok = (slot_to_store >= 0) & (slot_to_store < active_capacity)
    safe_slot = tl.where(slot_ok, slot_to_store, 0)
    if slot_ok:
        if maintain_slot_map:
            tl.store(active_slot_by_region + region, safe_slot.to(tl.int32))
        denominator_offsets = (
            safe_slot * denominator_stride_slot
            + head * denominator_stride_head
            + dim_offsets
        )
        numerator_offsets = (
            safe_slot * numerator_stride_slot
            + head * numerator_stride_head
            + dim_offsets
        )
        max_offset = safe_slot * max_stride_slot + head
        if use_active_numerator:
            tl.store(
                active_numerator + numerator_offsets, numerator, mask=dim_mask & slot_ok
            )
        tl.store(
            denominator + denominator_offsets, denominator_acc, mask=dim_mask & slot_ok
        )
        tl.store(max_logits + max_offset, max_acc, mask=slot_ok)


_CLEAR_COMPLETED_ALIGNMENT_DYNAMIC_ARGS = (
    "active_region_ids",
    "active_slot_by_region",
    "scratch_region_ids",
    "scratch_row_map",
)


@triton.jit(
    do_not_specialize_on_alignment=_CLEAR_COMPLETED_ALIGNMENT_DYNAMIC_ARGS,
)
def _step5_sparse_csa_compact_clear_completed_kernel(
    active_region_ids,
    active_slot_by_region,
    scratch_region_ids,
    scratch_row_map,
    region_block_size: tl.constexpr,
    active_capacity: tl.constexpr,
    USE_SLOT_MAP: tl.constexpr,
    MAINTAIN_SLOT_MAP: tl.constexpr,
) -> None:
    group_slot = tl.program_id(0)
    region_i32 = tl.load(scratch_region_ids + group_slot)
    if region_i32 < 0:
        return
    complete_row = tl.load(
        scratch_row_map + group_slot * region_block_size + region_block_size - 1
    )
    if complete_row < 0:
        return
    region = region_i32.to(tl.int64)
    if USE_SLOT_MAP:
        slot = tl.load(active_slot_by_region + region).to(tl.int64)
        slot_ok = (slot >= 0) & (slot < active_capacity)
        # active_slot_by_region is a reverse hint and may be stale. Clamp before
        # masked pointer arithmetic; slot_ok still controls the value.
        safe_slot = tl.where(slot_ok, slot, 0)
        cur = tl.load(active_region_ids + safe_slot, mask=slot_ok, other=-1)
        # The reverse map belongs to this completed region, so remove it even
        # if the mapped slot has already been reused by another region.  Slot
        # state is owned by active_region_ids and is only cleared on owner match.
        tl.store(active_slot_by_region + region, -1)
        if slot_ok & (cur == region):
            tl.store(active_region_ids + safe_slot, -1)
    else:
        if MAINTAIN_SLOT_MAP:
            tl.store(active_slot_by_region + region, -1)
        for slot in tl.range(0, active_capacity):
            active_region = tl.load(active_region_ids + slot)
            if active_region == region:
                tl.store(active_region_ids + slot, -1)


_CSA_FALLBACK_ALLOCATION_ERROR = (
    "Step5 DSA active-slot capacity is exhausted in the Triton fallback."
)


def _prepare_csa_fallback_allocation_status(
    allocation_success: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return a graph-safe status buffer for the Triton fallback.

    The CuTeDSL entry points receive a persistent status tensor from
    the attention implementation.  The legacy custom-op entry points predate
    that argument, so keep it optional and allocate a one-element device
    buffer when omitted.  In either case reset it before launching kernels;
    allocation failures are then published by the kernels and asserted by the
    caller without a host synchronization.
    """
    if allocation_success is None:
        allocation_success = torch.ones(
            (1,),
            device=device,
            dtype=torch.int32,
        )
    elif (
        allocation_success.dtype != torch.int32
        or allocation_success.device != device
        or tuple(allocation_success.shape) != (1,)
        or not allocation_success.is_contiguous()
    ):
        raise ValueError(
            "Step5 Triton CSA fallback requires contiguous int32 "
            "allocation_success=[1] on the cache device."
        )
    allocation_success.fill_(1)
    return allocation_success


def _step5_sparse_summary_cache_csa_compact_update_impl(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    active_slot_by_region: torch.Tensor | None = None,
    active_numerator: torch.Tensor | None = None,
    live_tokens: int = -1,
    allocation_success: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    allocation_success = _prepare_csa_fallback_allocation_status(
        allocation_success,
        device=sum_cache.device,
    )
    total_regions = int(
        active_slot_by_region.numel()
        if active_slot_by_region is not None
        else sum_cache.shape[0] * sum_cache.shape[1]
    )
    source_rows = int(index_k.shape[0])
    live_rows = source_rows if int(live_tokens) < 0 else int(live_tokens)
    region_block_size = int(region_block_size)
    if source_rows == 0:
        torch._assert_async(allocation_success, _CSA_FALLBACK_ALLOCATION_ERROR)
        return sum_cache, count_cache, denominator, max_logits
    if live_rows == 0:
        torch._assert_async(allocation_success, _CSA_FALLBACK_ALLOCATION_ERROR)
        return sum_cache, count_cache, denominator, max_logits
    block_d = triton.next_power_of_2(int(sum_cache.shape[3]))
    maintain_slot_map = active_slot_by_region is not None
    use_active_slot_map = maintain_slot_map and int(sum_cache.shape[2]) == 1
    scratch_rows = int(source_rows)
    scratch_region_ids = scratch_region_ids[:scratch_rows]
    scratch_row_map = scratch_row_map[:scratch_rows, :region_block_size]
    scratch_reset_map = scratch_reset_map[:scratch_rows]
    scratch_total = int(scratch_rows) * int(region_block_size)
    clear_block = 256
    _step5_sparse_csa_compact_clear_scratch_kernel[
        (triton.cdiv(max(scratch_total, scratch_rows), clear_block),)
    ](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        scratch_rows,
        region_block_size,
        max(scratch_total, scratch_rows),
        BLOCK_N=clear_block,
    )
    _step5_sparse_csa_compact_fill_scratch_kernel[(source_rows,)](
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        live_rows,
        total_regions,
        scratch_rows,
        region_block_size,
    )
    active_numerator_src = (
        active_numerator if active_numerator is not None else denominator
    )
    for process_complete in (True, False):
        _step5_sparse_summary_cache_csa_compact_update_ordered_kernel[
            (scratch_rows, int(sum_cache.shape[2]))
        ](
            sum_cache,
            count_cache,
            active_region_ids,
            active_slot_by_region if maintain_slot_map else active_region_ids,
            allocation_success,
            active_numerator if active_numerator is not None else denominator,
            denominator,
            max_logits,
            scratch_region_ids,
            scratch_row_map,
            scratch_reset_map,
            index_k,
            index_z,
            live_rows,
            total_regions,
            int(sum_cache.shape[1]),
            int(sum_cache.shape[2]),
            int(sum_cache.shape[3]),
            region_block_size,
            int(active_region_ids.numel()),
            int(sum_cache.stride(0)),
            int(sum_cache.stride(1)),
            int(sum_cache.stride(2)),
            int(count_cache.stride(0)),
            int(count_cache.stride(1)),
            int(count_cache.stride(2)),
            int(denominator.stride(0)),
            int(denominator.stride(1)),
            int((active_numerator_src).stride(0)),
            int((active_numerator_src).stride(1)),
            int(max_logits.stride(0)),
            use_active_slot_map=use_active_slot_map,
            maintain_slot_map=maintain_slot_map,
            use_active_numerator=active_numerator is not None,
            PROCESS_COMPLETE=process_complete,
            BLOCK_D=block_d,
        )
        # The single-head slot-map path clears both ownership tables in the
        # ordered complete pass. Launching the fallback clear afterwards is
        # redundant and, because the map is already -1, forces a full active
        # capacity scan for every completed region.
        if process_complete and not use_active_slot_map:
            _step5_sparse_csa_compact_clear_completed_kernel[(scratch_rows,)](
                active_region_ids,
                active_slot_by_region if maintain_slot_map else active_region_ids,
                scratch_region_ids,
                scratch_row_map,
                region_block_size,
                int(active_region_ids.numel()),
                USE_SLOT_MAP=use_active_slot_map,
                MAINTAIN_SLOT_MAP=maintain_slot_map,
            )
    torch._assert_async(allocation_success, _CSA_FALLBACK_ALLOCATION_ERROR)
    return sum_cache, count_cache, denominator, max_logits


def _step5_sparse_summary_cache_csa_compact_decode_update_impl(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    active_slot_by_region: torch.Tensor | None = None,
    active_numerator: torch.Tensor | None = None,
    live_tokens: int = -1,
    allocation_success: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    allocation_success = _prepare_csa_fallback_allocation_status(
        allocation_success,
        device=sum_cache.device,
    )
    total_regions = int(
        active_slot_by_region.numel()
        if active_slot_by_region is not None
        else sum_cache.shape[0] * sum_cache.shape[1]
    )
    source_rows = int(index_k.shape[0])
    live_rows = source_rows if int(live_tokens) < 0 else int(live_tokens)
    region_block_size = int(region_block_size)
    if source_rows == 0:
        torch._assert_async(allocation_success, _CSA_FALLBACK_ALLOCATION_ERROR)
        return sum_cache, count_cache, denominator, max_logits
    if live_rows == 0:
        torch._assert_async(allocation_success, _CSA_FALLBACK_ALLOCATION_ERROR)
        return sum_cache, count_cache, denominator, max_logits
    block_d = triton.next_power_of_2(int(sum_cache.shape[3]))
    maintain_slot_map = active_slot_by_region is not None
    use_fast_unique_region = (
        maintain_slot_map
        and active_numerator is not None
        and int(sum_cache.shape[2]) == 1
    )
    if use_fast_unique_region:
        _step5_sparse_summary_cache_csa_compact_decode_update_fast_kernel[
            (source_rows, int(sum_cache.shape[2]))
        ](
            sum_cache,
            count_cache,
            active_region_ids,
            active_slot_by_region,
            allocation_success,
            active_numerator,
            denominator,
            max_logits,
            flat_slot,
            reset_slots,
            token_valid,
            token_positions,
            index_k,
            index_z,
            source_rows,
            live_rows,
            total_regions,
            int(sum_cache.shape[1]),
            int(sum_cache.shape[2]),
            int(sum_cache.shape[3]),
            region_block_size,
            int(active_region_ids.numel()),
            int(sum_cache.stride(0)),
            int(sum_cache.stride(1)),
            int(sum_cache.stride(2)),
            int(count_cache.stride(0)),
            int(count_cache.stride(1)),
            int(count_cache.stride(2)),
            int(denominator.stride(0)),
            int(denominator.stride(1)),
            int(active_numerator.stride(0)),
            int(active_numerator.stride(1)),
            int(max_logits.stride(0)),
            BLOCK_D=block_d,
            ALLOC_PROBE=int(active_region_ids.numel()),
        )
    else:
        _step5_sparse_summary_cache_csa_compact_decode_update_kernel[
            (source_rows, int(sum_cache.shape[2]))
        ](
            sum_cache,
            count_cache,
            active_region_ids,
            active_slot_by_region if maintain_slot_map else active_region_ids,
            allocation_success,
            denominator,
            max_logits,
            flat_slot,
            reset_slots,
            token_valid,
            token_positions,
            index_k,
            index_z,
            source_rows,
            live_rows,
            total_regions,
            int(sum_cache.shape[1]),
            int(sum_cache.shape[2]),
            int(sum_cache.shape[3]),
            region_block_size,
            int(active_region_ids.numel()),
            int(sum_cache.stride(0)),
            int(sum_cache.stride(1)),
            int(sum_cache.stride(2)),
            int(count_cache.stride(0)),
            int(count_cache.stride(1)),
            int(count_cache.stride(2)),
            int(denominator.stride(0)),
            int(denominator.stride(1)),
            int(max_logits.stride(0)),
            MAINTAIN_SLOT_MAP=maintain_slot_map,
            BLOCK_D=block_d,
        )
        _step5_sparse_csa_compact_decode_clear_completed_kernel[(source_rows,)](
            active_region_ids,
            active_slot_by_region if maintain_slot_map else active_region_ids,
            flat_slot,
            token_valid,
            token_positions,
            source_rows,
            live_rows,
            total_regions,
            region_block_size,
            int(active_region_ids.numel()),
            MAINTAIN_SLOT_MAP=maintain_slot_map,
        )
    torch._assert_async(allocation_success, _CSA_FALLBACK_ALLOCATION_ERROR)
    return sum_cache, count_cache, denominator, max_logits


def _step5_sparse_summary_cache_csa_compact_update_with_slots_op_fake(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    live_tokens: int = -1,
    allocation_success: torch.Tensor | None = None,
) -> None:
    return None


def _step5_sparse_summary_cache_csa_compact_decode_update_with_slots_op_fake(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    live_tokens: int = -1,
    allocation_success: torch.Tensor | None = None,
) -> None:
    return None


def _step5_sparse_summary_cache_csa_compact_update_with_slots_op_impl(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    scratch_region_ids: torch.Tensor,
    scratch_row_map: torch.Tensor,
    scratch_reset_map: torch.Tensor,
    live_tokens: int = -1,
    allocation_success: torch.Tensor | None = None,
) -> None:
    _step5_sparse_summary_cache_csa_compact_update_impl(
        sum_cache,
        count_cache,
        active_region_ids,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        region_block_size,
        scratch_region_ids,
        scratch_row_map,
        scratch_reset_map,
        active_slot_by_region=active_slot_by_region,
        active_numerator=active_numerator,
        live_tokens=live_tokens,
        allocation_success=allocation_success,
    )


def _step5_sparse_summary_cache_csa_compact_decode_update_with_slots_op_impl(
    sum_cache: torch.Tensor,
    count_cache: torch.Tensor,
    active_region_ids: torch.Tensor,
    active_slot_by_region: torch.Tensor,
    active_numerator: torch.Tensor,
    denominator: torch.Tensor,
    max_logits: torch.Tensor,
    flat_slot: torch.Tensor,
    reset_slots: torch.Tensor,
    token_valid: torch.Tensor,
    token_positions: torch.Tensor,
    index_k: torch.Tensor,
    index_z: torch.Tensor,
    region_block_size: int,
    live_tokens: int = -1,
    allocation_success: torch.Tensor | None = None,
) -> None:
    _step5_sparse_summary_cache_csa_compact_decode_update_impl(
        sum_cache,
        count_cache,
        active_region_ids,
        denominator,
        max_logits,
        flat_slot,
        reset_slots,
        token_valid,
        token_positions,
        index_k,
        index_z,
        region_block_size,
        active_slot_by_region=active_slot_by_region,
        active_numerator=active_numerator,
        live_tokens=live_tokens,
        allocation_success=allocation_success,
    )


direct_register_custom_op(
    op_name="step5_sparse_summary_cache_csa_compact_update_with_slots",
    op_func=_step5_sparse_summary_cache_csa_compact_update_with_slots_op_impl,
    mutates_args=[
        "sum_cache",
        "count_cache",
        "active_region_ids",
        "active_slot_by_region",
        "active_numerator",
        "denominator",
        "max_logits",
        "scratch_region_ids",
        "scratch_row_map",
        "scratch_reset_map",
        "allocation_success",
    ],
    fake_impl=_step5_sparse_summary_cache_csa_compact_update_with_slots_op_fake,
)


direct_register_custom_op(
    op_name="step5_sparse_summary_cache_csa_compact_decode_update_with_slots",
    op_func=(_step5_sparse_summary_cache_csa_compact_decode_update_with_slots_op_impl),
    mutates_args=[
        "sum_cache",
        "count_cache",
        "active_region_ids",
        "active_slot_by_region",
        "active_numerator",
        "denominator",
        "max_logits",
        "allocation_success",
    ],
    fake_impl=(
        _step5_sparse_summary_cache_csa_compact_decode_update_with_slots_op_fake
    ),
)


@dataclass(frozen=True)
class Step5SparseSummaryCacheConfig:
    num_pages: int
    page_size: int
    region_block_size: int
    num_kv_heads: int
    proxy_dim: int

    def __post_init__(self) -> None:
        for name in (
            "num_pages",
            "page_size",
            "region_block_size",
            "num_kv_heads",
            "proxy_dim",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

    @staticmethod
    def max_region_fragments_per_block(
        *,
        block_size: int,
        region_block_size: int,
    ) -> int:
        block_size = int(block_size)
        region_block_size = int(region_block_size)
        if block_size <= 0 or region_block_size <= 0:
            raise ValueError(
                "Step5 sparse requires positive block_size and "
                "region_block_size, "
                f"got block_size={block_size}, "
                f"region_block_size={region_block_size}."
            )
        common = math.gcd(block_size, region_block_size)
        return (
            block_size + region_block_size - common + region_block_size - 1
        ) // region_block_size

    @property
    def max_region_fragments_per_page(self) -> int:
        return self.max_region_fragments_per_block(
            block_size=self.page_size,
            region_block_size=self.region_block_size,
        )

    @property
    def summaries_per_page(self) -> int:
        return self.max_region_fragments_per_page

    @property
    def sum_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_pages,
            self.summaries_per_page,
            self.num_kv_heads,
            self.proxy_dim,
        )

    @property
    def count_shape(self) -> tuple[int, int, int]:
        return (self.num_pages, self.summaries_per_page, self.num_kv_heads)


class Step5SparseSummaryCache:
    """Per-layer physical KV sidecar used by Step5 sparse selection.

    The first dimensions mirror FlashAttention KV cache pages. Local prefix
    cache hits reuse physical pages through ``block_table``, so summary sum/count
    follow the same physical KV block lifetime.
    """

    _step5_csa_active_region_ids: torch.Tensor
    _step5_csa_active_slot_by_region: torch.Tensor
    _step5_csa_allocation_success: torch.Tensor
    _step5_csa_numerator_cache: torch.Tensor
    _step5_csa_denominator_cache: torch.Tensor
    _step5_csa_max_cache: torch.Tensor
    mean_cache: torch.Tensor

    def __init__(
        self,
        *,
        config: Step5SparseSummaryCacheConfig,
        sum_cache: torch.Tensor,
        count_cache: torch.Tensor,
        mean_cache: torch.Tensor | None = None,
        blocks_per_scheduler_block: int = 1,
    ) -> None:
        self.config = config
        self.sum_cache = sum_cache
        self.count_cache = count_cache
        self.mean_cache = (
            mean_cache
            if mean_cache is not None
            else torch.zeros(
                self.config.sum_shape,
                device=sum_cache.device,
                dtype=torch.uint8,
            )
        )
        self.blocks_per_scheduler_block = max(1, int(blocks_per_scheduler_block))
        expected_sum_shape = (
            int(self.sum_cache.shape[0]),
            1,
            self.num_kv_heads,
            self.proxy_dim,
        )
        if tuple(self.sum_cache.shape) != expected_sum_shape:
            raise ValueError(
                "Step5 sparse sum_cache shape mismatch: expected "
                f"{expected_sum_shape}, got {tuple(self.sum_cache.shape)}."
            )
        expected_count_shape = (
            int(self.count_cache.shape[0]),
            1,
            self.num_kv_heads,
        )
        if tuple(self.count_cache.shape) != expected_count_shape:
            raise ValueError(
                "Step5 sparse count_cache shape mismatch: expected "
                f"{expected_count_shape}, got {tuple(self.count_cache.shape)}."
            )
        if tuple(self.mean_cache.shape) != self.config.sum_shape:
            raise ValueError(
                "Step5 sparse mean_cache shape mismatch: expected "
                f"{self.sum_shape}, got {tuple(self.mean_cache.shape)}."
            )
        if self.mean_cache.dtype != torch.uint8:
            raise ValueError(
                "Step5 sparse mean_cache must be uint8 FP8 storage, got "
                f"{self.mean_cache.dtype}."
            )

    @property
    def num_pages(self) -> int:
        return self.config.num_pages

    @property
    def page_size(self) -> int:
        return self.config.page_size

    @property
    def region_block_size(self) -> int:
        return self.config.region_block_size

    @property
    def num_kv_heads(self) -> int:
        return self.config.num_kv_heads

    @property
    def proxy_dim(self) -> int:
        return self.config.proxy_dim

    @property
    def summaries_per_page(self) -> int:
        return self.config.summaries_per_page

    @property
    def sum_shape(self) -> tuple[int, int, int, int]:
        return self.config.sum_shape

    @property
    def count_shape(self) -> tuple[int, int, int]:
        return self.config.count_shape

    @torch.inference_mode()
    def reset_runtime_state(
        self,
        *,
        reset_producer_scratch: bool = True,
    ) -> None:
        """Clear persistent DSA state without reallocating captured tensors."""
        if reset_producer_scratch:
            self.sum_cache.zero_()
            self.count_cache.zero_()
        self.mean_cache.zero_()

        fill_values = {
            "_step5_csa_active_region_ids": -1,
            "_step5_csa_active_slot_by_region": -1,
            "_step5_csa_allocation_success": 1,
            "_step5_csa_numerator_cache": 0,
            "_step5_csa_denominator_cache": 0,
            "_step5_csa_max_cache": float("-inf"),
            "_step5_csa_active_token_k": 0,
            "_step5_csa_active_token_z": 0,
            "_step5_csa_active_token_valid": 0,
        }
        for name, value in fill_values.items():
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                tensor.fill_(value)

        transaction = getattr(self, "_step5_mtp_transaction", None)
        reset_transaction = getattr(transaction, "reset_runtime_state", None)
        if reset_transaction is not None:
            reset_transaction()

    def reset_blocks(
        self, block_ids: torch.Tensor | list[int], *, clear_mean: bool = True
    ) -> None:
        if isinstance(block_ids, torch.Tensor):
            ids = block_ids
            if ids.device != self.sum_cache.device or ids.dtype != torch.long:
                ids = ids.to(device=self.sum_cache.device, dtype=torch.long)
        else:
            ids = torch.tensor(
                block_ids, device=self.sum_cache.device, dtype=torch.long
            )
        if ids.numel() == 0:
            return
        self._invalidate_mtp_transaction_for_reset(ids, blocks_per_scheduler_block=1)
        if self.sum_cache.device.type == "cuda":
            self._reset_scheduler_block_ids(
                ids, blocks_per_scheduler_block=1, clear_mean=clear_mean
            )
            return
        valid = (ids >= 0) & (ids < self.num_pages)
        if not bool(valid.any()):
            return
        ids = ids[valid].unique()
        reset_slots = self._active_slots_for_scheduler_blocks(
            ids,
            blocks_per_scheduler_block=1,
            active_region_ids=self._step5_csa_active_region_ids,
            active_slot_by_region=self._step5_csa_active_slot_by_region,
        )
        # sum/count are shared producer scratch; only the prefix-owned mean
        # payload and transaction state are reset per physical page.
        if clear_mean:
            self.mean_cache.index_fill_(0, ids, 0)
        self._reset_csa_pages(ids)
        if reset_slots is not None:
            active_token_k = getattr(self, "_step5_csa_active_token_k", None)
            active_token_z = getattr(self, "_step5_csa_active_token_z", None)
            active_token_valid = getattr(
                self,
                "_step5_csa_active_token_valid",
                None,
            )
            if isinstance(active_token_k, torch.Tensor):
                active_token_k.index_fill_(0, reset_slots, 0)
            if isinstance(active_token_z, torch.Tensor):
                active_token_z.index_fill_(0, reset_slots, 0)
            if isinstance(active_token_valid, torch.Tensor):
                active_token_valid.index_fill_(0, reset_slots, 0)

    def _invalidate_mtp_transaction_for_reset(
        self, block_ids: torch.Tensor, *, blocks_per_scheduler_block: int
    ) -> None:
        transaction = getattr(self, "_step5_mtp_transaction", None)
        invalidate = getattr(transaction, "invalidate_scheduler_blocks", None)
        if invalidate is None:
            return
        invalidate(
            block_ids,
            blocks_per_scheduler_block=int(blocks_per_scheduler_block),
            summaries_per_page=int(self.summaries_per_page),
        )

    def reset_scheduler_blocks(
        self, block_ids: torch.Tensor | list[int], *, clear_mean: bool = True
    ) -> None:
        if isinstance(block_ids, torch.Tensor):
            ids = block_ids
            if ids.device != self.sum_cache.device or ids.dtype != torch.long:
                ids = ids.to(device=self.sum_cache.device, dtype=torch.long)
        else:
            if not block_ids:
                return
            ids = torch.tensor(
                block_ids, device=self.sum_cache.device, dtype=torch.long
            )
        if ids.numel() == 0:
            return
        ratio = max(1, int(self.blocks_per_scheduler_block))
        self._invalidate_mtp_transaction_for_reset(
            ids, blocks_per_scheduler_block=ratio
        )
        if self.sum_cache.device.type == "cuda":
            self._reset_scheduler_block_ids(
                ids, blocks_per_scheduler_block=ratio, clear_mean=clear_mean
            )
            return
        if ratio == 1:
            reset_block_ids = ids
        else:
            reset_block_ids: list[int] = []
            for block_id in ids.tolist():
                base = int(block_id) * ratio
                reset_block_ids.extend(range(base, base + ratio))
        self.reset_blocks(reset_block_ids, clear_mean=clear_mean)

    def _reset_scheduler_block_ids(
        self,
        block_ids: torch.Tensor,
        *,
        blocks_per_scheduler_block: int,
        clear_mean: bool = True,
    ) -> None:
        active_region_ids = getattr(self, "_step5_csa_active_region_ids", None)
        active_slot_by_region = getattr(self, "_step5_csa_active_slot_by_region", None)
        active_numerator = getattr(self, "_step5_csa_numerator_cache", None)
        denominator = getattr(self, "_step5_csa_denominator_cache", None)
        max_logits = getattr(self, "_step5_csa_max_cache", None)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                active_region_ids,
                active_slot_by_region,
                active_numerator,
                denominator,
                max_logits,
            )
        ):
            raise RuntimeError(
                "Step5 sparse CSA state must be initialized before CUDA summary reset."
            )
        if denominator.ndim != 3:
            raise RuntimeError(
                "Step5 sparse CSA denominator scratch must be rank-3, got "
                f"{tuple(denominator.shape)}."
            )
        # A decode request may finish with an incomplete 8-token CSA tail.
        # The tail lives in per-slot staging buffers rather than in the
        # page-backed mean payload.  Capture the slots owned by the pages
        # before the reset kernel clears the reverse map, then invalidate the
        # staging rows as part of the same page lifecycle.
        active_token_k = getattr(self, "_step5_csa_active_token_k", None)
        active_token_z = getattr(self, "_step5_csa_active_token_z", None)
        active_token_valid = getattr(self, "_step5_csa_active_token_valid", None)
        reset_slots = self._active_slots_for_scheduler_blocks(
            block_ids,
            blocks_per_scheduler_block=blocks_per_scheduler_block,
            active_region_ids=active_region_ids,
            active_slot_by_region=active_slot_by_region,
        )
        block_d = triton.next_power_of_2(int(self.num_kv_heads) * int(self.proxy_dim))
        _step5_sparse_summary_reset_blocks_kernel[
            (
                int(block_ids.numel()),
                int(blocks_per_scheduler_block),
                int(self.summaries_per_page),
            )
        ](
            block_ids,
            self.mean_cache,
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            denominator,
            max_logits,
            int(block_ids.numel()),
            int(self.num_pages),
            MEAN_STRIDE_PAGE=int(self.mean_cache.stride(0)),
            MEAN_STRIDE_REGION=int(self.mean_cache.stride(1)),
            MEAN_STRIDE_HEAD=int(self.mean_cache.stride(2)),
            NUMERATOR_STRIDE_SLOT=int(active_numerator.stride(0)),
            NUMERATOR_STRIDE_HEAD=int(active_numerator.stride(1)),
            DENOMINATOR_STRIDE_SLOT=int(denominator.stride(0)),
            DENOMINATOR_STRIDE_HEAD=int(denominator.stride(1)),
            MAX_STRIDE_SLOT=int(max_logits.stride(0)),
            BLOCKS_PER_SCHEDULER_BLOCK=int(blocks_per_scheduler_block),
            SUMMARIES_PER_PAGE=int(self.summaries_per_page),
            NUM_KV_HEADS=int(self.num_kv_heads),
            PROXY_DIM=int(self.proxy_dim),
            ACTIVE_CAPACITY=int(active_region_ids.numel()),
            BLOCK_D=block_d,
            CLEAR_MEAN=clear_mean,
            num_warps=4,
        )
        if reset_slots is not None:
            if isinstance(active_token_k, torch.Tensor):
                active_token_k.index_fill_(0, reset_slots, 0)
            if isinstance(active_token_z, torch.Tensor):
                active_token_z.index_fill_(0, reset_slots, 0)
            if isinstance(active_token_valid, torch.Tensor):
                active_token_valid.index_fill_(0, reset_slots, 0)

    def _active_slots_for_scheduler_blocks(
        self,
        block_ids: torch.Tensor,
        *,
        blocks_per_scheduler_block: int,
        active_region_ids: torch.Tensor,
        active_slot_by_region: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return active CSA slots belonging to physical scheduler pages."""
        if block_ids.numel() == 0:
            return None
        block_ids = block_ids.to(
            device=active_region_ids.device,
            dtype=torch.long,
        )
        sub_blocks = torch.arange(
            int(blocks_per_scheduler_block),
            device=block_ids.device,
            dtype=torch.long,
        )
        pages = (
            block_ids.view(-1, 1) * int(blocks_per_scheduler_block)
            + sub_blocks.view(1, -1)
        ).flatten()
        valid_pages = (pages >= 0) & (pages < self.num_pages)
        if not bool(valid_pages.any()):
            return None
        pages = pages[valid_pages]
        fragments = torch.arange(
            int(self.summaries_per_page),
            device=pages.device,
            dtype=torch.long,
        )
        regions = (
            pages.view(-1, 1) * int(self.summaries_per_page) + fragments.view(1, -1)
        ).flatten()
        regions = regions[(regions >= 0) & (regions < active_slot_by_region.numel())]
        if regions.numel() == 0:
            return None
        mapped = active_slot_by_region.index_select(0, regions)
        valid_slots = (mapped >= 0) & (mapped < active_region_ids.numel())
        if not bool(valid_slots.any()):
            return None
        safe_slots = mapped.clamp_min(0)
        owners = active_region_ids.index_select(0, safe_slots)
        valid_slots &= owners == regions
        if not bool(valid_slots.any()):
            return None
        return safe_slots[valid_slots].to(torch.long).unique()

    def warmup_scheduler_block_reset(self) -> None:
        active_region_ids = getattr(self, "_step5_csa_active_region_ids", None)
        if (
            self.sum_cache.device.type == "cuda"
            and isinstance(active_region_ids, torch.Tensor)
            and active_region_ids.numel() > 0
        ):
            # Warm the fixed reset launch with a valid scheduler id.  The
            # active-region table is initialized with -1 and is state, not a
            # page-id input; passing it here used to make warmup depend on a
            # sentinel and could also feed -1 into the backing reset.
            warmup_block = torch.zeros(
                (1,), device=self.mean_cache.device, dtype=torch.long
            )
            reset_ratio = max(1, int(self.blocks_per_scheduler_block))
            self._invalidate_mtp_transaction_for_reset(
                warmup_block,
                blocks_per_scheduler_block=reset_ratio,
            )
            for clear_mean in (True, False):
                self._reset_scheduler_block_ids(
                    warmup_block,
                    blocks_per_scheduler_block=reset_ratio,
                    clear_mean=clear_mean,
                )

    def _reset_csa_pages(self, pages: torch.Tensor) -> None:
        active_region_ids = getattr(self, "_step5_csa_active_region_ids", None)
        active_slot_by_region = getattr(self, "_step5_csa_active_slot_by_region", None)
        active_numerator = getattr(self, "_step5_csa_numerator_cache", None)
        denominator = getattr(self, "_step5_csa_denominator_cache", None)
        max_logits = getattr(self, "_step5_csa_max_cache", None)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                active_region_ids,
                active_slot_by_region,
                active_numerator,
                denominator,
                max_logits,
            )
        ):
            raise RuntimeError(
                "Step5 sparse CSA state must be initialized before summary reset."
            )
        pages = pages.to(device=active_region_ids.device, dtype=torch.long)
        if int(pages.numel()) == 0:
            return
        valid_pages = pages[(pages >= 0) & (pages < self.num_pages)]
        if int(valid_pages.numel()) == 0:
            return
        fragments = torch.arange(
            int(self.summaries_per_page),
            device=pages.device,
            dtype=torch.long,
        )
        reset_regions = (
            valid_pages.view(-1, 1) * int(self.summaries_per_page)
            + fragments.view(1, -1)
        ).flatten()
        reset_regions = reset_regions[
            (reset_regions >= 0) & (reset_regions < active_slot_by_region.numel())
        ]
        if int(reset_regions.numel()) == 0:
            return
        active_slot_by_region.index_fill_(0, reset_regions, -1)
        reset_active = torch.isin(
            active_region_ids,
            reset_regions.to(
                device=active_region_ids.device,
                dtype=active_region_ids.dtype,
            ),
        )
        active_region_ids.masked_fill_(reset_active, -1)
        active_numerator.masked_fill_(reset_active.view(-1, 1, 1), 0.0)
        denominator.masked_fill_(reset_active.view(-1, 1, 1), 0.0)
        max_logits.masked_fill_(reset_active.view(-1, 1), float("-inf"))


class Step5SparseSummaryCacheBackend(AttentionBackend):
    class MetadataBuilder(AttentionMetadataBuilder[CommonAttentionMetadata]):
        _cudagraph_support = AttentionCGSupport.ALWAYS
        supports_update_block_table: bool = True
        reorder_batch_threshold: int | None = None

        def __init__(
            self,
            kv_cache_spec: Any,
            layer_names: list[str],
            vllm_config: Any,
            device: torch.device,
        ) -> None:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> CommonAttentionMetadata:
            del common_prefix_len, fast_build
            return common_attn_metadata

        def update_block_table(
            self,
            metadata: CommonAttentionMetadata,
            blk_table: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> CommonAttentionMetadata:
            return metadata.replace(
                block_table_tensor=blk_table,
                slot_mapping=slot_mapping,
            )

    class NoOpImpl(AttentionImpl[CommonAttentionMetadata]):
        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int | None = None,
            alibi_slopes: list[float] | None = None,
            sliding_window: int | None = None,
            kv_cache_dtype: str = "auto",
            logits_soft_cap: float | None = None,
            attn_type: str = "decoder",
            kv_sharing_target_layer_name: str | None = None,
        ) -> None:
            del (
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                alibi_slopes,
                sliding_window,
                kv_cache_dtype,
                logits_soft_cap,
                attn_type,
                kv_sharing_target_layer_name,
            )

        def forward(
            self,
            layer: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: CommonAttentionMetadata,
            output: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            raise RuntimeError(
                "Step5 sparse summary cache layer is a KV side buffer and must "
                "not run attention forward."
            )

    forward_includes_kv_cache_update: bool = False
    supported_dtypes = [torch.uint8]
    supported_kv_cache_dtypes = []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(256)]

    @staticmethod
    def get_name() -> str:
        return "STEP5_SPARSE_SUMMARY"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl[CommonAttentionMetadata]]:
        return Step5SparseSummaryCacheBackend.NoOpImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder[CommonAttentionMetadata]]:
        return Step5SparseSummaryCacheBackend.MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        assert int(block_size) > 0 and int(num_kv_heads) > 0 and int(head_size) > 0, (
            "Step5 sparse summary cache shape requires positive block_size, "
            "num_kv_heads and head_size, got "
            f"block_size={block_size}, num_kv_heads={num_kv_heads}, "
            f"head_size={head_size}."
        )
        return (int(num_blocks), int(block_size) * int(num_kv_heads) * int(head_size))

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (0, 1) if not include_num_layers_dimension else (0, 1, 2)


class Step5SparseSummaryCacheLayer(torch.nn.Module, AttentionLayerBase):
    """Step5-private layer that owns the sparse summary KV side buffer."""

    def __init__(
        self,
        *,
        prefix: str,
        target_impl: Any,
        sparse_config: Any,
        main_layer_name: str,
        static_forward_context: dict[str, Any],
    ) -> None:
        super().__init__()
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self
        self.layer_name = prefix
        self.impl = Step5SparseSummaryCacheBackend.NoOpImpl(1, 1, 1.0)
        self._target_impl = target_impl
        self._sparse_config = sparse_config
        self._main_layer_name = str(main_layer_name)
        self._kv_cache = torch.tensor([])
        self._kv_cache_block_size: int | None = None
        self._dsa_scratch_workspace: Step5DSAScratchWorkspace | None = None
        self._owns_dsa_scratch_workspace = False
        self._budget_bytes_per_page: int | None = None
        self._backing_bytes_per_page: int | None = None
        self._bound_summary_config: Step5SparseSummaryCacheConfig | None = None
        self._budget_only_cache: Step5SparseSummaryCache | None = None

    @property
    def kv_cache(self) -> torch.Tensor:
        return self._kv_cache

    @kv_cache.setter
    def kv_cache(self, value: torch.Tensor) -> None:
        self._kv_cache = value
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            self._clear_summary_cache_binding()
            return
        self._bind_summary_cache(value)

    def _clear_summary_cache_binding(self) -> None:
        """Release tensors owned by a previous KV-cache binding."""
        workspaces = (
            self._dsa_scratch_workspace,
            getattr(self._target_impl, "_dsa_scratch_workspace", None),
        )
        seen_workspaces: set[int] = set()
        profiled_workspace: Step5DSAScratchWorkspace | None = None
        for workspace in workspaces:
            if not isinstance(workspace, Step5DSAScratchWorkspace):
                continue
            workspace_id = id(workspace)
            if workspace_id in seen_workspaces:
                continue
            seen_workspaces.add(workspace_id)
            if workspace.memory_profiled:
                if (
                    profiled_workspace is not None
                    and profiled_workspace is not workspace
                ):
                    raise RuntimeError(
                        "Step5 side storage references multiple profiled DSA "
                        "scratch workspaces."
                    )
                profiled_workspace = workspace
            else:
                workspace.tensor_buffers_by_engine.clear()
                workspace.order_tokens_by_engine.clear()

        self._budget_only_cache = None
        self._bound_summary_config = None
        self._dsa_scratch_workspace = profiled_workspace
        if profiled_workspace is None:
            self._owns_dsa_scratch_workspace = False
        self._target_impl._summary_cache = None
        self._target_impl._summary_cache_config = None
        self._target_impl._dsa_scratch_bound = False

    @property
    def side_kv_cache(self) -> torch.Tensor:
        # Present the (nb, allocated_bytes) backing as a (1, nb, allocated_bytes)
        # region-major tensor so a generic block-id transfer treats it as a
        # single-region layer (shape[0] == 1), 1:1 aligned with the owner KV
        # blocks.  Storage and the summary kernels are unchanged.
        return self._kv_cache.unsqueeze(0)

    @property
    def main_layer_name(self) -> str:
        return self._main_layer_name

    @staticmethod
    def _side_storage_peers(
        forward_context: dict[str, Any],
    ) -> list[Step5SparseSummaryCacheLayer]:
        peers: list[Step5SparseSummaryCacheLayer] = []
        seen_layers: set[int] = set()
        for layer in forward_context.values():
            if not isinstance(layer, Step5SparseSummaryCacheLayer):
                continue
            layer_id = id(layer)
            if layer_id in seen_layers:
                continue
            seen_layers.add(layer_id)
            peers.append(layer)
        if not peers:
            raise RuntimeError("Step5 sparse summary side storage is missing.")
        peers.sort(key=lambda layer: layer.layer_name)
        return peers

    @classmethod
    def _ensure_shared_scratch_workspace(
        cls,
        forward_context: dict[str, Any],
    ) -> tuple[
        Step5SparseSummaryCacheLayer,
        Step5DSAScratchWorkspace,
        list[Step5SparseSummaryCacheLayer],
    ]:
        peers = cls._side_storage_peers(forward_context)
        owner = peers[0]
        workspace = owner._dsa_scratch_workspace
        if workspace is None:
            workspace = Step5DSAScratchWorkspace()
            owner._dsa_scratch_workspace = workspace
        for peer in peers:
            if (
                getattr(peer._target_impl, "_dsa_scratch_workspace", None)
                is not workspace
            ):
                peer._target_impl.bind_scratch_workspace(workspace)
            peer._dsa_scratch_workspace = workspace
            peer._owns_dsa_scratch_workspace = peer is owner
        return owner, workspace, peers

    def kv_cache_side_storage_memory_profile_key(
        self,
        forward_context: dict[str, Any],
    ) -> object:
        """Return the canonical owner of the model-wide DSA workspace."""
        return self._side_storage_peers(forward_context)[0]

    def prepare_kv_cache_side_storage_memory_profile(
        self,
        forward_context: dict[str, Any],
        device: torch.device,
    ) -> None:
        """Allocate the maximum fixed DSA scratch before KV cache planning."""
        _, workspace, peers = self._ensure_shared_scratch_workspace(forward_context)
        if workspace.memory_profiled:
            if not workspace.allocations_locked:
                raise RuntimeError(
                    "Step5 DSA scratch workspace is profiled but not locked."
                )
            return

        workspace.allocations_locked = False
        try:
            for peer in peers:
                block_size = peer._kv_cache_block_size
                if block_size is None:
                    raise RuntimeError(
                        "Step5 DSA scratch profiling ran before KV cache specs "
                        f"were created for {peer.layer_name!r}."
                    )
                target_impl = peer._target_impl
                config = Step5SparseSummaryCacheConfig(
                    # Fixed scratch sizing uses max_model_len. A single page
                    # is sufficient to supply the remaining layout metadata
                    # without allocating the page-lifetime sidecar itself.
                    num_pages=1,
                    page_size=int(block_size),
                    region_block_size=int(target_impl.sparse_region_block_size),
                    num_kv_heads=int(
                        getattr(
                            target_impl,
                            "summary_cache_num_proxy_kv_heads",
                            1,
                        )
                        or 1
                    ),
                    proxy_dim=int(getattr(peer._sparse_config, "proxy_dim", 0) or 0),
                )
                active_capacity = int(target_impl._csa_active_region_capacity())
                sum_cache = target_impl._get_dsa_tensor_buffer_at_least(
                    "csa_shared_sum_cache",
                    (
                        active_capacity,
                        1,
                        config.num_kv_heads,
                        config.proxy_dim,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                count_cache = target_impl._get_dsa_tensor_buffer_at_least(
                    "csa_shared_count_cache",
                    (active_capacity, 1, config.num_kv_heads),
                    device=device,
                    dtype=torch.float32,
                )
                profiling_cache = Step5SparseSummaryCache(
                    config=config,
                    sum_cache=sum_cache,
                    count_cache=count_cache,
                    mean_cache=torch.empty(
                        config.sum_shape,
                        device=device,
                        dtype=torch.uint8,
                    ),
                )
                target_impl.bind_summary_cache(
                    profiling_cache,
                    initialize_runtime_state=False,
                )
            workspace.memory_profiled = True
            workspace.allocations_locked = True
        except Exception:
            workspace.allocations_locked = False
            raise

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Step5SparseSummaryCacheBackend

    def get_kv_cache_spec(self, vllm_config: Any) -> KVCacheSpec | None:
        # The sparse summary cache is Step5-private side storage whose
        # lifetime is tied to the owner attention layer.  It must not expose a
        # generic KV spec here: the standard KV allocator may unify page sizes
        # and produce a backing shape that does not match the summary payload
        # layout.  bind_kv_cache_side_storage() allocates the exact backing
        # after the owner KV cache has been created.
        del vllm_config
        return None

    def _set_summary_cache_budget(self, block_size: int) -> int:
        self._kv_cache_block_size = block_size
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(
            getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1
        )
        payload_size_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=block_size,
            region_block_size=int(self._target_impl.sparse_region_block_size),
            proxy_dim=proxy_dim,
            num_kv_heads=num_proxy_kv_heads,
        )
        storage_head_size = max(
            1, (int(payload_size_bytes) + block_size - 1) // block_size
        )
        backing_page_size_bytes = storage_head_size * block_size
        reverse_map_page_size_bytes = (
            Step5SparseSummaryCacheConfig.max_region_fragments_per_block(
                block_size=block_size,
                region_block_size=int(self._target_impl.sparse_region_block_size),
            )
            * torch.tensor([], dtype=torch.int32).element_size()
        )
        self._backing_bytes_per_page = int(backing_page_size_bytes)
        self._budget_bytes_per_page = int(
            backing_page_size_bytes + reverse_map_page_size_bytes
        )
        return self._budget_bytes_per_page

    def _fixed_runtime_state_budget(self, block_size: int) -> int:
        del block_size
        return int(self._target_impl.csa_fixed_runtime_state_size_bytes())

    def bind_kv_cache_side_storage(
        self,
        forward_context: dict[str, Any],
    ) -> None:
        # The v1 KV connector protocol transfers one attention layer at a time
        # and has no side-buffer/version hook. Registering only the owner KV
        # cache would make a P/D or offload resume consume stale summary pages.
        # Fail during cache initialization rather than silently producing
        # incorrect sparse attention results.
        from vllm.distributed.kv_transfer import has_kv_transfer_group

        if has_kv_transfer_group():
            raise RuntimeError(
                "Step5 DSA sparse summary side storage is not compatible with "
                "v1 KV transfer/offload yet. Disable kv_transfer_config for "
                "Step5, or use a connector implementation that explicitly "
                "transfers the summary sidecar."
            )
        self._ensure_shared_scratch_workspace(forward_context)

        owner_layer = forward_context.get(self._main_layer_name)
        if owner_layer is None:
            raise RuntimeError(
                "Step5 sparse summary cache owner layer is missing: "
                f"{self._main_layer_name!r}."
            )
        owner_kv_cache = owner_layer.kv_cache
        if not isinstance(owner_kv_cache, torch.Tensor):
            raise RuntimeError(
                "Step5 sparse summary cache owner KV cache must be a tensor, "
                f"got {type(owner_kv_cache)!r}."
            )
        if owner_kv_cache.ndim != 4:
            raise RuntimeError(
                "Step5 sparse summary cache owner KV cache has invalid shape: "
                f"{tuple(owner_kv_cache.shape)}."
            )
        block_size = self._kv_cache_block_size
        if block_size is None:
            block_size = int(owner_kv_cache.shape[2])
        self._kv_cache_block_size = block_size
        summary_dtype = torch.uint8
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(
            getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1
        )
        payload_size_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=block_size,
            region_block_size=int(self._target_impl.sparse_region_block_size),
            proxy_dim=proxy_dim,
            num_kv_heads=num_proxy_kv_heads,
            sum_dtype=summary_dtype,
            count_dtype=summary_dtype,
        )
        backing_bytes = (
            (int(payload_size_bytes) + int(block_size) - 1) // int(block_size)
        ) * int(block_size)
        reverse_map_bytes = (
            Step5SparseSummaryCacheConfig.max_region_fragments_per_block(
                block_size=block_size,
                region_block_size=int(self._target_impl.sparse_region_block_size),
            )
            * torch.tensor([], dtype=torch.int32).element_size()
        )
        self._backing_bytes_per_page = int(backing_bytes)
        self._budget_bytes_per_page = int(backing_bytes + reverse_map_bytes)
        if int(owner_kv_cache.shape[2]) != int(block_size):
            raise RuntimeError(
                "Step5 sparse summary cache block size mismatch: "
                f"spec={block_size}, owner_kv_cache={tuple(owner_kv_cache.shape)}."
            )
        allocated_bytes = self._backing_bytes_per_page
        if allocated_bytes is None:
            raise RuntimeError(
                "Step5 sparse summary cache backing budget was not initialized."
            )
        backing = torch.zeros(
            (int(owner_kv_cache.shape[0]), int(allocated_bytes)),
            device=owner_kv_cache.device,
            dtype=torch.uint8,
        )
        self._kv_cache = backing
        self._bind_summary_cache(backing, summary_dtype=summary_dtype)
        blocks_per_scheduler_block = 1
        num_scheduler_blocks = int(owner_kv_cache.shape[0])
        if int(num_scheduler_blocks) > 0 and (
            int(backing.shape[0]) % int(num_scheduler_blocks) == 0
        ):
            blocks_per_scheduler_block = max(
                1, int(backing.shape[0]) // int(num_scheduler_blocks)
            )
        cache = self._target_impl._summary_cache
        if cache is None:
            raise RuntimeError("Step5 sparse summary cache side storage was not bound.")
        cache.blocks_per_scheduler_block = blocks_per_scheduler_block
        cache.warmup_scheduler_block_reset()
        self._budget_only_cache = cache
        log_summary_allocation = (
            logger.info_once if int(num_scheduler_blocks) > 64 else logger.debug
        )
        payload_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=block_size,
            region_block_size=int(self._target_impl.sparse_region_block_size),
            proxy_dim=int(getattr(self._sparse_config, "proxy_dim", 0) or 0),
            num_kv_heads=int(
                getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1
            ),
        )
        fixed_runtime_bytes = self._fixed_runtime_state_budget(block_size)
        waste_bytes = int(allocated_bytes) - int(payload_bytes)
        waste_ratio = waste_bytes / int(allocated_bytes) if allocated_bytes else 0.0
        log_summary_allocation(
            "Allocated Step5 sparse summary cache side storage: "
            "pages_per_layer=%d scheduler_blocks=%d "
            "blocks_per_scheduler_block=%d payload_bytes_per_page=%d "
            "allocated_bytes_per_page=%d reverse_map_bytes_per_page=%d "
            "budget_bytes_per_page=%d fixed_runtime_bytes_per_layer=%d "
            "padding_waste_ratio=%.2f%% "
            "sample_layer_allocated=%s GiB summary_dtype=%s device=%s",
            int(backing.shape[0]),
            int(num_scheduler_blocks),
            blocks_per_scheduler_block,
            int(payload_bytes),
            int(allocated_bytes),
            int(reverse_map_bytes),
            int(self._budget_bytes_per_page),
            int(fixed_runtime_bytes),
            waste_ratio * 100.0,
            f"{backing.numel() * backing.element_size() / (1 << 30):.2f}",
            str(summary_dtype).replace("torch.", ""),
            backing.device,
        )

    def zero_kv_cache_side_storage(self, block_ids: torch.Tensor | list[int]) -> None:
        budget_only_cache = self._budget_only_cache
        if budget_only_cache is None:
            return
        if any(
            not isinstance(getattr(budget_only_cache, name, None), torch.Tensor)
            for name in (
                "_step5_csa_active_region_ids",
                "_step5_csa_active_slot_by_region",
                "_step5_csa_numerator_cache",
                "_step5_csa_denominator_cache",
                "_step5_csa_max_cache",
            )
        ):
            return
        budget_only_cache.reset_scheduler_blocks(block_ids)

    def release_kv_cache_side_storage(
        self, block_ids: torch.Tensor | list[int]
    ) -> None:
        """Release completed request state while retaining prefix-cache payloads."""
        cache = self._budget_only_cache
        if cache is not None:
            cache.reset_scheduler_blocks(block_ids, clear_mean=False)

    def reset_kv_cache_side_storage_runtime_state(self) -> tuple[int, int]:
        """Clear startup dummy state through the generic side-storage hook."""
        cache = self._budget_only_cache
        if cache is None:
            return 0, 0
        cache.reset_runtime_state(reset_producer_scratch=True)

        num_scratch_buffers = 0
        workspace = self._dsa_scratch_workspace
        if getattr(self, "_owns_dsa_scratch_workspace", False) and isinstance(
            workspace, Step5DSAScratchWorkspace
        ):
            num_scratch_buffers = workspace.reset_runtime_state()
        return 1, num_scratch_buffers

    def copy_kv_cache_side_storage(
        self,
        block_copies: Any,
        num_blocks: int,
    ) -> None:
        """Copy side-cache pages for scheduler copy-on-write operations.

        Prefix-cache and COW paths copy the owner KV pages after allocation.
        The summary sidecar is not part of ``GPUModelRunner.kv_caches`` and
        therefore needs the same block mapping explicitly.
        """
        if self._kv_cache.numel() == 0 or block_copies is None:
            return
        # The worker protocol currently supplies a list, but materializing
        # here keeps this layer safe for connector implementations that pass a
        # one-shot iterator.  The mapping is consumed once for runtime-state
        # cloning and once more for the payload copy below.
        block_copies = list(block_copies)
        if not block_copies:
            return
        from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
        from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

        # The scheduler addresses logical blocks, while a side cache may hold
        # multiple physical rows per scheduler block (for example when the
        # owner attention backend expands a scheduler page into several
        # kernel pages).  Reset paths already expand these IDs through
        # ``blocks_per_scheduler_block``; COW must use the same mapping or it
        # would copy only the first physical row and leave the remaining rows
        # stale.
        budget_only_cache = self._budget_only_cache
        blocks_per_scheduler_block = max(
            1,
            int(
                getattr(
                    budget_only_cache,
                    "blocks_per_scheduler_block",
                    1,
                )
            ),
        )
        if blocks_per_scheduler_block > 1:
            expanded_block_copies = [
                KVCacheBlockCopy(
                    src_block_id=(
                        int(block_copy.src_block_id) * blocks_per_scheduler_block
                        + offset
                    ),
                    dst_block_id=(
                        int(block_copy.dst_block_id) * blocks_per_scheduler_block
                        + offset
                    ),
                )
                for block_copy in block_copies
                for offset in range(blocks_per_scheduler_block)
            ]
            block_copies = expanded_block_copies
            # ``_kv_cache`` is the physical side-cache backing.  Prefer its
            # actual row count over the scheduler count supplied by the
            # generic lifecycle hook once the mapping has been expanded.
            num_blocks = int(self._kv_cache.shape[0])

        # A partial prefix block can own an incomplete CSA region.  Its
        # persistent state lives in the fixed-capacity active-slot arrays, not
        # in ``mean_cache`` alone.  Clone that state before copying the page
        # payload; otherwise a COW destination starts with a valid mean but
        # no numerator/denominator/max (or decode-tail staging), and the next
        # MTP/graph update observes a different history from the source page.
        self._copy_csa_runtime_state(block_copies)

        if (
            budget_only_cache is not None
            and getattr(budget_only_cache, "_step5_mtp_transaction", None) is not None
        ):
            destinations = sorted(
                {
                    int(copy.dst_block_id)
                    for copy in block_copies
                    if copy.src_block_id != copy.dst_block_id
                }
            )
            if destinations:
                # A copied destination changes owner even when its physical
                # page number stays the same. Unchanged source pages stay live.
                budget_only_cache._invalidate_mtp_transaction_for_reset(
                    torch.tensor(
                        destinations,
                        device=budget_only_cache.mean_cache.device,
                        dtype=torch.long,
                    ),
                    blocks_per_scheduler_block=1,
                )

        copy_kv_cache_blocks_inplace(
            [self._kv_cache],
            num_blocks,
            block_copies,
        )

    def _copy_csa_runtime_state(self, block_copies: Any) -> None:
        """Clone active CSA state for physical page COW mappings.

        ``active_region_ids`` is authoritative; ``active_slot_by_region`` is
        only a reverse lookup hint and can be stale after slot reuse.  We
        snapshot every live source slot before clearing destinations so
        overlapping copy lists remain deterministic.  This hook runs between
        scheduler zeroing and the model forward, outside CUDA-graph capture.
        """
        cache = self._budget_only_cache
        if cache is None or not block_copies:
            return
        state_names = (
            "_step5_csa_active_region_ids",
            "_step5_csa_active_slot_by_region",
            "_step5_csa_numerator_cache",
            "_step5_csa_denominator_cache",
            "_step5_csa_max_cache",
            "_step5_csa_active_token_k",
            "_step5_csa_active_token_z",
            "_step5_csa_active_token_valid",
        )
        states = tuple(getattr(cache, name, None) for name in state_names)
        if not all(isinstance(state, torch.Tensor) for state in states):
            # The side cache may be bound before CSA runtime state is
            # initialized (e.g. during profiling).  There is no state to copy
            # in that phase; the page payload copy remains valid.
            return
        (
            active_region_ids,
            active_slot_by_region,
            active_numerator,
            active_denominator,
            active_max,
            active_token_k,
            active_token_z,
            active_token_valid,
        ) = states
        assert isinstance(active_region_ids, torch.Tensor)
        assert isinstance(active_slot_by_region, torch.Tensor)
        assert isinstance(active_numerator, torch.Tensor)
        assert isinstance(active_denominator, torch.Tensor)
        assert isinstance(active_max, torch.Tensor)
        assert isinstance(active_token_k, torch.Tensor)
        assert isinstance(active_token_z, torch.Tensor)
        assert isinstance(active_token_valid, torch.Tensor)

        num_pages = int(getattr(cache, "num_pages", 0))
        summaries_per_page = int(getattr(cache, "summaries_per_page", 0))
        active_capacity = int(active_region_ids.numel())
        if num_pages <= 0 or summaries_per_page <= 0 or active_capacity <= 0:
            return

        # Normalize and validate the physical page mappings.  The generic
        # payload copier receives KVCacheBlockCopy objects, but this helper is
        # intentionally independent of the scheduler block geometry.
        physical_pairs: list[tuple[int, int]] = []
        seen_destinations: dict[int, int] = {}
        for block_copy in block_copies:
            src_page = int(block_copy.src_block_id)
            dst_page = int(block_copy.dst_block_id)
            if src_page == dst_page:
                continue
            if not (0 <= src_page < num_pages and 0 <= dst_page < num_pages):
                raise ValueError(
                    "Step5 DSA side-storage COW page is out of range: "
                    f"src={src_page}, dst={dst_page}, num_pages={num_pages}"
                )
            previous_src = seen_destinations.get(dst_page)
            if previous_src is not None:
                if previous_src != src_page:
                    raise ValueError(
                        "Step5 DSA side-storage COW has conflicting sources "
                        f"for destination page {dst_page}: "
                        f"{previous_src} and {src_page}"
                    )
                # Duplicate copies are harmless for payload assignment but
                # would otherwise double-count active-slot allocations.
                continue
            seen_destinations[dst_page] = src_page
            physical_pairs.append((src_page, dst_page))
        if not physical_pairs:
            return

        slot_ids = torch.arange(
            active_capacity,
            device=active_region_ids.device,
            dtype=torch.long,
        )
        snapshots: list[tuple[int, torch.Tensor, tuple[torch.Tensor, ...]]] = []
        destination_regions: list[torch.Tensor] = []
        total_slots_needed = 0

        # Snapshot all source state before any destination is cleared.  The
        # source active map is authoritative; this also repairs around a stale
        # reverse-map hint instead of silently dropping a live region.
        for src_page, dst_page in physical_pairs:
            dst_begin = dst_page * summaries_per_page
            dst_regions_for_page = torch.arange(
                dst_begin,
                dst_begin + summaries_per_page,
                device=active_region_ids.device,
                dtype=torch.long,
            )
            invalid_destination_regions = (dst_regions_for_page < 0) | (
                dst_regions_for_page >= active_slot_by_region.numel()
            )
            if bool(invalid_destination_regions.any()):
                raise ValueError(
                    "Step5 DSA side-storage COW destination region is out "
                    f"of range: dst_page={dst_page}"
                )
            # Include every destination page, even one with no active source
            # tail.  This clears stale destination ownership before the mean
            # payload is copied.
            destination_regions.append(dst_regions_for_page)

            src_begin = src_page * summaries_per_page
            src_end = src_begin + summaries_per_page
            live_mask = (active_region_ids >= src_begin) & (active_region_ids < src_end)
            src_slots = slot_ids[live_mask]
            if src_slots.numel() == 0:
                continue
            src_regions = active_region_ids[src_slots].clone()
            dst_regions = src_regions + ((dst_page - src_page) * summaries_per_page)
            invalid_regions = (dst_regions < 0) | (
                dst_regions >= active_slot_by_region.numel()
            )
            if bool(invalid_regions.any()):
                raise ValueError(
                    "Step5 DSA side-storage COW region is out of range: "
                    f"src_page={src_page}, dst_page={dst_page}"
                )
            snapshots.append(
                (
                    dst_page,
                    dst_regions,
                    (
                        active_numerator[src_slots].clone(),
                        active_denominator[src_slots].clone(),
                        active_max[src_slots].clone(),
                        active_token_k[src_slots].clone(),
                        active_token_z[src_slots].clone(),
                        active_token_valid[src_slots].clone(),
                    ),
                )
            )
            total_slots_needed += int(src_slots.numel())

        # Clear any old destination ownership and staging rows.  The generic
        # zeroer normally did this already, but keeping the hook idempotent
        # makes direct connector/COW callers safe as well.
        destination_regions_flat = torch.cat(destination_regions)
        destination_regions_flat = destination_regions_flat.unique()
        # ``active_region_ids`` is authoritative. The reverse map is only a
        # lookup hint and can be stale after a prior interrupted/failed
        # allocation, so using it to find destination owners can leak a live
        # slot or create duplicate ownership during COW. Find every live
        # destination owner directly and clear all duplicates defensively.
        destination_region_ids = destination_regions_flat.to(
            device=active_region_ids.device,
            dtype=active_region_ids.dtype,
        )
        matched_slots = torch.where(
            torch.isin(active_region_ids, destination_region_ids)
        )[0].to(torch.long)

        # Check capacity before mutating any destination state.  A destination
        # slot that currently owns one of the copied regions will be released
        # below and is therefore available for a new clone.  If this check
        # fails, leave both source and destination state untouched.
        free_slot_count = int((active_region_ids < 0).sum().item())
        available_slots = free_slot_count + int(matched_slots.numel())
        if available_slots < total_slots_needed:
            raise RuntimeError(
                "Step5 DSA active-slot capacity is exhausted during KV COW: "
                f"needed={total_slots_needed}, available={available_slots}"
            )

        if matched_slots.numel() > 0:
            active_region_ids.index_fill_(0, matched_slots, -1)
            active_numerator.index_fill_(0, matched_slots, 0)
            active_denominator.index_fill_(0, matched_slots, 0)
            active_max.index_fill_(0, matched_slots, float("-inf"))
            active_token_k.index_fill_(0, matched_slots, 0)
            active_token_z.index_fill_(0, matched_slots, 0)
            active_token_valid.index_fill_(0, matched_slots, 0)
        active_slot_by_region.index_fill_(0, destination_regions_flat, -1)

        free_slots = torch.where(active_region_ids < 0)[0]
        if int(free_slots.numel()) < total_slots_needed:
            raise RuntimeError(
                "Step5 DSA active-slot capacity is exhausted during KV COW: "
                f"needed={total_slots_needed}, available={int(free_slots.numel())}"
            )

        cursor = 0
        for _, dst_regions, state_snapshot in snapshots:
            count = int(dst_regions.numel())
            dst_slots = free_slots[cursor : cursor + count]
            cursor += count
            (
                numerator,
                denominator,
                max_logits,
                token_k,
                token_z,
                token_valid,
            ) = state_snapshot
            active_region_ids.index_copy_(0, dst_slots, dst_regions)
            active_slot_by_region.index_copy_(
                0,
                dst_regions,
                dst_slots.to(dtype=active_slot_by_region.dtype),
            )
            active_numerator.index_copy_(0, dst_slots, numerator)
            active_denominator.index_copy_(0, dst_slots, denominator)
            active_max.index_copy_(0, dst_slots, max_logits)
            active_token_k.index_copy_(0, dst_slots, token_k)
            active_token_z.index_copy_(0, dst_slots, token_z)
            active_token_valid.index_copy_(0, dst_slots, token_valid)

    def _bind_summary_cache(
        self,
        backing: torch.Tensor,
        *,
        summary_dtype: torch.dtype | None = None,
    ) -> None:
        block_size = self._kv_cache_block_size
        if block_size is None:
            raise RuntimeError(
                "Step5 sparse summary cache was bound before its KV spec was created."
            )
        if backing.dtype is not torch.uint8:
            raise RuntimeError(
                "Step5 sparse summary cache backing tensor must be uint8, "
                f"got {backing.dtype}."
            )
        if not backing.is_contiguous():
            raise RuntimeError(
                "Step5 sparse summary cache backing tensor must be contiguous, "
                f"got stride={backing.stride()}."
            )
        proxy_dim = int(getattr(self._sparse_config, "proxy_dim", 0) or 0)
        num_proxy_kv_heads = int(
            getattr(self._target_impl, "summary_cache_num_proxy_kv_heads", 1) or 1
        )
        config = Step5SparseSummaryCacheConfig(
            num_pages=int(backing.shape[0]),
            page_size=int(block_size),
            region_block_size=int(self._target_impl.sparse_region_block_size),
            num_kv_heads=num_proxy_kv_heads,
            proxy_dim=proxy_dim,
        )
        summary_dtype = summary_dtype or torch.uint8
        expected_bytes = step5_sparse_summary_cache_bytes_per_block(
            block_size=config.page_size,
            region_block_size=config.region_block_size,
            proxy_dim=config.proxy_dim,
            num_kv_heads=config.num_kv_heads,
            sum_dtype=torch.uint8,
            count_dtype=torch.uint8,
        )
        if int(backing.shape[1]) < expected_bytes:
            raise RuntimeError(
                "Step5 sparse summary cache backing tensor shape mismatch: "
                f"expected at least bytes_per_page={expected_bytes}, "
                f"got shape={tuple(backing.shape)}."
            )
        sum_elems = (
            int(config.summaries_per_page)
            * int(config.num_kv_heads)
            * int(config.proxy_dim)
        )
        if int(backing.shape[1]) < sum_elems:
            raise RuntimeError(
                "Step5 sparse mean cache backing is too small: "
                f"need={sum_elems}, got={int(backing.shape[1])}"
            )
        mean_cache = backing[:, :sum_elems].view(config.sum_shape)
        active_capacity = self._target_impl._csa_active_region_capacity()
        sum_cache = self._target_impl._get_dsa_tensor_buffer_at_least(
            "csa_shared_sum_cache",
            (active_capacity, 1, config.num_kv_heads, config.proxy_dim),
            device=backing.device,
            dtype=torch.float32,
        )
        count_cache = self._target_impl._get_dsa_tensor_buffer_at_least(
            "csa_shared_count_cache",
            (active_capacity, 1, config.num_kv_heads),
            device=backing.device,
            dtype=torch.float32,
        )
        summary_cache = Step5SparseSummaryCache(
            config=config,
            sum_cache=sum_cache,
            count_cache=count_cache,
            mean_cache=mean_cache,
        )
        self._target_impl.bind_summary_cache(summary_cache)
        self._bound_summary_config = config
