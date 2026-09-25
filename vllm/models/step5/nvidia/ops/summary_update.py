# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered updates of compressed index keys."""

import torch

from vllm.triton_utils import tl, triton

libdevice = tl.extra.cuda.libdevice


@triton.jit
def _maximum(x, y):
    return tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _update_decode_values(
    row,
    slot,
    dim,
    key,
    z,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    active = slot >= 0
    page = tl.where(active, slot // BLOCK_SIZE, PAGES + row)
    fresh = (slot % 8 == 0) | ~active
    last = tl.load(LastSlot + page)
    tl.device_assert(
        fresh | (last == slot - 1), "Step5 CSA tail missing for a partial region"
    )
    numerator = tl.load(Numerator + page * 256 + dim)
    denominator = tl.load(Denominator + page * 256 + dim)
    maximum = tl.load(Maximum + page)
    numerator = tl.where(fresh, 0, numerator)
    denominator = tl.where(fresh, 0, denominator)
    maximum = tl.where(fresh, -float("inf"), maximum)
    updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
    scale = tl.where(denominator > 0, libdevice.exp(maximum - updated_max), 0)
    weight = libdevice.exp(z - updated_max)
    numerator = numerator * scale + weight * key
    denominator = denominator * scale + weight
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    region = tl.maximum(slot, 0) % BLOCK_SIZE // 8
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(Mean + (page * (BLOCK_SIZE // 8) + region) * 256 + dim, summary_fp8)
    tl.store(Numerator + page * 256 + dim, numerator)
    tl.store(Denominator + page * 256 + dim, denominator)
    tl.store(Maximum + page, updated_max)
    tl.store(LastSlot + page, tl.where(active & (slot % 8 != 7), slot, -1))


@triton.jit
def _update_decode(
    Key,
    Z,
    Slots,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    KEY_STRIDE: tl.constexpr,
    Z_STRIDE: tl.constexpr,
    PAGES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.arange(0, 256)
    slot = tl.load(Slots + row)
    key = tl.load(Key + row * KEY_STRIDE + dim).to(tl.float32)
    z = tl.load(Z + row * Z_STRIDE + dim).to(tl.float32)
    _update_decode_values(
        row,
        slot,
        dim,
        key,
        z,
        Mean,
        Numerator,
        Denominator,
        Maximum,
        LastSlot,
        PAGES,
        BLOCK_SIZE,
    )


def update_decode(
    key, z, slots, mean, numerator, denominator, maximum, last_slot, pages, block_size
):
    if slots.numel():
        _update_decode[(slots.numel(),)](
            key,
            z,
            slots,
            mean,
            numerator,
            denominator,
            maximum,
            last_slot,
            key.stride(0),
            z.stride(0),
            pages,
            block_size,
            num_warps=4,
            enable_fp_fusion=False,
            debug=True,
        )


@triton.jit
def _accumulate_prefill(
    Key,
    Z,
    Metadata,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    State,
    Valid,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    region_row = tl.program_id(0)
    metadata = Metadata + region_row * 12
    region = tl.load(metadata)
    first = tl.load(metadata + 1)
    page = region * 8 // BLOCK_SIZE
    fresh = first == 0
    last_slot = tl.load(LastSlot + page)
    tl.store(Valid + region_row, fresh | (last_slot == region * 8 + first - 1))
    dim = tl.arange(0, 256)
    numerator = tl.load(Numerator + page * 256 + dim)
    denominator = tl.load(Denominator + page * 256 + dim)
    maximum = tl.load(Maximum + page)
    numerator = tl.where(fresh, 0, numerator)
    denominator = tl.where(fresh, 0, denominator)
    maximum = tl.where(fresh, -float("inf"), maximum)
    for offset in range(8):
        row = tl.load(metadata + 4 + offset)
        valid = row >= 0
        key = tl.load(
            Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.float32)
        z = tl.load(
            Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.float32)
        updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
        scale = tl.where(denominator > 0, libdevice.exp(maximum - updated_max), 0)
        weight = libdevice.exp(z - updated_max)
        updated_num = numerator * scale + weight * key
        updated_den = denominator * scale + weight
        numerator = tl.where(valid, updated_num, numerator)
        denominator = tl.where(valid, updated_den, denominator)
        maximum = tl.where(valid, updated_max, maximum)
    tl.store(State + region_row * 513 + dim, numerator)
    tl.store(State + region_row * 513 + 256 + dim, denominator)
    tl.store(State + region_row * 513 + 512, maximum)


@triton.jit
def _commit_prefill(
    Metadata,
    State,
    Mean,
    Numerator,
    Denominator,
    Maximum,
    LastSlot,
    BLOCK_SIZE: tl.constexpr,
):
    region_row = tl.program_id(0)
    metadata = Metadata + region_row * 12
    region = tl.load(metadata)
    final_slot = tl.load(metadata + 2)
    final_region = tl.load(metadata + 3) != 0
    page = region * 8 // BLOCK_SIZE
    dim = tl.arange(0, 256)
    numerator = tl.load(State + region_row * 513 + dim)
    denominator = tl.load(State + region_row * 513 + 256 + dim)
    maximum = tl.load(State + region_row * 513 + 512)
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(Mean + region * 256 + dim, summary_fp8)
    tl.store(Numerator + page * 256 + dim, numerator, mask=final_region)
    tl.store(Denominator + page * 256 + dim, denominator, mask=final_region)
    tl.store(Maximum + page, maximum, mask=final_region)
    tl.store(LastSlot + page, final_slot, mask=final_region)


def update_prefill(
    key, z, groups, mean, numerator, denominator, maximum, last_slot, block_size
):
    if not groups:
        return
    regions = list(groups)
    last = {region * 8 // block_size: region for region in regions}
    metadata = []
    for region, rows in groups.items():
        offsets = [i for i, row in enumerate(rows) if row >= 0]
        final_slot = region * 8 + offsets[-1]
        metadata.append(
            [
                region,
                offsets[0],
                final_slot if offsets[-1] != 7 else -1,
                int(last[region * 8 // block_size] == region),
                *rows,
            ]
        )
    metadata = torch.tensor(metadata, dtype=torch.int64, device=key.device)
    state = torch.empty((len(groups), 513), dtype=torch.float32, device=key.device)
    valid = torch.empty(len(groups), dtype=torch.bool, device=key.device)
    _accumulate_prefill[(len(groups),)](
        key,
        z,
        metadata,
        numerator,
        denominator,
        maximum,
        last_slot,
        state,
        valid,
        key.stride(0),
        key.stride(2),
        z.stride(0),
        z.stride(2),
        block_size,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Validate before any cache mutation, including all summary writes.
    if not bool(valid.all()):
        raise RuntimeError("Step5 CSA tail missing for a partial region")
    _commit_prefill[(len(groups),)](
        metadata,
        state,
        mean,
        numerator,
        denominator,
        maximum,
        last_slot,
        block_size,
        num_warps=4,
        enable_fp_fusion=False,
    )
