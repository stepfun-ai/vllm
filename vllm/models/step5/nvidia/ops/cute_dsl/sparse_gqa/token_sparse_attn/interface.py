# Copyright (c) 2026 StepFun Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
from typing import Optional

import torch

import cutlass
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.token_sparse_attn.prefill_paged_sm90_gqa import (
    TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillPagedGQA,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.token_sparse_attn.prefill_union_paged_sm90_gqa import (
    TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillUnionGQA,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.token_sparse_attn.decode_sm90_gqa import (
    TokenWiseFlashAttnFwdSm90GQADecode,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.token_sparse_attn.decode_mtp_union_exact_mask_sm90_gqa import (
    TokenWiseFlashAttnFwdSm90GQADecodeMTPUnionExactMask,
)
from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.token_sparse_attn.splitkv_merge_sm90_gqa import (
    merge_variable_split_nat_lse_states_sm90_gqa,
)

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _require_cuda_contiguous_tensor(
    tensor: Optional[torch.Tensor],
    *,
    tensor_name: str,
) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{tensor_name} must be a tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{tensor_name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(
            f"{tensor_name} must be dense row-major, got stride={tuple(int(v) for v in tensor.stride())}"
        )
    return tensor


def _require_cuda_kv_tensor(
    tensor: torch.Tensor, *, tensor_name: str
) -> torch.Tensor:
    # CuTe consumes the tensor's actual strides. Packed K+V allocations have
    # non-contiguous halves, but still satisfy the kernels' vector alignment.
    if tensor is None or tensor.device.type != "cuda":
        raise ValueError(f"{tensor_name} must be a CUDA tensor")
    if tensor.ndim != 4 or tensor.stride(-1) != 1:
        raise ValueError(f"{tensor_name} requires rank 4 and contiguous head dim")
    if tensor.data_ptr() % 16 or any(
        stride <= 0 or stride * tensor.element_size() % 16
        for stride in tensor.stride()[:-1]
    ):
        raise ValueError(f"{tensor_name} requires 16-byte-aligned positive strides")
    return tensor


def _require_paged_qkv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    tensor_name: str,
    q_heads: tuple[int, ...],
    head_dims: tuple[int, ...],
    allow_multi_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    q = _require_cuda_contiguous_tensor(q, tensor_name="q")
    k_cache = _require_cuda_kv_tensor(k_cache, tensor_name="k_cache")
    v_cache = _require_cuda_kv_tensor(v_cache, tensor_name="v_cache")
    if q.ndim != 3:
        raise ValueError(
            f"{tensor_name} expects q shape (Tq, Hq, D), got {tuple(q.shape)}"
        )
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q/k_cache/v_cache must have the same dtype")
    logical_q_heads = int(q.shape[1])
    head_dim = int(q.shape[2])
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            f"{tensor_name} expects k_cache/v_cache rank 4, "
            f"got {tuple(k_cache.shape)} and {tuple(v_cache.shape)}"
        )
    kv_heads = int(k_cache.shape[2])
    if not allow_multi_kv and kv_heads != 1:
        raise ValueError(
            f"{tensor_name} expects a single local KV head, got {kv_heads}"
        )
    if int(v_cache.shape[2]) != kv_heads:
        raise ValueError(
            f"{tensor_name} K/V local head mismatch: {kv_heads} vs {int(v_cache.shape[2])}"
        )
    q_heads_per_kv = logical_q_heads // kv_heads if kv_heads > 0 else 0
    if (
        kv_heads <= 0
        or logical_q_heads != q_heads_per_kv * kv_heads
        or q_heads_per_kv not in q_heads
        or head_dim not in head_dims
    ):
        raise ValueError(
            f"{tensor_name} expects q_heads_per_kv in {q_heads} and head_dim in {head_dims}, "
            f"got q={tuple(q.shape)}, k_cache={tuple(k_cache.shape)}"
        )
    expected_kv_tail = (16, kv_heads, head_dim)
    if tuple(k_cache.shape[1:]) != expected_kv_tail:
        raise ValueError(
            f"{tensor_name} expects k_cache shape (num_pages, 16, 1, {head_dim}), "
            f"got {tuple(k_cache.shape)}"
        )
    if tuple(v_cache.shape[1:]) != expected_kv_tail:
        raise ValueError(
            f"{tensor_name} expects v_cache shape (num_pages, 16, 1, {head_dim}), "
            f"got {tuple(v_cache.shape)}"
        )
    return q, k_cache, v_cache, q_heads_per_kv, head_dim


def _get_cutlass_dtype(dtype: torch.dtype):
    if dtype not in torch2cute_dtype_map:
        raise TypeError("Only fp16/bf16 are supported")
    return torch2cute_dtype_map[dtype]


@functools.cache
def _get_token_wise_flash_attn_prefill_sm90_gqa_op(
    *,
    dtype: torch.dtype,
    q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int = 16,
    num_threads: int = 256,
    q_per_cta: int = 4,
) -> TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillPagedGQA:
    return TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillPagedGQA(
        dtype=_get_cutlass_dtype(dtype),
        logical_q_heads=int(q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        block_n=int(block_n),
        num_threads=int(num_threads),
        q_per_cta=int(q_per_cta),
    )


def _expected_union_q_group(q_heads: int) -> int:
    if int(q_heads) == 4:
        return 32
    if int(q_heads) == 8:
        return 16
    if int(q_heads) == 16:
        return 8
    raise ValueError(
        f"GQA grouped-union prefill supports q_heads in (4, 8, 16), got {int(q_heads)}"
    )


def _resolve_union_q_group(*, q_group: Optional[int], q_heads: int) -> int:
    expected = _expected_union_q_group(q_heads)
    if q_group is None:
        return expected
    if int(q_group) != expected:
        raise ValueError(
            "GQA grouped-union prefill requires "
            f"q_group={expected} when q_heads_per_kv={int(q_heads)}, "
            f"got q_group={int(q_group)}"
        )
    return int(q_group)


@functools.cache
def _get_token_wise_flash_attn_prefill_union_sm90_gqa_op(
    *,
    dtype: torch.dtype,
    q_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_n: int = 64,
    num_threads: int = 512,
    q_group: Optional[int] = None,
) -> TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillUnionGQA:
    resolved_q_group = _resolve_union_q_group(q_group=q_group, q_heads=q_heads)
    return TokenWiseFlashAttnFwdSm90ManualMbarrierMmaPrefillUnionGQA(
        dtype=_get_cutlass_dtype(dtype),
        logical_q_heads=int(q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        block_n=int(block_n),
        num_threads=int(num_threads),
        q_group=resolved_q_group,
    )


@functools.cache
def _get_token_wise_flash_attn_decode_sm90_gqa_op(
    *,
    dtype: torch.dtype,
    q_heads: int,
    head_dim: int,
    head_dim_v: int,
    topk_windows: int,
    mtp_q_len: int = 1,
    batch_invariant: bool = False,
    block_n: int = 64,
    num_threads: int = 256,
    variable_split_max: int = 1,
    sm_count: int | None = None,
) -> TokenWiseFlashAttnFwdSm90GQADecode:
    return TokenWiseFlashAttnFwdSm90GQADecode(
        dtype=_get_cutlass_dtype(dtype),
        logical_q_heads=int(q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        topk_windows=int(topk_windows),
        mtp_q_len=int(mtp_q_len),
        batch_invariant=bool(batch_invariant),
        block_n=int(block_n),
        num_threads=int(num_threads),
        variable_split_max=int(variable_split_max),
        sm_count=None if sm_count is None else int(sm_count),
    )


@functools.cache
def _get_token_wise_flash_attn_decode_mtp_union_exact_mask_sm90_gqa_op(
    *,
    dtype: torch.dtype,
    q_heads: int,
    head_dim: int,
    head_dim_v: int,
) -> TokenWiseFlashAttnFwdSm90GQADecodeMTPUnionExactMask:
    return TokenWiseFlashAttnFwdSm90GQADecodeMTPUnionExactMask(
        dtype=_get_cutlass_dtype(dtype),
        logical_q_heads=int(q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
    )


def token_wise_flash_attn_prefill_sm90_gqa_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    region_counts: torch.Tensor,
    region_phys_indices: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    seq_q: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    stream=None,
):
    q, k_cache, v_cache, q_heads, _ = _require_paged_qkv(
        q,
        k_cache,
        v_cache,
        tensor_name="gqa fixed-spec sparse prefill",
        q_heads=(4, 8, 16),
        head_dims=(128, 192),
        allow_multi_kv=True,
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(q.shape[2]))
    if out is None:
        out = torch.empty_like(q)

    op = _get_token_wise_flash_attn_prefill_sm90_gqa_op(
        dtype=q.dtype,
        q_heads=q_heads,
        head_dim=int(q.shape[2]),
        head_dim_v=int(v_cache.shape[3]),
    )
    return op.run(
        q,
        k_cache,
        v_cache,
        out,
        lse=lse,
        region_counts=region_counts,
        region_phys_indices=region_phys_indices,
        work_q_global=work_q_global,
        work_q_local=work_q_local,
        work_q_len=work_q_len,
        seq_q=seq_q,
        softmax_scale=softmax_scale,
        stream=stream,
    )


def token_wise_flash_attn_prefill_union_sm90_gqa_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_local: torch.Tensor,
    work_q_len: torch.Tensor,
    union_logical_indices: Optional[torch.Tensor] = None,
    exact_mask_bits: Optional[torch.Tensor] = None,
    causal_limits: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    stream=None,
):
    if causal_limits is None:
        raise ValueError("GQA grouped-union prefill requires causal_limits")
    q, k_cache, v_cache, q_heads, head_dim = _require_paged_qkv(
        q,
        k_cache,
        v_cache,
        tensor_name="gqa grouped-union prefill",
        q_heads=(4, 8, 16),
        head_dims=(128, 192),
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(head_dim))
    if out is None:
        out = torch.empty_like(q)
    out = _require_cuda_contiguous_tensor(out, tensor_name="out")
    op = _get_token_wise_flash_attn_prefill_union_sm90_gqa_op(
        dtype=q.dtype,
        q_heads=q_heads,
        head_dim=head_dim,
        head_dim_v=int(v_cache.shape[3]),
    )
    return op.run(
        q,
        k_cache,
        v_cache,
        out,
        lse=lse,
        union_counts=union_counts,
        union_phys_indices=union_phys_indices,
        union_logical_indices=union_logical_indices,
        exact_mask_bits=exact_mask_bits,
        work_q_global=work_q_global,
        work_q_local=work_q_local,
        work_q_len=work_q_len,
        causal_limits=causal_limits,
        softmax_scale=float(softmax_scale),
        stream=stream,
    )


def token_wise_flash_attn_decode_sm90_gqa_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    region_counts: torch.Tensor,
    region_packed_indices: torch.Tensor,
    kv_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    variable_split_max: int = 1,
    mtp_q_len: int = 1,
    batch_invariant: bool = False,
    query_start_loc: Optional[torch.Tensor] = None,
    valid_rows: Optional[torch.Tensor] = None,
    stream=None,
):
    q = _require_cuda_contiguous_tensor(q, tensor_name="q")
    sm_count = None
    if int(variable_split_max) > 1:
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    op = _get_token_wise_flash_attn_decode_sm90_gqa_op(
        dtype=q.dtype,
        q_heads=int(q.shape[1]),
        head_dim=int(q.shape[2]),
        head_dim_v=int(v_cache.shape[3]),
        topk_windows=int(region_packed_indices.shape[-1]),
        mtp_q_len=int(mtp_q_len),
        batch_invariant=bool(batch_invariant),
        variable_split_max=int(variable_split_max),
        sm_count=None if sm_count is None else int(sm_count),
    )
    if out is None:
        _, _, work_items = op.get_variable_split_plan(int(q.shape[0]))
        out = torch.empty(
            (work_items, int(q.shape[1]), int(q.shape[2])),
            dtype=q.dtype,
            device=q.device,
        )
    out = _require_cuda_contiguous_tensor(out, tensor_name="out")
    return op.run(
        q,
        k_cache,
        v_cache,
        out,
        lse=lse,
        region_counts=region_counts,
        region_packed_indices=region_packed_indices,
        kv_seqlens=kv_seqlens,
        query_start_loc=query_start_loc,
        valid_rows=valid_rows,
        softmax_scale=softmax_scale,
        stream=stream,
    )


def token_wise_flash_attn_decode_mtp_union_exact_mask_sm90_gqa_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    union_counts: torch.Tensor,
    union_phys_indices: torch.Tensor,
    union_logical_indices: torch.Tensor,
    exact_mask_bits: torch.Tensor,
    work_q_global: torch.Tensor,
    work_q_input_local: torch.Tensor,
    work_q_output_base: torch.Tensor,
    work_q_len: torch.Tensor,
    causal_limits: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    stream=None,
) -> torch.Tensor:
    q, k_cache, v_cache, q_heads, head_dim = _require_paged_qkv(
        q,
        k_cache,
        v_cache,
        tensor_name="gqa MTP union exact-mask decode",
        q_heads=(16,),
        head_dims=(128, 192),
    )
    partial_out = _require_cuda_contiguous_tensor(
        partial_out, tensor_name="partial_out"
    )
    partial_lse = _require_cuda_contiguous_tensor(
        partial_lse, tensor_name="partial_lse"
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(head_dim))
    op = _get_token_wise_flash_attn_decode_mtp_union_exact_mask_sm90_gqa_op(
        dtype=q.dtype,
        q_heads=q_heads,
        head_dim=head_dim,
        head_dim_v=int(v_cache.shape[3]),
    )
    return op.run(
        q,
        k_cache,
        v_cache,
        partial_out,
        lse=partial_lse,
        union_phys_indices=union_phys_indices,
        union_logical_indices=union_logical_indices,
        union_counts=union_counts,
        exact_mask_bits=exact_mask_bits,
        work_q_global=work_q_global,
        work_q_input_local=work_q_input_local,
        work_q_local=work_q_output_base,
        work_q_len=work_q_len,
        causal_limits=causal_limits,
        softmax_scale=float(softmax_scale),
        stream=stream,
    )


def get_token_wise_flash_attn_decode_variable_split_plan_sm90_gqa(
    q: torch.Tensor,
    *,
    topk_windows: int,
    variable_split_max: int,
    batch_invariant: bool = False,
) -> tuple[int, int, int]:
    """Return the cached direct-decode split plan for a runtime batch.

    The plan is computed by the same CuTeDSL decode wrapper that launches the
    kernel. Keeping this query next to the wrapper avoids duplicating the SM
    balancing heuristic in the vLLM integration and lets that integration
    allocate graph-stable partial-output storage before launch.
    """
    q = _require_cuda_contiguous_tensor(q, tensor_name="q")
    split_max = int(variable_split_max)
    sm_count = None
    if split_max > 1:
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    op = _get_token_wise_flash_attn_decode_sm90_gqa_op(
        dtype=q.dtype,
        q_heads=int(q.shape[1]),
        head_dim=int(q.shape[2]),
        head_dim_v=int(q.shape[2]),
        topk_windows=int(topk_windows),
        variable_split_max=split_max,
        batch_invariant=bool(batch_invariant),
        sm_count=None if sm_count is None else int(sm_count),
    )
    return op.get_variable_split_plan(int(q.shape[0]))


def token_wise_flash_attn_decode_sm90_gqa_plan(
    batch: int,
    *,
    dtype: torch.dtype,
    q_heads: int,
    head_dim: int,
    head_dim_v: int,
    topk_windows: int,
    variable_split_max: int = 1,
    mtp_q_len: int = 1,
    batch_invariant: bool = False,
    sm_count: Optional[int] = None,
) -> tuple[int, int, int]:
    """Return (n_split4, n_split2, work_items) for a decode split-KV plan.

    Lets callers size partial (out, lse) workspaces without launching the kernel.
    work_items == batch when variable_split_max == 1 (no split).
    """
    op = _get_token_wise_flash_attn_decode_sm90_gqa_op(
        dtype=dtype,
        q_heads=int(q_heads),
        head_dim=int(head_dim),
        head_dim_v=int(head_dim_v),
        topk_windows=int(topk_windows),
        mtp_q_len=int(mtp_q_len),
        batch_invariant=bool(batch_invariant),
        variable_split_max=int(variable_split_max),
        sm_count=None if sm_count is None else int(sm_count),
    )
    return op.get_variable_split_plan(int(batch))


def token_wise_flash_attn_decode_split_sm90_gqa_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    region_counts: torch.Tensor,
    region_packed_indices: torch.Tensor,
    kv_seqlens: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    variable_split_max: int | str = "auto",
    stream=None,
):
    q = _require_cuda_contiguous_tensor(q, tensor_name="q")
    if isinstance(variable_split_max, str):
        if variable_split_max.strip().lower() != "auto":
            raise ValueError(
                "split decode string policy must be 'auto', got "
                f"{variable_split_max!r}"
            )
        variable_split_max = 4
    if int(variable_split_max) not in (2, 4):
        raise ValueError(
            "split decode requires variable_split_max in (2, 4), got "
            f"{variable_split_max}"
        )
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    op = _get_token_wise_flash_attn_decode_sm90_gqa_op(
        dtype=q.dtype,
        q_heads=int(q.shape[1]),
        head_dim=int(q.shape[2]),
        head_dim_v=int(v_cache.shape[3]),
        topk_windows=int(region_packed_indices.shape[-1]),
        variable_split_max=int(variable_split_max),
        sm_count=int(sm_count),
    )
    n_split4, n_split2, work_items = op.get_variable_split_plan(int(q.shape[0]))
    partial_out = _require_cuda_contiguous_tensor(
        partial_out, tensor_name="partial_out"
    )
    partial_lse = _require_cuda_contiguous_tensor(
        partial_lse, tensor_name="partial_lse"
    )
    if int(partial_out.shape[0]) < work_items or int(partial_lse.shape[0]) < work_items:
        raise ValueError(
            "split decode workspace is too small: "
            f"work_items={work_items}, partial_out={tuple(partial_out.shape)}, "
            f"partial_lse={tuple(partial_lse.shape)}"
        )
    partial_out = partial_out[:work_items]
    partial_lse = partial_lse[:work_items]
    op.run(
        q,
        k_cache,
        v_cache,
        partial_out,
        lse=partial_lse,
        region_counts=region_counts,
        region_packed_indices=region_packed_indices,
        kv_seqlens=kv_seqlens,
        softmax_scale=softmax_scale,
        stream=stream,
    )
    merge_variable_split_nat_lse_states_sm90_gqa(
        partial_out,
        partial_lse,
        out,
        lse,
        n_split4=n_split4,
        n_split2=n_split2,
        stream=stream,
    )
    return out


__all__ = [
    "get_token_wise_flash_attn_decode_variable_split_plan_sm90_gqa",
    "token_wise_flash_attn_decode_mtp_union_exact_mask_sm90_gqa_func",
    "token_wise_flash_attn_decode_sm90_gqa_func",
    "token_wise_flash_attn_decode_sm90_gqa_plan",
    "token_wise_flash_attn_decode_split_sm90_gqa_func",
    "token_wise_flash_attn_prefill_sm90_gqa_func",
    "token_wise_flash_attn_prefill_union_sm90_gqa_func",
]
