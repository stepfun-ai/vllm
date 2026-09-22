# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.step5.kernels import (
    is_supported_step5_qknorm_cache_rotary,
)
from vllm.models.step5.nvidia.ops.cute_dsl import (
    fused_qknorm_rope_forward_impl,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


@pytest.mark.parametrize(
    "head_dim,rotary_pairs,expected",
    [
        (64, 0, True),
        (128, 16, True),
        (128, 32, True),
        (192, 96, True),
        (128, 19, False),
        (128, -4, False),
        (128, 68, False),
    ],
)
def test_step5_qknorm_cache_rotary_dispatch(
    head_dim: int,
    rotary_pairs: int,
    expected: bool,
) -> None:
    assert is_supported_step5_qknorm_cache_rotary(head_dim, rotary_pairs) is expected


def _reference(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_pairs: int,
    epsilon: float,
    norm_weight_bias: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # CuTeDSL is the bitwise source of truth for the fused native cache op.
    return fused_qknorm_rope_forward_impl(
        qkv.contiguous(),
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        head_dim,
        num_q_heads,
        num_kv_heads,
        rotary_pairs,
        epsilon,
        norm_weight_bias,
    )


def _make_cache(
    layout: str,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    if layout == "PACKED_NHD":
        storage = torch.empty((*shape[:-1], 2 * head_dim), device="cuda", dtype=dtype)
        key, value = storage.split(head_dim, dim=-1)
        key.fill_(17)
        value.fill_(19)
        return key, value
    if layout == "NHD":
        return (
            torch.full(shape, 17, device="cuda", dtype=dtype),
            torch.full(shape, 19, device="cuda", dtype=dtype),
        )
    key_storage = torch.full(
        (num_blocks, num_kv_heads, block_size, head_dim),
        17,
        device="cuda",
        dtype=dtype,
    )
    value_storage = torch.full(
        (num_blocks, num_kv_heads, block_size, head_dim),
        19,
        device="cuda",
        dtype=dtype,
    )
    return key_storage.permute(0, 2, 1, 3), value_storage.permute(0, 2, 1, 3)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The Step5 QKNorm+RoPE+cache kernel is CUDA-only",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype_mode", ["input", "float32"])
@pytest.mark.parametrize("layout", ["NHD", "HND", "PACKED_NHD"])
@pytest.mark.parametrize(
    "head_dim,rotary_pairs",
    [(64, 16), (128, 16), (128, 32), (192, 32), (192, 48), (192, 96), (256, 64)],
)
@torch.inference_mode()
def test_step5_qknorm_rope_cache_matches_reference(
    dtype: torch.dtype,
    weight_dtype_mode: str,
    layout: str,
    head_dim: int,
    rotary_pairs: int,
) -> None:
    set_random_seed(13)
    num_tokens = 5
    num_q_heads = 6
    num_kv_heads = 2
    num_blocks = 3
    block_size = 16
    packed_width = (num_q_heads + 2 * num_kv_heads) * head_dim

    # Step5 can split QKV from a wider QKVG projection. Keep that row stride
    # and use the real chunked RoPE-cache stride.
    qkvg = torch.randn(
        num_tokens,
        packed_width + num_q_heads,
        device="cuda",
        dtype=dtype,
    )
    qkv = qkvg[:, :packed_width]
    assert not qkv.is_contiguous()
    weight_dtype = dtype if weight_dtype_mode == "input" else torch.float32
    q_weight = torch.randn(head_dim, device="cuda", dtype=weight_dtype) * 0.1
    k_weight = torch.randn(head_dim, device="cuda", dtype=weight_dtype) * 0.1

    max_position = 32
    rope_base = torch.randn(
        max_position,
        2 * rotary_pairs,
        device="cuda",
        dtype=dtype,
    )
    cos, sin = rope_base.chunk(2, dim=-1)
    positions = torch.tensor([1, 7, 3, 9, 2], device="cuda", dtype=torch.long)
    # The final two padded QKV rows deliberately have no slot mapping.
    slot_mapping = torch.tensor(
        [0, block_size + 3, -1],
        device="cuda",
        dtype=torch.long,
    )
    epsilon = 1e-5
    norm_weight_bias = 1.0

    q_ref, k_ref, v_ref = _reference(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_pairs,
        epsilon,
        norm_weight_bias,
    )
    key_cache, value_cache = _make_cache(
        layout,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype,
    )
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    expected_key_cache[0, 0] = k_ref[0].view(num_kv_heads, head_dim)
    expected_value_cache[0, 0] = v_ref[0].view(num_kv_heads, head_dim)
    expected_key_cache[1, 3] = k_ref[1].view(num_kv_heads, head_dim)
    expected_value_cache[1, 3] = v_ref[1].view(num_kv_heads, head_dim)

    q_out = torch.empty_like(q_ref)
    k_out = torch.empty_like(k_ref)
    v_out = torch.empty_like(v_ref)
    torch.ops._C.optimus_fused_qknorm_rope_cache_bitwise(
        q_out,
        k_out,
        v_out,
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        slot_mapping,
        key_cache,
        value_cache,
        head_dim,
        num_q_heads,
        num_kv_heads,
        rotary_pairs,
        epsilon,
        norm_weight_bias,
    )

    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert torch.equal(v_out, v_ref)
    assert torch.equal(key_cache, expected_key_cache)
    assert torch.equal(value_cache, expected_value_cache)

    fast_key_cache, fast_value_cache = _make_cache(
        layout,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype,
    )
    expected_fast_key_cache = fast_key_cache.clone()
    expected_fast_value_cache = fast_value_cache.clone()
    q_fast = torch.empty_like(q_ref)
    k_fast = torch.empty_like(k_ref)
    v_fast = torch.empty_like(v_ref)
    torch.ops._C.optimus_fused_qknorm_rope_cache(
        q_fast,
        k_fast,
        v_fast,
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        slot_mapping,
        fast_key_cache,
        fast_value_cache,
        head_dim,
        num_q_heads,
        num_kv_heads,
        rotary_pairs,
        epsilon,
        norm_weight_bias,
    )
    expected_fast_key_cache[0, 0] = k_fast[0].view(num_kv_heads, head_dim)
    expected_fast_value_cache[0, 0] = v_fast[0].view(num_kv_heads, head_dim)
    expected_fast_key_cache[1, 3] = k_fast[1].view(num_kv_heads, head_dim)
    expected_fast_value_cache[1, 3] = v_fast[1].view(num_kv_heads, head_dim)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(q_fast, q_ref, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(k_fast, k_ref, rtol=tolerance, atol=tolerance)
    if rotary_pairs != head_dim // 4:
        assert torch.equal(q_fast, q_ref)
        assert torch.equal(k_fast, k_ref)
    assert torch.equal(v_fast, v_ref)
    assert torch.equal(fast_key_cache, expected_fast_key_cache)
    assert torch.equal(fast_value_cache, expected_fast_value_cache)
