# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16]
IS_NEOX = [True, False]
EPS_VALUES = [1e-5, 1e-6]
SEEDS = [13]
PARTIAL_ROPE = [True, False]
CUDA_DEVICES = ["cuda:0"]


def _apply_qk_norm_rope(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rope: RotaryEmbedding,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> torch.Tensor:
    q_size = num_heads_q * head_dim
    kv_size = num_heads_kv * head_dim

    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
    q_by_head = q_norm.forward_native(q_by_head)
    assert isinstance(q_by_head, torch.Tensor)
    q = q_by_head.view(q.shape)

    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
    k_by_head = k_norm.forward_native(k_by_head)
    assert isinstance(k_by_head, torch.Tensor)
    k = k_by_head.view(k.shape)

    q, k = rope.forward_native(positions, q, k)
    return torch.cat([q, k, v], dim=-1)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_norm_rope custom op requires cuda and rocm platform",
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_neox", IS_NEOX)
@pytest.mark.parametrize("eps", EPS_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
@torch.inference_mode()
def test_fused_qk_norm_rope_matches_reference(
    default_vllm_config,
    device: str,
    dtype: torch.dtype,
    is_neox: bool,
    eps: float,
    seed: int,
    rotary_ratio: float,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    num_tokens = 4

    total_dim = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_fused = qkv_base.clone()
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)

    q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
    q_norm.weight.data.normal_(mean=1.0, std=0.1)
    k_norm.weight.data.normal_(mean=1.0, std=0.1)
    q_weight = q_norm.weight.data
    k_weight = k_norm.weight.data
    rotary_dim = int(head_dim * rotary_ratio)
    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=is_neox,
        dtype=dtype,
    ).to(device)

    ref_result = _apply_qk_norm_rope(
        qkv=qkv_base,
        positions=positions,
        q_norm=q_norm,
        k_norm=k_norm,
        rope=rope,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
    )

    opcheck(
        torch.ops._C.fused_qk_norm_rope,
        (
            qkv_fused.clone(),
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        ),
    )

    torch.ops._C.fused_qk_norm_rope(
        qkv_fused,
        num_heads,
        num_kv_heads,
        num_kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        rope.cos_sin_cache,
        is_neox,
        positions.view(-1),
    )

    if dtype == torch.float16:
        ATOL, RTOL = (2e-3, 2e-3)
    else:
        ATOL, RTOL = (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_fused,
        ref_result,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only cache fusion")
@pytest.mark.parametrize(
    "head_dim,rotary_dim",
    [(64, 32), (128, 64), (192, 0), (192, 64), (192, 192), (256, 128)],
)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["NHD", "HND", "PACKED"])
@pytest.mark.parametrize("num_tokens", [1, 129])
@torch.inference_mode()
def test_qknorm_rope_cache_preserves_norm_precision_and_slots(
    default_vllm_config, head_dim, rotary_dim, weight_dtype, layout, num_tokens
):
    """Cover Step-5 heads, strided projections and padded KV writes."""
    from vllm import _custom_ops as ops

    torch.manual_seed(17)
    nq, nkv, block_size = 8, 2, 16
    width = (nq + 2 * nkv) * head_dim
    qkvg = torch.randn(num_tokens, width + nq, device="cuda", dtype=torch.bfloat16)
    qkv = qkvg[:, :width]
    weights = [
        torch.randn(head_dim, device="cuda", dtype=weight_dtype) * 0.1 for _ in range(2)
    ]
    # Keep a real table even in NoPE mode; the kernel must not read it.
    cache_dim = rotary_dim or 64
    rope = RotaryEmbedding(
        head_dim, cache_dim, 2048, 10000.0, True, torch.bfloat16
    ).cuda()
    positions = torch.randint(2048, (num_tokens,), device="cuda")
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    if layout == "PACKED":
        storage = torch.full(
            (num_blocks, nkv, block_size, 2 * head_dim),
            17,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_cache, value_cache = storage.transpose(1, 2).split(head_dim, dim=-1)
    else:
        shape = (num_blocks, block_size, nkv, head_dim)
        if layout == "HND":
            shape = (num_blocks, nkv, block_size, head_dim)
        key_cache, value_cache = [
            torch.full(shape, 17, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        ]
        if layout == "HND":
            key_cache, value_cache = [
                x.transpose(1, 2) for x in (key_cache, value_cache)
            ]
    # Nonsequential slots, a negative slot, and a shorter mapping than QKV.
    slots = torch.arange(num_tokens, device="cuda", dtype=torch.long).flip(0)
    if num_tokens > 1:
        slots[1] = -1
        slots = slots[:-1]
    q_in, k_in, v_in = qkv.split([nq * head_dim, nkv * head_dim, nkv * head_dim], -1)
    ref = []
    for x, w in zip((q_in, k_in), weights):
        xf = x.reshape(num_tokens, -1, head_dim).float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
        ref.append((normalized * (w.float() + 1)).to(x.dtype).reshape(x.shape))
    q_ref, k_ref = ref
    if rotary_dim:
        q_ref, k_ref = rope(positions, q_ref, k_ref)
    expected_k, expected_v = key_cache.clone(), value_cache.clone()
    valid = slots >= 0
    expected_k[slots[valid] // block_size, slots[valid] % block_size] = k_ref[
        : slots.numel()
    ][valid].reshape(-1, nkv, head_dim)
    expected_v[slots[valid] // block_size, slots[valid] % block_size] = v_in[
        : slots.numel()
    ][valid].reshape(-1, nkv, head_dim)
    q, k, v = ops.fused_qknorm_rope_cache(
        qkv,
        weights[0],
        weights[1],
        rope.cos_sin_cache,
        positions,
        slots,
        key_cache,
        value_cache,
        nq,
        nkv,
        head_dim,
        rotary_dim,
        1e-5,
        norm_weight_bias=1.0,
    )
    torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v, v_in, rtol=0, atol=0)
    torch.testing.assert_close(key_cache, expected_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(value_cache, expected_v, rtol=0, atol=0)
    # Inserted keys must match the materialized K output exactly.
    torch.testing.assert_close(
        key_cache[slots[valid] // block_size, slots[valid] % block_size],
        k[: slots.numel()][valid].reshape(-1, nkv, head_dim),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only cache fusion")
@pytest.mark.parametrize("rotary_dim", [0, 64, 192])
@torch.inference_mode()
def test_qknorm_rope_cache_cuda_graph_updates_positions_and_slots(
    default_vllm_config, rotary_dim
):
    """Replay must read new positions/slots and update the intended cache pages."""
    from vllm import _custom_ops as ops

    head_dim, nq, nkv = 192, 8, 1
    qkv = torch.randn(2, (nq + 2 * nkv) * head_dim, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(head_dim, device="cuda") * 0.1 for _ in range(2)]
    positions = torch.tensor([1, 3], device="cuda")
    slots = torch.tensor([0, -1], device="cuda")
    rope = RotaryEmbedding(
        head_dim, rotary_dim or 64, 128, 10000.0, True, torch.bfloat16
    ).cuda()
    caches = [
        torch.zeros(2, 16, nkv, head_dim, device="cuda", dtype=qkv.dtype)
        for _ in range(2)
    ]
    args = (
        qkv,
        weights[0],
        weights[1],
        rope.cos_sin_cache,
        positions,
        slots,
        *caches,
        nq,
        nkv,
        head_dim,
        rotary_dim,
        1e-5,
        1.0,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.fused_qknorm_rope_cache(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = ops.fused_qknorm_rope_cache(*args)
    qkv.add_(0.125)
    positions.copy_(torch.tensor([7, 19], device="cuda"))
    slots.copy_(torch.tensor([18, 4], device="cuda"))
    for cache in caches:
        cache.zero_()
    graph.replay()
    graph_cache = [x.clone() for x in caches]
    for cache in caches:
        cache.zero_()
    eager = ops.fused_qknorm_rope_cache(*args)
    for actual, expected in zip(captured + tuple(graph_cache), eager + tuple(caches)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
