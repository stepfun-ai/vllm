# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The joint Step5 QKV/indexer preparation used by decode and MTP verify."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.step5.nvidia.ops.indexer_norm_rope import indexer_norm_rope
from vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope import (
    qkv_indexer_norm_rope,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@pytest.mark.parametrize("groups,heads", [(1, 8), (4, 16)])
@pytest.mark.parametrize("fp8", [False, True])
def test_joint_prepare_without_csa_matches_separate_kernels(groups, heads, fp8):
    """MTP verify (and DEP's grouped layout) take the joint kernel without the
    CSA update; it must match the separate preparation bitwise."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    tokens, block, pages = 16, 32, 8
    qkv = torch.randn(tokens, (heads + 2) * groups * 192, device=device).bfloat16()
    q_weight, k_weight = (torch.rand(192, device=device).bfloat16() for _ in "qk")
    rope = torch.randn(4096, 64, device=device).bfloat16()
    positions = torch.randint(0, 4096, (tokens,), device=device)
    slots = torch.randperm(pages * block, device=device)[:tokens]
    index_q = torch.randn(tokens, groups * heads * 256, device=device).bfloat16()
    index_k = torch.randn(tokens, 256, device=device).bfloat16()
    weights = torch.randn(tokens, groups * heads, device=device).bfloat16()
    iq_weight = torch.rand(256, device=device).bfloat16()
    ik_weight = torch.rand(256, device=device).bfloat16()
    ik_bias = torch.randn(256, device=device).bfloat16()
    index_rope = torch.randn(4096, 64, device=device).float()
    scale = torch.full((), 0.5, device=device)
    dtype = torch.uint8 if fp8 else torch.bfloat16

    def cache():
        kv = torch.zeros(pages, groups, block, 384, device=device, dtype=dtype)
        return kv, *kv.transpose(1, 2).split(192, dim=-1)

    kv, key_cache, value_cache = cache()
    main = (
        qkv,
        q_weight,
        k_weight,
        rope,
        slots,
        key_cache,
        value_cache,
        groups * heads,
        groups,
        64,
        1e-6,
        0.0,
        scale,
        scale,
    )
    index_args = (
        index_q,
        index_k,
        weights,
        iq_weight,
        ik_weight,
        ik_bias,
        index_rope,
        positions,
        1e-6,
        False,
    )
    joint = qkv_indexer_norm_rope(*index_args, main_inputs=main)

    ref_kv, ref_key_cache, ref_value_cache = cache()
    q, k, v = ops.fused_qknorm_rope_cache(
        qkv,
        q_weight,
        k_weight,
        rope,
        positions,
        slots,
        ref_key_cache,
        ref_value_cache,
        groups * heads,
        groups,
        192,
        64,
        1e-6,
        0.0,
        "fp8_e4m3" if fp8 else "auto",
        scale,
        scale,
    )
    index = indexer_norm_rope(*index_args, num_groups=groups)
    for actual, expected in zip(joint, (*index, q, k, v)):
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8).reshape(-1),
            expected.contiguous().view(torch.uint8).reshape(-1),
            rtol=0,
            atol=0,
        )
    assert torch.equal(kv, ref_kv)
