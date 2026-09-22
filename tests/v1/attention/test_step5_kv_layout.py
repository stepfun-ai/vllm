# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5's canonical cache must preserve values and physical page identities."""

import math

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.models.step5.sparse_attention import (
    Step5DSAAttentionImpl,
    Step5SplitKVFlashAttentionBackend,
    Step5SplitKVFlashAttentionImpl,
    _step5_sparse_kv_view,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout


def _cache(block_size, heads, dim, device="cpu", dtype=torch.float32):
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=dim, dtype=dtype
    )
    # Two layers exercise main's actual allocation/view helpers, including
    # the nonzero storage offset of the second layer.
    raw = torch.empty(2 * 3 * spec.page_size_bytes, dtype=torch.int8, device=device)
    return dense_kv_cache_views(raw, spec, 3, 2, KVCacheLayout.LBNHC)


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("dim", [128, 192])
def test_step5_canonical_cache_preserves_page_ids_and_aliases(block_size, heads, dim):
    views = _cache(block_size, heads, dim)
    for layer, cache in enumerate(views):
        cache.copy_(torch.arange(cache.numel()).reshape(cache.shape) + layer * 1e7)
        sparse = _step5_sparse_kv_view(cache)
        impl = object.__new__(Step5DSAAttentionImpl)
        impl.head_size = dim
        key, value = impl._sparse_gqa_kv_cache_for_kernel(sparse)
        swa_key, swa_value = Step5SplitKVFlashAttentionImpl._split_kv_cache(impl, cache)
        ratio = block_size // 16
        for block in range(3):
            for subpage in range(ratio):
                page = block * ratio + subpage
                for half, pages, swa in ((0, key, swa_key), (1, value, swa_value)):
                    expected = cache[
                        block,
                        :,
                        subpage * 16 : (subpage + 1) * 16,
                        half * dim : (half + 1) * dim,
                    ].transpose(0, 1)
                    assert torch.equal(pages[page], expected)
                    assert pages[page].data_ptr() == expected.data_ptr()
                    assert (
                        swa[block, subpage * 16].data_ptr() == pages[page, 0].data_ptr()
                    )
        key[-1, -1, -1, -1] = -17
        value[-1, -1, -1, -1] = -29
        assert cache[-1, -1, -1, dim - 1] == -17
        assert cache[-1, -1, -1, -1] == -29
    assert Step5SplitKVFlashAttentionBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.LBNHC,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Step5 DSA kernels require Hopper (SM90)",
)
@pytest.mark.parametrize("dim", [128, 192])
@pytest.mark.parametrize("query_len", [1, 4])
def test_step5_packed_decode_matches_split_cache_and_reference(dim, query_len):
    from tests.kernels.core.test_step5_decode_batch_invariance import decode, make_case

    case = make_case(regions=64, batch=4, head_dim=dim)
    query, dense_key, dense_value, counts, packed, seq_lens = case
    # 8 scheduler blocks, each split into four 16-token kernel pages.
    storage = torch.empty((8, 64, 1, 2 * dim), device="cuda", dtype=query.dtype)
    canonical = storage.transpose(1, 2)
    sparse = _step5_sparse_kv_view(canonical)
    impl = object.__new__(Step5DSAAttentionImpl)
    key, value = impl._sparse_gqa_kv_cache_for_kernel(sparse)
    key.copy_(dense_key)
    value.copy_(dense_value)
    assert not key.is_contiguous() and not value.is_contiguous()
    expected, expected_lse, _ = decode(case, query_len)
    actual, actual_lse, _ = decode(
        (query, key, value, counts, packed, seq_lens), query_len
    )
    torch.cuda.synchronize()
    # Only storage strides changed, so the same arithmetic must be bitwise equal.
    assert torch.equal(actual, expected)
    assert torch.equal(actual_lse, expected_lse)
    scores = torch.einsum("bhd,nd->bhn", query.float(), dense_key.flatten(0, 2).float())
    reference = torch.einsum(
        "bhn,nd->bhd",
        (scores / math.sqrt(dim)).softmax(-1),
        dense_value.flatten(0, 2).float(),
    )
    torch.testing.assert_close(actual.float(), reference, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("num_blocks,num_heads", [(3, 1), (7, 2)])
def test_step5_summary_sidecar_uses_owner_block_axis(
    monkeypatch, num_blocks, num_heads
):
    from types import SimpleNamespace

    import vllm.distributed.kv_transfer as transfer
    from vllm.models.step5.sparse_summary_cache import Step5SparseSummaryCacheLayer

    monkeypatch.setattr(transfer, "has_kv_transfer_group", lambda: False)
    layer = object.__new__(Step5SparseSummaryCacheLayer)
    layer._main_layer_name = "owner"
    layer._kv_cache_block_size = 64
    layer._sparse_config = SimpleNamespace(proxy_dim=256)
    cache = SimpleNamespace(warmup_scheduler_block_reset=lambda: None)
    layer._target_impl = SimpleNamespace(
        summary_cache_num_proxy_kv_heads=1,
        sparse_region_block_size=8,
        _summary_cache=cache,
    )
    layer._ensure_shared_scratch_workspace = lambda context: None
    layer._bind_summary_cache = lambda backing, **kwargs: None
    layer._fixed_runtime_state_budget = lambda block_size: 0
    owner_cache = torch.empty(num_blocks, num_heads, 64, 384)
    layer.bind_kv_cache_side_storage({"owner": SimpleNamespace(kv_cache=owner_cache)})
    assert layer._kv_cache.shape[0] == num_blocks
    assert cache.blocks_per_scheduler_block == 1
    assert torch.count_nonzero(layer._kv_cache) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [128, 192])
def test_step5_sparse_interface_accepts_packed_views(dim):
    from importlib import import_module

    from vllm.models.step5.nvidia.ops.cute_dsl._cutlass_compat import apply_patches

    # Production imports the kernel module through the model's lazy facade.
    apply_patches()
    interface = import_module(
        "vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa.token_sparse_attn.interface"
    )
    _require_paged_qkv = interface._require_paged_qkv

    q = torch.empty(8, 16, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(2, 16, 1, 2 * dim, device="cuda", dtype=q.dtype)
    key, value = kv.split(dim, dim=-1)
    checked = _require_paged_qkv(
        q, key, value, tensor_name="test", q_heads=(16,), head_dims=(128, 192)
    )
    assert checked[1] is key and checked[2] is value
    with pytest.raises(ValueError, match="contiguous head dim"):
        _require_paged_qkv(
            q,
            key[..., ::2],
            value[..., ::2],
            tensor_name="test",
            q_heads=(16,),
            head_dims=(128, 192),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Step5 DSA kernels require Hopper (SM90)",
)
@pytest.mark.parametrize("dim", [128, 192])
def test_step5_packed_prefill_matches_split_cache_and_reference(dim):
    from vllm.models.step5.nvidia.ops.cute_dsl.sparse_gqa import (
        token_wise_flash_attn_prefill_union_sm90_gqa_func,
    )

    torch.manual_seed(20260922)
    q = torch.randn(8, 16, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 16, 1, 2 * dim, device="cuda", dtype=q.dtype)
    key, value = kv.split(dim, dim=-1)
    meta = dict(
        union_counts=torch.tensor([2], device="cuda", dtype=torch.int32),
        union_phys_indices=torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
        union_logical_indices=torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
        exact_mask_bits=torch.full((1, 2), 255, device="cuda", dtype=torch.int32),
        work_q_global=torch.tensor([0], device="cuda", dtype=torch.int32),
        work_q_local=torch.tensor([0], device="cuda", dtype=torch.int32),
        work_q_len=torch.tensor([8], device="cuda", dtype=torch.int32),
        causal_limits=torch.full((1, 8), 16, device="cuda", dtype=torch.int32),
    )
    expected, actual = torch.empty_like(q), torch.empty_like(q)
    token_wise_flash_attn_prefill_union_sm90_gqa_func(
        q, key.contiguous(), value.contiguous(), out=expected, **meta
    )
    token_wise_flash_attn_prefill_union_sm90_gqa_func(q, key, value, out=actual, **meta)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
    scores = torch.einsum("bhd,nd->bhn", q.float(), key.flatten(0, 2).float())
    reference = torch.einsum(
        "bhn,nd->bhd",
        (scores / math.sqrt(dim)).softmax(-1),
        value.flatten(0, 2).float(),
    )
    torch.testing.assert_close(actual.float(), reference, atol=2e-3, rtol=2e-2)
