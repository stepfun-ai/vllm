# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 sliding-window decode must match FA2 with the headwise gate."""

import pytest
import torch

from vllm.models.step5.nvidia.ops.swa_decode import swa_decode_gated

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((9, 0), (10, 3)),
    reason="Hopper or B300",
)


@pytest.mark.parametrize("heads,kv_heads", [(8, 1), (64, 4)])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("splits", [1, 8])
@pytest.mark.parametrize("block", [32, 64])
@pytest.mark.parametrize("pack", [1, 4])
def test_swa_decode_matches_fa2(heads, kv_heads, query_len, splits, block, pack):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    if query_len % pack or pack * heads // kv_heads > 64:
        pytest.skip("pack must divide query_len and fit 64 rows")
    torch.manual_seed(0)
    device = torch.device("cuda")
    dim, window = 192, 512
    # Contexts shorter than, equal to and longer than the window; the last
    # request is graph padding with no visible keys.
    contexts = [5, 512, 700, 9000, 0]
    requests = len(contexts)
    per_request = max(contexts) // block + 2
    pages = requests * per_request
    raw = torch.randn(pages, block, kv_heads, 2 * dim, device=device).bfloat16()
    key, value = raw[..., :dim], raw[..., dim:]
    table = torch.randperm(pages, device=device).int().view(requests, per_request)
    seq_lens = torch.tensor(contexts, device=device, dtype=torch.int32)
    tokens = requests * query_len
    query = torch.randn(tokens, heads, dim, device=device).bfloat16()
    gate = torch.randn(tokens, heads, device=device).bfloat16()
    actual = torch.full_like(query, float("nan"))
    swa_decode_gated(
        query,
        key,
        value,
        table,
        seq_lens,
        gate,
        actual,
        scale=dim**-0.5,
        window=window,
        query_len=query_len,
        splits=splits,
        pack=pack,
    )
    live = [r for r, c in enumerate(contexts) if c >= query_len]
    rows = torch.cat([torch.arange(r * query_len, (r + 1) * query_len) for r in live])
    expected = torch.empty_like(query[rows])
    starts = torch.arange(len(live) + 1, device=device, dtype=torch.int32) * query_len
    flash_attn_varlen_func(
        q=query[rows],
        k=key,
        v=value,
        out=expected,
        cu_seqlens_q=starts,
        seqused_k=seq_lens[live],
        max_seqlen_q=query_len,
        max_seqlen_k=max(contexts),
        softmax_scale=dim**-0.5,
        causal=True,
        window_size=[window - 1, 0],
        block_table=table[live],
        fa_version=2,
        headwise_gate=gate[rows],
    )
    torch.testing.assert_close(actual[rows], expected, rtol=0.02, atol=4e-3)
    assert torch.equal(actual[-query_len:], torch.zeros_like(actual[-query_len:]))
