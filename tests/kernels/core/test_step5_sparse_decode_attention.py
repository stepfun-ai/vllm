# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct paged DSA decode attention against a dense reference."""

import pytest
import torch

from vllm.models.step5.nvidia.ops.sparse_decode_attention import (
    sparse_decode_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((9, 0), (10, 3)),
    reason="Hopper or B300",
)


def _case(groups, heads, query_len, fp8, contexts, block=64, topk=512):
    device = torch.device("cuda")
    dim = 192
    requests = len(contexts)
    per_request = max(contexts) // block + 2
    pages = requests * per_request
    kv = torch.randn(pages, block, groups, 2 * dim, device=device).bfloat16()
    kv = kv.transpose(1, 2)
    table = torch.randperm(pages, device=device).int().view(requests, per_request)
    rows = requests * query_len * groups
    positions = torch.empty(rows, dtype=torch.long, device=device)
    regions = torch.full((rows, topk + 1), -1, dtype=torch.long, device=device)
    for row in range(rows):
        request, token = row // (groups * query_len), row // groups % query_len
        position = contexts[request] - query_len + token
        positions[row] = position
        current = position // 8
        history = torch.randperm(current, device=device)[: min(topk, current)]
        picked = torch.cat(
            [history.sort().values, torch.tensor([current], device=device)]
        )
        regions[row, topk + 1 - picked.numel() :] = picked
    active = torch.ones(rows, dtype=torch.bool, device=device)
    active[: groups * query_len] = False  # first request is padding
    query = torch.randn(rows, heads, dim, device=device).bfloat16()
    scales = (None, None)
    reference_kv = kv.float()
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
        scales = (torch.tensor(0.5, device=device), torch.tensor(0.25, device=device))
        reference_kv = kv.float()
        reference_kv[..., :dim] *= 0.5
        reference_kv[..., dim:] *= 0.25
    expected = torch.zeros(rows, heads, dim, device=device)
    for row in range(rows):
        if not active[row]:
            continue
        request, group = row // (groups * query_len), row % groups
        tokens = (
            regions[row][regions[row] >= 0][:, None] * 8
            + torch.arange(8, device=device)
        ).flatten()
        tokens = tokens[tokens <= positions[row]]
        page = table[request, tokens // block].long()
        cached = reference_kv[page, group, tokens % block]
        key, value = cached[:, :dim], cached[:, dim:]
        weights = (query[row].float() @ key.T * dim**-0.5).softmax(-1)
        expected[row] = weights @ value
    return query, kv, regions, table, positions, active, scales, expected


@pytest.mark.parametrize("groups,heads", [(1, 8), (4, 16)])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("splits", [1, 8])
def test_sparse_decode_matches_reference(groups, heads, query_len, fp8, splits):
    torch.manual_seed(3)
    # Short (left-padded selection), medium, and beyond the top-k budget.
    query, kv, regions, table, positions, active, scales, expected = _case(
        groups, heads, query_len, fp8, [300, 97, 1024, 6000]
    )
    output = torch.full_like(query, float("nan"))
    sparse_decode_attention(
        query,
        kv,
        regions,
        table,
        positions,
        active,
        output,
        scale=192**-0.5,
        query_len=query_len,
        splits=splits,
        k_scale=scales[0],
        v_scale=scales[1],
    )
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=3e-3)


@pytest.mark.parametrize("groups,heads", [(1, 8), (4, 16)])
@pytest.mark.parametrize("splits", [1, 8])
def test_sparse_decode_fused_gate_matches_gate_kernel(groups, heads, splits):
    """The fused gate equals attention followed by the headwise gate kernel."""
    from vllm.models.step5.nvidia.ops.attention_gate import headwise_sigmoid_gate

    torch.manual_seed(7)
    query, kv, regions, table, positions, active, _, _ = _case(
        groups, heads, 1, False, [300, 2000, 5000]
    )
    tokens = query.shape[0] // groups
    # Row-strided like a projection split, one gate per head of every group.
    gate = torch.randn(tokens, groups * heads + 5, device=query.device).bfloat16()
    gate = gate[:, : groups * heads]
    args = (query, kv, regions, table, positions, active)
    plain = torch.empty_like(query)
    sparse_decode_attention(*args, plain, scale=192**-0.5, splits=splits)
    expected = headwise_sigmoid_gate(plain.reshape(tokens, -1), gate)
    fused = torch.empty_like(query)
    sparse_decode_attention(*args, fused, scale=192**-0.5, splits=splits, gate=gate)
    torch.testing.assert_close(fused.reshape(tokens, -1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("topk", [2, 7, 512])
def test_unsorted_selection_keeps_decode_layout(topk):
    """sort=False picks the same regions, left-padded with current last."""
    from vllm.models.step5.nvidia.ops.sparse_indexer import select_regions

    torch.manual_seed(0)
    device = torch.device("cuda")
    lengths = [3, 511, 512, 700, 2048]
    scores = torch.randn(len(lengths), 2048, device=device)
    scores[3, :100] = 0.5
    current = torch.tensor(lengths, device=device)
    expected = select_regions(scores, current, topk)
    actual = select_regions(scores, current, topk, backend="per_row", sort=False)
    for row, length in enumerate(lengths):
        live = min(length, topk)
        assert (actual[row, : topk - live] == -1).all()
        assert actual[row, topk] == length
        assert torch.equal(actual[row], expected[row])
        reference = scores[row, :length].argsort(descending=True, stable=True)[:live]
        assert torch.equal(
            actual[row, topk - live : topk].sort().values, reference.sort().values
        )


@pytest.mark.parametrize("heads,next_n", [(4, 2), (8, 1), (8, 2), (16, 1)])
def test_hopper_paged_scores_match_reference(heads, next_n):
    """Packed scales, query pairs and page boundaries preserve weighted logits."""
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import paged_mqa_logits_sm90

    torch.manual_seed(29)
    device = torch.device("cuda")
    batch, regions, block, dim = 3, 257, 32, 256
    pages_per_request = (regions + block - 1) // block
    pages = batch * pages_per_request
    table = torch.randperm(pages, device=device).int().view(batch, pages_per_request)
    lengths = torch.tensor([0, 33, regions], dtype=torch.int32, device=device)
    keys = torch.randn(pages, block, dim, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(pages, block, device=device) + 0.25
    packed = torch.empty((pages, block * (dim + 4)), dtype=torch.uint8, device=device)
    packed[:, : block * dim] = keys.view(torch.uint8).reshape(pages, -1)
    packed[:, block * dim :] = scales.view(torch.uint8).reshape(pages, -1)
    packed = packed.view(pages, block, 1, dim + 4)
    q = torch.randn(batch, next_n, heads + 4, dim, device=device).to(
        torch.float8_e4m3fn
    )
    q = q[:, :, :heads]  # Preserve the strides of a projection split.
    weights = torch.randn(batch * next_n, heads + 4, device=device)[:, :heads]
    actual = paged_mqa_logits_sm90(q, packed, weights, lengths, table, regions)
    for request, length in enumerate(lengths.tolist()):
        rows = slice(request * next_n, (request + 1) * next_n)
        positions = torch.arange(length, device=device)
        physical = table[request, positions // block].long()
        k = keys.float()[physical, positions % block].double()
        dot = torch.einsum("nhd,rd->nhr", q[request].double(), k).clamp_min(0)
        expected = (dot * weights[rows].double()[:, :, None]).sum(1)
        expected *= scales[physical, positions % block].double()
        torch.testing.assert_close(
            actual[rows, :length].double(), expected, rtol=2e-5, atol=1e-4
        )
        assert torch.isneginf(actual[rows, length:]).all()
