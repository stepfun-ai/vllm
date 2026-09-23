# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B300 DSA integration through MRV2 metadata: prefill and graph replay."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.step5.nvidia.flashinfer_sparse import Step5FlashInferSparse
from vllm.models.step5.nvidia.sparse_attention import (
    Step5DSAAttentionBackend,
    Step5DSAAttentionImpl,
    Step5DSAMetadataBuilder,
)
from vllm.models.step5.nvidia.sparse_summary_cache import Step5SparseSummaryCache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup


@pytest.fixture
def mrv2_metadata(default_vllm_config):
    """Use the runner's real metadata path instead of hand-built DSA metadata."""
    config = default_vllm_config
    config.model_config = SimpleNamespace(
        get_num_attention_heads=lambda _: 8,
        uses_alibi=False,
        rswa_window=None,
        is_mm_prefix_lm=False,
        hf_config=SimpleNamespace(model_type="step5"),
    )
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=192, dtype=torch.bfloat16
    )
    group = AttentionGroup(Step5DSAAttentionBackend, ["step5"], spec, 0)
    group.metadata_builders = [
        Step5DSAMetadataBuilder(spec, ["step5"], config, torch.device("cuda"))
    ]
    cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["step5"], spec)],
    )

    def build(
        *,
        num_actual_tokens,
        num_actual_reqs,
        max_query_len,
        seq_lens,
        slot_mapping,
        block_table,
        query_start_loc_cpu=None,
        for_capture=False,
    ):
        if query_start_loc_cpu is None:
            query_start_loc_cpu = torch.arange(num_actual_reqs + 1)
        starts = query_start_loc_cpu.to(torch.int32)
        return build_attn_metadata(
            attn_groups=[[group]],
            num_reqs=num_actual_reqs,
            num_tokens=num_actual_tokens,
            query_start_loc_gpu=starts.to(seq_lens.device),
            query_start_loc_cpu=starts,
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=block_table.shape[1] * 16,
            block_tables=[block_table],
            slot_mappings=slot_mapping.unsqueeze(0),
            kv_cache_config=cache_config,
            for_cudagraph_capture=for_capture,
        )["step5"]

    return build


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("n,topk", [(41, 2), (4105, 512)])
def test_dsa_chunked_prefill_and_decode_match_causal_reference(n, topk, mrv2_metadata):
    torch.manual_seed(7)
    device = torch.device("cuda")
    dim, heads = 192, 8
    q = torch.randn(n, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(n, 1, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(n, 1, 4, 256, device=device, dtype=torch.bfloat16)
    ik = torch.randn(n, 1, 256, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(ik)
    weights = torch.randn(n, 1, 4, device=device)
    pages = (n + 15) // 16
    table = torch.randperm(pages, device=device).int().unsqueeze(0)
    physical = table[0, torch.arange(n, device=device) // 16].long()
    offsets = torch.arange(n, device=device) % 16
    slots = physical * 16 + offsets
    kv = torch.zeros(pages, 1, 16, 2 * dim, device=device, dtype=q.dtype)
    kv[physical, 0, offsets, :dim] = k[:, 0]
    kv[physical, 0, offsets, dim:] = v[:, 0]
    impl = object.__new__(Step5DSAAttentionImpl)
    impl.adapter = Step5FlashInferSparse(device)
    impl.sparse_topk = topk
    impl.max_num_regions = (n + 7) // 8
    impl.scale = dim**-0.5

    def run(chunks):
        impl.summary_cache = Step5SparseSummaryCache(pages, 16, device)
        out = torch.empty_like(q)
        start = 0
        for count in chunks:
            end = start + count
            metadata = mrv2_metadata(
                num_actual_tokens=count,
                max_query_len=count,
                num_actual_reqs=1,
                query_start_loc_cpu=torch.tensor([0, count]),
                seq_lens=torch.tensor([end], device=device),
                slot_mapping=slots[start:end],
                block_table=table,
            )
            impl.forward(
                None,
                q[start:end],
                k[start:end],
                v[start:end],
                kv,
                metadata,
                output=out[start:end],
                dsa_proxy_query=iq[start:end],
                dsa_proxy_key=ik[start:end],
                dsa_proxy_weights=weights[start:end],
                dsa_proxy_z=z[start:end],
            )
            start = end
        return out

    full = run([n])
    incremental = run([13, n - 27] + [1] * 14)
    torch.testing.assert_close(full, incremental, rtol=0.015, atol=0.003)
    # Before top-k truncates history, the adapter must equal dense causal GQA.
    score = torch.einsum("qhd,kd->hqk", q[:24].float(), k[:24, 0].float()) / math.sqrt(
        dim
    )
    mask = (
        torch.arange(24, device=device)[None, :]
        > torch.arange(24, device=device)[:, None]
    )
    score.masked_fill_(mask, float("-inf"))
    expected = torch.einsum("hqk,kd->qhd", score.softmax(-1), v[:24, 0].float())
    torch.testing.assert_close(full[:24].float(), expected, rtol=0.025, atol=0.004)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("n,topk", [(43, 2), (4107, 512)])
def test_full_decode_graph_replay_matches_eager_with_padding(n, topk, mrv2_metadata):
    """Replay updates CSA across regions/pages and selects fresh KV each step."""
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, dim, heads = 2, 192, 8
    per_request = (n + 15) // 16
    pages = batch * per_request
    table = torch.randperm(pages, device=device).reshape(batch, per_request).int()
    q = torch.randn(batch, n, heads, dim, device=device, dtype=torch.bfloat16)
    iq = torch.randn(batch, n, 1, 4, 256, device=device, dtype=torch.bfloat16)
    ik = torch.randn(batch, n, 1, 256, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(ik)
    weights = torch.randn(batch, n, 1, 4, device=device)
    kv = torch.randn(pages, 1, 16, 2 * dim, device=device, dtype=q.dtype)
    impl = object.__new__(Step5DSAAttentionImpl)
    impl.adapter = Step5FlashInferSparse(device)
    impl.sparse_topk = topk
    impl.max_num_regions = (n + 7) // 8
    impl.scale = dim**-0.5
    cache = Step5SparseSummaryCache(pages, 16, device, batch)
    reference = Step5SparseSummaryCache(pages, 16, device, batch)
    prefix = n - 19
    for req in range(batch):
        logical = torch.arange(prefix, device=device)
        slots = (table[req, logical // 16].long() * 16 + logical % 16).tolist()
        for state in (cache, reference):
            state.update(ik[req, :prefix], z[req, :prefix], slots)
    inputs = [t[:, 0].clone() for t in (q, iq, ik, weights, z)]
    output = torch.empty_like(inputs[0])
    metadata = mrv2_metadata(
        for_capture=True,
        num_actual_tokens=batch,
        num_actual_reqs=batch,
        max_query_len=1,
        seq_lens=torch.ones(batch, device=device, dtype=torch.int32),
        slot_mapping=torch.full((batch,), -1, device=device, dtype=torch.long),
        block_table=table,
    )

    def forward():
        return impl.forward(
            None,
            inputs[0],
            None,
            None,
            kv,
            metadata,
            output=output,
            dsa_proxy_query=inputs[1],
            dsa_proxy_key=inputs[2],
            dsa_proxy_weights=inputs[3],
            dsa_proxy_z=inputs[4],
        )

    impl.summary_cache = cache
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    cursors = [prefix, prefix]
    for step in range(19):
        active = [True, step >= 5]
        expected = torch.zeros_like(output)
        for req in range(batch):
            pos = cursors[req]
            for dst, src in zip(inputs, (q, iq, ik, weights, z)):
                dst[req].copy_(src[req, pos])
            metadata.seq_lens[req] = pos + 1
            slot = int(table[req, pos // 16]) * 16 + pos % 16 if active[req] else -1
            metadata.slot_mapping[req] = slot
            if not active[req]:
                continue
            # Force the independently planned, variable-length eager path.
            ref_metadata = mrv2_metadata(
                num_actual_tokens=1,
                num_actual_reqs=1,
                max_query_len=2,
                query_start_loc_cpu=torch.tensor([0, 1]),
                seq_lens=metadata.seq_lens[req : req + 1],
                slot_mapping=metadata.slot_mapping[req : req + 1],
                block_table=table[req : req + 1],
            )
            impl.summary_cache = reference
            impl.forward(
                None,
                inputs[0][req : req + 1],
                None,
                None,
                kv,
                ref_metadata,
                output=expected[req : req + 1],
                dsa_proxy_query=inputs[1][req : req + 1],
                dsa_proxy_key=inputs[2][req : req + 1],
                dsa_proxy_weights=inputs[3][req : req + 1],
                dsa_proxy_z=inputs[4][req : req + 1],
            )
            cursors[req] += 1
        impl.summary_cache = cache
        graph.replay()
        torch.testing.assert_close(
            output,
            expected,
            rtol=0.015,
            atol=0.003,
            msg=lambda m, step=step: f"step={step}\n{m}",
        )
        assert torch.equal(cache.mean, reference.mean)
