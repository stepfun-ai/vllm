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
        dtype=torch.bfloat16,
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
        block_size=16,
        request_state=None,
        impl=None,
        groups=1,
        seq_lens_cpu_upper_bound=None,
        positions=None,
    ):
        """``impl`` with ``seq_lens_cpu_upper_bound`` plans in the builder."""
        if query_start_loc_cpu is None:
            query_start_loc_cpu = torch.arange(num_actual_reqs + 1)
        starts = query_start_loc_cpu.to(torch.int32)
        builder = group.metadata_builders[0]
        builder.step5_request_state = request_state
        builder.step5_impl = impl
        builder.kv_cache_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=groups,
            head_size=192,
            dtype=torch.bfloat16,
        )
        return build_attn_metadata(
            attn_groups=[[group]],
            num_reqs=num_actual_reqs,
            num_tokens=num_actual_tokens,
            query_start_loc_gpu=starts.pin_memory().to(
                seq_lens.device, non_blocking=True
            ),
            query_start_loc_cpu=starts,
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=block_table.shape[1] * block_size,
            block_tables=[block_table],
            slot_mappings=slot_mapping.unsqueeze(0),
            kv_cache_config=cache_config,
            for_cudagraph_capture=for_capture,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            positions=positions,
        )["step5"]

    return build


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("n,topk", [(41, 2), (97, 512), (4105, 512)])
@pytest.mark.parametrize("groups,heads_per_group", [(1, 8), (2, 16), (4, 16)])
@pytest.mark.parametrize("block", [16, 32])
def test_dsa_chunked_prefill_and_decode_match_causal_reference(
    n, topk, groups, heads_per_group, block, mrv2_metadata, workspace_init
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    dim, heads = 192, groups * heads_per_group
    q = torch.randn(n, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(n, groups, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(n, groups, 4, 256, device=device, dtype=torch.bfloat16)
    ik = torch.randn(n, 1, 256, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(ik)
    weights = torch.randn(n, groups, 4, device=device)
    pages = (n + block - 1) // block
    table = torch.randperm(pages, device=device).int().unsqueeze(0)
    physical = table[0, torch.arange(n, device=device) // block].long()
    offsets = torch.arange(n, device=device) % block
    slots = physical * block + offsets
    kv = torch.zeros(
        pages, block, groups, 2 * dim, device=device, dtype=q.dtype
    ).transpose(1, 2)
    kv[physical, :, offsets, :dim] = k
    kv[physical, :, offsets, dim:] = v
    impl = object.__new__(Step5DSAAttentionImpl)
    impl.adapter = Step5FlashInferSparse(device)
    impl.sparse_topk = topk
    impl.max_num_regions = (n + 7) // 8
    impl.scale = dim**-0.5

    def run(chunks, group=None):
        group_slice = slice(None) if group is None else slice(group, group + 1)
        head_slice = (
            slice(None)
            if group is None
            else slice(group * heads_per_group, (group + 1) * heads_per_group)
        )
        run_q = q[:, head_slice].contiguous()
        run_kv = kv if group is None else kv[:, group_slice].contiguous()
        impl.summary_cache = Step5SparseSummaryCache(pages, block, device)
        out = torch.full_like(run_q, float("nan"))
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
                block_size=block,
            )
            impl.forward(
                None,
                run_q[start:end],
                k[start:end, group_slice],
                v[start:end, group_slice],
                run_kv,
                metadata,
                output=out[start:end],
                dsa_proxy_query=iq[start:end, group_slice],
                dsa_proxy_key=ik[start:end],
                dsa_proxy_weights=weights[start:end, group_slice],
                dsa_proxy_z=z[start:end],
            )
            start = end
        return out

    full = run([n])
    incremental = run([13, n - 27] + [1] * 14)
    torch.testing.assert_close(full, incremental, rtol=0.015, atol=0.003)
    if groups > 1:
        for group in range(groups):
            independent = run([n], group=group)
            torch.testing.assert_close(
                full[:, group * heads_per_group : (group + 1) * heads_per_group],
                independent,
                rtol=0,
                atol=0,
            )
    # Before top-k truncates history, the adapter must equal dense causal GQA.
    score = torch.einsum(
        "qhd,khd->hqk",
        q[:24].float(),
        k[:24].repeat_interleave(heads_per_group, dim=1).float(),
    ) / math.sqrt(dim)
    mask = (
        torch.arange(24, device=device)[None, :]
        > torch.arange(24, device=device)[:, None]
    )
    score.masked_fill_(mask, float("-inf"))
    expected = torch.einsum(
        "hqk,khd->qhd",
        score.softmax(-1),
        v[:24].repeat_interleave(heads_per_group, dim=1).float(),
    )
    torch.testing.assert_close(full[:24].float(), expected, rtol=0.025, atol=0.004)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("block", [16, 32])
def test_fp8_prefill_matches_bf16_on_dequantized_cache(
    groups, block, mrv2_metadata, workspace_init
):
    """FP8 and strided multi-group caches stage pages for the paged kernels.

    Staging dequantizes exactly into BF16, so FP8 must equal BF16 run on the
    dequantized values, including past the top-k history limit.
    """
    torch.manual_seed(11)
    device = torch.device("cuda")
    n, topk, dim, heads_per_group = 4200, 512, 192, 16
    heads = groups * heads_per_group
    q = torch.randn(n, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(n, groups, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    iq = torch.randn(n, groups, 4, 256, device=device, dtype=torch.bfloat16)
    ik = torch.randn(n, 1, 256, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(ik)
    weights = torch.randn(n, groups, 4, device=device)
    pages = (n + block - 1) // block
    table = torch.randperm(pages, device=device).int().unsqueeze(0)
    physical = table[0, torch.arange(n, device=device) // block].long()
    offsets = torch.arange(n, device=device) % block
    slots = physical * block + offsets
    kv = torch.zeros(
        pages, block, groups, 2 * dim, device=device, dtype=torch.float8_e4m3fn
    ).transpose(1, 2)
    kv[physical, :, offsets, :dim] = (k * 4).to(kv.dtype)
    kv[physical, :, offsets, dim:] = (v * 2).to(kv.dtype)
    layer = SimpleNamespace(
        _k_scale=torch.tensor(0.25, device=device),
        _v_scale=torch.tensor(0.5, device=device),
    )
    dequantized = torch.cat(
        (
            kv[..., :dim].to(torch.bfloat16) * 0.25,
            kv[..., dim:].to(torch.bfloat16) * 0.5,
        ),
        dim=-1,
    )

    def run(cache):
        impl = object.__new__(Step5DSAAttentionImpl)
        impl.adapter = Step5FlashInferSparse(device)
        impl.sparse_topk = topk
        impl.max_num_regions = (n + 7) // 8
        impl.scale = dim**-0.5
        impl.summary_cache = Step5SparseSummaryCache(pages, block, device)
        out = torch.full_like(q, float("nan"))
        metadata = mrv2_metadata(
            num_actual_tokens=n,
            max_query_len=n,
            num_actual_reqs=1,
            query_start_loc_cpu=torch.tensor([0, n]),
            seq_lens=torch.tensor([n], device=device),
            slot_mapping=slots,
            block_table=table,
            block_size=block,
        )
        impl.forward(
            layer,
            q,
            k,
            v,
            cache,
            metadata,
            output=out,
            dsa_proxy_query=iq,
            dsa_proxy_key=ik,
            dsa_proxy_weights=weights,
            dsa_proxy_z=z,
        )
        return out

    torch.testing.assert_close(run(kv), run(dequantized), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("n,topk", [(43, 2), (4107, 512)])
@pytest.mark.parametrize("output_padding", [0, 2])
@pytest.mark.parametrize("groups", [1, 4])
def test_full_decode_graph_replay_matches_eager_with_padding(
    n, topk, output_padding, groups, mrv2_metadata, workspace_init
):
    """Replay updates CSA across regions/pages and selects fresh KV each step."""
    torch.manual_seed(17)
    device = torch.device("cuda")
    batch, dim, heads = 2, 192, 8 if groups == 1 else groups * 16
    per_request = (n + 15) // 16
    pages = batch * per_request
    table = torch.randperm(pages, device=device).reshape(batch, per_request).int()
    q = torch.randn(batch, n, heads, dim, device=device, dtype=torch.bfloat16)
    iq = torch.randn(batch, n, groups, 4, 256, device=device, dtype=torch.bfloat16)
    ik = torch.randn(batch, n, 1, 256, device=device, dtype=torch.bfloat16)
    z = torch.randn_like(ik)
    weights = torch.randn(batch, n, groups, 4, device=device)
    kv = torch.randn(
        pages, 16, groups, 2 * dim, device=device, dtype=q.dtype
    ).transpose(1, 2)
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
    output = torch.full(
        (batch + output_padding, heads, dim),
        float("nan"),
        device=device,
        dtype=inputs[0].dtype,
    )
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
        active = [False, False] if step == 0 else [True, step >= 5]
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
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            output,
            expected,
            rtol=0.015,
            atol=0.003,
            msg=lambda m, step=step: f"step={step}\n{m}",
        )
        assert torch.equal(cache.mean, reference.mean)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
# BF16 4-head queries use the packed scoring fallback; the production FP8
# group-major layout with block 64 uses direct summary scores.
@pytest.mark.parametrize("block,fp8_indexer", [(16, False), (64, True)])
def test_mtp_verify_graph_decode_matches_prefill_path(
    groups, num_speculative_tokens, block, fp8_indexer, mrv2_metadata, workspace_init
):
    """MTP verify as multi-token decode, replayed from one FULL CUDA graph.

    Each step verifies k + 1 tokens per request and rejects a random suffix;
    the graph must match the eager prefill path, including summary undo.
    """
    torch.manual_seed(23)
    device = torch.device("cuda")
    width = num_speculative_tokens + 1
    batch, dim, topk = 3, 192, 512
    index_heads = 8 if fp8_indexer else 4
    heads = 8 if groups == 1 else groups * 16
    history = [4150, 4203, 700]
    total = max(history) + 12 * width
    per_request = (total + block - 1) // block
    pages = batch * per_request
    table = torch.randperm(pages, device=device).reshape(batch, per_request).int()
    request_state = torch.tensor([5, 0, 2], device=device)
    q_all = torch.randn(batch, total, heads, dim, device=device).bfloat16()
    k_all = torch.randn(batch, total, groups, dim, device=device).bfloat16()
    v_all = torch.randn_like(k_all)
    iq_all = torch.randn(batch, total, groups, index_heads, 256, device=device)
    iq_all = iq_all.to(torch.float8_e4m3fn if fp8_indexer else torch.bfloat16)
    ik_all = torch.randn(batch, total, 1, 256, device=device).bfloat16()
    z_all = torch.randn_like(ik_all)
    w_all = torch.randn(batch, total, groups, index_heads, device=device)
    kv = torch.zeros(
        pages, block, groups, 2 * dim, device=device, dtype=torch.bfloat16
    ).transpose(1, 2)

    def slots_of(req, positions):
        return table[req, positions // block].long() * block + positions % block

    def write_kv(req, positions):
        physical = table[req, positions // block].long()
        kv[physical, :, positions % block, :dim] = k_all[req, positions]
        kv[physical, :, positions % block, dim:] = v_all[req, positions]

    def make_impl():
        impl = object.__new__(Step5DSAAttentionImpl)
        impl.adapter = Step5FlashInferSparse(device)
        impl.sparse_topk = topk
        impl.max_num_regions = per_request * block // 8
        impl.scale = dim**-0.5
        impl.num_speculative_tokens = num_speculative_tokens
        impl.summary_cache = Step5SparseSummaryCache(
            pages, block, device, 8, request_journal_width=width
        )
        return impl

    def run_prefill_path(impl, rows, seq_lens, slots, starts, out):
        metadata = mrv2_metadata(
            num_actual_tokens=rows.numel(),
            num_actual_reqs=len(starts) - 1,
            # One more than any query forces the eager prefill path.
            max_query_len=int((starts[1:] - starts[:-1]).max()) + 1,
            query_start_loc_cpu=starts,
            seq_lens=seq_lens,
            slot_mapping=slots,
            block_table=table[: len(starts) - 1],
            request_state=request_state[: len(starts) - 1],
        )
        req, pos = rows // total, rows % total

        def group_major(t):
            return t.transpose(0, 1).contiguous().transpose(0, 1)

        impl.forward(
            None,
            q_all[req, pos],
            None,
            None,
            kv,
            metadata,
            output=out,
            # The indexer's group-major layout, as in the model.
            dsa_proxy_query=group_major(iq_all[req, pos]),
            dsa_proxy_key=ik_all[req, pos],
            dsa_proxy_weights=group_major(w_all[req, pos]),
            dsa_proxy_z=z_all[req, pos],
        )

    graph_impl, reference = make_impl(), make_impl()
    lengths = list(history)
    # Committed history: one prefill batch through both implementations.
    rows = torch.cat(
        [r * total + torch.arange(h, device=device) for r, h in enumerate(history)]
    )
    for req, h in enumerate(history):
        write_kv(req, torch.arange(h, device=device))
    starts = torch.tensor([0, *torch.tensor(history).cumsum(0).tolist()])
    seq = torch.tensor(history, device=device, dtype=torch.int32)
    slots = torch.cat(
        [slots_of(r, torch.arange(h, device=device)) for r, h in enumerate(history)]
    )
    for impl in (graph_impl, reference):
        scratch = torch.empty(rows.numel(), heads, dim, device=device).bfloat16()
        run_prefill_path(impl, rows, seq, slots, starts, scratch)

    # Persistent graph inputs; warm up with inactive rows before capture.
    tokens = batch * width
    seq_buf = torch.ones(batch, device=device, dtype=torch.int32)
    slot_buf = torch.full((tokens,), -1, device=device, dtype=torch.long)
    q_buf = torch.zeros(tokens, heads, dim, device=device).bfloat16()
    # Group-major storage, as the indexer produces for the decode kernels.
    iq_buf = torch.zeros(
        groups, tokens, index_heads, 256, device=device, dtype=iq_all.dtype
    ).transpose(0, 1)
    ik_buf = torch.zeros(tokens, 1, 256, device=device).bfloat16()
    z_buf = torch.zeros_like(ik_buf)
    w_buf = torch.zeros(groups, tokens, index_heads, device=device).transpose(0, 1)
    out_buf = torch.empty_like(q_buf)
    uniform_starts = torch.arange(batch + 1) * width
    metadata = mrv2_metadata(
        for_capture=True,
        num_actual_tokens=tokens,
        num_actual_reqs=batch,
        max_query_len=width,
        query_start_loc_cpu=uniform_starts,
        seq_lens=seq_buf,
        slot_mapping=slot_buf,
        block_table=table,
        request_state=request_state,
    )

    def forward():
        graph_impl.forward(
            None,
            q_buf,
            None,
            None,
            kv,
            metadata,
            output=out_buf,
            dsa_proxy_query=iq_buf,
            dsa_proxy_key=ik_buf,
            dsa_proxy_weights=w_buf,
            dsa_proxy_z=z_buf,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()

    generator = torch.Generator().manual_seed(29)
    for step in range(12):
        positions = torch.cat(
            [lengths[r] + torch.arange(width, device=device) for r in range(batch)]
        )
        owners = torch.arange(batch, device=device).repeat_interleave(width)
        rows = owners * total + positions
        for req in range(batch):
            write_kv(req, positions[req * width : (req + 1) * width])
        seq_buf.copy_(torch.tensor(lengths, device=device) + width)
        slot_buf.copy_(slots_of(owners, positions))
        for buf, src in zip(
            (q_buf, iq_buf, ik_buf, z_buf, w_buf),
            (q_all, iq_all, ik_all, z_all, w_all),
        ):
            buf.copy_(src[owners, positions])
        graph.replay()
        expected = torch.empty_like(out_buf)
        run_prefill_path(
            reference,
            rows,
            seq_buf.clone(),
            slot_buf.clone(),
            uniform_starts,
            expected,
        )
        torch.testing.assert_close(
            out_buf,
            expected,
            rtol=0.015,
            atol=0.003,
            msg=lambda m, step=step: f"step={step}\n{m}",
        )
        assert torch.equal(graph_impl.summary_cache.mean, reference.summary_cache.mean)
        for req in range(batch):
            lengths[req] += int(torch.randint(1, width + 1, (), generator=generator))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
def test_builder_planned_prefill_is_sync_free(
    groups, fp8, speculative, mrv2_metadata, workspace_init
):
    """The metadata builder plans prefill so the forward never syncs.

    With speculative decoding the host only has upper-bound lengths; the
    plan must still match exact in-forward planning bit for bit.
    """
    torch.manual_seed(31)
    device = torch.device("cuda")
    dim, block, topk, hpg = 192, 32, 512, 8 if groups == 1 else 16
    heads = groups * hpg
    width = 3
    history, prompt = 4150, 700
    total = history + 64
    per_request = (total + block - 1) // block
    pages = 2 * per_request
    table = torch.randperm(pages, device=device).reshape(2, per_request).int()
    q_all = torch.randn(2, total, heads, dim, device=device).bfloat16()
    k_all = torch.randn(2, total, groups, dim, device=device).bfloat16()
    v_all = torch.randn_like(k_all)
    iq_all = torch.randn(2, total, groups, 4, 256, device=device).bfloat16()
    ik_all = torch.randn(2, total, 1, 256, device=device).bfloat16()
    z_all = torch.randn_like(ik_all)
    w_all = torch.randn(2, total, groups, 4, device=device)
    kv = torch.zeros(pages, block, groups, 2 * dim, device=device).bfloat16()
    kv = kv.transpose(1, 2)
    for req in range(2):
        pos = torch.arange(total, device=device)
        physical = table[req, pos // block].long()
        kv[physical, :, pos % block, :dim] = k_all[req]
        kv[physical, :, pos % block, dim:] = v_all[req]
    if fp8:
        kv = kv.to(torch.float8_e4m3fn)
    layer = SimpleNamespace(
        _k_scale=torch.ones((), device=device), _v_scale=torch.ones((), device=device)
    )
    request_state = torch.tensor([3, 1], device=device)

    def make_impl():
        impl = object.__new__(Step5DSAAttentionImpl)
        impl.adapter = Step5FlashInferSparse(device)
        impl.sparse_topk = topk
        impl.max_num_regions = per_request * block // 8
        impl.scale = dim**-0.5
        impl.num_heads = heads
        impl.kv_cache_dtype = "fp8" if fp8 else "auto"
        impl.num_speculative_tokens = width - 1 if speculative else 0
        impl.summary_cache = Step5SparseSummaryCache(
            pages, block, device, 4, request_journal_width=width if speculative else 0
        )
        return impl

    def run(impl, owners, positions, seq, *, plan_in_builder, upper_bound=None):
        n = owners.numel()
        starts = torch.tensor([0, int((owners == 0).sum()), n])
        slots = table[owners, positions // block].long() * block + positions % block
        seq = torch.tensor(seq, device=device, dtype=torch.int32)
        upper = seq.cpu() if upper_bound is None else torch.tensor(upper_bound)
        torch.accelerator.synchronize()
        # Builder planning must not sync either: it overlaps the previous step.
        torch.cuda.set_sync_debug_mode("error" if plan_in_builder else "default")
        metadata = mrv2_metadata(
            num_actual_tokens=n,
            num_actual_reqs=2,
            max_query_len=int((starts[1:] - starts[:-1]).max()),
            query_start_loc_cpu=starts,
            seq_lens=seq,
            slot_mapping=slots,
            block_table=table,
            block_size=block,
            groups=groups,
            request_state=request_state,
            impl=impl if plan_in_builder else None,
            seq_lens_cpu_upper_bound=upper,
            positions=positions,
        )
        torch.cuda.set_sync_debug_mode("default")
        assert (metadata.step5_runtime is not None) == plan_in_builder
        out = torch.empty(n, heads, dim, device=device).bfloat16()
        torch.accelerator.synchronize()
        # Any CPU-GPU synchronization inside the step raises.
        mode = "error" if plan_in_builder else "default"
        torch.cuda.set_sync_debug_mode(mode)
        try:
            impl.forward(
                layer,
                q_all[owners, positions],
                None,
                None,
                kv,
                metadata,
                output=out,
                dsa_proxy_query=iq_all[owners, positions],
                dsa_proxy_key=ik_all[owners, positions],
                dsa_proxy_weights=w_all[owners, positions],
                dsa_proxy_z=z_all[owners, positions],
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        return out

    planned, reference = make_impl(), make_impl()
    # History for request 1, planned in the forward on both.
    owners = torch.tensor([0] * 8 + [1] * history, device=device)
    positions = torch.cat(
        [torch.arange(8, device=device), torch.arange(history, device=device)]
    )
    for impl in (planned, reference):
        run(impl, owners, positions, [8, history], plan_in_builder=False)
    # Mixed step: request 0 prefills a chunk; request 1 verifies a window (or
    # prefills a short chunk). Speculative hosts see an upper bound for 1.
    owners = torch.tensor([0] * prompt + [1] * width, device=device)
    positions = torch.cat(
        [
            8 + torch.arange(prompt, device=device),
            history + torch.arange(width, device=device),
        ]
    )
    seq = [8 + prompt, history + width]
    upper = [8 + prompt, history + width + 2] if speculative else None
    expected = run(reference, owners, positions, seq, plan_in_builder=False)
    actual = run(
        planned, owners, positions, seq, plan_in_builder=True, upper_bound=upper
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(planned.summary_cache.mean, reference.summary_cache.mean)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="B300 integration",
)
def test_grouped_decode_rows_match_per_group_kernels():
    """DEP folds (token, KV group) into rows; each row must see its group only."""
    from vllm.models.step5.nvidia.ops.sparse_decode import prepare_sparse_decode
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import (
        decode_summary_scores,
    )

    torch.manual_seed(13)
    device = torch.device("cuda")
    n, groups, block, dim, cols = 3, 4, 64, 192, 20
    pages = n * cols
    table = torch.randperm(pages, device=device).int().view(n, cols)
    seq_lens = torch.tensor([37, 700, 1280], device=device)
    slots = (table[torch.arange(n), (seq_lens - 1) // block] * block).long() + (
        seq_lens - 1
    ) % block
    slots[0] = -1
    mean = torch.randint(0, 120, (pages, block // 8, 256), device=device).to(
        torch.uint8
    )
    # The indexer emits group-major storage viewed as [tokens, groups, ...].
    query = (
        torch.randn(groups, n, 8, 256, device=device)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
    )
    weights = torch.rand(groups, n, 8, device=device).transpose(0, 1)
    capacity = 160
    grouped, grouped_meta = decode_summary_scores(
        query,
        weights,
        mean,
        table,
        slots,
        seq_lens,
        capacity,
        num_sms=148,
        row_groups=groups,
    )
    repeated, repeated_meta = decode_summary_scores(
        query.reshape(n * groups, 1, 8, 256),
        weights.reshape(n * groups, 1, 8),
        mean,
        *(t.repeat_interleave(groups, 0) for t in (table, slots, seq_lens)),
        capacity,
        num_sms=148,
    )
    for name in ("active", "positions", "current", "lengths", "topk_lengths"):
        assert torch.equal(getattr(grouped_meta, name), getattr(repeated_meta, name))
    for row, length in enumerate(grouped_meta.lengths.tolist()):
        if grouped_meta.active[row]:
            assert torch.equal(grouped[row, :length], repeated[row, :length])

    kv = torch.randn(pages, block, groups, 2 * dim, device=device).bfloat16()
    kv = kv.transpose(1, 2)
    regions = torch.full((n * groups, 33), -1, dtype=torch.int32, device=device)
    for row in range(n * groups):
        live = int(seq_lens[row // groups] + 7) // 8
        picked = torch.randperm(live - 1, device=device)[: min(32, live - 1)]
        regions[row, 33 - picked.numel() - 1 : 32] = picked.sort().values.int()
        regions[row, 32] = live - 1
    positions = (seq_lens - 1).repeat_interleave(groups)
    active = torch.ones(n * groups, dtype=torch.bool, device=device)
    mask = torch.empty(n * groups * 33, dtype=torch.uint8, device=device)
    key, value = prepare_sparse_decode(
        kv, regions, table, positions, active, mask, None
    )
    for group in range(groups):
        rows = torch.arange(group, n * groups, groups, device=device)
        group_mask = torch.empty(n * 33, dtype=torch.uint8, device=device)
        group_key, group_value = prepare_sparse_decode(
            kv[:, group : group + 1],
            regions[rows],
            table,
            positions[rows],
            active[rows],
            group_mask,
            None,
        )
        tokens = key.view(n, groups, 33 * 8, dim)[:, group]
        assert torch.equal(tokens, group_key.view(n, 33 * 8, dim))
        assert torch.equal(
            value.view(n, groups, 33 * 8, dim)[:, group],
            group_value.view(n, 33 * 8, dim),
        )
        assert torch.equal(mask.view(n, groups, 33)[:, group], group_mask.view(n, 33))
