# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residual carry preserves materialized hidden states at model boundaries."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.step5.nvidia import model as step5
from vllm.models.step5.nvidia import mtp
from vllm.models.step5.nvidia.layernorm import Step5RMSNorm
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA norm")


class _Attention(nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states * 0.25


class _FFN(nn.Module):
    def __init__(self, fp32):
        super().__init__()
        self.fp32 = fp32

    def forward(self, hidden_states):
        if self.fp32:
            hidden_states = hidden_states.float()
        return hidden_states * 0.5


def _model(hidden=128, fp32=True, depth=4):
    model = step5.Step5Model.__new__(step5.Step5Model)
    nn.Module.__init__(model)
    model.fp32_residual_connection = fp32
    model.use_sequence_parallel = False
    model.start_layer, model.end_layer = 0, depth
    model.layers = nn.ModuleList()
    for index in range(depth):
        layer = step5.Step5DecoderLayer.__new__(step5.Step5DecoderLayer)
        nn.Module.__init__(layer)
        layer.fp32_residual_connection = fp32
        layer.use_moe = False
        layer.use_sequence_parallel = False
        layer.self_attn = _Attention()
        layer.mlp = _FFN(index % 2 == 1)
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = Step5RMSNorm(hidden, eps=1e-5, zero_centered=True).cuda()
            norm.weight.data.normal_(std=0.1)
            setattr(layer, name, norm)
        model.layers.append(layer)
    return model


def _sequence_parallel(monkeypatch, model):
    # Emulate rank zero of TP2. Tests repeat input rows so the peer shard
    # is identical; collectives exercise residual layout and boundary order.
    def shard(x):
        return x[: (x.shape[0] + 1) // 2]

    def gather(x):
        return torch.cat((x, x))

    model.use_sequence_parallel = True
    for layer in model.layers:
        layer.use_sequence_parallel = True
    monkeypatch.setattr(step5, "sp_shard", shard)
    monkeypatch.setattr(step5, "sp_all_gather", gather)
    monkeypatch.setattr(step5, "sp_reduce_scatter", shard)
    monkeypatch.setattr(
        step5,
        "_step5_sp_padding",
        lambda x: torch.zeros((x.shape[0] + 1) // 2, device=x.device, dtype=torch.bool),
    )
    monkeypatch.setattr(mtp, "sp_shard", shard)
    monkeypatch.setattr(mtp, "sp_all_gather", gather)
    monkeypatch.setattr(mtp, "_step5_sp_padding", step5._step5_sp_padding)


def _group(monkeypatch, first=True, last=True):
    monkeypatch.setattr(
        step5,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=first, is_last_rank=last),
    )


def _materialized(model, positions, hidden):
    # Materializing every layer forces the unfused input-norm path.
    hidden = model._cast_for_residual(hidden)
    for layer in model.layers[model.start_layer : model.end_layer]:
        ffn, residual = layer(positions, hidden)
        hidden = layer._add_for_residual(ffn, residual)
    return hidden


def _exact(actual, expected):
    assert actual.dtype == expected.dtype
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int16
    torch.testing.assert_close(actual.view(bits), expected.view(bits), rtol=0, atol=0)


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("rows,hidden", [(1, 128), (8, 4096)])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_residual_carry_matches_layerwise_materialization(
    monkeypatch, fp32, rows, hidden, sequence_parallel
):
    torch.manual_seed(782)
    _group(monkeypatch)
    model = _model(hidden, fp32)
    positions = torch.arange(rows, device="cuda")
    source = torch.randn(rows, hidden, device="cuda").bfloat16()
    if sequence_parallel:
        source = source[:1].expand(rows, hidden).clone()
    expected = _materialized(model, positions, source.clone())
    if sequence_parallel:
        _sequence_parallel(monkeypatch, model)
    actual = model(positions, positions, inputs_embeds=source.clone())
    _exact(actual, expected)


@pytest.mark.parametrize("rows", [1, 8])
def test_residual_carry_graph_reads_new_inputs(monkeypatch, rows):
    _group(monkeypatch)
    model = _model(4096)
    positions = torch.arange(rows, device="cuda")
    source = torch.randn(rows, 4096, device="cuda").bfloat16()
    model(positions, positions, inputs_embeds=source)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = model(positions, positions, inputs_embeds=source)
    for scale in (0.01, 1.0, 100.0):
        source.normal_().mul_(scale)
        expected = _materialized(model, positions, source.clone())
        graph.replay()
        _exact(actual, expected)


@pytest.mark.parametrize("depth", [0, 4])
def test_pp_boundary_returns_materialized_hidden_states(monkeypatch, depth):
    model = _model(depth=depth)
    positions = torch.arange(3, device="cuda")
    source = torch.randn(3, 128, device="cuda").bfloat16()
    expected = _materialized(model, positions, source.clone())
    model.end_layer = depth // 2
    _group(monkeypatch, last=False)
    intermediate = model(positions, positions, inputs_embeds=source.clone())
    assert isinstance(intermediate, IntermediateTensors)
    assert set(intermediate.tensors) == {"hidden_states"}
    assert intermediate["hidden_states"].dtype == torch.float32
    _exact(
        intermediate["hidden_states"],
        _materialized(model, positions, source.clone()),
    )
    model.start_layer, model.end_layer = depth // 2, depth
    _group(monkeypatch, first=False)
    actual = model(positions, positions, intermediate_tensors=intermediate)
    _exact(actual, expected)


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_mtp_returns_materialized_decoder_output(monkeypatch, fp32, sequence_parallel):
    predictor = mtp.Step5MultiTokenPredictorLayer.__new__(
        mtp.Step5MultiTokenPredictorLayer
    )
    nn.Module.__init__(predictor)
    predictor.fp32_residual_connection = fp32
    predictor.enorm = predictor.hnorm = nn.Identity()
    predictor.eh_proj = nn.Linear(256, 128, bias=False).cuda().bfloat16()
    predictor.mtp_block = _model(fp32=fp32, depth=1).layers[0]
    positions = torch.arange(3, device="cuda")
    embed = torch.randn(3, 128, device="cuda").bfloat16()
    previous = torch.randn_like(embed)
    if sequence_parallel:
        embed = embed[:1].expand_as(embed).clone()
        previous = previous[:1].expand_as(previous).clone()
    joined = predictor.eh_proj(torch.cat((embed, previous), dim=-1))
    if fp32:
        joined = joined.float()
    ffn, residual = predictor.mtp_block(positions, joined)
    expected = predictor.mtp_block._add_for_residual(ffn, residual)
    if sequence_parallel:
        _sequence_parallel(monkeypatch, SimpleNamespace(layers=[predictor.mtp_block]))
    actual = predictor(positions, positions, previous.clone(), embed.clone())
    _exact(actual, expected)
