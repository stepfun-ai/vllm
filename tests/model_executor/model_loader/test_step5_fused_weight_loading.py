# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.models.step5 import kernels as step5_kernels
from vllm.models.step5 import layernorm as step5_layernorm
from vllm.models.step5 import model as step5_model
from vllm.models.step5 import mtp as step5_mtp
from vllm.models.step5.kernels import (
    checkpoint_has_step5_sparse_config,
    get_step5_sparse_config,
    is_supported_step5_qknorm_cache_rotary,
)


def _step5_custom_linear_projection(kind, weight):
    if kind == "indexer":
        owner = SimpleNamespace(weight=weight)
        forward = step5_model.Step5SparseIndexerIndexTPLinear.forward
        return lambda inputs: forward(owner, inputs)[0]
    owner = SimpleNamespace(weight=weight, qkv_size=weight.shape[0] // 2)
    forward = step5_model.Step5FusedQKVIndexerLinear.forward
    return lambda inputs: torch.cat(forward(owner, inputs), dim=-1)


@pytest.mark.parametrize("kind", ["indexer", "fused"])
@pytest.mark.parametrize(
    "invariant,cuda_alike", [(True, True), (False, True), (True, False)]
)
def test_step5_custom_linear_dispatch(monkeypatch, kind, invariant, cuda_alike):
    monkeypatch.setattr(step5_model.envs, "VLLM_BATCH_INVARIANT", invariant)
    monkeypatch.setattr(
        step5_model.current_platform, "is_cuda_alike", lambda: cuda_alike
    )
    weight = torch.ones((4, 8))
    inputs = torch.ones((2, 8))
    calls = []

    def invariant_linear(input_tensor, weights, bias=None):
        calls.append((input_tensor, weights, bias))
        return torch.full((2, 4), 17.0)

    monkeypatch.setattr(
        step5_model, "linear_batch_invariant", invariant_linear, raising=False
    )
    output = _step5_custom_linear_projection(kind, weight)(inputs)
    if invariant and cuda_alike:
        assert len(calls) == 1
        assert calls[0][0] is inputs
        assert calls[0][1] is weight
        assert calls[0][2] is None
        assert torch.equal(output, torch.full((2, 4), 17.0))
    else:
        assert not calls
        assert torch.equal(output, torch.full((2, 4), 8.0))


@pytest.mark.parametrize(
    ("use_rope", "is_cuda", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_step5_fused_qknorm_requires_cuda(
    monkeypatch,
    use_rope,
    is_cuda,
    expected,
):
    monkeypatch.setattr(step5_model.current_platform, "is_cuda", lambda: is_cuda)

    assert step5_model._can_use_step5_fused_qknorm(use_rope) is expected


@pytest.mark.parametrize(
    ("enabled", "num_tokens", "expected"),
    [
        (False, 1, False),
        (True, 1, True),
        (True, 256, True),
        (True, 257, False),
        (True, 16384, False),
    ],
)
def test_step5_qknorm_cache_fusion_is_limited_to_small_workloads(
    enabled,
    num_tokens,
    expected,
):
    assert step5_model._use_step5_qknorm_cache_fusion(enabled, num_tokens) is expected


@pytest.mark.parametrize(
    ("num_tokens", "expected_path", "expected_kv_dependency"),
    [
        (256, "cache", "cache-write"),
        (257, "separate", None),
    ],
)
def test_step5_attention_dispatches_qknorm_at_cache_fusion_boundary(
    monkeypatch,
    num_tokens,
    expected_path,
    expected_kv_dependency,
):
    calls = []
    kv_dependencies = []
    qkv = torch.arange(num_tokens * 3, dtype=torch.float32).view(num_tokens, 3)

    def cache_fusion(*args, **kwargs):
        calls.append("cache")
        return qkv[:, :1], qkv[:, 1:2], qkv[:, 2:], "cache-write"

    def separate_fusion(*args, **kwargs):
        calls.append("separate")
        return qkv[:, :1], qkv[:, 1:2], qkv[:, 2:]

    class FakeAttention:
        layer_name = "layers.0.self_attn.attn"

        def __call__(self, q, k, v, *, kv_cache_dummy_dep):
            kv_dependencies.append(kv_cache_dummy_dep)
            return q + k + v

    monkeypatch.setattr(
        step5_model,
        "fused_qknorm_rope_cache_forward_impl",
        cache_fusion,
    )
    monkeypatch.setattr(
        step5_model,
        "fused_qknorm_rope_forward_impl",
        separate_fusion,
    )

    attention = SimpleNamespace(
        use_dsa_backend=False,
        fuse_qkv_gate=False,
        qkv_proj=lambda hidden_states: (qkv, None),
        use_step5_fused_qknorm_cache=True,
        use_step5_fused_qknorm=True,
        q_norm=SimpleNamespace(variance_epsilon=1e-5, weight=torch.ones(1)),
        k_norm=SimpleNamespace(weight=torch.ones(1)),
        rope_cos=torch.ones(1),
        rope_sin=torch.zeros(1),
        head_dim=1,
        num_heads=1,
        num_kv_heads=1,
        rotary_dim=2,
        zero_centered=False,
        attn=FakeAttention(),
        sparse_attn=None,
        use_head_wise_attn_gate=False,
        o_proj=lambda output: (output, None),
    )

    output = step5_model.Step5Attention.forward(
        attention,
        torch.arange(num_tokens),
        torch.empty(num_tokens, 1),
    )

    assert calls == [expected_path]
    assert kv_dependencies == [expected_kv_dependency]
    torch.testing.assert_close(output, qkv.sum(dim=-1, keepdim=True))


def test_step5_attention_quantizes_projections_but_not_kv_cache(monkeypatch):
    quant_config = object()
    projection_quant_configs = []
    attention_quant_configs = []

    class FakeProjection(nn.Module):
        def __init__(self, *_args, **kwargs):
            super().__init__()
            self.params_dtype = torch.bfloat16
            self.bias = None
            projection_quant_configs.append(kwargs["quant_config"])

    class FakeAttention(nn.Module):
        def __init__(self, *_args, **kwargs):
            super().__init__()
            attention_quant_configs.append(kwargs["quant_config"])

    class FakeRotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cos_sin_cache", torch.zeros(1, 8))

    monkeypatch.setattr(step5_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(step5_model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(step5_model, "QKVParallelLinear", FakeProjection)
    monkeypatch.setattr(step5_model, "RowParallelLinear", FakeProjection)
    monkeypatch.setattr(step5_model, "Attention", FakeAttention)
    monkeypatch.setattr(
        step5_model, "get_rope", lambda **_kwargs: FakeRotaryEmbedding()
    )
    monkeypatch.setattr(
        step5_model, "make_step5_rms_norm", lambda *_args, **_kwargs: nn.Identity()
    )

    step5_model.Step5Attention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        max_position=16,
        head_dim=4,
        quant_config=quant_config,
        prefix="model.layers.0.self_attn",
        layer_types=["full_attention"],
        use_rope_layers=[True],
        zero_centered=False,
    )

    assert projection_quant_configs == [quant_config, quant_config]
    assert attention_quant_configs == [None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind,output_width", [("indexer", 4), ("fused", 5008)])
@pytest.mark.parametrize("rows", [2, 3, 4, 7, 160])
def test_step5_custom_linear_batch_invariance(monkeypatch, kind, output_width, rows):
    monkeypatch.setattr(step5_model.envs, "VLLM_BATCH_INVARIANT", True)
    torch.manual_seed(20260906)
    weight = torch.randn((output_width, 4096), device="cuda", dtype=torch.bfloat16)
    storage = torch.randn((rows, 4128), device="cuda", dtype=torch.bfloat16)
    inputs = storage[:, :4096]
    forward = _step5_custom_linear_projection(kind, weight)
    actual = forward(inputs)
    for row in sorted({0, rows // 2, rows - 1}):
        expected = forward(inputs[row : row + 1])
        torch.testing.assert_close(actual[row], expected[0], rtol=0, atol=0)
    oracle = torch.nn.functional.linear(inputs.float(), weight.float())
    torch.testing.assert_close(actual.float(), oracle, rtol=0.005, atol=0.01)


def _make_fused_projection(*loaded_shards: str, has_gate: bool = True):
    projection = step5_model.Step5FusedQKVIndexerLinear.__new__(
        step5_model.Step5FusedQKVIndexerLinear
    )
    nn.Module.__init__(projection)
    projection.has_gate = has_gate
    projection._loaded_shards = set(loaded_shards)
    return projection


def test_step5_moe_layer_indices_accept_matching_config_aliases():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        moe_layers_enum="1,3",
        moe_layer_list=[3, 1],
    )

    assert step5_model._get_step5_moe_layer_indices(config) == {1, 3}


def test_step5_moe_layer_indices_falls_back_to_list():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        moe_layers_enum=None,
        moe_layer_list=[2],
    )

    assert step5_model._get_step5_moe_layer_indices(config) == {2}


def test_step5_moe_layer_indices_rejects_conflicting_aliases():
    config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        moe_layers_enum="1,3",
        moe_layer_list=[1, 2],
    )

    with pytest.raises(ValueError, match="describe different layers"):
        step5_model._get_step5_moe_layer_indices(config)


def test_step5_pp_stage_without_local_moe_layers_is_valid():
    model = SimpleNamespace(moe_layers=[])

    step5_model._set_step5_moe_protocol_metadata(model, None)

    assert model.num_moe_layers == 0
    assert model.num_logical_experts == 0
    assert model.num_physical_experts == 0
    assert model.num_local_physical_experts == 0
    assert model.num_routed_experts == 0
    assert model.num_redundant_experts == 0


@pytest.mark.parametrize(
    ("tp_size", "dp_size", "fuse_all_reduce", "reduce_results"),
    [
        (1, 1, False, True),
        (4, 1, True, False),
        (4, 2, False, True),
    ],
)
def test_step5_moe_reduce_policy(
    tp_size,
    dp_size,
    fuse_all_reduce,
    reduce_results,
):
    assert step5_model._step5_moe_reduce_policy(tp_size, dp_size) == (
        fuse_all_reduce,
        reduce_results,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_step5_router_bias_kernel_matches_reference(indices_dtype):
    from vllm.models.step5.nvidia.ops.triton.router_bias import (
        router_bias_triton_func,
    )

    torch.manual_seed(20260916)
    gating_output = torch.randn(7, 352, device="cuda", dtype=torch.float32)
    gating_output[3, 17] = torch.nan
    gating_output[4].fill_(-46.75)
    correction_bias = torch.randn(352, device="cuda") * 0.05
    correction_bias[4:] += torch.arange(348, device="cuda")
    scaling_factor = 2.5

    actual_weights, actual_ids = router_bias_triton_func(
        gating_output,
        correction_bias,
        topk=8,
        renormalize=True,
        routed_scaling_factor=scaling_factor,
        indices_dtype=indices_dtype,
    )

    scores = gating_output.sigmoid()
    expected_ids = torch.topk(scores + correction_bias, k=8, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20
    expected_weights *= scaling_factor
    expected_weights[3].zero_()
    expected_ids[3].zero_()

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual_ids, expected_ids.to(indices_dtype))


class _FakeMoeRunner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.eplb_calls = []
        self.update_expert_map_calls = 0

    def get_expert_weights(self):
        return [self.weight]

    def set_eplb_state(self, **kwargs):
        self.eplb_calls.append(kwargs)

    def update_expert_map(self):
        self.update_expert_map_calls += 1


def _make_mtp_with_optional_moe(*, use_moe: bool):
    mtp = object.__new__(step5_mtp.Step5MTP)
    nn.Module.__init__(mtp)
    mtp.model = nn.Module()

    predictor_layer = nn.Module()
    predictor_layer.mtp_block = nn.Module()
    predictor_layer.mtp_block.use_moe = use_moe
    if use_moe:
        moe = step5_model.Step5MoE.__new__(step5_model.Step5MoE)
        nn.Module.__init__(moe)
        moe.experts = _FakeMoeRunner()
        moe.n_logical_experts = 8
        moe.n_physical_experts = 10
        moe.n_local_physical_experts = 5
        moe.n_routed_experts = 8
        moe.n_redundant_experts = 2
        predictor_layer.mtp_block.moe = moe

    mtp.model.layers = nn.ModuleDict({"4": predictor_layer})
    moe_blocks = step5_mtp._get_step5_mtp_moe_blocks(mtp.model)
    mtp.moe_layers = [moe.experts for moe in moe_blocks]
    step5_model._set_step5_moe_protocol_metadata(
        mtp,
        moe_blocks[0] if moe_blocks else None,
    )
    return mtp


def test_dense_step5_mtp_is_not_registered_as_moe():
    mtp = _make_mtp_with_optional_moe(use_moe=False)

    assert mtp.moe_layers == []
    assert mtp.num_moe_layers == 0
    assert not mtp.model.layers["4"].mtp_block.use_moe
    assert step5_mtp._is_step5_mtp_dense(mtp.model)
    assert not is_mixture_of_experts(mtp)


def test_step5_mtp_dense_flag_is_false_when_a_draft_block_is_moe():
    mtp = _make_mtp_with_optional_moe(use_moe=True)

    assert not step5_mtp._is_step5_mtp_dense(mtp.model)


def test_step5_mtp_exposes_moe_runner_to_eplb_and_updates_metadata():
    mtp = _make_mtp_with_optional_moe(use_moe=True)
    runner = mtp.moe_layers[0]
    expert_load_view = torch.zeros(1)
    logical_to_physical_map = torch.zeros(1, dtype=torch.int32)
    logical_replica_count = torch.ones(1, dtype=torch.int32)

    assert is_mixture_of_experts(mtp)
    mtp.set_eplb_state(
        expert_load_view,
        logical_to_physical_map,
        logical_replica_count,
    )
    mtp.update_physical_experts_metadata(
        num_physical_experts=12,
        num_local_physical_experts=5,
    )

    assert mtp.expert_weights == [[runner.weight]]
    assert runner.eplb_calls == [
        {
            "moe_layer_idx": 0,
            "expert_load_view": expert_load_view,
            "logical_to_physical_map": logical_to_physical_map,
            "logical_replica_count": logical_replica_count,
        }
    ]
    assert runner.update_expert_map_calls == 1
    assert mtp.num_physical_experts == 12
    assert mtp.num_redundant_experts == 4


def test_step5_mtp_rejects_local_physical_expert_count_changes():
    mtp = _make_mtp_with_optional_moe(use_moe=True)

    with pytest.raises(ValueError, match="cannot change the number of local"):
        mtp.update_physical_experts_metadata(
            num_physical_experts=12,
            num_local_physical_experts=4,
        )


def test_step5_dsa_rejects_kv_transfer_before_cache_binding():
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_transfer_instance=True)
    )

    with pytest.raises(ValueError, match="not compatible with KV transfer"):
        step5_model._verify_step5_dsa_kv_transfer_compatibility(vllm_config)


@pytest.mark.parametrize("tp_size", [4, 8, 16])
def test_step5_dsa_parallel_geometry_accepts_supported_layout(tp_size):
    step5_model._validate_step5_dsa_parallel_geometry(
        total_num_heads=64,
        total_num_kv_heads=4,
        indexer_num_heads=16,
        index_tp_size=4,
        tp_size=tp_size,
    )


@pytest.mark.parametrize(("tp_size", "local_groups"), [(1, 4), (2, 2)])
def test_step5_dsa_parallel_geometry_rejects_multiple_local_groups(
    tp_size,
    local_groups,
):
    with pytest.raises(
        ValueError,
        match=rf"exactly one local.*local_groups={local_groups}",
    ):
        step5_model._validate_step5_dsa_parallel_geometry(
            total_num_heads=64,
            total_num_kv_heads=4,
            indexer_num_heads=16,
            index_tp_size=4,
            tp_size=tp_size,
        )


def test_step5_tp16_provider_replication_matches_main_kv_geometry():
    mappings = [
        step5_model._step5_replicated_group_rank(
            total_groups=4,
            tp_size=16,
            tp_rank=rank,
        )
        for rank in range(16)
    ]

    assert mappings == [
        *((0, 4),) * 4,
        *((1, 4),) * 4,
        *((2, 4),) * 4,
        *((3, 4),) * 4,
    ]


def test_step5_dsa_parallel_geometry_rejects_provider_group_mismatch():
    with pytest.raises(ValueError, match="provider groups must align"):
        step5_model._validate_step5_dsa_parallel_geometry(
            total_num_heads=64,
            total_num_kv_heads=4,
            indexer_num_heads=16,
            index_tp_size=8,
            tp_size=8,
        )


def test_step5_dsa_parallel_geometry_rejects_incompatible_local_groups():
    with pytest.raises(ValueError, match="incompatible with group count"):
        step5_model._validate_step5_dsa_parallel_geometry(
            total_num_heads=64,
            total_num_kv_heads=3,
            indexer_num_heads=15,
            index_tp_size=3,
            tp_size=8,
        )


def test_fused_qkv_indexer_validation_rejects_missing_shard():
    module = nn.Module()
    module.projection = _make_fused_projection(
        "q",
        "k",
        "v",
        "index_q",
        "index_k",
        "index_z",
    )

    with pytest.raises(ValueError, match=r"missing shards \['index_g'\]"):
        step5_model._validate_fused_qkv_indexer_weights(module)


def test_fused_qkv_indexer_validation_reports_and_resets_complete_weight():
    module = nn.Module()
    module.projection = _make_fused_projection(
        "q",
        "k",
        "v",
        "index_q",
        "index_k",
        "index_z",
        "index_g",
    )

    assert step5_model._validate_fused_qkv_indexer_weights(module) == {
        "projection.weight"
    }
    assert module.projection._loaded_shards == set()


def test_fused_qkv_indexer_accepts_local_fused_checkpoint_weight():
    projection = _make_fused_projection()
    projection._fused_checkpoint_shard_sizes = ()
    param = nn.Parameter(torch.zeros(8, 4))
    loaded_weight = torch.arange(32, dtype=torch.float32).view(8, 4)

    projection.weight_loader_v2(param, loaded_weight)

    torch.testing.assert_close(param, loaded_weight)
    assert projection._loaded_shards == projection.required_shard_ids()


def test_fused_qkv_indexer_rejects_malformed_fused_checkpoint_weight():
    projection = _make_fused_projection()
    projection._fused_checkpoint_shard_sizes = (
        ("q", 2),
        ("k", 1),
        ("v", 1),
        ("index_q", 2),
        ("index_k", 1),
        ("index_z", 1),
        ("index_g", 1),
    )
    param = nn.Parameter(torch.zeros(8, 4))

    with pytest.raises(ValueError, match="expected local 8 or global 9"):
        projection.weight_loader_v2(param, torch.zeros(7, 4))


def test_step5_full_load_validates_but_weight_transfer_can_be_partial(monkeypatch):
    calls: list[str] = []

    class FakeStep5Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = _make_fused_projection()

    class FakeLoader:
        def __init__(self, _module, **_kwargs):
            pass

        def load_weights(self, _weights, *, mapper):
            del mapper
            calls.append("load")
            return set()

    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        hf_to_vllm_mapper=step5_model.Step5ForCausalLM.hf_to_vllm_mapper,
        model=FakeStep5Model(),
    )
    monkeypatch.setattr(step5_model, "AutoWeightsLoader", FakeLoader)

    with pytest.raises(ValueError, match="Incomplete fused qkv"):
        step5_model.Step5ForCausalLM.load_weights(model, [])

    with disable_mtp_completeness_check():
        assert step5_model.Step5ForCausalLM.load_weights(model, []) == set()
    assert calls == ["load", "load"]


def _make_weight_module() -> nn.Module:
    module = nn.Module()
    module.weight = nn.Parameter(torch.zeros(1))
    return module


def test_step5_main_loader_accepts_base_prefixes_and_skips_later_layers():
    model = step5_model.Step5Model.__new__(step5_model.Step5Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_make_weight_module(), _make_weight_module()])
    model.config = SimpleNamespace(
        num_attention_groups=2,
        num_hidden_layers=2,
        moe_num_experts=0,
    )
    model.vllm_config = SimpleNamespace(quant_config=None)
    model.moe_num_experts = 0

    loaded = step5_model.Step5Model.load_weights(
        model,
        [
            ("model.layers.0.weight", torch.tensor([1.0])),
            (
                "model.layers.0.moe.experts.3.gate_proj.weight",
                torch.tensor([5.0]),
            ),
            ("layers.1.weight", torch.tensor([2.0])),
            ("model.layers.2.weight", torch.tensor([3.0])),
            ("layers.3.weight", torch.tensor([4.0])),
        ],
    )

    assert loaded == {"layers.0.weight", "layers.1.weight"}
    torch.testing.assert_close(model.layers[0].weight, torch.tensor([1.0]))
    torch.testing.assert_close(model.layers[1].weight, torch.tensor([2.0]))


def test_step5_mtp_loader_selects_and_rewrites_only_draft_layers():
    mtp = step5_mtp.Step5MTP.__new__(step5_mtp.Step5MTP)
    nn.Module.__init__(mtp)
    mtp.config = SimpleNamespace(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        moe_num_experts=0,
    )
    mtp.model = nn.Module()
    mtp.model.layers = nn.ModuleDict({"2": nn.Module()})
    mtp.model.layers["2"].mtp_block = _make_weight_module()
    mtp.model.mtp_start_layer_idx = 2
    mtp.model.num_mtp_layers = 1

    with disable_mtp_completeness_check():
        loaded = step5_mtp.Step5MTP.load_weights(
            mtp,
            [
                ("model.layers.1.weight", torch.tensor([1.0])),
                ("layers.2.weight", torch.tensor([2.0])),
                ("model.layers.3.weight", torch.tensor([3.0])),
            ],
        )

    assert loaded == {"model.layers.2.mtp_block.weight"}
    torch.testing.assert_close(
        mtp.model.layers["2"].mtp_block.weight,
        torch.tensor([2.0]),
    )


def test_step5_sparse_config_prefers_native_section_and_keeps_legacy_alias():
    native = SimpleNamespace(
        step5_sparse_config={"enabled": False},
        step3p5_sparse_config={"enabled": True},
        sparse_config={"enabled": True},
    )
    legacy = SimpleNamespace(step3p5_sparse_config={"enabled": True})

    assert get_step5_sparse_config(native) is None
    assert get_step5_sparse_config(legacy) is not None


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(step5_sparse_config={"enabled": True}), True),
        (SimpleNamespace(step5_sparse_config={}), True),
        # ``enabled`` is a runtime DSA switch; section presence still
        # advertises the split-KV/replay contract required by the checkpoint.
        (SimpleNamespace(step5_sparse_config={"enabled": False}), True),
        (SimpleNamespace(), False),
    ],
)
def test_step5_sparse_checkpoint_capability_tracks_declared_layout(config, expected):
    assert checkpoint_has_step5_sparse_config(config) is expected


def test_step5_sparse_checkpoint_capability_honors_force_enable_for_declared_section(
    monkeypatch,
):
    monkeypatch.setattr(step5_kernels.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(
        step5_kernels.envs,
        "VLLM_STEP5_SPARSE",
        True,
        raising=False,
    )

    assert checkpoint_has_step5_sparse_config(
        SimpleNamespace(step5_sparse_config={"enabled": False})
    )
    assert not checkpoint_has_step5_sparse_config(SimpleNamespace())


def test_step5_sparse_force_enable_rejects_native_dense_checkpoint(monkeypatch):
    monkeypatch.setattr(step5_kernels.envs, "is_set", lambda _name: True)
    monkeypatch.setattr(
        step5_kernels.envs,
        "VLLM_STEP5_SPARSE",
        True,
        raising=False,
    )

    with pytest.raises(ValueError, match="requires a checkpoint-declared"):
        get_step5_sparse_config(SimpleNamespace())


def test_step5_sparse_config_retains_ssmax_compatibility_metadata():
    config = SimpleNamespace(
        step5_sparse_config={
            "enabled": True,
            "sparse_indexer_softmax_variant": "ssmax",
        }
    )

    sparse_config = get_step5_sparse_config(config)

    assert sparse_config is not None
    assert sparse_config.sparse_indexer_softmax_variant == "ssmax"


def test_step5_sparse_config_rejects_stable_topk_above_selector_limit():
    config = SimpleNamespace(
        step5_sparse_config={
            "enabled": True,
            "topk": 513,
        }
    )

    with pytest.raises(ValueError, match=r"stable top-k currently requires topk"):
        get_step5_sparse_config(config)


def test_step5_packed_mapping_exposes_dsa_partitions():
    assert step5_model.Step5ForCausalLM.packed_modules_mapping["qkv_indexer_proj"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "sparse_indexer_q",
        "sparse_indexer_k",
        "sparse_indexer_z",
        "g_proj",
    ]


@pytest.mark.parametrize(
    "model_cls",
    [
        step5_model.Step5ForCausalLM,
        step5_mtp.Step5MTP,
    ],
)
def test_step5_quantized_models_enable_weight_tracking_by_default(model_cls):
    assert model_cls._enable_weights_track_by_default


def test_step5_mtp_local_argmax_selects_the_current_step_head():
    calls: list[tuple[object, torch.Tensor]] = []
    hidden_states = torch.ones(2, 3)

    class _SharedHead:
        def __init__(self, name):
            self.head = name

        def __call__(self, states):
            return states + 1

    def get_top_tokens(head, states):
        calls.append((head, states))
        return torch.tensor([7, 8])

    predictor = SimpleNamespace(
        num_mtp_layers=2,
        mtp_start_layer_idx=10,
        layers={
            "10": SimpleNamespace(shared_head=_SharedHead("head-0")),
            "11": SimpleNamespace(shared_head=_SharedHead("head-1")),
        },
        logits_processor=SimpleNamespace(get_top_tokens=get_top_tokens),
    )

    result = step5_mtp.Step5MultiTokenPredictor.get_top_tokens(
        predictor,
        hidden_states,
        spec_step_idx=3,
    )

    assert torch.equal(result, torch.tensor([7, 8]))
    assert len(calls) == 1
    assert calls[0][0] == "head-1"
    torch.testing.assert_close(calls[0][1], hidden_states + 1)


def test_step5_rms_norm_falls_back_without_optional_op(monkeypatch):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(step5_layernorm, "_has_step5_legacy_rms_norm_op", lambda: False)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual = step5_layernorm.apply_step5_rms_norm(
        x,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected = step5_layernorm._step5_rms_norm_fallback(
        x,
        weight,
        1e-6,
        True,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("platform_is_cuda", [False, True])
def test_step5_rms_norm_optional_op_requires_cuda_tensor_and_platform(
    monkeypatch,
    platform_is_cuda,
):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(
        step5_layernorm.current_platform,
        "is_cuda",
        lambda: platform_is_cuda,
    )
    monkeypatch.setattr(step5_layernorm, "_has_step5_legacy_rms_norm_op", lambda: True)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual = step5_layernorm.apply_step5_rms_norm(
        x,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected = step5_layernorm._step5_rms_norm_fallback(
        x,
        weight,
        1e-6,
        True,
    )

    torch.testing.assert_close(actual, expected)


def test_step5_fused_rms_norm_falls_back_without_custom_ops(monkeypatch):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(step5_layernorm, "_has_step5_legacy_rms_norm_op", lambda: False)
    monkeypatch.setattr(
        step5_layernorm,
        "_has_step5_fused_add_rms_norm_op",
        lambda: False,
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual, residual_out = step5_layernorm.apply_step5_fused_add_rms_norm(
        x,
        residual,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected_residual = x + residual
    expected = step5_layernorm._step5_rms_norm_fallback(
        expected_residual,
        weight,
        1e-6,
        True,
    )

    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("platform_is_cuda", [False, True])
def test_step5_fused_rms_norm_optional_ops_require_cuda_tensor_and_platform(
    monkeypatch,
    platform_is_cuda,
):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(
        step5_layernorm.current_platform,
        "is_cuda",
        lambda: platform_is_cuda,
    )
    monkeypatch.setattr(step5_layernorm, "_has_step5_legacy_rms_norm_op", lambda: True)
    monkeypatch.setattr(
        step5_layernorm,
        "_has_step5_fused_add_rms_norm_op",
        lambda: True,
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual, residual_out = step5_layernorm.apply_step5_fused_add_rms_norm(
        x,
        residual,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected_residual = x + residual
    expected = step5_layernorm._step5_rms_norm_fallback(
        expected_residual,
        weight,
        1e-6,
        True,
    )

    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(actual, expected)


def test_step5_fused_rms_norm_fp16_fallback_preserves_residual_dtype(monkeypatch):
    monkeypatch.setenv("VLLM_STEP_CC_LEVEL", "0")
    monkeypatch.setattr(step5_layernorm, "_has_step5_legacy_rms_norm_op", lambda: False)
    monkeypatch.setattr(
        step5_layernorm,
        "_has_step5_fused_add_rms_norm_op",
        lambda: False,
    )
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    residual = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    weight = torch.tensor([0.25, -0.5], dtype=torch.float32)

    actual, residual_out = step5_layernorm.apply_step5_fused_add_rms_norm(
        x,
        residual,
        weight,
        1e-6,
        zero_centered=True,
    )
    expected_residual = (x.float() + residual.float()).to(torch.float16)
    expected = step5_layernorm._step5_rms_norm_fallback(
        expected_residual,
        weight,
        1e-6,
        True,
    )

    assert residual_out.dtype == torch.float16
    assert actual.dtype == torch.float16
    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qknorm_cache_falls_back_when_native_extension_op_is_missing(monkeypatch):
    device = torch.device("cuda")
    qkv = torch.zeros((1, 6), dtype=torch.bfloat16, device=device)
    q = torch.full((1, 2), 1, dtype=qkv.dtype, device=device)
    k = torch.full((1, 2), 2, dtype=qkv.dtype, device=device)
    v = torch.full((1, 2), 3, dtype=qkv.dtype, device=device)
    kv_cache = torch.zeros((2, 1, 4, 1, 2), dtype=qkv.dtype, device=device)
    slot_mapping = torch.zeros((1,), dtype=torch.int64, device=device)
    updates: list[tuple[torch.Tensor, torch.Tensor]] = []

    class FakeImpl:
        @staticmethod
        def do_kv_cache_update(
            _layer,
            key,
            value,
            _kv_cache,
            _slot_mapping,
        ):
            updates.append((key, value))

    attn_layer = SimpleNamespace(impl=FakeImpl())
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.attention.get_attention_context",
        lambda _layer_name: (None, attn_layer, kv_cache, slot_mapping),
    )
    monkeypatch.setattr(
        step5_kernels,
        "_get_step5_qknorm_cache_op",
        lambda: None,
    )
    monkeypatch.setattr(
        step5_kernels,
        "_fused_qknorm_rope_forward_impl",
        lambda *_args, **_kwargs: (q, k, v),
    )

    actual_q, actual_k, actual_v, _ = (
        step5_kernels._fused_qknorm_rope_cache_forward_impl_op(
            qkv,
            torch.ones(2, dtype=torch.float32, device=device),
            torch.ones(2, dtype=torch.float32, device=device),
            torch.ones((4, 1), dtype=qkv.dtype, device=device),
            torch.zeros((4, 1), dtype=qkv.dtype, device=device),
            torch.zeros((1,), dtype=torch.int64, device=device),
            head_dim=2,
            num_q_head=1,
            num_kv_head=1,
            rotary_dim=0,
            layer_name="layers.0.self_attn",
        )
    )

    assert actual_q is q
    assert actual_k is k
    assert actual_v is v
    assert len(updates) == 1
    torch.testing.assert_close(updates[0][0], k.view(1, 1, 2))
    torch.testing.assert_close(updates[0][1], v.view(1, 1, 2))


@pytest.mark.parametrize(
    ("cc_level", "op_name"),
    [
        (0, "optimus_fused_qknorm_rope_cache_bitwise"),
        (3, "optimus_fused_qknorm_rope_cache"),
    ],
)
def test_qknorm_cache_selects_step5_op_for_cc_level(
    monkeypatch,
    cc_level,
    op_name,
):
    selected_op = object()
    namespace = SimpleNamespace(**{op_name: selected_op})
    monkeypatch.setattr(
        step5_kernels,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(_C=namespace)),
    )
    monkeypatch.setattr(
        step5_kernels,
        "envs",
        SimpleNamespace(VLLM_STEP_CC_LEVEL=cc_level),
    )

    assert step5_kernels._get_step5_qknorm_cache_op() is selected_op


@pytest.mark.parametrize("cc_level", [0, 3])
def test_qknorm_cache_selector_returns_none_when_extension_op_is_missing(
    monkeypatch,
    cc_level,
):
    monkeypatch.setattr(
        step5_kernels,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(_C=SimpleNamespace())),
    )
    monkeypatch.setattr(
        step5_kernels,
        "envs",
        SimpleNamespace(VLLM_STEP_CC_LEVEL=cc_level),
    )

    assert step5_kernels._get_step5_qknorm_cache_op() is None


@pytest.mark.parametrize(
    ("head_dim", "rotary_pairs", "supported"),
    [
        (64, 32, True),
        (128, 64, True),
        (192, 64, True),
        (192, 96, True),
        (256, 128, True),
        (80, 40, False),
    ],
)
def test_qknorm_cache_rotary_layout_support(
    head_dim,
    rotary_pairs,
    supported,
):
    assert is_supported_step5_qknorm_cache_rotary(head_dim, rotary_pairs) is supported


def test_router_bias_lazy_imports_model_scoped_kernel(monkeypatch):
    calls = []
    imported_modules = []
    expected = (object(), object())

    def router(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    def import_module(name):
        imported_modules.append(name)
        assert name == "vllm.models.step5.nvidia.ops.triton.router_bias"
        return SimpleNamespace(router_bias_triton_func=router)

    monkeypatch.setattr(step5_kernels.importlib, "import_module", import_module)

    gating_output = SimpleNamespace(is_cuda=True)
    router_bias = object()
    actual = step5_kernels.router_bias_func(
        object(),
        gating_output,
        topk=2,
        renormalize=True,
        router_bias=router_bias,
        routed_scaling_factor=3.0,
        indices_dtype=torch.int64,
    )

    assert actual is expected
    assert imported_modules == ["vllm.models.step5.nvidia.ops.triton.router_bias"]
    assert calls == [
        (
            (gating_output, router_bias, 2),
            {
                "renormalize": True,
                "routed_scaling_factor": 3.0,
                "nan_row_i_out": 0,
                "indices_dtype": torch.int64,
            },
        )
    ]


@pytest.mark.parametrize("renormalize", [False, True])
def test_router_bias_cpu_fallback_preserves_reference_semantics(
    monkeypatch, renormalize
):
    def unexpected_import(_name):
        raise AssertionError("CPU fallback must not import the CUDA Triton kernel")

    monkeypatch.setattr(step5_kernels.importlib, "import_module", unexpected_import)

    gating_output = torch.tensor(
        [[0.0, 1.0, -1.0], [float("nan"), 0.0, 1.0]], dtype=torch.float32
    )
    router_bias = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float32)
    weights, ids = step5_kernels.router_bias_func(
        gating_output,
        gating_output,
        topk=2,
        renormalize=renormalize,
        router_bias=router_bias,
        routed_scaling_factor=1.5,
        nan_row_i_out=7,
        indices_dtype=torch.int64,
    )

    scores = gating_output[0].sigmoid()
    expected_ids = torch.topk(scores + router_bias, k=2).indices
    expected_weights = scores.gather(0, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum() + 1e-20
    expected_weights *= 1.5

    torch.testing.assert_close(weights[0], expected_weights)
    torch.testing.assert_close(weights[1], torch.zeros(2))
    assert ids.dtype == torch.int64
    assert ids[0].tolist() == expected_ids.tolist()
    assert ids[1].tolist() == [7, 7]
