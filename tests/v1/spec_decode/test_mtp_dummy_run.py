# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as proposer_module
import vllm.v1.spec_decode.step3p5 as step3p5_module
from vllm.config import CUDAGraphMode
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.utils import CpuGpuBuffer


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    (
        "num_mtp_layers",
        "is_step_mtp",
        "is_graph_capturing",
        "expected_steps",
        "expected_dispatches",
    ),
    [
        (3, True, True, [0, 1, 2], 1),
        (3, False, True, [0, 1, 2], 2),
        (1, True, False, [None, None, None], 1),
        (1, False, False, [None, None, None], 2),
        (1, False, True, [None], 1),
    ],
)
def test_mtp_dummy_run_captures_each_distinct_module(
    monkeypatch,
    num_mtp_layers,
    is_step_mtp,
    is_graph_capturing,
    expected_steps,
    expected_dispatches,
):
    calls = []
    dispatches = []
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        use_multi_module_mtp=lambda: num_mtp_layers > 1,
        use_step_mtp=lambda: is_step_mtp,
    )
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_nextn_predict_layers=num_mtp_layers)
    )
    proposer.num_speculative_tokens = 3
    proposer.parallel_drafting = False
    proposer._draft_attn_layer_names = set()
    proposer.vllm_config = SimpleNamespace()
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer._get_positions = lambda num_tokens: torch.arange(num_tokens)

    def determine_batch_execution_and_padding(num_tokens, use_cudagraphs=True):
        dispatches.append((num_tokens, use_cudagraphs))
        return (
            CUDAGraphMode.PIECEWISE,
            num_tokens,
            None,
        )

    proposer._determine_batch_execution_and_padding = (
        determine_batch_execution_and_padding
    )
    proposer.model = lambda **kwargs: calls.append(kwargs.get("spec_step_idx"))
    monkeypatch.setattr(
        proposer_module,
        "set_forward_context",
        lambda *_args, **_kwargs: nullcontext(),
    )

    proposer.dummy_run(
        num_tokens=4,
        is_graph_capturing=is_graph_capturing,
    )

    assert calls == expected_steps
    assert len(dispatches) == expected_dispatches


@pytest.mark.cpu_test
def test_step_mtp_dummy_and_runtime_reuse_inputs_embeds_address(monkeypatch):
    calls = []
    embed_calls = []
    proposer = object.__new__(Step3p5MTPProposer)
    proposer.speculative_config = SimpleNamespace(
        use_multi_module_mtp=lambda: True,
        use_step_mtp=lambda: True,
    )
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_nextn_predict_layers=3)
    )
    proposer.num_speculative_tokens = 3
    proposer.parallel_drafting = False
    proposer._draft_attn_layer_names = set()
    proposer.vllm_config = SimpleNamespace()
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.method = "mtp"
    proposer.input_ids = torch.arange(8, dtype=torch.int32)
    proposer.inputs_embeds = torch.zeros((8, 4), dtype=torch.float32)
    proposer._get_positions = lambda num_tokens: torch.arange(num_tokens)
    proposer._determine_batch_execution_and_padding = (
        lambda num_tokens, use_cudagraphs=True: (
            CUDAGraphMode.PIECEWISE,
            num_tokens,
            None,
        )
    )

    class FakeModel:
        def embed_input_ids(self, input_ids):
            embed_calls.append(input_ids.clone())
            return input_ids[:, None].expand(-1, 4).float()

        def __call__(self, **kwargs):
            calls.append(kwargs["inputs_embeds"])

    proposer.model = FakeModel()
    monkeypatch.setattr(
        proposer_module,
        "set_forward_context",
        lambda *_args, **_kwargs: nullcontext(),
    )

    proposer.dummy_run(num_tokens=4, is_graph_capturing=True)
    proposer.input_ids[:4].add_(10)
    runtime_kwargs, _ = proposer.build_model_inputs_first_pass(2, 4, None)

    assert len(calls) == 3
    assert len(embed_calls) == 4
    runtime_inputs_embeds = runtime_kwargs["inputs_embeds"]
    assert runtime_inputs_embeds is not None
    assert all(
        captured_inputs_embeds.data_ptr() == proposer.inputs_embeds.data_ptr()
        for captured_inputs_embeds in calls
    )
    assert runtime_inputs_embeds.data_ptr() == proposer.inputs_embeds.data_ptr()
    assert torch.equal(embed_calls[-1], proposer.input_ids[:4])
    assert torch.equal(
        runtime_inputs_embeds,
        proposer.input_ids[:4, None].expand(-1, 4).float(),
    )


@pytest.mark.cpu_test
def test_step3p5_runtime_keeps_packed_query_for_every_module(monkeypatch):
    batch_size = 2
    num_tokens = 6
    model_calls = []
    shift_calls = []
    call_order = []
    metadata_calls = []
    slot_mapping_calls = []
    sample_index = 0
    proposer = object.__new__(Step3p5MTPProposer)
    proposer.speculative_config = SimpleNamespace(use_step_mtp=lambda: True)
    proposer.parallel_drafting = False
    proposer.allowed_attn_types = None
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = True
    proposer.vllm_config = SimpleNamespace()
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.inputs_embeds = torch.zeros((8, 4), dtype=torch.float32)
    proposer.hidden_states = torch.zeros((8, 4), dtype=torch.float32)
    proposer.positions = torch.arange(8)
    proposer.last_token_indices = torch.tensor([2, 5], dtype=torch.int64)
    proposer._last_draft_probs = None

    common_attn_metadata = SimpleNamespace(
        batch_size=lambda: batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
        seq_lens_cpu_upper_bound=torch.tensor([3, 3], dtype=torch.int32),
        dcp_local_seq_lens=None,
        dcp_local_seq_lens_cpu=None,
        slot_mapping=None,
        positions=proposer.positions[:num_tokens],
        max_seq_len=3,
    )

    class FakeModel:
        def embed_input_ids(self, input_ids):
            return input_ids[:, None].expand(-1, 4).float()

        def __call__(self, **kwargs):
            call_order.append(f"module{kwargs['spec_step_idx']}")
            model_calls.append({**kwargs, "positions": kwargs["positions"].clone()})
            return torch.full(
                (num_tokens, 4),
                kwargs["spec_step_idx"] + 1,
                dtype=torch.float32,
            )

    proposer.model = FakeModel()
    proposer.model_returns_tuple = lambda: False
    proposer._prepare_packed_inputs = lambda **_kwargs: (
        num_tokens,
        torch.zeros(batch_size, dtype=torch.int32),
    )

    def rebuild_slot_mappings(_num_tokens, metadata):
        positions = proposer.positions[:num_tokens].clone()
        proposer._per_group_slot_mappings = {
            0: positions + 100,
            1: positions + 200,
        }
        metadata.slot_mapping = proposer._per_group_slot_mappings[0]

    proposer._rebuild_packed_slot_mappings = rebuild_slot_mappings

    def prepare_hidden(target_hidden_states, *_args):
        proposer.hidden_states[:num_tokens].copy_(target_hidden_states)

    proposer._prepare_packed_hidden_states = prepare_hidden
    proposer._cache_packed_inputs = lambda *_args: call_order.append("cache")

    def build_metadata(metadata, draft_index=0):
        metadata_calls.append(
            (
                draft_index,
                metadata.seq_lens.clone(),
                metadata.max_seq_len,
                metadata.positions.clone(),
            )
        )
        return [], {}

    proposer.build_per_group_and_layer_attn_metadata = build_metadata
    proposer._determine_batch_execution_and_padding = lambda *_args, **_kwargs: (
        CUDAGraphMode.PIECEWISE,
        num_tokens,
        None,
    )
    proposer._get_positions = lambda num_tokens: proposer.positions[:num_tokens]
    proposer._get_slot_mapping = lambda num_tokens, *_args, **_kwargs: {
        f"group{gid}": slots[:num_tokens]
        for gid, slots in proposer._per_group_slot_mappings.items()
    }

    def build_first_pass(_num_tokens, num_input_tokens, _mm_embed_inputs):
        input_ids, inputs_embeds = proposer._prepare_text_model_inputs(num_input_tokens)
        return {
            "input_ids": input_ids,
            "positions": proposer._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
            "hidden_states": proposer.hidden_states[:num_input_tokens],
            "spec_step_idx": 0,
        }, num_input_tokens

    proposer.build_model_inputs_first_pass = build_first_pass

    def shift_packed_inputs(
        draft_token_ids,
        _prefill_token_overrides,
        override_index,
        *_args,
    ):
        shift_calls.append(override_index)
        proposer.inputs_embeds[:num_tokens].fill_(draft_token_ids[0].item())

    proposer._shift_packed_inputs = shift_packed_inputs

    def sample_draft_tokens(_hidden_states, _sampling_metadata, spec_step_idx):
        nonlocal sample_index
        sample_index += 1
        return (
            torch.full(
                (batch_size,),
                10 + spec_step_idx,
                dtype=torch.int64,
            ),
            None,
        )

    proposer._sample_draft_tokens_for_step = sample_draft_tokens

    def capture_forward_context(*_args, slot_mapping, **_kwargs):
        slot_mapping_calls.append(
            {name: slots.clone() for name, slots in slot_mapping.items()}
        )
        return nullcontext()

    monkeypatch.setattr(
        step3p5_module,
        "set_forward_context",
        capture_forward_context,
    )

    result = proposer.propose(
        num_speculative_tokens=3,
        target_token_ids=torch.zeros(num_tokens, dtype=torch.int64),
        target_positions=torch.arange(num_tokens, dtype=torch.int64),
        target_hidden_states=torch.zeros((num_tokens, 4)),
        next_token_ids=torch.zeros(batch_size, dtype=torch.int64),
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=SimpleNamespace(),
        history_slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        prefill_token_overrides=torch.full((batch_size, 3), -1),
    )

    assert result.shape == (batch_size, 3)
    assert sample_index == 3
    assert len(model_calls) == 3
    assert shift_calls == [1, 2]
    assert call_order == ["cache", "module0", "module1", "module2"]
    assert [call["positions"].tolist() for call in model_calls] == [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
    ]
    assert [
        (draft_index, seq_lens.tolist(), max_seq_len, positions.tolist())
        for draft_index, seq_lens, max_seq_len, positions in metadata_calls
    ] == [
        (0, [3, 3], 3, [0, 1, 2, 3, 4, 5]),
        (1, [3, 3], 3, [0, 1, 2, 3, 4, 5]),
        (2, [3, 3], 3, [0, 1, 2, 3, 4, 5]),
    ]
    assert [
        {name: slots.tolist() for name, slots in slot_mapping.items()}
        for slot_mapping in slot_mapping_calls
    ] == [
        {
            "group0": [100, 101, 102, 103, 104, 105],
            "group1": [200, 201, 202, 203, 204, 205],
        },
        {
            "group0": [100, 101, 102, 103, 104, 105],
            "group1": [200, 201, 202, 203, 204, 205],
        },
        {
            "group0": [100, 101, 102, 103, 104, 105],
            "group1": [200, 201, 202, 203, 204, 205],
        },
    ]
    assert common_attn_metadata.seq_lens_cpu_upper_bound.tolist() == [3, 3]
    assert all(
        call["inputs_embeds"].shape == (num_tokens, 4)
        and call["inputs_embeds"].data_ptr() == proposer.inputs_embeds.data_ptr()
        and call["hidden_states"].shape == (num_tokens, 4)
        for call in model_calls
    )


def _make_history_only_step_mtp(
    max_num_reqs: int = 3,
    max_speculative_tokens: int = 3,
) -> Step3p5MTPProposer:
    proposer = object.__new__(Step3p5MTPProposer)
    proposer.max_batch_size = max_num_reqs
    proposer.cached_history_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    proposer._history_slot_mapping = CpuGpuBuffer(
        max_num_reqs,
        dtype=torch.int32,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    proposer._prefill_token_overrides = CpuGpuBuffer(
        max_num_reqs,
        max_speculative_tokens,
        dtype=torch.int32,
        device=torch.device("cpu"),
        pin_memory=False,
    )
    proposer._prefill_token_overrides.cpu.fill_(-1)
    proposer._prefill_token_overrides.gpu.fill_(-1)
    proposer._history_slot_by_req_id = {}
    proposer._free_history_slots = list(range(max_num_reqs - 1, -1, -1))
    return proposer


@pytest.mark.cpu_test
def test_step3p5_rejects_compact_drafter_batches(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(disable_padded_drafter_batch=True)
    )

    def fake_eagle_init(self, vllm_config, *_args, **_kwargs):
        self.speculative_config = vllm_config.speculative_config

    monkeypatch.setattr(step3p5_module.EagleProposer, "__init__", fake_eagle_init)

    with pytest.raises(
        NotImplementedError,
        match="Step MTP requires padded drafter batches",
    ):
        Step3p5MTPProposer(config, torch.device("cpu"))


@pytest.mark.cpu_test
def test_step3p5_rejects_dynamic_speculative_token_counts(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            disable_padded_drafter_batch=False,
            num_speculative_tokens_per_batch_size=[(1, 8, 3), (9, 16, 1)],
        )
    )

    def fake_eagle_init(self, vllm_config, *_args, **_kwargs):
        self.speculative_config = vllm_config.speculative_config

    monkeypatch.setattr(step3p5_module.EagleProposer, "__init__", fake_eagle_init)

    with pytest.raises(
        NotImplementedError,
        match="does not support dynamic speculative token counts",
    ):
        Step3p5MTPProposer(config, torch.device("cpu"))


@pytest.mark.cpu_test
@pytest.mark.parametrize("has_rejected_tokens", [False, True])
def test_step3p5_recomputes_dcp_local_lens_after_reprefill(
    monkeypatch,
    has_rejected_tokens,
):
    launches = []

    class FakePrepareKernel:
        def __getitem__(self, _grid):
            def launch(*args, **_kwargs):
                args[3].sub_(torch.tensor([1, 2], dtype=torch.int32))

            return launch

    def fake_prepare_dcp_local_seq_lens(
        out,
        seq_lens,
        num_reqs,
        dcp_size,
        dcp_rank,
        cp_interleave,
    ):
        launches.append((seq_lens.clone(), num_reqs, dcp_size, dcp_rank, cp_interleave))
        out.copy_(torch.tensor([20, 29], dtype=torch.int32))

    monkeypatch.setattr(
        step3p5_module,
        "_prepare_step_mtp_input_buffers_kernel",
        FakePrepareKernel(),
    )
    monkeypatch.setattr(
        step3p5_module,
        "maybe_prepare_dcp_local_seq_lens",
        fake_prepare_dcp_local_seq_lens,
    )

    proposer = object.__new__(Step3p5MTPProposer)
    proposer.last_token_indices = torch.zeros(2, dtype=torch.int64)
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.positions = torch.zeros(8, dtype=torch.int64)
    proposer.cached_draft_input_ids = torch.zeros((2, 2), dtype=torch.int32)
    proposer.cached_history_lens = torch.full((2,), 2, dtype=torch.int32)
    proposer._zero_num_rejected = torch.zeros(2, dtype=torch.int32)
    proposer._history_window_size = 2
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.dcp_world_size = 2
    proposer.dcp_rank = 1
    proposer.cp_kv_cache_interleave_size = 2

    common_attn_metadata = SimpleNamespace(
        batch_size=lambda: 2,
        seq_lens=torch.tensor([41, 60], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        dcp_local_seq_lens=torch.zeros(2, dtype=torch.int32),
    )

    proposer._prepare_packed_inputs(
        target_token_ids=torch.arange(8, dtype=torch.int32),
        target_positions=torch.arange(8, dtype=torch.int64),
        next_token_ids=torch.tensor([100, 101], dtype=torch.int32),
        common_attn_metadata=common_attn_metadata,
        num_rejected_tokens_gpu=(
            torch.tensor([2, 3], dtype=torch.int32) if has_rejected_tokens else None
        ),
        history_slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        prefill_token_overrides=torch.full((2, 3), -1, dtype=torch.int32),
    )

    assert len(launches) == 1
    seq_lens, num_reqs, dcp_size, dcp_rank, cp_interleave = launches[0]
    assert torch.equal(seq_lens, torch.tensor([40, 58], dtype=torch.int32))
    assert (num_reqs, dcp_size, dcp_rank, cp_interleave) == (2, 2, 1, 2)
    assert common_attn_metadata.dcp_local_seq_lens.tolist() == [20, 29]


@pytest.mark.cpu_test
def test_step3p5_history_slots_survive_unscheduled_gap_and_release():
    proposer = _make_history_only_step_mtp()
    requests = {
        req_id: SimpleNamespace(num_computed_tokens=0) for req_id in ("a", "b", "c")
    }
    scheduled = {req_id: 1 for req_id in requests}

    slots, _ = proposer.prepare_request_state(
        ["a", "b"], requests, scheduled, np.array([False, False])
    )
    slot_a, slot_b = slots.tolist()
    proposer.cached_history_lens[slot_a] = 2

    slots, _ = proposer.prepare_request_state(
        ["b"], requests, scheduled, np.array([False])
    )
    assert slots.tolist() == [slot_b]

    slots, _ = proposer.prepare_request_state(
        ["a"], requests, scheduled, np.array([False])
    )
    assert slots.tolist() == [slot_a]
    assert proposer.cached_history_lens[slot_a].item() == 2

    proposer.release_request_history("a")
    assert proposer.cached_history_lens[slot_a].item() == 0
    slots, _ = proposer.prepare_request_state(
        ["c"], requests, scheduled, np.array([False])
    )
    assert slots.tolist() == [slot_a]


@pytest.mark.cpu_test
def test_step3p5_request_state_builds_full_prefill_lookahead():
    proposer = _make_history_only_step_mtp(max_num_reqs=1)
    token_ids = list(range(16))
    request = SimpleNamespace(
        num_computed_tokens=4,
        get_token_id=lambda index: token_ids[index] if index < len(token_ids) else -1,
    )

    _, overrides = proposer.prepare_request_state(
        ["req"],
        {"req": request},
        {"req": 2},
        np.array([True]),
    )

    assert overrides.tolist() == [[6, 7, 8]]


@pytest.mark.cpu_test
def test_step3p5_request_state_pads_rows_and_skips_load_only_requests():
    proposer = _make_history_only_step_mtp(max_num_reqs=4)
    live_request = SimpleNamespace(
        num_computed_tokens=0,
        get_token_id=lambda index: index,
    )
    load_only_request = SimpleNamespace(
        num_computed_tokens=0,
        get_token_id=lambda _index: pytest.fail(
            "load-only rows must not read prompt tokens"
        ),
    )

    slots, overrides = proposer.prepare_request_state(
        ["live", "load-only"],
        {
            "live": live_request,
            "load-only": load_only_request,
        },
        {"live": 1, "load-only": 0},
        np.array([False, True]),
        num_rows=4,
    )

    assert slots.tolist() == [0, -1, -1, -1]
    assert torch.all(overrides == -1)
    assert proposer._history_slot_by_req_id == {"live": 0}


@pytest.mark.cpu_test
def test_step3p5_request_state_validates_batch_before_allocating_history():
    proposer = _make_history_only_step_mtp(max_num_reqs=2)
    valid_request = SimpleNamespace(
        num_computed_tokens=0,
        get_token_id=lambda index: index,
    )
    prompt_embed_request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=4,
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(4, 4),
        prompt_is_token_ids=None,
        mm_features=[],
        get_token_id=lambda _index: pytest.fail(
            "embedding positions must be rejected before reading token IDs"
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="cannot read ahead through multimodal embeddings",
    ):
        proposer.prepare_request_state(
            ["valid", "prompt-embed"],
            {
                "valid": valid_request,
                "prompt-embed": prompt_embed_request,
            },
            {"valid": 1, "prompt-embed": 1},
            np.array([False, True]),
        )

    assert proposer._history_slot_by_req_id == {}


@pytest.mark.cpu_test
def test_step3p5_rejects_prompt_embed_terminal_token_before_base_prepare(
    monkeypatch,
):
    proposer = object.__new__(Step3p5MTPProposer)
    request = SimpleNamespace(
        num_prompt_tokens=4,
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(4, 4),
        prompt_is_token_ids=None,
        mm_features=[],
    )
    gpu_input_batch = SimpleNamespace(
        req_ids=["load-only", "req"],
        num_tokens_no_spec=np.array([0, 4], dtype=np.int32),
    )

    def uses_embedding(request, position):
        assert position >= 0
        return Step3p5MTPProposer._lookahead_position_uses_embedding(request, position)

    proposer._lookahead_position_uses_embedding = uses_embedding
    monkeypatch.setattr(
        step3p5_module.EagleProposer,
        "prepare_next_token_ids_padded",
        lambda *_args, **_kwargs: pytest.fail(
            "base preparation must not read an unknown prompt-embed token ID"
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="requires terminal token IDs",
    ):
        proposer.prepare_next_token_ids_padded(
            torch.zeros((2, 1), dtype=torch.int32),
            {"load-only": request, "req": request},
            gpu_input_batch,
            torch.zeros(2, dtype=torch.bool),
        )


@pytest.mark.cpu_test
def test_step3p5_rejects_current_mm_embedding_when_repair_needs_packing():
    proposer = object.__new__(Step3p5MTPProposer)
    request = SimpleNamespace(
        num_computed_tokens=4,
        num_prompt_tokens=8,
        prompt_token_ids=list(range(8)),
        prompt_embeds=torch.zeros(8, 4),
        prompt_is_token_ids=[True, True, True, True, True, True, False, True],
        mm_features=[],
    )

    with pytest.raises(
        NotImplementedError,
        match="cannot repack a current multimodal embedding",
    ):
        proposer.validate_current_query_mm_repair(
            ["req"],
            {"req": request},
            {"req": 2},
        )

    request.prompt_is_token_ids[6] = True
    proposer.validate_current_query_mm_repair(
        ["req"],
        {"req": request},
        {"req": 2},
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("prompt_token_ids", "prompt_embeds", "prompt_is_token_ids", "mm_position"),
    [
        (
            list(range(8)),
            torch.zeros(8, 4),
            [True, True, True, True, True, True, False, True],
            None,
        ),
        (None, torch.zeros(8, 4), None, None),
        (list(range(8)), None, None, PlaceholderRange(offset=6, length=1)),
        (
            list(range(8)),
            None,
            None,
            PlaceholderRange(
                offset=5,
                length=3,
                is_embed=torch.tensor([False, True, False]),
            ),
        ),
    ],
)
def test_step3p5_request_state_rejects_multimodal_prefill_lookahead(
    prompt_token_ids,
    prompt_embeds,
    prompt_is_token_ids,
    mm_position,
):
    proposer = _make_history_only_step_mtp(max_num_reqs=1)
    mm_features = (
        [] if mm_position is None else [SimpleNamespace(mm_position=mm_position)]
    )
    request = SimpleNamespace(
        num_computed_tokens=4,
        num_prompt_tokens=8,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=prompt_embeds,
        prompt_is_token_ids=prompt_is_token_ids,
        mm_features=mm_features,
        get_token_id=lambda index: index,
    )

    with pytest.raises(
        NotImplementedError,
        match="cannot read ahead through multimodal embeddings",
    ):
        proposer.prepare_request_state(
            ["req"],
            {"req": request},
            {"req": 2},
            np.array([True]),
        )
    assert proposer._history_slot_by_req_id == {}


@pytest.mark.cpu_test
def test_non_step_text_mtp_keeps_internal_embedding_path():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(use_step_mtp=lambda: False)
    proposer.input_ids = torch.arange(8, dtype=torch.int32)
    proposer.inputs_embeds = torch.zeros((8, 4), dtype=torch.float32)
    proposer.model = object()

    input_ids, inputs_embeds = proposer._prepare_text_model_inputs(4)

    assert input_ids.data_ptr() == proposer.input_ids.data_ptr()
    assert inputs_embeds is None


@pytest.mark.cpu_test
def test_idle_dp_dummy_masks_proposer_owned_slot_mappings(monkeypatch):
    private_slot_mapping = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
    forwarded_slot_mappings = []
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        use_multi_module_mtp=lambda: False,
        use_step_mtp=lambda: False,
    )
    proposer.num_speculative_tokens = 1
    proposer.parallel_drafting = False
    proposer._draft_attn_layer_names = {"draft"}
    proposer.vllm_config = SimpleNamespace()
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.inputs_embeds = torch.zeros((8, 4), dtype=torch.float32)
    proposer._get_positions = lambda num_tokens: torch.arange(num_tokens)
    proposer._get_slot_mapping = lambda _num_tokens, _mapping: {
        "draft": private_slot_mapping
    }
    proposer._determine_batch_execution_and_padding = (
        lambda num_tokens, use_cudagraphs=True: (
            CUDAGraphMode.PIECEWISE,
            num_tokens,
            None,
        )
    )
    proposer.model = lambda **_kwargs: None

    def capture_forward_context(
        _attn_metadata,
        _vllm_config,
        *,
        slot_mapping,
        **_kwargs,
    ):
        forwarded_slot_mappings.append(slot_mapping)
        return nullcontext()

    monkeypatch.setattr(
        proposer_module,
        "set_forward_context",
        capture_forward_context,
    )

    proposer.dummy_run(
        num_tokens=4,
        slot_mappings={"draft": torch.full((4,), -1, dtype=torch.int64)},
        disable_kv_cache_writes=True,
    )

    assert len(forwarded_slot_mappings) == 1
    assert torch.equal(private_slot_mapping, torch.full_like(private_slot_mapping, -1))
    assert torch.equal(
        forwarded_slot_mappings[0]["draft"],
        torch.full_like(private_slot_mapping, -1),
    )


@pytest.mark.cpu_test
def test_padded_next_token_warmup_covers_each_block_size(monkeypatch):
    next_token_launches = []
    prepare_input_launches = []

    class FakeNextTokenKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                sampled_token_ids = args[0]
                next_token_launches.append(
                    {
                        "grid": grid,
                        "rows": args[7],
                        "width": args[6],
                        "stride": args[8],
                        "block_size": kwargs["BLOCK_SIZE_TOKENS"],
                        "shape": sampled_token_ids.shape,
                    }
                )

            return launch

    class FakePrepareInputKernel:
        def __getitem__(self, grid):
            def launch(*args, **_kwargs):
                prepare_input_launches.append(
                    {
                        "grid": grid,
                        "shapes": [arg.shape for arg in args[:5]],
                        "dtypes": [arg.dtype for arg in args[:5]],
                        "num_reqs": args[5],
                    }
                )

            return launch

    monkeypatch.setattr(
        proposer_module,
        "_eagle_prepare_next_token_padded",
        lambda *args, **kw: FakeNextTokenKernel()[(args[7],)](*args, **kw),
    )
    monkeypatch.setattr(
        proposer_module,
        "_eagle_prepare_inputs_padded",
        lambda *args, **kw: FakePrepareInputKernel()[(args[5],)](*args, **kw),
    )

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.num_speculative_tokens = 3

    proposer.warmup_prepare_next_token_ids_padded(
        num_reqs=1,
        vocab_size=128815,
    )

    assert next_token_launches == [
        {
            "grid": (1,),
            "rows": 1,
            "width": 1,
            "stride": 1,
            "block_size": 1,
            "shape": torch.Size([1, 1]),
        },
        {
            "grid": (1,),
            "rows": 1,
            "width": 2,
            "stride": 2,
            "block_size": 2,
            "shape": torch.Size([1, 2]),
        },
        {
            "grid": (1,),
            "rows": 1,
            "width": 4,
            "stride": 4,
            "block_size": 4,
            "shape": torch.Size([1, 4]),
        },
    ]
    assert prepare_input_launches == [
        {
            "grid": (1,),
            "shapes": [
                torch.Size([1]),
                torch.Size([1]),
                torch.Size([2]),
                torch.Size([1]),
                torch.Size([1]),
            ],
            "dtypes": [torch.int32] * 5,
            "num_reqs": 1,
        }
    ]


@pytest.mark.cpu_test
def test_eagle_step_warmup_uses_runtime_cache_contract(monkeypatch):
    launches = []

    def fake_launch(**kwargs):
        launches.append(kwargs)

    monkeypatch.setattr(
        proposer_module,
        "eagle_step_update_slot_mapping_and_metadata",
        fake_launch,
    )

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.num_speculative_tokens = 3
    proposer.constant_draft_positions = False
    proposer.block_size = 64
    proposer.max_model_len = 32768
    block_table = torch.zeros((3, 512), dtype=torch.int32)

    proposer.warmup_eagle_step_slot_mapping(3, block_table)

    assert len(launches) == 1
    launch = launches[0]
    assert launch["positions_1d"].shape == torch.Size([3])
    assert launch["positions_1d"].dtype == torch.int64
    assert launch["block_table_tensor"] is block_table
    assert launch["seq_lens"].shape == torch.Size([3])
    assert launch["seq_lens"].dtype == torch.int32
    assert launch["block_size"] == 64
    assert launch["max_model_len"] == 32768
    assert launch["out_clamped_positions"].dtype == torch.int64
    assert launch["out_slot_mapping"].dtype == torch.int64
