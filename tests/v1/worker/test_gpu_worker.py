# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.gpu_worker import maybe_rocm_profiling_fallback
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_fingerprint_includes_valid_vocab_size(plan_env):
    """Sampler buffer sizing must not reuse a plan across vocab contracts."""
    fp = startup_plan.compute_plan_fingerprint
    config_a = SimpleNamespace(
        compute_hash=lambda: "same-model-graph",
        model_config=SimpleNamespace(valid_vocab_size=128815),
        speculative_config=None,
    )
    config_b = SimpleNamespace(
        compute_hash=lambda: "same-model-graph",
        model_config=SimpleNamespace(valid_vocab_size=129024),
        speculative_config=None,
    )
    assert fp(config_a, 0, 1) != fp(config_b, 0, 1)


def test_startup_plan_fingerprint_includes_draft_valid_vocab_size(plan_env):
    """Probabilistic draft logits are sized from the draft vocabulary."""
    fp = startup_plan.compute_plan_fingerprint

    def config(draft_valid_vocab_size):
        return SimpleNamespace(
            compute_hash=lambda: "same-model-graph",
            model_config=SimpleNamespace(valid_vocab_size=128815),
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(
                    valid_vocab_size=draft_valid_vocab_size
                )
            ),
        )

    assert fp(config(64000), 0, 1) != fp(config(65000), 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


@pytest.mark.cpu_test
def test_fault_tolerance_cleanup_uses_request_state_removal_hook():
    from vllm.v1.worker.sentinel.gpu_worker_sentinel import WorkerSentinel

    req_state = object()
    remove_request = Mock()
    model_runner = SimpleNamespace(
        execute_model_state=None,
        requests={"request-0": req_state},
        num_prompt_logprobs={"request-0": 1},
        input_batch=SimpleNamespace(
            req_id_to_index={"request-0": 0},
            remove_request=remove_request,
            condense=Mock(),
            refresh_metadata=Mock(),
            req_prompt_embeds={},
        ),
        _on_request_state_removed=Mock(),
    )
    sentinel = SimpleNamespace(
        worker=SimpleNamespace(model_runner=model_runner, use_v2_model_runner=False)
    )

    WorkerSentinel._clean_worker_state(sentinel)

    model_runner._on_request_state_removed.assert_called_once_with(
        "request-0", req_state
    )
    assert model_runner.requests == {}
    assert model_runner.num_prompt_logprobs == {}
    remove_request.assert_called_once_with("request-0")


@pytest.mark.cpu_test
def test_side_storage_is_prepared_inside_main_memory_profile(monkeypatch):
    from vllm.v1.worker import gpu_worker

    events: list[str] = []

    @contextmanager
    def fake_memory_profiling(*_args, **_kwargs):
        events.append("profile_enter")
        yield SimpleNamespace(
            total_consumed=100,
            transient_peak_headroom=20,
            non_kv_cache_memory=200,
            after_profile=SimpleNamespace(free_memory=900),
        )
        events.append("profile_exit")

    worker = object.__new__(gpu_worker.Worker)
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None,
        gpu_memory_utilization=0.9,
    )
    worker.compilation_config = SimpleNamespace(static_forward_context={})
    worker.device = torch.device("cpu")
    worker.model_runner = SimpleNamespace(
        model_memory_usage=0,
        profile_run=lambda: events.append("model_profile"),
    )
    worker.init_snapshot = SimpleNamespace(free_memory=1_000, total_memory=2_000)
    worker.requested_memory = 800
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=None)
    )
    worker.model_config = SimpleNamespace(multimodal_config=None)
    worker.parallel_config = SimpleNamespace(_api_process_count=1)

    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda _worker: None)
    monkeypatch.setattr(
        gpu_worker,
        "deep_gemm_kernel_warmup",
        lambda _worker: events.append("deep_gemm"),
    )
    monkeypatch.setattr(gpu_worker, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(
        gpu_worker,
        "prepare_kv_cache_side_storage_for_memory_profiling",
        lambda _context, _device: events.append("side_storage_prepare"),
    )
    monkeypatch.setattr(
        gpu_worker.current_platform,
        "is_cuda_alike",
        lambda: False,
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_mm_ipc_gpu_memory",
        lambda num_bytes, *_args: num_bytes,
    )

    available_memory = worker.determine_available_memory()

    assert available_memory == 600
    assert events == [
        "profile_enter",
        "deep_gemm",
        "side_storage_prepare",
        "model_profile",
        "profile_exit",
    ]


@pytest.mark.cpu_test
def test_attention_backend_warmup_runs_prefill_then_uniform_decode():
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.warmup.kernel_warmup import (
        _warmup_attention_backends,
    )
    from vllm.v1.attention.backend import AttentionBackend

    class NoWarmupBackend(AttentionBackend):
        pass

    class WarmupBackend(AttentionBackend):
        @staticmethod
        def get_attention_warmup_num_tokens() -> int:
            return 16

        @staticmethod
        def get_attention_warmup_decode_query_len() -> int:
            return 1

    calls: list[dict[str, object]] = []
    runner = SimpleNamespace(
        attn_groups=[
            [
                SimpleNamespace(backend=NoWarmupBackend),
                SimpleNamespace(backend=WarmupBackend),
            ]
        ],
        max_num_tokens=8,
        max_num_reqs=4,
        uniform_decode_query_len=3,
        _dummy_run=lambda **kwargs: calls.append(kwargs),
    )

    _warmup_attention_backends(runner)

    assert calls == [
        {
            "num_tokens": 8,
            "cudagraph_runtime_mode": CUDAGraphMode.NONE,
            "force_attention": True,
            "create_mixed_batch": True,
            "skip_eplb": True,
        },
        {
            "num_tokens": 6,
            "cudagraph_runtime_mode": CUDAGraphMode.NONE,
            "force_attention": True,
            "uniform_decode": True,
            "skip_eplb": True,
        },
    ]


@pytest.mark.cpu_test
def test_runtime_idle_dp_dummy_disables_kv_cache_writes():
    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    dummy_run = Mock()
    worker.model_runner = SimpleNamespace(
        uniform_decode_query_len=3,
        _dummy_run=dummy_run,
    )

    worker.execute_dummy_batch()

    dummy_run.assert_called_once_with(
        3,
        uniform_decode=True,
        disable_kv_cache_writes=True,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("batch_invariant", "max_rows", "expected_rows"),
    [
        (False, 256, [256]),
        (True, 1, [1]),
        (True, 2, [2, 1]),
        (True, 256, [256, 1, 2]),
    ],
)
def test_v1_sampler_warmup_covers_batch_invariant_row_classes(
    monkeypatch,
    batch_invariant,
    max_rows,
    expected_rows,
):
    from vllm.v1.worker import gpu_worker

    calls: list[int] = []
    runner = SimpleNamespace(
        _dummy_sampler_run=lambda *, hidden_states: calls.append(hidden_states.shape[0])
    )
    hidden_states = torch.empty(max_rows, 8)
    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_BATCH_INVARIANT",
        batch_invariant,
    )

    gpu_worker._warmup_v1_sampler(runner, hidden_states)

    assert calls == expected_rows


# Memory accounting of the profiling run (Worker.determine_available_memory).

# The fallback reads only the sign of the measured drop and this process's torch
# reservation; free memory is only logged, so no amount here is a device size.
ANY_FREE_MEMORY = 8 * GiB_bytes
MEASURED_DROP = 4 * GiB_bytes
TORCH_RESERVED = 3 * GiB_bytes
RELEASED_BY_OTHERS = 2 * GiB_bytes


def _snapshot(free_memory, torch_memory=0):
    return SimpleNamespace(free_memory=free_memory, torch_memory=torch_memory)


def _profile_result(consumed, reserved_before=0, reserved_after=0):
    """A result whose free-memory readings agree with `consumed`, which
    `memory_profiling` derives as the drop in free memory, negative when it grew."""
    return SimpleNamespace(
        total_consumed=consumed,
        transient_peak_headroom=0,
        before_create=_snapshot(ANY_FREE_MEMORY, reserved_before),
        after_profile=_snapshot(ANY_FREE_MEMORY - consumed, reserved_after),
    )


@pytest.fixture
def rocm(request):
    with patch.object(
        gpu_worker, "current_platform", SimpleNamespace(is_rocm=lambda: request.param)
    ):
        yield request.param


@pytest.mark.parametrize("rocm", [True, False], indirect=True)
def test_profiling_fallback_declines_when_free_memory_dropped(rocm):
    """The profiling measurement is kept as-is whenever free memory dropped."""
    result = _profile_result(consumed=MEASURED_DROP)

    assert maybe_rocm_profiling_fallback(result) is None


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_replaces_a_released_measurement(rocm):
    """A negative measurement describes the rest of the device, so it is replaced
    by this process's reservation, which the rest of the device cannot move."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_after=TORCH_RESERVED,
    )

    assert maybe_rocm_profiling_fallback(result) == TORCH_RESERVED


@pytest.mark.parametrize("rocm", [True], indirect=True)
def test_profiling_fallback_never_returns_a_negative_amount(rocm):
    """A reservation that shrank across the run cannot become negative usage."""
    result = _profile_result(
        consumed=-RELEASED_BY_OTHERS,
        reserved_before=TORCH_RESERVED,
        reserved_after=0,
    )

    assert maybe_rocm_profiling_fallback(result) == 0


@pytest.mark.parametrize("rocm", [False], indirect=True)
def test_profiling_fallback_declines_off_rocm(rocm):
    """Platforms that account frees eagerly keep reporting the error, so the
    caller's assertion stays reachable there."""
    result = _profile_result(consumed=-RELEASED_BY_OTHERS)

    assert maybe_rocm_profiling_fallback(result) is None


class _OrderedHandle:
    """Send handle that logs when it is waited."""

    def __init__(self, log: list[str], name: str):
        self.log = log
        self.name = name

    def is_completed(self) -> bool:
        return True

    def wait(self) -> None:
        self.log.append(f"wait:{self.name}")


def test_execute_model_waits_previous_pp_send_before_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    """Previous device handles are waited before the forward pass; the
    metadata handle is left to the GroupCoordinator's reaper."""
    import torch

    from vllm.sequence import IntermediateTensors

    log: list[str] = []
    previous_tensor_send = _OrderedHandle(log, "prev-tensor")
    metadata_handle = _OrderedHandle(log, "meta")
    tensor_handle = _OrderedHandle(log, "tensor")

    def isend_tensor_dict(tensors, all_gather_group=None, all_gather_tensors=None):
        log.append("isend")
        return [metadata_handle, tensor_handle]

    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=False,
        isend_tensor_dict=isend_tensor_dict,
    )
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: SimpleNamespace())

    def run_model(scheduler_output, intermediate_tensors):
        log.append("forward")
        return IntermediateTensors({"hidden_states": torch.zeros(1)})

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2, distributed_executor_backend="mp"
            ),
        ),
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(execute_model=run_model),
        annotate_profile=lambda scheduler_output: nullcontext(),
        _pp_send_work=[previous_tensor_send],
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, num_scheduled_tokens={"r0": 4}
    )

    assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None

    assert log == ["wait:prev-tensor", "forward", "isend"]
    assert worker._pp_send_work == [tensor_handle]
