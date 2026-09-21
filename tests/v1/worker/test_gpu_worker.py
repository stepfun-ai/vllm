# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
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
