# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest

import vllm.config.speculative as speculative_module
from vllm.config import ModelConfig, ParallelConfig
from vllm.config.compilation import CompilationMode
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.transformers_utils.configs.step5 import Step5VLConfig


@pytest.mark.parametrize("model_type", ["step3p5_mtp", "step5_mtp"])
def test_step_mtp_models_use_step_proposer(model_type):
    speculative_config = object.__new__(SpeculativeConfig)
    speculative_config.method = "mtp"
    speculative_config.draft_model_config = cast(
        ModelConfig, SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )

    assert speculative_config.use_step_mtp()


def test_non_step_mtp_model_does_not_use_step_proposer():
    speculative_config = object.__new__(SpeculativeConfig)
    speculative_config.method = "mtp"
    speculative_config.draft_model_config = cast(
        ModelConfig,
        SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_mtp")),
    )

    assert not speculative_config.use_step_mtp()


def test_step5_vl_mtp_promotes_the_nested_text_config():
    target_config = Step5VLConfig(
        text_config={
            "architectures": ["Step5ForCausalLM"],
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 3,
        }
    )

    draft_config = SpeculativeConfig.hf_config_override(target_config)

    assert draft_config is target_config.text_config
    assert draft_config.model_type == "step5_mtp"
    assert draft_config.architectures == ["Step5MTP"]
    assert draft_config.n_predict == 3


class _DraftModelConfigCaptured(RuntimeError):
    pass


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    (
        "draft_model",
        "draft_revision",
        "draft_code_revision",
        "expected_hf_config_path",
        "expected_revision",
        "expected_code_revision",
    ),
    [
        (
            "target-model",
            None,
            None,
            "target-config",
            "target-revision",
            "target-code-revision",
        ),
        ("independent-draft", None, None, None, None, None),
        (
            "target-model",
            "draft-revision",
            "draft-code-revision",
            "target-config",
            "draft-revision",
            "draft-code-revision",
        ),
    ],
)
def test_draft_config_source_inheritance(
    monkeypatch,
    draft_model,
    draft_revision,
    draft_code_revision,
    expected_hf_config_path,
    expected_revision,
    expected_code_revision,
):
    captured = {}

    def capture_model_config(**kwargs):
        captured.update(kwargs)
        raise _DraftModelConfigCaptured

    monkeypatch.setattr(speculative_module, "ModelConfig", capture_model_config)
    target_model_config = SimpleNamespace(
        model="target-model",
        hf_config_path="target-config",
        revision="target-revision",
        code_revision="target-code-revision",
        hf_overrides=None,
        tokenizer="target-tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="bfloat16",
        seed=0,
        tokenizer_revision=None,
        max_model_len=32768,
        enforce_eager=False,
        max_logprobs=20,
        config_format="auto",
    )

    with pytest.raises(_DraftModelConfigCaptured):
        SpeculativeConfig(
            model=draft_model,
            method="mtp",
            num_speculative_tokens=3,
            revision=draft_revision,
            code_revision=draft_code_revision,
            target_model_config=cast(ModelConfig, target_model_config),
            target_parallel_config=cast(ParallelConfig, SimpleNamespace()),
        )

    assert captured["hf_config_path"] == expected_hf_config_path
    assert captured["revision"] == expected_revision
    assert captured["code_revision"] == expected_code_revision


@pytest.mark.cpu_test
def test_heterogeneous_draft_uses_draft_tokenizer_revision(monkeypatch):
    captured = {}

    def capture_model_config(**kwargs):
        captured.update(kwargs)
        raise _DraftModelConfigCaptured

    monkeypatch.setattr(speculative_module, "ModelConfig", capture_model_config)
    target_model_config = SimpleNamespace(
        model="target-model",
        hf_config_path=None,
        revision="target-revision",
        code_revision=None,
        hf_overrides=None,
        tokenizer="target-tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="bfloat16",
        seed=0,
        tokenizer_revision="target-tokenizer-revision",
        max_model_len=32768,
        enforce_eager=False,
        max_logprobs=20,
        config_format="auto",
    )

    with pytest.raises(_DraftModelConfigCaptured):
        SpeculativeConfig(
            model="independent-draft",
            method="draft_model",
            num_speculative_tokens=3,
            revision="draft-revision",
            use_heterogeneous_vocab=True,
            target_model_config=cast(ModelConfig, target_model_config),
            target_parallel_config=cast(ParallelConfig, SimpleNamespace()),
        )

    assert captured["tokenizer"] == "independent-draft"
    assert captured["tokenizer_revision"] == "draft-revision"


@pytest.mark.parametrize(
    "architecture",
    ["Step5ForCausalLM", "Step5ForConditionalGeneration", "Step5MTP"],
)
def test_step5_v2_model_runner_support_excludes_mtp(architecture):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            use_mla=False,
            enable_return_routed_experts=False,
            logits_processors=[],
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False, mamba_cache_mode="none"
        ),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(
        cast(VllmConfig, config)
    )

    assert unsupported == (["Step5 MTP"] if architecture == "Step5MTP" else [])
