# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


class FakeTokenizer:
    def __len__(self) -> int:
        return 1024


class FakeSparseTokenizer:
    def __len__(self) -> int:
        return 2

    def get_vocab(self) -> dict[str, int]:
        return {"zero": 0, "sparse-added-token": 1000}


class FakeBadWordsTokenizer:
    max_token_id = 1023

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [1000]


class FakeSparseBadWordsTokenizer:
    vocab_size = 2

    def __len__(self) -> int:
        return 2

    def get_vocab(self) -> dict[str, int]:
        return {"zero": 0, "sparse-token": 1000}

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return [1000]


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None
    valid_vocab_size: int | None = 1000
    vocab_size: int = 1024

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_valid_vocab_size(self) -> int:
        return (
            self.valid_vocab_size
            if self.valid_vocab_size is not None
            else self.vocab_size
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


def test_valid_vocab_size_bounds_token_specific_sampling_params():
    model_config = MockModelConfig()
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        SamplingParams(logprob_token_ids=[1000]).verify(model_config, None, None, None)
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        SamplingParams(logit_bias={1000: 1.0}).verify(model_config, None, None, None)


def test_full_logprobs_uses_valid_vocab_size():
    model_config = MockModelConfig(max_logprobs=1000)
    SamplingParams(logprobs=-1, prompt_logprobs=-1).verify(
        model_config, None, None, None
    )


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_logprobs_cannot_exceed_valid_vocab_size(field: str):
    model_config = MockModelConfig(max_logprobs=1024)
    with pytest.raises(VLLMValidationError, match="greater than max allowed: 1000"):
        SamplingParams(**{field: 1001}).verify(model_config, None, None, None)


def test_allowed_token_ids_uses_valid_vocab_size():
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        SamplingParams(allowed_token_ids=[1000]).verify(
            MockModelConfig(), None, None, FakeTokenizer()
        )


def test_allowed_token_ids_use_token_id_upper_bound_for_sparse_vocab():
    model_config = MockModelConfig(valid_vocab_size=1001)
    tokenizer = FakeSparseTokenizer()

    SamplingParams(allowed_token_ids=[1000]).verify(
        model_config,
        None,
        None,
        tokenizer,
    )
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        SamplingParams(allowed_token_ids=[1001]).verify(
            model_config,
            None,
            None,
            tokenizer,
        )


def test_stop_token_ids_use_valid_vocab_size():
    model_config = MockModelConfig()
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        SamplingParams(stop_token_ids=[1000]).verify(model_config, None, None, None)


def test_generation_config_eos_uses_valid_vocab_size():
    params = SamplingParams(min_tokens=1)
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        params.update_from_generation_config(
            {"eos_token_id": [1000]},
            eos_token_id=2,
            model_config=MockModelConfig(),
        )


def test_bad_words_generated_ids_use_valid_vocab_size():
    tokenizer = FakeBadWordsTokenizer()

    params = SamplingParams(bad_words=["foo"])
    params.update_from_tokenizer(tokenizer)
    assert params.bad_words_token_ids == [[1000]]

    with pytest.raises(
        VLLMValidationError,
        match="model vocabulary size is 1000",
    ):
        SamplingParams(bad_words=["foo"]).update_from_tokenizer(
            tokenizer,
            model_config=MockModelConfig(),
        )


def test_bad_words_use_sparse_tokenizer_upper_bound():
    params = SamplingParams(bad_words=["foo"])
    params.update_from_tokenizer(FakeSparseBadWordsTokenizer())

    assert params.bad_words_token_ids == [[1000]]


def test_models_without_valid_vocab_reject_out_of_model_token_ids():
    model_config = MockModelConfig(valid_vocab_size=None, vocab_size=1000)
    params = SamplingParams(stop_token_ids=[1000])
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        params.verify(model_config, None, None, FakeTokenizer())


def test_verify_leaves_logits_processors_to_admission():
    """verify() is runner-agnostic; LP validation lives in the admission
    layer, so an unimportable FQCN must not fail verify()."""
    SamplingParams().verify(
        MockModelConfig(logits_processors=["no.such:Cls"]), None, None, None
    )


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]
