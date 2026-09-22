# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)
from vllm.models.step5 import model as step5_model


@pytest.mark.parametrize(
    ("tie_word_embeddings", "expected_skip_prefixes"),
    [
        (False, ["vision_model."]),
        (
            True,
            ["vision_model.", "lm_head."],
        ),
    ],
)
def test_step5_text_loader_skips_multimodal_weights_and_tied_lm_head(
    monkeypatch,
    tie_word_embeddings,
    expected_skip_prefixes,
):
    observed_skip_prefixes = None

    class FakeLoader:
        def __init__(self, _module):
            pass

        def load_weights(self, _weights, *, mapper):
            nonlocal observed_skip_prefixes
            observed_skip_prefixes = list(mapper.orig_to_new_prefix)
            for prefix in observed_skip_prefixes:
                assert mapper._map_name(prefix + "weight") is None
            return set()

    monkeypatch.setattr(step5_model, "AutoWeightsLoader", FakeLoader)
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tie_word_embeddings),
        hf_to_vllm_mapper=step5_model.Step5ForCausalLM.hf_to_vllm_mapper,
        model=nn.Module(),
    )

    assert step5_model.Step5ForCausalLM.load_weights(model, []) == set()
    assert observed_skip_prefixes == expected_skip_prefixes


class _DummyStep5TextModel(nn.Module):
    hf_to_vllm_mapper = step5_model.Step5ForCausalLM.hf_to_vllm_mapper

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()


def test_step5_text_loader_accepts_known_multimodal_checkpoint_weights():
    model = _DummyStep5TextModel()
    weights = [
        ("vision_model.transformer.layers.0.weight", torch.empty(0)),
        ("vit_large_projector.weight", torch.empty(0)),
    ]

    with disable_mtp_completeness_check():
        assert step5_model.Step5ForCausalLM.load_weights(model, weights) == set()


@pytest.mark.parametrize(
    "unexpected_name",
    [
        "vision_model_extra.weight",
        "vit_large_projector.bias",
        "vit_large_projector.weight_scale",
        "unknown.weight",
    ],
)
def test_step5_text_loader_rejects_other_unexpected_weights(unexpected_name):
    model = _DummyStep5TextModel()

    with (
        disable_mtp_completeness_check(),
        pytest.raises(ValueError, match="There is no module or parameter"),
    ):
        step5_model.Step5ForCausalLM.load_weights(
            model, [(unexpected_name, torch.empty(0))]
        )
