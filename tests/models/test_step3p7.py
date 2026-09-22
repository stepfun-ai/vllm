# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from torch import nn

import vllm.model_executor.models.step3p7 as step3p7_module
from vllm.model_executor.models.step3p7 import Step3p7ForConditionalGeneration
from vllm.model_executor.models.step5_vl import Step5ForConditionalGeneration


class _DummyLanguageModel(nn.Module):
    supports_pp = True

    def forward(self, input_ids, positions, intermediate_tensors=None):
        raise AssertionError("not exercised by this constructor test")

    def __init__(self) -> None:
        super().__init__()
        self.make_empty_intermediate_tensors = object()


def test_step5_checkpoint_weights_use_the_step3p7_mapper():
    mapper = Step5ForConditionalGeneration.hf_to_vllm_mapper

    assert mapper.apply_list(
        [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "vision_model.transformer.resblocks.0.attn.in_proj_weight",
            "vision_model.transformer.resblocks.0.mlp.c_fc.weight",
            "vit_large_projector.weight",
        ]
    ) == [
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "vision_model.transformer.resblocks.0.attn.qkv_proj.weight",
        "vision_model.transformer.resblocks.0.mlp.fc1.weight",
        "vit_large_projector.weight",
    ]


@pytest.mark.parametrize(
    "model_cls",
    [Step3p7ForConditionalGeneration, Step5ForConditionalGeneration],
)
def test_step3p7_family_initializes_encoder_cudagraph_model_config(
    monkeypatch, model_cls
):
    vision_quant_configs = []
    projector_quant_configs = []

    def fake_perception_encoder(*args, **kwargs):
        vision_quant_configs.append(kwargs["quant_config"])
        return nn.Identity()

    def fake_projector(*args, **kwargs):
        projector_quant_configs.append(kwargs["quant_config"])
        return nn.Identity()

    monkeypatch.setattr(
        Step3p7ForConditionalGeneration,
        "_mark_tower_model",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        Step3p7ForConditionalGeneration,
        "_mark_language_model",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(step3p7_module, "PerceptionEncoder", fake_perception_encoder)
    monkeypatch.setattr(step3p7_module, "ColumnParallelLinear", fake_projector)
    monkeypatch.setattr(step3p7_module, "get_act_fn", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        step3p7_module,
        "init_vllm_registered_model",
        lambda *args, **kwargs: _DummyLanguageModel(),
    )

    hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(
            hidden_act="quick_gelu",
            image_size=728,
            patch_size=14,
            width=1536,
        ),
        text_config=SimpleNamespace(hidden_size=4096),
        understand_projector_stride=2,
        projector_bias=False,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        multimodal_config=SimpleNamespace(mm_encoder_tp_mode="tensor"),
        max_model_len=1024,
    )
    model_config.get_multimodal_config = lambda: model_config.multimodal_config
    quant_config = object()
    model = model_cls(
        vllm_config=SimpleNamespace(
            model_config=model_config, quant_config=quant_config
        )
    )

    assert model.model_config is model_config
    assert vision_quant_configs == [quant_config]
    assert projector_quant_configs == [quant_config]
    assert model.get_encoder_cudagraph_budget_range(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=2048))
    ) == (169, 1024)
