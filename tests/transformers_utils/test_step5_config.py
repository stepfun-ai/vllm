# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.step5 import (
    PerceptionEncoderConfig,
    Step5Config,
    Step5MTPConfig,
    Step5VLConfig,
)


def test_step5_config_defaults_to_independent_lm_head():
    config = Step5Config()

    assert config.tie_word_embeddings is False
    assert Step5Config(**config.to_dict()).tie_word_embeddings is False


def test_step5_config_preserves_explicit_tied_embeddings():
    config = Step5Config(tie_word_embeddings=True)

    assert config.tie_word_embeddings is True
    assert Step5Config(**config.to_dict()).tie_word_embeddings is True


def test_step5_config_roundtrips_dense_and_mtp_layer_types():
    layer_types = ["sliding_attention", "full_attention", "full_attention"]
    config = Step5Config(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        layer_types=layer_types,
    )

    assert config.layer_types == layer_types[:2]
    assert config.layer_types_with_mtp == layer_types

    restored = Step5Config(**config.to_dict())
    assert restored.layer_types == layer_types[:2]
    assert restored.layer_types_with_mtp == layer_types


def test_step5_config_rejects_layer_types_missing_mtp_entries():
    with pytest.raises(ValueError, match="expected at least 3 entries"):
        Step5Config(
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            layer_types=["sliding_attention", "full_attention"],
        )


def test_step5_config_rejects_conflicting_serialized_layer_types():
    with pytest.raises(ValueError, match="must match the dense prefix"):
        Step5Config(
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            layer_types=["full_attention", "full_attention"],
            layer_types_with_mtp=[
                "sliding_attention",
                "full_attention",
                "full_attention",
            ],
        )


def test_step5_mtp_config_has_independent_model_type():
    config = Step5MTPConfig()

    assert config.model_type == "step5_mtp"
    assert config.to_dict()["model_type"] == "step5_mtp"
    assert config.architectures == ["Step5MTP"]
    assert config.num_nextn_predict_layers == 1
    assert Step5Config.model_type == "step5"


def test_step5_mtp_standalone_config_loads_without_remote_code(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "step5_mtp",
                "vocab_size": 128896,
                "num_hidden_layers": 92,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, Step5MTPConfig)
    assert config.model_type == "step5_mtp"
    assert config.architectures == ["Step5MTP"]


def test_step5_vl_config_uses_native_nested_configs():
    config = Step5VLConfig(
        vision_config={
            "model_type": "perception_encoder",
            "image_size": 728,
            "patch_size": 14,
            "width": 1536,
            "layers": 47,
            "heads": 16,
            "use_cls_token": False,
            "hidden_act": "quick_gelu",
        },
        text_config={
            "model_type": "step5",
            "architectures": ["Step5ForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 2,
            "quantization_config": {"quant_method": "fp8"},
        },
        image_token_id=128001,
    )

    assert isinstance(config.vision_config, PerceptionEncoderConfig)
    assert config.vision_config.use_abs_posemb is True
    assert config.vision_config.use_rope2d is True
    assert config.vision_config.use_ln_pre is True
    assert isinstance(config.text_config, Step5Config)
    assert config.text_config.architectures == ["Step5ForCausalLM"]
    assert config.quantization_config == {"quant_method": "fp8"}
    assert config.architectures == ["Step5ForConditionalGeneration"]
    assert config.hidden_size == 4096
    assert config.text_config.moe_router_activation == "sigmoid"

    restored = Step5VLConfig(**config.to_dict())
    assert isinstance(restored.vision_config, PerceptionEncoderConfig)
    assert isinstance(restored.text_config, Step5Config)
    assert restored.model_type == "step5_vl"
    assert restored.architectures == ["Step5ForConditionalGeneration"]


def test_step5_vl_config_loads_without_remote_code(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Step5ForConditionalGeneration"],
                "model_type": "step5_vl",
                "image_token_id": 128001,
                "understand_projector_stride": 2,
                "projector_bias": False,
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "modules_to_not_convert": [
                        "vision_model.transformer.resblocks.0.attn.qkv_proj",
                        "vit_large_projector",
                    ],
                },
                "vision_config": {
                    "model_type": "perception_encoder",
                    "image_size": 728,
                    "patch_size": 14,
                    "width": 1536,
                    "layers": 47,
                    "heads": 16,
                    "use_cls_token": False,
                    "hidden_act": "quick_gelu",
                },
                "text_config": {
                    "architectures": ["Step5ForCausalLM"],
                    "model_type": "step5",
                    "hidden_size": 4096,
                    "num_hidden_layers": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, Step5VLConfig)
    assert isinstance(config.vision_config, PerceptionEncoderConfig)
    assert isinstance(config.text_config, Step5Config)
    assert config.model_type == "step5_vl"
    assert config.architectures == ["Step5ForConditionalGeneration"]
    assert config.quantization_config["quant_method"] == "fp8"
    assert config.quantization_config["modules_to_not_convert"] == [
        "vision_model.transformer.resblocks.0.attn.qkv_proj",
        "vit_large_projector",
    ]
    assert not hasattr(config.text_config, "quantization_config")
