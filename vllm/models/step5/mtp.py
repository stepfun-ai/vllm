# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .layernorm import Step5RMSNorm, get_step5_norm_dtype
from .model import (
    STEP5_PACKED_MODULES_MAPPING,
    Step5DecoderLayer,
    Step5MoE,
    _reset_fused_qkv_indexer_load_state,
    _set_step5_moe_protocol_metadata,
    _validate_fused_qkv_indexer_weights,
)

logger = init_logger(__name__)


def _get_mtp_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        return vllm_config
    return replace(vllm_config, model_config=speculative_config.draft_model_config)


def _get_mtp_config(vllm_config: VllmConfig) -> PretrainedConfig:
    return _get_mtp_vllm_config(vllm_config).model_config.hf_text_config


def _build_mtp_norm(config: PretrainedConfig) -> nn.Module:
    return Step5RMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        zero_centered=config.zero_centered,
        dtype=get_step5_norm_dtype(config),
    )


class Step5MTPSharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Keep MTP normalization and residual precision aligned with the target
        # model configuration.
        self.fp32_residual_connection = config.fp32_residual_connection
        self.norm = _build_mtp_norm(config)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.bfloat16)
        return self.norm(hidden_states)


@support_torch_compile
class Step5MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        mtp_vllm_config = _get_mtp_vllm_config(vllm_config)
        config = mtp_vllm_config.model_config.hf_text_config
        self.fp32_residual_connection = config.fp32_residual_connection
        quant_config = vllm_config.quant_config
        self.enorm = _build_mtp_norm(config)
        self.hnorm = _build_mtp_norm(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = Step5MTPSharedHead(
            config=config, quant_config=quant_config, prefix=f"{prefix}.shared_head"
        )
        self.mtp_block = Step5DecoderLayer(
            mtp_vllm_config,
            prefix=f"{prefix}.mtp_block",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        if self.fp32_residual_connection:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
            previous_hidden_states = previous_hidden_states.to(torch.bfloat16)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        if self.fp32_residual_connection:
            hidden_states = hidden_states.to(torch.float32)

        hidden_states = self.mtp_block(positions=positions, hidden_states=hidden_states)
        return hidden_states


class Step5MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = _get_mtp_config(vllm_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Step5 MTP requires num_nextn_predict_layers > 0, got "
                f"{self.num_mtp_layers}."
            )
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Step5MultiTokenPredictorLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


def _get_step5_mtp_moe_blocks(
    model: Step5MultiTokenPredictor,
) -> list[Step5MoE]:
    blocks: list[Step5MoE] = []
    for predictor_layer in model.layers.values():
        mtp_block = getattr(predictor_layer, "mtp_block", None)
        moe = getattr(mtp_block, "moe", None)
        if isinstance(moe, Step5MoE):
            blocks.append(moe)
    return blocks


def _is_step5_mtp_dense(model: Step5MultiTokenPredictor) -> bool:
    """Return whether every draft transformer block is dense."""
    return not _get_step5_mtp_moe_blocks(model)


class Step5MTP(nn.Module, MixtureOfExperts):
    packed_modules_mapping = STEP5_PACKED_MODULES_MAPPING
    _enable_weights_track_by_default = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = _get_mtp_config(vllm_config)
        self.vllm_config = vllm_config
        self.model = Step5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.num_moe_layers: int
        self.num_expert_groups: int
        self.num_shared_experts: int
        self.num_logical_experts: int
        self.num_physical_experts: int
        self.num_local_physical_experts: int
        self.num_routed_experts: int
        self.num_redundant_experts: int
        moe_blocks = _get_step5_mtp_moe_blocks(self.model)
        # Expose the actual draft topology so dense MTP forwards do not
        # inherit the target model's DP collectives.
        self.is_dense_mtp = not moe_blocks
        self.moe_layers = [moe.experts for moe in moe_blocks]
        _set_step5_moe_protocol_metadata(
            self,
            moe_blocks[0] if moe_blocks else None,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError(
                "Step5 MTP EPLB cannot change the number of local physical "
                f"experts: expected={self.num_local_physical_experts}, "
                f"got={num_local_physical_experts}."
            )
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in _get_step5_mtp_moe_blocks(self.model):
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.mtp_validation import (
            is_mtp_completeness_check_enabled,
        )
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
        )

        validate_completeness = is_mtp_completeness_check_enabled()
        if validate_completeness:
            _reset_fused_qkv_indexer_load_state(self)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkvg_proj", "q_proj", 0),
            ("qkvg_proj", "k_proj", 1),
            ("qkvg_proj", "v_proj", 2),
            ("qkvg_proj", "g_proj", 3),
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("qkv_indexer_proj", "q_proj", "q"),
            ("qkv_indexer_proj", "k_proj", "k"),
            ("qkv_indexer_proj", "v_proj", "v"),
            ("qkv_indexer_proj", "sparse_indexer_q", "index_q"),
            ("qkv_indexer_proj", "sparse_indexer_k", "index_k"),
            ("qkv_indexer_proj", "sparse_indexer_z", "index_z"),
            ("qkv_indexer_proj", "g_proj", "index_g"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        base_layer = (
            "base_layer." if any(".base_layer." in name for name in params_dict) else ""
        )

        routed_experts = (
            "routed_experts."
            if any(".experts.routed_experts." in name for name in params_dict)
            else ""
        )
        expert_prefix = f".moe.experts.{routed_experts}{base_layer}"
        expert_params_mapping = [
            (f"{expert_prefix}w13_weight", ".moe.gate_proj.weight", "w1"),
            (f"{expert_prefix}w13_weight", ".moe.up_proj.weight", "w3"),
            (f"{expert_prefix}w2_weight", ".moe.down_proj.weight", "w2"),
            (
                f"{expert_prefix}w13_weight_scale_2",
                ".moe.gate_proj.weight_scale_2",
                "w1",
            ),
            (
                f"{expert_prefix}w13_weight_scale_2",
                ".moe.up_proj.weight_scale_2",
                "w3",
            ),
            (
                f"{expert_prefix}w2_weight_scale_2",
                ".moe.down_proj.weight_scale_2",
                "w2",
            ),
            (
                f"{expert_prefix}w13_weight_scale",
                ".moe.gate_proj.weight_scale",
                "w1",
            ),
            (
                f"{expert_prefix}w13_weight_scale",
                ".moe.up_proj.weight_scale",
                "w3",
            ),
            (
                f"{expert_prefix}w2_weight_scale",
                ".moe.down_proj.weight_scale",
                "w2",
            ),
            (
                f"{expert_prefix}w13_input_scale",
                ".moe.gate_proj.input_scale",
                "w1",
            ),
            (
                f"{expert_prefix}w13_input_scale",
                ".moe.up_proj.input_scale",
                "w3",
            ),
            (
                f"{expert_prefix}w2_input_scale",
                ".moe.down_proj.input_scale",
                "w2",
            ),
        ]

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("layers."):
                name = f"model.{name}"
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if "embed_tokens" not in name and spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if "experts" in name or "moe" in name:
                    continue
                replaced_name = name.replace(weight_name, param_name)
                if replaced_name not in params_dict:
                    continue
                name = replaced_name
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, shard_id = mapping
                    if weight_name not in name:
                        continue
                    replaced_name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if (
                        replaced_name.endswith(".bias")
                        or replaced_name.endswith("_bias")
                    ) and replaced_name not in params_dict:
                        continue
                    if replaced_name not in params_dict:
                        continue
                    name = replaced_name
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    moe_expert_num = self.config.moe_num_experts
                    if (
                        loaded_weight.ndim == 0
                        or loaded_weight.shape[0] != moe_expert_num
                    ):
                        raise ValueError(
                            "Step5 MTP expert tensor has an unexpected leading "
                            f"dimension: expected {moe_expert_num}, got "
                            f"{tuple(loaded_weight.shape)} for {name}."
                        )
                    for expert_id in range(moe_expert_num):
                        loaded_weight_expert = loaded_weight[expert_id]
                        weight_loader(
                            param,
                            loaded_weight_expert,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    loaded_params.add(name)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        or "tok_embeddings" in name
                    ):
                        continue

                    if spec_layer is not None and ".transformer." in name:
                        name = name.replace(".transformer.", ".")
                    if "shared_head" in name:
                        name = name.replace("shared_head.output", "shared_head.head")
                    if "embed_tokens" in name:
                        assert (
                            hasattr(self.config, "num_nextn_predict_layers")
                            and self.config.num_nextn_predict_layers > 0
                        )
                        name = "model.embed_tokens.weight"
                    if name not in params_dict:
                        logger.warning_once(
                            "Skipping unexpected MTP checkpoint weight: %s", name
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        if validate_completeness:
            loaded_params.update(_validate_fused_qkv_indexer_weights(self))
        missing_params = set(params_dict).difference(loaded_params)
        if missing_params and is_mtp_completeness_check_enabled():
            param_name_example = min(missing_params)
            raise RuntimeError(
                "Some parameters like "
                f"{param_name_example} are not in the checkpoint and will falsely "
                "use random initialization"
            )
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int | None, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(
                f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
            )
        return name
