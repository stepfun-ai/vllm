# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def router_bias_func(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    router_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    nan_row_i_out: int = 0,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts with biased scores, but weight them with unbiased scores."""
    del hidden_states
    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "Step5 router indices_dtype must be torch.int32 or torch.int64, "
            f"got {indices_dtype}."
        )

    if router_bias is None:
        raise ValueError("Step5 router_bias must be provided.")
    if not 0 < topk <= gating_output.shape[-1]:
        raise ValueError("Step5 router topk must be between 1 and the expert count.")
    scores = gating_output.float().sigmoid()
    selection_scores = scores + router_bias.to(
        device=gating_output.device, dtype=torch.float32
    )
    # Match the former kernel's lower-expert-ID tie break.
    topk_ids = torch.argsort(selection_scores, dim=-1, descending=True, stable=True)[
        :, :topk
    ]
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights *= routed_scaling_factor

    invalid_rows = gating_output.isnan().any(dim=-1, keepdim=True)
    topk_weights = torch.where(invalid_rows, 0, topk_weights)
    topk_ids = torch.where(invalid_rows, nan_row_i_out, topk_ids)
    return topk_weights, topk_ids.to(indices_dtype)
