# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 normalization layers."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNorm

_NORM_DTYPE_TO_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def get_step5_norm_dtype(config: Any) -> torch.dtype:
    norm_dtype = config.norm_dtype.lower()
    if norm_dtype in _NORM_DTYPE_TO_TORCH_DTYPE:
        return _NORM_DTYPE_TO_TORCH_DTYPE[norm_dtype]
    raise ValueError(f"Unknown norm_dtype: {norm_dtype!r}")


def _step5_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool,
) -> torch.Tensor:
    compute = x.float()
    variance = compute.pow(2).mean(dim=-1, keepdim=True)
    compute = compute * torch.rsqrt(variance + variance_epsilon)
    scale = weight.float()
    if zero_centered:
        scale = scale + 1.0
    return (compute * scale).to(x.dtype)


def apply_step5_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    out: torch.Tensor | None = None,
    zero_centered: bool = False,
) -> torch.Tensor:
    result = _step5_rms_norm(x, weight, variance_epsilon, zero_centered)
    if out is None:
        return result
    if out.shape != result.shape or out.dtype != result.dtype:
        raise ValueError(
            "Step5 RMSNorm output buffer must match the computed output "
            f"shape/dtype, got out={tuple(out.shape)}/{out.dtype}, "
            f"result={tuple(result.shape)}/{result.dtype}."
        )
    out.copy_(result)
    return out


def apply_step5_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    residual_out = (
        (x.float() + residual.float()).to(residual.dtype)
        if orig_dtype == torch.float16
        else x + residual
    )
    output = _step5_rms_norm(
        residual_out,
        weight,
        variance_epsilon,
        zero_centered,
    ).to(orig_dtype)
    return output, residual_out


class Step5RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        zero_centered: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.zero_centered = zero_centered

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        fp16_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            if output is not None or fp16_out:
                raise ValueError(
                    "Residual Step5RMSNorm does not support output buffers or fp16_out."
                )
            return apply_step5_add_rms_norm(
                x,
                residual,
                self.weight,
                self.variance_epsilon,
                zero_centered=self.zero_centered,
            )

        if fp16_out:
            raise ValueError("Step5RMSNorm does not support fp16_out.")
        return apply_step5_rms_norm(
            x,
            self.weight,
            self.variance_epsilon,
            out=output,
            zero_centered=self.zero_centered,
        )


def make_step5_rms_norm(
    hidden_size: int,
    eps: float = 1e-6,
    zero_centered: bool = False,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if zero_centered:
        return Step5RMSNorm(hidden_size, eps, zero_centered, dtype=dtype)
    return RMSNorm(hidden_size, eps, dtype=dtype)


class Step5IndexerLayerNorm(nn.Module):
    """Per-head LayerNorm for the Step5 sparse-attention indexer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            x.unflatten(-1, (-1, self.hidden_size)),
            (self.hidden_size,),
            self.weight,
            self.bias,
            self.variance_epsilon,
        )
        return normalized.flatten(-2, -1)
