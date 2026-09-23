# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 model entry point.

The DSA attention backends live in this package too (``sparse_attention``,
``sparse_summary_cache``) because they are Step5-specific and are selected by
the model rather than by platform detection. The current implementation uses
PyTorch and FlashInfer on B300 in eager mode, without speculative decoding.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Step5ForCausalLM
    from .mtp import Step5MTP

__all__ = ["Step5ForCausalLM", "Step5MTP"]


def __getattr__(name: str):
    if name == "Step5ForCausalLM":
        from .model import Step5ForCausalLM

        return Step5ForCausalLM
    if name == "Step5MTP":
        from .mtp import Step5MTP

        return Step5MTP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
