# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.step5.sparse_summary_cache import Step5SparseSummaryCache


@pytest.mark.parametrize("chunk", [1, 3, 8, 13])
def test_csa_chunking_preserves_history_and_partial_tail(chunk):
    torch.manual_seed(3)
    key = torch.randn(29, 1, 256).bfloat16()
    z = torch.randn_like(key)
    full = Step5SparseSummaryCache(4, 16, torch.device("cpu"))
    chunked = Step5SparseSummaryCache(4, 16, torch.device("cpu"))
    # Logical sequence crosses non-contiguous physical pages.
    slots = list(range(32, 48)) + list(range(13))
    full.update(key, z, slots)
    for start in range(0, len(slots), chunk):
        chunked.update(
            key[start : start + chunk],
            z[start : start + chunk],
            slots[start : start + chunk],
        )
    assert torch.equal(full.mean, chunked.mean)
    for name in ("last_slot", "numerator", "denominator", "maximum"):
        torch.testing.assert_close(
            getattr(full, name), getattr(chunked, name), rtol=0, atol=0
        )
    reference = (z[:8].float().softmax(0) * key[:8].float()).sum(0)
    actual = full.mean[2, 0].view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(actual, reference[0], rtol=0.07, atol=0.015)


def test_summary_cow_copies_partial_tail_and_release_keeps_prefix():
    torch.manual_seed(4)
    cache = Step5SparseSummaryCache(4, 16, torch.device("cpu"))
    key = torch.randn(8, 1, 256).bfloat16()
    z = torch.randn_like(key)
    cache.update(key[:3], z[:3], [0, 1, 2])
    cache.copy([SimpleNamespace(src_block_id=0, dst_block_id=2)])
    cache.update(key[3:], z[3:], list(range(35, 40)))
    cache.update(key[3:], z[3:], list(range(3, 8)))
    assert torch.equal(cache.mean[0], cache.mean[2])
    expected = cache.mean[2].clone()
    cache.reset([2], clear_mean=False)
    assert torch.equal(expected, cache.mean[2])
    cache.reset([2])
    assert not cache.mean[2].any()
    assert (cache.last_slot == -1).all()


def test_partial_region_without_history_fails():
    cache = Step5SparseSummaryCache(1, 16, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="tail missing"):
        cache.update(torch.ones(1, 1, 256), torch.ones(1, 1, 256), [3])
