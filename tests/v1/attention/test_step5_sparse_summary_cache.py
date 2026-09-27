# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.step5.nvidia.sparse_summary_cache import Step5SparseSummaryCache


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


def _assert_same_summaries(reference, actual):
    torch.testing.assert_close(
        actual.mean.view(torch.float8_e4m3fn).float().cpu(),
        reference.mean.view(torch.float8_e4m3fn).float(),
        rtol=0.13,
        atol=1e-3,
    )
    pages = reference.pages
    torch.testing.assert_close(
        actual.last_slot[:pages].cpu(), reference.last_slot[:pages]
    )
    # Accumulators are only live for pages with an open region.
    open_pages = reference.last_slot[:pages] >= 0
    for name in ("numerator", "denominator", "maximum"):
        torch.testing.assert_close(
            getattr(actual, name)[:pages][open_pages.cuda()].cpu(),
            getattr(reference, name)[:pages][open_pages],
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_speculative_tokens", [1, 3])
def test_cuda_request_journal_matches_reference(num_speculative_tokens):
    """The device journal must undo rejected drafts like the CPU reference.

    Covers mixed prefill/verify batches, page crossings, skipped requests,
    request slot and page reuse after a finish, and a COW prefix fork.
    """
    torch.manual_seed(5)
    generator = torch.Generator().manual_seed(6)
    device = torch.device("cuda")
    width = num_speculative_tokens + 1
    block_size, pages = 16, 128
    reference = Step5SparseSummaryCache(pages, block_size, torch.device("cpu"))
    actual = Step5SparseSummaryCache(
        pages, block_size, device, max_num_seqs=8, request_journal_width=width
    )
    free_pages = list(torch.randperm(pages, generator=generator).tolist())
    free_states = list(range(8))
    # Request: [page table, accepted length, pending prompt tokens, state index].
    requests: list[list] = []

    def add_request(prompt, table=None, length=0):
        table = table or [free_pages.pop() for _ in range(8)]
        requests.append([table, length, prompt, free_states.pop(0)])

    for prompt in (5, 13, 20, 31):
        add_request(prompt)
    for step in range(24):
        if step == 8:
            add_request(11)
        if step == 12:
            # Finish a request; its state slot and zeroed pages are reused.
            table, _, _, state = requests.pop(0)
            for cache in (reference, actual):
                cache.reset(table)
            free_pages.extend(table)
            free_states.insert(0, state)
            add_request(9)
        if step == 16:
            # A prefix hit copies a committed full page for a new request.
            src = requests[0][0][0]
            dst = free_pages.pop()
            copy = [SimpleNamespace(src_block_id=src, dst_block_id=dst)]
            for cache in (reference, actual):
                cache.reset([dst])
                cache.copy(copy)
            fork = [dst] + [free_pages.pop() for _ in range(7)]
            add_request(6, table=fork, length=block_size)
        scheduled = [
            req for i, req in enumerate(requests) if not (step % 5 == 4 and i == 1)
        ]
        slots, starts, tables, seq_lens, states = [], [0], [], [], []
        for table, length, prompt, state in scheduled:
            count = prompt + width
            for position in range(length, length + count):
                slots.append(table[position // block_size] * block_size + position % 16)
            starts.append(len(slots))
            tables.append(table)
            seq_lens.append(length + count)
            states.append(state)
        key = torch.randn(len(slots), 1, 256, generator=generator).bfloat16()
        z = torch.randn(len(slots), 1, 256, generator=generator).bfloat16()
        reference.update_speculative(
            key, z, slots, starts, tables, num_speculative_tokens
        )
        actual.update_request_journal(
            key.to(device),
            z.to(device),
            torch.tensor(slots, device=device),
            torch.tensor(starts, device=device, dtype=torch.int32),
            torch.tensor(seq_lens, device=device, dtype=torch.int32),
            torch.tensor(states, device=device),
            actual.speculative_prefix_metadata(slots, starts, device),
        )
        torch.accelerator.synchronize()
        _assert_same_summaries(reference, actual)
        for req in scheduled:
            accepted = int(torch.randint(1, width + 1, (), generator=generator))
            req[1] += req[2] + accepted
            req[2] = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vectorized_prefill_metadata_matches_dict_grouping():
    """Step-level numpy metadata must equal the per-row dict construction."""
    import numpy as np

    from vllm.models.step5.nvidia.ops.summary_update import (
        make_prefill_metadata,
        make_prefill_metadata_from_slots,
    )

    rng = np.random.default_rng(0)
    device = torch.device("cuda")
    cache = Step5SparseSummaryCache(64, 16, device, request_journal_width=3)
    for trial in range(20):
        # Contiguous runs from shuffled pages, with padding rows mixed in.
        pages = rng.permutation(64)[: rng.integers(1, 6)]
        slots = np.concatenate(
            [page * 16 + np.arange(rng.integers(0, 16), 16) for page in pages]
        )
        slots = np.where(rng.random(slots.size) < 0.1, -1, slots)
        groups = Step5SparseSummaryCache._group_slots(slots.tolist())
        actual = make_prefill_metadata_from_slots(slots, 16, device)
        if not groups:
            assert actual is None
            continue
        expected = make_prefill_metadata(groups, 16, device)
        assert torch.equal(actual, expected), trial

        starts = np.sort(
            np.concatenate(([0, slots.size], rng.integers(0, slots.size, 3)))
        )
        prefix = [-1] * slots.size
        for start, end in zip(starts, starts[1:]):
            if start < end and slots[start] >= 0:
                for row in range(start, max(start, end - 3)):
                    prefix[row] = int(slots[row])
        expected = cache.prefill_metadata(np.array(prefix), device)
        actual = cache.speculative_prefix_metadata(slots, starts, device)
        assert (actual is None and expected is None) or torch.equal(actual, expected)
