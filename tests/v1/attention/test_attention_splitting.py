# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.test_attention_backends import BATCH_SPECS
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    _make_metadata_with_slice,
    maybe_create_ubatch_slices,
    slice_query_start_locs,
    split_attn_metadata,
)


@pytest.fixture
def sample_query_start_loc():
    """Sample query_start_loc tensor for testing."""
    return torch.tensor([0, 5, 12, 20, 35, 50])


def test_basic_slice_middle(sample_query_start_loc):
    """Test slicing from middle of tensor."""
    req_slice = slice(1, 3)  # slice from index 1 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 7, 15])
    assert torch.equal(result, expected)


def test_slice_from_beginning(sample_query_start_loc):
    """Test slicing from the beginning of tensor."""
    req_slice = slice(0, 2)  # slice from index 0 to 2
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12])
    assert torch.equal(result, expected)


def test_slice_to_end(sample_query_start_loc):
    """Test slicing to the end of tensor."""
    req_slice = slice(3, 5)  # slice from index 3 to 5 (last index)
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15, 30])
    assert torch.equal(result, expected)


def test_single_element_slice(sample_query_start_loc):
    """Test slice that results in single element."""
    req_slice = slice(2, 3)  # slice from index 2 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 8])
    assert torch.equal(result, expected)


def test_full_tensor_slice(sample_query_start_loc):
    """Test slicing the entire tensor."""
    req_slice = slice(0, 5)  # slice entire tensor
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12, 20, 35, 50])
    assert torch.equal(result, expected)


def test_slice_bounds_edge_cases(sample_query_start_loc):
    # Test slice that goes exactly to the last element
    req_slice = slice(4, 5)  # Last index
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15])
    assert torch.equal(result, expected)


@pytest.fixture
def small_decode_metadata():
    """Create metadata for small decode batch."""
    batch_spec = BATCH_SPECS["small_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


@pytest.fixture
def large_decode_metadata():
    """Create metadata for small decode batch."""
    batch_spec = BATCH_SPECS["large_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


@pytest.fixture
def mixed_small_metadata():
    """Create metadata for mixed small batch."""
    batch_spec = BATCH_SPECS["mixed_small"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec, block_size=16, device=device)


# Tests for _make_metadata_with_slice
def test_make_metadata_with_slice_decode_batch(small_decode_metadata):
    """Test slicing decode batch metadata"""
    small_decode_metadata.is_prefilling = torch.tensor([False] * 5)
    # Split first request only
    ubatch_slice = UBatchSlice(slice(0, 1), slice(0, 1))

    result = _make_metadata_with_slice(ubatch_slice, small_decode_metadata)

    # Check sliced results
    assert result.num_reqs == 1  # slice(0, 1) gives 1 requests
    assert result.num_actual_tokens == 1  # slice(0, 1) gives 1 token
    assert result.max_query_len == 1
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1]))
    assert torch.equal(result.seq_lens, torch.tensor([32]))
    assert result.is_prefilling is not None
    assert result.is_prefilling.tolist() == [False]


def test_make_metadata_with_slice_mixed_batch(mixed_small_metadata):
    """Test slicing mixed batch metadata"""
    mixed_small_metadata.is_prefilling = torch.tensor([False, False, True])
    ubatch_slice = UBatchSlice(slice(1, 3), slice(1, 7))  # Requests 1-3, tokens 1-7

    result = _make_metadata_with_slice(ubatch_slice, mixed_small_metadata)

    assert result.num_reqs == 2  # slice(1, 3) gives 2 requests
    assert result.num_actual_tokens == 6  # slice(1, 7) gives 6 tokens
    assert result.max_query_len == 5
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1, 6]))
    assert torch.equal(result.seq_lens, torch.tensor([40, 48]))
    assert result.is_prefilling is not None
    assert result.is_prefilling.tolist() == [False, True]


def test_unpadded_preserves_prefill_phase(small_decode_metadata):
    small_decode_metadata.is_prefilling = torch.tensor(
        [False, True, False, True, False]
    )

    result = small_decode_metadata.unpadded(
        num_actual_tokens=1,
        num_actual_reqs=1,
    )

    assert result.is_prefilling is not None
    assert result.is_prefilling.tolist() == [False]


def test_split_attn_metadata_decode_batch(large_decode_metadata):
    """Test splitting decode batch into two equal parts."""
    num_tokens = large_decode_metadata.num_reqs
    mid_point = num_tokens // 2
    ubatch_slices = [
        UBatchSlice(slice(0, mid_point), slice(0, mid_point)),
        UBatchSlice(slice(mid_point, num_tokens), slice(mid_point, num_tokens)),
    ]

    results = split_attn_metadata(ubatch_slices, large_decode_metadata)

    assert len(results) == 2

    # Check first split
    assert results[0].num_reqs == mid_point
    assert results[0].num_actual_tokens == mid_point
    assert torch.equal(results[0].seq_lens, torch.tensor([2048] * mid_point))

    # Check second split
    assert results[1].num_reqs == mid_point
    assert results[1].num_actual_tokens == mid_point
    assert torch.equal(results[1].seq_lens, torch.tensor([2048] * mid_point))


def apply_split_decodes_and_prefills(
    query_lens: list[int],
    decode_threshold: int,
    require_uniform: bool,
    padded_num_tokens: int | None = None,
    is_prefilling: list[bool] | None = None,
    treat_short_extends_as_decodes: bool = True,
):
    """Helper function to apply split_decodes_and_prefills and return
    the results."""
    device = torch.device("cpu")
    seq_lens = [10 * (i + 1) for i in range(len(query_lens))]
    common_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=device,
    )

    if padded_num_tokens is not None:
        common_metadata.num_actual_tokens = padded_num_tokens
    if is_prefilling is not None:
        common_metadata.is_prefilling = torch.tensor(is_prefilling)

    return split_decodes_and_prefills(
        common_metadata,
        decode_threshold=decode_threshold,
        require_uniform=require_uniform,
        treat_short_extends_as_decodes=treat_short_extends_as_decodes,
    )


def test_split_decodes_and_prefills_nonuniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, False)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_short_decodes():
    query_lens = [1, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 7
    assert num_prefills == 0
    assert num_decode_tokens == sum(query_lens)
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_nonuniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, False)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_nonuniform_mixed_batch():
    query_lens = [2, 1, 3, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, False)
    )
    assert num_decodes == 4  # 2, 1, 3, 4 are all <= 4
    assert num_prefills == 4  # 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 10  # 2 + 1 + 3 + 4
    assert num_prefill_tokens == 26  # 5 + 6 + 7 + 8


@pytest.mark.parametrize("reset_replay", [False, True])
def test_mtp_q1_reset_replay_classification(reset_replay):
    metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[100, 101], query_lens=[2, 1]),
        block_size=16,
        device=torch.device("cpu"),
    )
    metadata.is_prefilling = torch.tensor([False, reset_replay])
    result = split_decodes_and_prefills(
        metadata, decode_threshold=2, treat_short_extends_as_decodes=False
    )
    if reset_replay:
        assert result == (1, 1, 2, 1)
    else:
        assert result == (2, 0, 3, 0)


def test_split_decodes_and_prefills_uniform_all_ones():
    query_lens = [1, 1, 1]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 1, True)
    )
    assert num_decodes == 3
    assert num_prefills == 0
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 0


def test_split_decodes_and_prefills_uniform_short_extend():
    result = apply_split_decodes_and_prefills(
        [1, 1],
        decode_threshold=1,
        require_uniform=True,
        is_prefilling=[False, True],
        treat_short_extends_as_decodes=False,
    )
    assert result == (1, 1, 1, 1)


def test_split_decodes_and_prefills_uniform_all_short_decodes():
    query_lens = [2, 2, 1, 3, 2, 1, 2]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 2
    assert num_prefills == 5
    assert num_decode_tokens == 4
    assert num_prefill_tokens == (1 + 3 + 2 + 1 + 2)


def test_split_decodes_and_prefills_uniform_all_prefills():
    query_lens = [4, 5, 6, 7]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True)
    )
    assert num_decodes == 0
    assert num_prefills == 4
    assert num_decode_tokens == 0
    assert num_prefill_tokens == sum(query_lens)


def test_split_decodes_and_prefills_uniform_mixed_batch_all_uniform_decodes():
    query_lens = [2, 2, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 3  # 2, 2, 2 are all <= 4 and uniform
    assert num_prefills == 5  # 4, 5, 6, 7, 8 are all > 4
    assert num_decode_tokens == 6  # 2 + 2 + 2
    assert num_prefill_tokens == 30  # 4 + 5 + 6 + 7 + 8


def test_split_decodes_and_prefills_uniform_mixed_batch_non_uniform_decodes():
    query_lens = [2, 1, 2, 4, 5, 6, 7, 8]
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 4, True)
    )
    assert num_decodes == 1  # only the first 2 is taken as decode
    assert num_prefills == 7  # 1, 2, 4, 5, 6, 7, 8 are all > 4 or non-uniform
    assert num_decode_tokens == 2  # only the first 2
    assert num_prefill_tokens == (sum(query_lens) - 2)  # rest of the tokens


def test_split_decodes_and_prefills_uniform_padded_batch_all_same():
    """Uniform batch where all query lengths are identical with 0 length padded reqs."""
    # All query lengths are 2, with decode_threshold=3 (so 2 <= 3)
    # This triggers the padded uniform path at line 891
    query_lens = [2, 2, 2, 0]
    padded_num_tokens = 8
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        apply_split_decodes_and_prefills(query_lens, 3, True, padded_num_tokens)
    )
    # With uniform batch, all requests are treated as decodes
    assert num_decodes == 4
    assert num_prefills == 0
    assert num_decode_tokens == padded_num_tokens
    assert num_prefill_tokens == 0


@pytest.mark.parametrize(
    "seq_lens,query_lens,split_point,expected_first_reqs,expected_second_reqs",
    [
        # Split in the middle of request 1
        ([32, 40], [8, 8], 12, 2, 1),
        # Split inside the first request
        ([32, 40], [8, 8], 4, 1, 2),
    ],
)
def test_prefill_split_across_ubatches(
    seq_lens, query_lens, split_point, expected_first_reqs, expected_second_reqs
):
    """Test splitting a prefill across ubatches."""
    import numpy as np

    device = torch.device("cpu")
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(batch_spec, block_size=16, device=device)

    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    qsl_np = common.query_start_loc_cpu.numpy()
    num_tokens = common.num_actual_tokens

    ubatch_slices, _ = maybe_create_ubatch_slices(
        True,
        num_scheduled_tokens,
        num_tokens,
        batch_spec.batch_size,
        split_point=split_point,
        num_ubatches=2,
    )
    assert ubatch_slices is not None and len(ubatch_slices) == 2

    first_meta = _make_metadata_with_slice(ubatch_slices[0], common)
    second_meta = _make_metadata_with_slice(ubatch_slices[1], common)

    # Token counts match the split
    assert first_meta.num_actual_tokens == split_point
    assert second_meta.num_actual_tokens == num_tokens - split_point

    # Number of requests per ubatch
    assert first_meta.num_reqs == expected_first_reqs
    assert second_meta.num_reqs == expected_second_reqs

    # Identify which request is split and how many tokens are in the first chunk
    split_req_idx = int(np.searchsorted(qsl_np, split_point, side="right") - 1)
    tokens_in_first_chunk = split_point - int(qsl_np[split_req_idx])
    orig_q_lens = common.query_start_loc_cpu[1:] - common.query_start_loc_cpu[:-1]

    # Check query length continuity: first-chunk + second-chunk == original qlen
    # First ubatch last request query length
    qlen_first_last = int(
        first_meta.query_start_loc_cpu[-1] - first_meta.query_start_loc_cpu[-2]
    )
    # Second ubatch first request query length
    qlen_second_first = int(
        second_meta.query_start_loc_cpu[1] - second_meta.query_start_loc_cpu[0]
    )
    assert qlen_first_last == tokens_in_first_chunk
    assert qlen_first_last + qlen_second_first == int(orig_q_lens[split_req_idx])

    # Check seq_lens adjustments
    # Context lengths per original request
    context_lens = [s - q for s, q in zip(seq_lens, query_lens)]

    # First ubatch: last request's seq_len should be
    #  context + tokens_in_first_chunk
    expected_seqlen = context_lens[split_req_idx] + tokens_in_first_chunk
    assert int(first_meta.seq_lens[-1]) == expected_seqlen

    # For full preceding requests in first ubatch, seq_lens should match
    #  originals
    for i in range(first_meta.num_reqs - 1):
        assert int(first_meta.seq_lens[i]) == seq_lens[i]

    # Second ubatch: first request (continuation) seq_len should be full
    #  original
    assert int(second_meta.seq_lens[0]) == seq_lens[split_req_idx]
    # Any following full requests in second ubatch should match originals
    for j in range(1, second_meta.num_reqs):
        # Map to original request index
        orig_idx = split_req_idx + j
        assert int(second_meta.seq_lens[j]) == seq_lens[orig_idx]


def test_build_attention_metadata_preserves_async_prefill_phase():
    """Async attention metadata keeps the exact host-side prefill phase."""
    from unittest.mock import MagicMock, patch

    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    num_reqs = 3
    num_reqs_padded = 5

    # Real rows [0-2] have known computed/prompt values; padded rows [3-4]
    # carry stale data from a prior prefill (num_computed < num_prompt → True).
    num_computed = torch.tensor([50, 100, 200, 10, 20], dtype=torch.int32)
    num_prompt = torch.tensor([50, 200, 200, 100, 200], dtype=torch.int32)

    runner = MagicMock()
    runner.kv_cache_config.kv_cache_groups = [
        MagicMock()
    ]  # non-empty: skip early return
    runner.attn_groups = [[]]  # empty inner list: inner loop never runs
    runner.input_batch.num_computed_tokens_cpu_tensor = num_computed
    runner.input_batch.num_prompt_tokens_cpu_tensor = num_prompt
    runner.input_batch.is_prefilling = torch.tensor(
        [False, True, False, True, True], dtype=torch.bool
    )
    runner.optimistic_seq_lens_cpu = torch.tensor([100, 200, 300, 0, 0])
    runner.query_start_loc.gpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)
    runner.query_start_loc.cpu = torch.zeros(num_reqs_padded + 1, dtype=torch.int32)
    runner.seq_lens = torch.zeros(num_reqs_padded, dtype=torch.int32)
    runner.positions = torch.zeros(num_reqs_padded, dtype=torch.int64)
    runner.routed_experts_initialized = False
    runner.use_async_spec_decode = True
    runner.dcp_world_size = 1
    runner.speculative_config = object()
    runner.drafter = object()
    runner.is_mm_prefix_lm = False
    runner._get_encoder_seq_lens.return_value = (None, None)

    captured: dict[str, CommonAttentionMetadata] = {}
    original_unpadded = CommonAttentionMetadata.unpadded

    def capturing_unpadded(self, *args, **kwargs):
        captured["producer"] = self
        result = original_unpadded(self, *args, **kwargs)
        captured["unpadded"] = result
        return result

    with patch.object(CommonAttentionMetadata, "unpadded", capturing_unpadded):
        _, spec_decode_metadata = GPUModelRunner._build_attention_metadata(
            runner,
            num_tokens=num_reqs,
            num_reqs=num_reqs,
            max_query_len=1,
            num_tokens_padded=num_reqs_padded,
            num_reqs_padded=num_reqs_padded,
            slot_mappings={0: torch.zeros(num_reqs_padded, dtype=torch.int64)},
        )

    producer_metadata = captured["producer"]
    unpadded_metadata = captured["unpadded"]
    assert spec_decode_metadata is unpadded_metadata
    captured_is_prefilling = producer_metadata.is_prefilling
    assert isinstance(captured_is_prefilling, torch.Tensor)
    assert producer_metadata.seq_lens_cpu_upper_bound.tolist() == [100, 200, 300, 0, 0]
    assert unpadded_metadata.seq_lens_cpu_upper_bound.tolist() == [100, 200, 300]
    assert unpadded_metadata.is_prefilling is not None
    assert unpadded_metadata.is_prefilling.tolist() == [False, True, False]
    assert not captured_is_prefilling[0]  # decode  (50 >= 50)
    assert captured_is_prefilling[1]  # prefill (100 < 200)
    assert not captured_is_prefilling[2]  # decode  (200 >= 200)
    assert not captured_is_prefilling[3]  # stale data (10 < 100) zeroed
    assert not captured_is_prefilling[4]  # stale data (20 < 200) zeroed


def test_build_attention_metadata_does_not_cache_across_ubatches():
    from types import SimpleNamespace

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    class Builder:
        supports_update_block_table = True

        def __init__(self, ubatch_id):
            self.ubatch_id = ubatch_id
            self.build_calls = 0
            self.update_calls = 0

        def build(self, common_prefix_len, common_attn_metadata):
            self.build_calls += 1
            return SimpleNamespace(ubatch_id=self.ubatch_id)

        def update_block_table(self, metadata, block_table, slot_mapping):
            self.update_calls += 1
            return metadata

    builders = [Builder(0), Builder(1)]
    attn_group = SimpleNamespace(
        layer_names=["layer"],
        get_metadata_builder=lambda ubatch_id=0: builders[ubatch_id],
    )
    kv_cache_spec = object()
    block_table = torch.zeros((2, 1), dtype=torch.int32)
    input_batch = SimpleNamespace(
        block_table={
            0: SimpleNamespace(
                get_device_tensor=lambda num_reqs: block_table[:num_reqs]
            )
        },
        num_computed_tokens_cpu_tensor=torch.tensor([8, 16], dtype=torch.int32),
        num_prompt_tokens_cpu_tensor=torch.tensor([8, 16], dtype=torch.int32),
        is_prefilling=torch.zeros(2, dtype=torch.bool),
    )
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_cache_spec)]
        ),
        attn_groups=[[attn_group]],
        input_batch=input_batch,
        query_start_loc=SimpleNamespace(
            gpu=query_start_loc,
            cpu=query_start_loc.clone(),
        ),
        seq_lens=torch.tensor([9, 17], dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.tensor([9, 17], dtype=torch.int32),
        positions=torch.tensor([8, 16], dtype=torch.int64),
        routed_experts_initialized=False,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=SimpleNamespace(rswa_window=None),
        cache_config=SimpleNamespace(
            use_replayssm=False,
            kv_sharing_fast_prefill=False,
        ),
        dcp_world_size=1,
        speculative_config=None,
        _get_encoder_seq_lens=lambda *args, **kwargs: (None, None),
    )

    metadata, _ = GPUModelRunner._build_attention_metadata(
        runner,
        num_tokens=2,
        num_reqs=2,
        max_query_len=1,
        ubatch_slices=[
            UBatchSlice(slice(0, 1), slice(0, 1)),
            UBatchSlice(slice(1, 2), slice(1, 2)),
        ],
        slot_mappings={0: torch.tensor([0, 1], dtype=torch.int64)},
    )

    assert [builder.build_calls for builder in builders] == [1, 1]
    assert [builder.update_calls for builder in builders] == [0, 0]
    assert metadata[0]["layer"].ubatch_id == 0
    assert metadata[1]["layer"].ubatch_id == 1
