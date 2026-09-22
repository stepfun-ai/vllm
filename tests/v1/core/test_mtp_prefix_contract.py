# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    _max_memory_usage_bytes_from_groups,
    _project_kv_cache_groups_to_worker,
    generate_scheduler_kv_cache_config,
    init_none_hash,
    resolve_dcp_kv_block_size,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from .test_prefix_caching import make_request

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _initialize_block_hash():
    init_none_hash(sha256)


@pytest.mark.parametrize(
    ("remaining_tokens", "expected_num_new_tokens"),
    [
        (0, 10),
        (1, 7),
        (2, 7),
        (3, 7),
    ],
)
def test_reserve_multi_module_mtp_prefill_lookahead(
    remaining_tokens: int,
    expected_num_new_tokens: int,
):
    scheduler = SimpleNamespace(num_prefill_lookahead=3)
    request = SimpleNamespace(num_tokens=10)
    requested_tokens = request.num_tokens - remaining_tokens

    num_new_tokens = Scheduler._reserve_prefill_lookahead(
        scheduler,
        request,
        num_computed_tokens=0,
        num_new_tokens=requested_tokens,
    )

    assert num_new_tokens == expected_num_new_tokens


def test_swa_retention_includes_mtp_tail_eagle_proof_and_scheduler_alignment():
    spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=9,
        extra_retained_tokens=2,
    )

    assert (
        spec.retained_window_size(
            scheduler_block_size=16,
            retain_eagle_proof=False,
        )
        == 11
    )
    assert (
        spec.retained_window_size(
            scheduler_block_size=16,
            retain_eagle_proof=True,
        )
        == 30
    )
    assert (
        spec.max_admission_blocks_per_request(
            max_in_flight_tokens=5,
            max_model_len=100,
            scheduler_block_size=16,
            retain_eagle_proof=True,
        )
        == 10
    )
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        max_in_flight_tokens=5,
        model_config=SimpleNamespace(max_model_len=100),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            block_size=4,
            enable_prefix_caching=True,
        ),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
    )
    assert spec.max_memory_usage_bytes(vllm_config) == 5 * spec.page_size_bytes

    block_pool = BlockPool(
        num_gpu_blocks=32,
        enable_caching=True,
        hash_block_size=spec.block_size,
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=16,
        max_admission_blocks_per_request=10,
    )
    manager.use_eagle = True

    assert manager.get_num_skipped_tokens(29) == 0
    assert manager.get_num_skipped_tokens(30) == 1
    assert manager.get_num_skipped_tokens(34) == 5


def test_swa_startup_sizing_matches_hybrid_runtime_admission_cap():
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    swa_spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=9,
        extra_retained_tokens=2,
    )
    groups = [
        KVCacheGroupSpec(["target"], full_spec),
        KVCacheGroupSpec(["draft-0"], swa_spec, is_eagle_group=True),
        # Equal specs share one hybrid lookup group, so both retain proof KV.
        KVCacheGroupSpec(["draft-1"], swa_spec),
    ]
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        max_in_flight_tokens=5,
        model_config=SimpleNamespace(max_model_len=100),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            block_size=4,
            enable_prefix_caching=True,
        ),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
        kv_transfer_config=None,
    )

    # Full attention needs 7 blocks; each equal SWA group needs the same
    # 10-block proof-aware cap. All page sizes are 128 bytes.
    assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == 27 * 128

    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
        ),
        max_model_len=100,
        max_in_flight_tokens=5,
        scheduler_block_size=16,
        hash_block_size=4,
        enable_caching=True,
        use_eagle=True,
    )
    assert [
        single_manager._max_admission_blocks_per_request
        for single_manager in manager.coordinator.single_type_managers
    ] == [None, 10, 10]


def test_uniform_swa_startup_sizing_matches_scheduler_proof_closure():
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    eagle_swa_spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=9,
        extra_retained_tokens=2,
    )
    other_swa_spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=13,
        extra_retained_tokens=2,
    )

    def uniform(name: str, spec):
        return UniformTypeKVCacheSpecs(
            block_size=spec.block_size,
            kv_cache_specs={name: spec},
        )

    groups = [
        KVCacheGroupSpec(["target"], uniform("target", full_spec)),
        KVCacheGroupSpec(
            ["draft-0"],
            uniform("draft-0", eagle_swa_spec),
            is_eagle_group=True,
        ),
        # The wrappers compare unequal because their layer-name dictionaries
        # differ, but the scheduler representatives compare equal.
        KVCacheGroupSpec(["draft-1"], uniform("draft-1", eagle_swa_spec)),
        KVCacheGroupSpec(["local"], uniform("local", other_swa_spec)),
    ]
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        max_in_flight_tokens=5,
        model_config=SimpleNamespace(max_model_len=100),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            block_size=4,
            enable_prefix_caching=True,
            prefix_match_unit=None,
        ),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
        kv_transfer_config=None,
    )

    # Full attention needs 7 blocks. The EAGLE group and its equal sibling
    # both need 10 proof-aware blocks; the distinct non-EAGLE SWA group needs 6.
    assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == 33 * 128

    worker_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    assert [group.is_eagle_group for group in scheduler_config.kv_cache_groups] == [
        False,
        True,
        False,
        False,
    ]

    manager = KVCacheManager(
        scheduler_config,
        max_model_len=100,
        max_in_flight_tokens=5,
        scheduler_block_size=16,
        hash_block_size=4,
        enable_caching=True,
        use_eagle=True,
    )
    assert [
        single_manager._max_admission_blocks_per_request
        for single_manager in manager.coordinator.single_type_managers
    ] == [None, 10, 10, 6]


def test_single_uniform_swa_group_uses_proof_aware_recursive_sizing():
    first = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=9,
        extra_retained_tokens=2,
    )
    second = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=9,
        extra_retained_tokens=2,
    )
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=4,
        kv_cache_specs={"draft-0": first, "draft-1": second},
    )
    groups = [
        KVCacheGroupSpec(
            ["draft-0", "draft-1"],
            uniform_spec,
            is_eagle_group=True,
        )
    ]
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        max_in_flight_tokens=5,
        model_config=SimpleNamespace(max_model_len=100),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            block_size=4,
            enable_prefix_caching=True,
        ),
        speculative_config=SimpleNamespace(use_eagle=lambda: True),
        kv_transfer_config=None,
    )

    # Each child needs 7 blocks at scheduler_block_size=4.
    assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == 7 * (
        first.page_size_bytes + second.page_size_bytes
    )

    scheduler_config = generate_scheduler_kv_cache_config(
        [
            KVCacheConfig(
                num_blocks=32,
                kv_cache_tensors=[],
                kv_cache_groups=groups,
            )
        ]
    )
    manager = KVCacheManager(
        scheduler_config,
        max_model_len=100,
        max_in_flight_tokens=5,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_caching=True,
        use_eagle=True,
    )
    assert (
        manager.coordinator.single_type_managers[0]._max_admission_blocks_per_request
        == 7
    )


def test_chunked_local_manager_construction_uses_compatible_admission_contract():
    spec = ChunkedLocalAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=8,
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["local"], spec)],
        ),
        max_model_len=100,
        max_in_flight_tokens=5,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_caching=True,
    )

    assert (
        manager.coordinator.single_type_managers[0]._max_admission_blocks_per_request
        == 4
    )


def test_uniform_scheduler_block_size_matches_runtime_projection_under_dcp():
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    uniform_full_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"attention": full_spec},
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
    )
    groups = [
        KVCacheGroupSpec(["attention"], uniform_full_spec),
        KVCacheGroupSpec(["mamba"], mamba_spec),
    ]
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
            prefix_match_unit=None,
        ),
        kv_transfer_config=None,
    )

    startup_block_size = math.lcm(
        *(resolve_dcp_kv_block_size(group.kv_cache_spec, 2) for group in groups)
    )
    scheduler_config = generate_scheduler_kv_cache_config(
        [
            KVCacheConfig(
                num_blocks=32,
                kv_cache_tensors=[],
                kv_cache_groups=groups,
            )
        ]
    )
    runtime_block_size, _ = resolve_kv_cache_block_sizes(
        scheduler_config,
        vllm_config,
    )

    assert startup_block_size == runtime_block_size == 32


def test_pp_projection_preserves_global_eagle_group_metadata():
    target_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    draft_spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float32,
        sliding_window=9,
    )
    global_groups = [
        KVCacheGroupSpec(["target"], target_spec),
        KVCacheGroupSpec(["draft"], draft_spec, is_eagle_group=True),
    ]

    stage0_groups = _project_kv_cache_groups_to_worker(
        global_groups,
        {"target": target_spec},
    )
    stage1_groups = _project_kv_cache_groups_to_worker(
        global_groups,
        {"draft": draft_spec},
    )

    assert [group.is_eagle_group for group in stage0_groups] == [False, True]
    assert [group.is_eagle_group for group in stage1_groups] == [False, True]
    scheduler_config = generate_scheduler_kv_cache_config(
        [
            KVCacheConfig(
                num_blocks=32,
                kv_cache_tensors=[],
                kv_cache_groups=stage0_groups,
            ),
            KVCacheConfig(
                num_blocks=32,
                kv_cache_tensors=[],
                kv_cache_groups=stage1_groups,
            ),
        ]
    )
    assert [group.is_eagle_group for group in scheduler_config.kv_cache_groups] == [
        False,
        True,
    ]


def test_multi_module_mtp_only_publishes_finalized_draft_kv():
    scheduler_block_size = 4
    draft_block_size = 1
    target_spec = FullAttentionSpec(
        block_size=scheduler_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    draft_spec = FullAttentionSpec(
        block_size=draft_block_size,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["target"], target_spec),
                KVCacheGroupSpec(["draft"], draft_spec, is_eagle_group=True),
            ],
        ),
        max_model_len=128,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=draft_block_size,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=3,
    )
    request = make_request(
        "mtp-prefill",
        list(range(10)),
        draft_block_size,
        sha256,
    )
    computed_blocks, num_computed_tokens, _ = manager.get_computed_blocks(request)
    assert num_computed_tokens == 0

    assert manager.allocate_slots(
        request,
        num_new_tokens=7,
        new_computed_blocks=computed_blocks,
    )
    draft_manager = manager.coordinator.single_type_managers[1]
    draft_blocks = draft_manager.req_to_blocks[request.request_id]
    assert draft_manager.num_cached_block[request.request_id] == 5
    assert [block.block_hash is not None for block in draft_blocks] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]

    request.num_computed_tokens = 7
    assert manager.allocate_slots(request, num_new_tokens=3)
    assert draft_manager.num_cached_block[request.request_id] == 8
    assert [block.block_hash is not None for block in draft_blocks] == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_prefix_replay_ceiling_does_not_duplicate_eagle_rewind():
    block_size = 4
    replay_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    ).with_side_storage(prefix_cache_recompute_tokens=block_size)
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["draft"],
                    replay_spec,
                    is_eagle_group=True,
                )
            ],
        ),
        max_model_len=128,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
        use_eagle=True,
    )
    prefix = list(range(4 * block_size))
    producer = make_request("producer", prefix, block_size, sha256)
    producer_blocks, _, _ = manager.get_computed_blocks(producer)
    assert manager.allocate_slots(
        producer,
        num_new_tokens=len(prefix),
        new_computed_blocks=producer_blocks,
    )
    manager.free(producer)

    replay = make_request("replay", prefix + [999], block_size, sha256)
    assert manager._get_cache_hit_limits(replay) == (4 * block_size, 13)

    computed_blocks, num_computed_tokens, _ = manager.get_computed_blocks(replay)

    assert num_computed_tokens == 3 * block_size
    assert len(computed_blocks.blocks[0]) == 3
