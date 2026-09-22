# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    SlidingWindowSpec,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _zero_kv_blocks_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("cuda")
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layers_in_one_group_may_use_different_kernel_pages_per_block():
    """Derive kernel pages per block from each layer's allocation."""
    device = torch.device("cuda")
    num_blocks = 4
    spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=2,
    )
    # Two kernel pages per logical block.
    wide = torch.ones((num_blocks * 2, 3), dtype=torch.int32, device=device)
    # One kernel page per logical block, carved out of a larger allocation so
    # an over-strided write lands in the guard region instead of faulting or
    # silently hitting another tensor.
    narrow_backing = torch.ones((num_blocks * 3, 5), dtype=torch.int32, device=device)
    narrow = narrow_backing[:num_blocks]

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                None,
                ["wide", "narrow"],
                spec,
                0,
            )
        ],
        kernel_block_sizes=[1],
        static_forward_context={
            "wide": SimpleNamespace(kv_cache=wide),
            "narrow": SimpleNamespace(kv_cache=narrow),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected_wide = torch.ones_like(wide)
    expected_wide[2 * (num_blocks - 1) :] = 0
    assert torch.equal(wide, expected_wide)

    expected_narrow = torch.ones_like(narrow_backing)
    expected_narrow[num_blocks - 1] = 0
    assert torch.equal(narrow_backing, expected_narrow)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    blk_size = min(1 << (max_ps - 1).bit_length(), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        (max_ps + blk_size - 1) // blk_size,
        blk_size,
        2,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device("cuda")
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones(
        (num_blocks, block_stride_el), dtype=torch.int32, device=device
    )

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(
            [backing.data_ptr() + page_offset_el * backing.element_size()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([block_stride_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = SlidingWindowSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=1,
    )
    storages = {
        name: torch.ones((1, page_size), dtype=torch.int32)
        for name, page_size in zip(layer_names, page_sizes)
    }
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, [name], spec, group_id)
            for group_id, name in enumerate(layer_names)
        ],
        kernel_block_sizes=[1] * n_segs,
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=1,
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == page_sizes
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    assert captured_grids == [(n_blocks, n_segs, max_chunks)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_for_all_block_counts():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel.
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,  # max_chunks
        page_size_el,  # blk_size
        1,  # n_segs
    )

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device("cuda")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
        num_blocks=num_blocks,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout


class _BlockFirstBackend:
    @staticmethod
    def get_kv_cache_block_dim(*_args, **_kwargs):
        return 0


def _make_block_view(
    raw: torch.Tensor,
    *,
    num_blocks: int,
    offset_bytes: int,
    block_stride_bytes: int,
    block_size: int,
    head_size: int,
) -> torch.Tensor:
    """Build a blocks-first [B, H, N, 2D] view into packed int8 storage."""
    assert raw.numel() % 4 == 0
    page_elements = block_size * 2 * head_size
    return raw.view(torch.int32).as_strided(
        size=(num_blocks, 1, block_size, 2 * head_size),
        stride=(block_stride_bytes // 4, page_elements, 2 * head_size, 1),
        storage_offset=offset_bytes // 4,
    )


def _full_spec(*, block_size: int, head_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.float32,
    )


@pytest.mark.cpu_test
def test_packed_zeroer_metadata_uses_block_stride_and_page_span():
    """Packed layers step by the whole row but clear only their own page."""
    num_blocks = 3
    page_bytes = 64
    block_stride_bytes = 3 * page_bytes
    raw = torch.ones(num_blocks * block_stride_bytes, dtype=torch.int8)
    spec = _full_spec(block_size=4, head_size=2)
    views = {
        name: _make_block_view(
            raw,
            num_blocks=num_blocks,
            offset_bytes=offset,
            block_stride_bytes=block_stride_bytes,
            block_size=spec.block_size,
            head_size=spec.head_size,
        )
        for name, offset in (("layer.0", 0), ("layer.1", page_bytes))
    }

    zeroer = KVBlockZeroer(
        num_blocks=num_blocks,
        device=torch.device("cpu"),
        attn_groups_iter=[AttentionGroup(_BlockFirstBackend, list(views), spec, 0)],
        kernel_block_sizes=[spec.block_size],
        static_forward_context={
            name: SimpleNamespace(kv_cache=view) for name, view in views.items()
        },
    )

    assert zeroer._meta is not None
    seg_addrs, seg_strides, seg_pages, max_chunks, blk_size, n_segs = zeroer._meta
    assert n_segs == 2
    assert (seg_strides * 4 == block_stride_bytes).all()
    assert (seg_pages * 4 == page_bytes).all()
    assert max_chunks == 1
    assert blk_size == page_bytes // 4
    assert sorted(seg_addrs.tolist()) == sorted(
        view.data_ptr() for view in views.values()
    )


@pytest.mark.cpu_test
def test_overlaid_zeroer_metadata_keeps_widest_page_span():
    """Overlaid addresses are deduplicated without losing the wider page."""
    num_blocks = 2
    block_stride_bytes = 256
    raw = torch.ones(num_blocks * block_stride_bytes, dtype=torch.int8)
    small_spec = _full_spec(block_size=4, head_size=2)  # 64 bytes
    large_spec = _full_spec(block_size=4, head_size=4)  # 128 bytes
    views = {
        "small": _make_block_view(
            raw,
            num_blocks=num_blocks,
            offset_bytes=0,
            block_stride_bytes=block_stride_bytes,
            block_size=small_spec.block_size,
            head_size=small_spec.head_size,
        ),
        "large": _make_block_view(
            raw,
            num_blocks=num_blocks,
            offset_bytes=0,
            block_stride_bytes=block_stride_bytes,
            block_size=large_spec.block_size,
            head_size=large_spec.head_size,
        ),
    }

    zeroer = KVBlockZeroer(
        num_blocks=num_blocks,
        device=torch.device("cpu"),
        attn_groups_iter=[
            AttentionGroup(_BlockFirstBackend, ["small"], small_spec, 0),
            AttentionGroup(_BlockFirstBackend, ["large"], large_spec, 1),
        ],
        kernel_block_sizes=[small_spec.block_size, large_spec.block_size],
        static_forward_context={
            name: SimpleNamespace(kv_cache=view) for name, view in views.items()
        },
    )

    assert zeroer._meta is not None
    _, seg_strides, seg_pages, _, _, n_segs = zeroer._meta
    assert n_segs == 1
    assert int(seg_strides[0]) * 4 == block_stride_bytes
    assert int(seg_pages[0]) * 4 == large_spec.page_size_bytes


@pytest.mark.cpu_test
def test_virtual_block_split_zeroer_metadata_uses_logical_stride():
    """A scheduler block split into kernel pages keeps logical stepping."""
    num_scheduler_blocks = 2
    ratio = 2
    kernel_block_size = 4
    physical_page_bytes = 64
    physical_blocks = num_scheduler_blocks * ratio
    raw = torch.ones(physical_blocks * physical_page_bytes, dtype=torch.int8)
    spec = _full_spec(block_size=kernel_block_size * ratio, head_size=2)
    view = _make_block_view(
        raw,
        num_blocks=physical_blocks,
        offset_bytes=0,
        block_stride_bytes=physical_page_bytes,
        block_size=kernel_block_size,
        head_size=spec.head_size,
    )

    zeroer = KVBlockZeroer(
        num_blocks=num_scheduler_blocks,
        device=torch.device("cpu"),
        attn_groups_iter=[
            AttentionGroup(_BlockFirstBackend, ["layer"], spec, 0),
        ],
        kernel_block_sizes=[kernel_block_size],
        static_forward_context={"layer": SimpleNamespace(kv_cache=view)},
    )

    assert zeroer._meta is not None
    seg_addrs, seg_strides, seg_pages, _, _, n_segs = zeroer._meta
    assert n_segs == ratio
    assert (seg_strides * 4 == physical_page_bytes * ratio).all()
    assert (seg_pages * 4 == physical_page_bytes).all()
    assert seg_addrs.tolist()[1] - seg_addrs.tolist()[0] == physical_page_bytes


@pytest.mark.cpu_test
def test_side_storage_resets_use_only_owner_group_block_ids():
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = torch.device("cpu")
    zeroer._meta = None
    received: dict[int, list[int]] = {}
    zeroer._side_storage_zero_hooks = {
        0: [lambda ids: received.setdefault(0, ids.tolist())],
        2: [lambda ids: received.setdefault(2, ids.tolist())],
    }

    zeroer.zero_block_ids([], [[1, 3], [4], [2, 5]])

    assert received == {0: [1, 3], 2: [2, 5]}


@pytest.mark.cpu_test
def test_side_storage_warmup_deduplicates_aliases():
    received: list[list[int]] = []

    class SideStorage:
        main_layer_name = "owner"

        def bind_kv_cache_side_storage(self, _forward_context):
            raise AssertionError("not used")

        def zero_kv_cache_side_storage(self, block_ids):
            received.append(block_ids.tolist())

        def reset_kv_cache_side_storage_runtime_state(self):
            raise AssertionError("not used")

        def copy_kv_cache_side_storage(self, _block_copies, _num_blocks):
            raise AssertionError("not used")

    side_storage = SideStorage()
    spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=8,
    )
    group = SimpleNamespace(
        layer_names=["owner"],
        kv_cache_group_id=0,
        kv_cache_spec=spec,
    )
    zeroer = KVBlockZeroer(
        num_blocks=8,
        device=torch.device("cpu"),
        attn_groups_iter=[group],
        kernel_block_sizes=[2],
        static_forward_context={
            "side": side_storage,
            "side_alias": side_storage,
        },
        zero_base_kv_cache=False,
    )

    zeroer.warmup(num_kv_blocks=1)

    assert received == [[0]]


@pytest.mark.cpu_test
def test_side_storage_only_zeroer_does_not_build_base_kv_segments():
    received: list[list[int]] = []

    class Backend:
        @staticmethod
        def get_kv_cache_block_dim(*_args, **_kwargs):
            raise AssertionError("base KV metadata must not be built")

    class SideStorage:
        main_layer_name = "owner"

        def bind_kv_cache_side_storage(self, _forward_context):
            raise AssertionError("not used")

        def zero_kv_cache_side_storage(self, block_ids):
            received.append(block_ids.tolist())

        def reset_kv_cache_side_storage_runtime_state(self):
            raise AssertionError("not used")

        def copy_kv_cache_side_storage(self, _block_copies, _num_blocks):
            raise AssertionError("not used")

    group = SimpleNamespace(
        backend=Backend,
        layer_names=["owner"],
        kv_cache_group_id=0,
        kv_cache_spec=FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        ),
    )
    zeroer = KVBlockZeroer(
        num_blocks=8,
        device=torch.device("cpu"),
        attn_groups_iter=[group],
        kernel_block_sizes=[16],
        static_forward_context={"side": SideStorage()},
        zero_base_kv_cache=False,
    )

    assert zeroer._meta is None
    zeroer.zero_block_ids([7], [[7]])
    assert received == [[7]]


@pytest.mark.cpu_test
def test_side_storage_zeroing_contract_requires_an_owner_hook():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    ).with_side_storage(requires_zeroing=True)
    group = SimpleNamespace(
        backend=object(),
        layer_names=["owner"],
        kv_cache_group_id=0,
        kv_cache_spec=spec,
    )

    with pytest.raises(
        ValueError,
        match="declares side-storage zeroing but no KVCacheSideStorageLayer",
    ):
        KVBlockZeroer(
            num_blocks=8,
            device=torch.device("cpu"),
            attn_groups_iter=[group],
            kernel_block_sizes=[16],
            static_forward_context={},
            zero_base_kv_cache=False,
        )


@pytest.mark.cpu_test
def test_v1_runner_preserves_base_kv_zeroing_policy(monkeypatch):
    import vllm.v1.worker.gpu_model_runner as runner_module

    captured: dict[str, object] = {}

    class CapturingZeroer:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner._kernel_block_sizes = []
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.runner_only_attn_layers = set()
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(
        needs_base_kv_cache_zeroing=False, num_blocks=8
    )
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(())
    monkeypatch.setattr(runner_module, "KVBlockZeroer", CapturingZeroer)

    runner._init_kv_zero_meta()

    assert captured["zero_base_kv_cache"] is False


@pytest.mark.cpu_test
def test_v2_runner_preserves_base_kv_zeroing_policy(monkeypatch):
    import vllm.v1.worker.gpu.model_runner as runner_module

    captured: dict[str, object] = {}

    class CapturingZeroer:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.attn_groups = []
    runner.kernel_block_sizes = []
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(
        needs_base_kv_cache_zeroing=False, num_blocks=8
    )
    monkeypatch.setattr(runner_module, "KVBlockZeroer", CapturingZeroer)

    runner._init_kv_zero_meta()

    assert captured["zero_base_kv_cache"] is False
