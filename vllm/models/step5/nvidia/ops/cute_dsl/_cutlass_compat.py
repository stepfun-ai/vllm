"""Process-local compatibility patches for the pinned CUTLASS/CuTe runtime.

The `nvidia-cutlass-dsl` version shipped with vLLM is missing
implementations of ``mark_layout_dynamic`` and ``mark_compact_shape_dynamic``
for ``cutlass.cute.runtime._FakeTensor``.  The TVM-FFI compilation path uses
``make_fake_tensor`` to describe kernel arguments, so the missing methods
cause failures during compilation. These patches only modify live Python
objects; they never rewrite the installed CUTLASS package.
"""

from __future__ import annotations

import inspect
import re
import threading
from importlib.metadata import PackageNotFoundError, version
from types import UnionType
from typing import Any, Callable, Iterable, Sequence, Tuple, get_args, get_origin

_EXPECTED_CUTLASS_DSL_VERSION = "4.7.1"
_REQUIRED_TORCH_DTYPES = (
    "uint16",
    "uint32",
    "uint64",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
)


def verify_torch_dtype_compat() -> bool:
    """Verify the pinned torch build exposes CUTLASS-required dtypes."""
    try:
        import torch
    except ModuleNotFoundError:
        return False

    missing = [name for name in _REQUIRED_TORCH_DTYPES if not hasattr(torch, name)]
    if missing:
        raise RuntimeError(
            "The installed torch build is incompatible with the Step5 CUTLASS "
            f"bindings; missing dtypes: {', '.join(missing)}."
        )
    return True


def verify_cutlass_runtime_compat() -> bool:
    """Fail fast unless the installed CUTLASS runtime has required upstream fixes."""
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    if installed_version != _EXPECTED_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            "Step5 requires nvidia-cutlass-dsl "
            f"{_EXPECTED_CUTLASS_DSL_VERSION}, got {installed_version}."
        )

    try:
        from cutlass.base_dsl.dsl import BaseDSL  # type: ignore
        from cutlass.cute.runtime import _Tensor, from_dlpack  # type: ignore
    except ModuleNotFoundError:
        return False

    missing: list[str] = []
    if "enable_tvm_ffi" not in inspect.signature(from_dlpack).parameters:
        missing.append("from_dlpack(enable_tvm_ffi=...)")
    try:
        tensor_init_source = inspect.getsource(_Tensor.__init__)
    except (OSError, TypeError):
        tensor_init_source = ""
    if "elif enable_tvm_ffi" not in tensor_init_source:
        missing.append("_Tensor TVM-FFI branch ordering")
    try:
        generate_mlir_source = inspect.getsource(BaseDSL.generate_mlir)
    except (OSError, TypeError):
        generate_mlir_source = ""
    if "enable_multithreading(False)" not in generate_mlir_source:
        missing.append("BaseDSL.generate_mlir threading guard")
    if missing:
        raise RuntimeError(
            "The installed nvidia-cutlass-dsl runtime is missing required "
            f"compatibility fixes: {', '.join(missing)}."
        )
    return True


def patch_cute_arch_proxy_enums() -> bool:
    """Backfill arch fence enum shims removed from newer CUTLASS Python APIs."""
    try:
        import cutlass.cute.arch as arch  # type: ignore
    except ModuleNotFoundError:
        return False

    if not hasattr(arch, "ProxyKind"):
        arch.ProxyKind = type(  # type: ignore[attr-defined]
            "ProxyKind",
            (),
            {
                "alias": "alias",
                "async_global": "async.global",
                "async_shared": "async.shared",
                "tensormap": "tensormap",
                "generic": "generic",
            },
        )
    if not hasattr(arch, "SharedSpace"):
        arch.SharedSpace = type(  # type: ignore[attr-defined]
            "SharedSpace",
            (),
            {
                "shared_cta": "cta",
                "shared_cluster": "cluster",
            },
        )
    return True


def patch_cute_core_type_aliases() -> bool:
    """Backfill CuTe type aliases moved out of cute.core in newer CUTLASS DSL."""
    try:
        import cutlass.cute as cute  # type: ignore
    except ModuleNotFoundError:
        return False

    core = getattr(cute, "core", None)
    if core is None:
        return True
    for name in ("ThrMma",):
        if not hasattr(core, name) and hasattr(cute, name):
            setattr(core, name, getattr(cute, name))
    if not hasattr(cute, "make_fragment") and hasattr(cute, "make_rmem_tensor"):
        cute.make_fragment = cute.make_rmem_tensor  # type: ignore[attr-defined]
    return True


def _deduce_stride_order(stride: Sequence[int]) -> Tuple[int, ...]:
    """Reproduce torch.Tensor.dim_order() for compact layouts."""
    indexed = list(enumerate(stride))
    # Sort by stride magnitude from outermost (largest stride) to innermost.
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    return tuple(idx for idx, _ in indexed)


def _infer_stride_order(tensor: Any) -> Tuple[int, ...] | None:
    try:
        return tuple(tensor.dim_order())
    except Exception:
        pass

    try:
        return _deduce_stride_order(tuple(tensor.stride()))
    except Exception:
        return None


def _needs_patch(fake_tensor_cls) -> bool:
    try:
        from cutlass.cute.typing import Tensor as _CuteTensor  # type: ignore
    except ModuleNotFoundError:
        return False
    base_layout = getattr(_CuteTensor, "mark_layout_dynamic", None)
    base_compact = getattr(_CuteTensor, "mark_compact_shape_dynamic", None)
    return (
        getattr(fake_tensor_cls, "mark_layout_dynamic", None) is base_layout
        or getattr(fake_tensor_cls, "mark_compact_shape_dynamic", None) is base_compact
    )


def patch_fake_tensor_dynamic_methods() -> bool:
    """Install ``mark_layout_dynamic``/``mark_compact_shape_dynamic`` on _FakeTensor.

    Returns:
        True if the patch is applied or determined to be unnecessary (no need to
        retry in this process). False if the patch couldn't be evaluated because
        CUTLASS/CuTe isn't importable (callers may retry later).
    """
    try:
        from cutlass.cute.runtime import _FakeTensor  # type: ignore
        from cutlass.cute.typing import SymInt, sym_int64
    except ModuleNotFoundError:
        return False

    def _validate_dynamic_methods() -> None:
        from cutlass import Float32  # type: ignore

        probe = _FakeTensor(Float32, (8, 8), stride=(8, 1))
        if probe.mark_layout_dynamic(leading_dim=1) is not probe:
            raise RuntimeError(
                "CUTLASS _FakeTensor.mark_layout_dynamic must return self."
            )
        if probe.mark_compact_shape_dynamic(
            mode=0,
            stride_order=(0, 1),
            divisibility=1,
        ) is not probe:
            raise RuntimeError(
                "CUTLASS _FakeTensor.mark_compact_shape_dynamic must return self."
            )
        if not isinstance(probe.shape[0], SymInt) or not isinstance(
            probe.stride[0], SymInt
        ):
            raise RuntimeError(
                "CUTLASS _FakeTensor dynamic-layout methods did not produce "
                "symbolic shape and stride metadata."
            )

    if getattr(_FakeTensor, "_optimus_dynamic_patch", False):
        _validate_dynamic_methods()
        return True
    if not _needs_patch(_FakeTensor):
        _validate_dynamic_methods()
        return True

    # ``cutlass.cute.typing.SymInt`` is only used for ``isinstance`` checks.
    SymIntTuple = (SymInt,)

    def _coerce_sym_int(value, divisibility: int):
        if isinstance(value, SymIntTuple):
            return value
        return sym_int64(divisibility=divisibility)

    def _get_layout(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        typed_tensor = getattr(self, "_typed_tensor", None)
        if typed_tensor is not None:
            return tuple(typed_tensor.shape), tuple(typed_tensor.stride)
        return tuple(self._shape), tuple(self._stride)

    def _set_layout(
        self,
        shape: Sequence[Any],
        stride: Sequence[Any],
    ) -> None:
        typed_tensor = getattr(self, "_typed_tensor", None)
        if typed_tensor is not None:
            typed_tensor._shape = tuple(shape)
            typed_tensor._stride = tuple(stride)
            return
        self._shape = tuple(shape)
        self._stride = tuple(stride)

    def mark_layout_dynamic(self, leading_dim: int | None = None):
        shape, original_stride = _get_layout(self)
        stride = list(original_stride)
        if leading_dim is None:
            stride_ones = [
                idx for idx, value in enumerate(original_stride) if value == 1
            ]
            if len(stride_ones) == 1:
                leading_dim = stride_ones[0]
            elif len(stride_ones) > 1:
                nondegenerate = [
                    idx
                    for idx in stride_ones
                    if isinstance(shape[idx], int) and shape[idx] > 1
                ]
                if len(nondegenerate) != 1:
                    raise ValueError(
                        "Unable to deduce leading_dim for a layout with "
                        "multiple stride-1 dimensions."
                    )
                leading_dim = nondegenerate[0]
        if leading_dim is not None:
            if not (0 <= leading_dim < len(original_stride)):
                raise ValueError(f"leading_dim {leading_dim} out of range.")
            if original_stride[leading_dim] != 1:
                raise ValueError(
                    f"Leading dimension {leading_dim} must have stride 1, "
                    f"got {original_stride[leading_dim]}."
                )
        for dim, value in enumerate(original_stride):
            if dim != leading_dim:
                stride[dim] = _coerce_sym_int(value, 1)
        _set_layout(self, shape, stride)
        self._optimus_leading_dim = leading_dim
        try:
            self._optimus_stride_order = _deduce_stride_order(original_stride)
        except (TypeError, ValueError):
            self._optimus_stride_order = None
        return self

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: Iterable[int] | None = None,
        divisibility: int = 1,
    ):
        shape, current_stride = _get_layout(self)
        rank = len(shape)
        if not (0 <= mode < rank):
            raise ValueError(f"mode {mode} out of range for shape rank {rank}.")
        if divisibility <= 0:
            raise ValueError(f"divisibility must be positive, got {divisibility}.")
        if stride_order is None:
            stride_order = getattr(self, "_optimus_stride_order", None)
            if stride_order is None:
                stride_order = _deduce_stride_order(current_stride)
        else:
            stride_order = tuple(stride_order)
        if len(stride_order) != rank or set(stride_order) != set(range(rank)):
            raise ValueError(
                f"stride_order must be a permutation of range({rank}), "
                f"got {stride_order}."
            )

        shape = list(shape)
        shape[mode] = _coerce_sym_int(shape[mode], divisibility)

        dynamic_prefix = stride_order[: stride_order.index(mode)]
        stride = list(current_stride)
        for dim in dynamic_prefix:
            stride[dim] = _coerce_sym_int(stride[dim], divisibility)
        _set_layout(self, shape, stride)
        self._optimus_stride_order = stride_order
        return self

    original_layout = _FakeTensor.mark_layout_dynamic
    original_compact = _FakeTensor.mark_compact_shape_dynamic
    _FakeTensor.mark_layout_dynamic = mark_layout_dynamic  # type: ignore[attr-defined]
    _FakeTensor.mark_compact_shape_dynamic = mark_compact_shape_dynamic  # type: ignore[attr-defined]
    try:
        _validate_dynamic_methods()
    except Exception:
        _FakeTensor.mark_layout_dynamic = original_layout
        _FakeTensor.mark_compact_shape_dynamic = original_compact
        raise
    _FakeTensor._optimus_dynamic_patch = True  # type: ignore[attr-defined]
    return True


def _is_cute_tensor_annotation(annotation: Any) -> bool:
    try:
        from cutlass.cute.typing import Tensor as CuteTensor  # type: ignore
    except ModuleNotFoundError:
        return False

    origin = get_origin(annotation)
    if origin is not None:
        return any(_is_cute_tensor_annotation(arg) for arg in get_args(annotation))

    if isinstance(annotation, UnionType):
        return any(_is_cute_tensor_annotation(arg) for arg in get_args(annotation))

    return isinstance(annotation, type) and issubclass(annotation, CuteTensor)


def patch_cutlass_tensor_arg_validation() -> bool:
    """Allow direct torch.Tensor arguments for cute.Tensor annotations."""
    try:
        import torch
        from cutlass.cutlass_dsl.cutlass import CutlassBaseDSL  # type: ignore
    except ModuleNotFoundError:
        return False

    if getattr(CutlassBaseDSL, "_optimus_tensor_arg_patch", False):
        return True

    original_validate_arg = CutlassBaseDSL._validate_arg

    def _validate_arg_with_torch_tensor(
        self,
        arg: Any,
        arg_index: int,
        arg_name: str,
        arg_annotation: Any,
    ):
        if arg is not None and _is_cute_tensor_annotation(arg_annotation):
            if isinstance(arg, torch.Tensor) or hasattr(arg, "__dlpack__"):
                return None
        return original_validate_arg(self, arg, arg_index, arg_name, arg_annotation)

    CutlassBaseDSL._validate_arg = _validate_arg_with_torch_tensor  # type: ignore[assignment]
    CutlassBaseDSL._optimus_tensor_arg_patch = True
    return True


def _infer_assumed_align_bytes(tensor: Any, max_align: int = 16) -> int:
    """Use real pointer alignment for conservative TVM-FFI tensor assumptions."""
    try:
        ptr = int(tensor.data_ptr())
    except Exception:
        return 1

    align = 1
    while align < max_align and (ptr % (align * 2) == 0):
        align *= 2
    try:
        element_size = max(1, int(tensor.element_size()))
    except Exception:
        element_size = 1
    return max(align, element_size)


def _make_runtime_tensor_converter() -> Callable[[Any], Any]:
    import torch
    from cutlass import Float8E4M3FN
    from cutlass.cute.runtime import from_dlpack as cute_from_dlpack

    float8_element_map = {torch.float8_e4m3fn: Float8E4M3FN}
    if hasattr(torch, "float8_e5m2"):
        from cutlass import Float8E5M2

        float8_element_map[torch.float8_e5m2] = Float8E5M2

    def _convert(tensor: Any):
        logical_dtype = getattr(tensor, "dtype", None)
        element_override = float8_element_map.get(logical_dtype)
        dlpack_source = tensor.view(torch.uint8) if element_override is not None else tensor
        cute_tensor = cute_from_dlpack(
            dlpack_source,
            assumed_align=_infer_assumed_align_bytes(tensor),
            enable_tvm_ffi=True,
        )

        if element_override is not None:
            cute_tensor.element_type = element_override

        strides = tuple(tensor.stride())
        stride_ones = [idx for idx, value in enumerate(strides) if value == 1]
        leading_dim = stride_ones[0] if len(stride_ones) == 1 else None

        if leading_dim is None:
            return cute_tensor.mark_layout_dynamic()

        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        tensor_rank = getattr(tensor, "ndim", None)
        # Keep <=2D runtime tensors aligned with the
        # dynamic-dim0 ABI used by qknorm/rmsnorm so execution no longer falls
        # back to fully dynamic layouts like (?,?):(?,1).
        if tensor_rank is None or tensor_rank == 0 or tensor_rank > 2:
            return cute_tensor

        stride_order = _infer_stride_order(tensor)
        try:
            if stride_order is None:
                return cute_tensor.mark_compact_shape_dynamic(mode=0, divisibility=1)
            return cute_tensor.mark_compact_shape_dynamic(
                mode=0,
                stride_order=stride_order,
                divisibility=1,
            )
        except RuntimeError as err:
            if "stride_order" not in str(err) and "could not be deduced" not in str(err):
                raise
            return cute_tensor.mark_compact_shape_dynamic(mode=0, divisibility=1)

    return _convert


_RUNTIME_TENSOR_CONVERTER: Callable[[Any], Any] | None = None


def _get_runtime_tensor_converter() -> Callable[[Any], Any]:
    global _RUNTIME_TENSOR_CONVERTER
    if _RUNTIME_TENSOR_CONVERTER is None:
        _RUNTIME_TENSOR_CONVERTER = _make_runtime_tensor_converter()
    return _RUNTIME_TENSOR_CONVERTER


def _to_cute_tensor_for_optimus_runtime(tensor: Any):
    return _get_runtime_tensor_converter()(tensor)


def patch_cutlass_tensor_adapter() -> bool:
    """Route torch.Tensor through the version-aware tensor conversion."""
    try:
        import torch
        from cutlass.cute.runtime import TensorAdapter
    except ModuleNotFoundError:
        return False

    if getattr(TensorAdapter, "_optimus_tensor_adapter_patch", False):
        return True

    original_init = TensorAdapter.__init__
    convert_tensor = _get_runtime_tensor_converter()

    def _tensor_adapter_init_with_tvm_ffi(self, arg):
        self._c_pointers_cache = None
        self._mlir_types_cache = None
        if isinstance(arg, torch.Tensor):
            self._arg = convert_tensor(arg)
            return
        original_init(self, arg)

    def _tensor_adapter_c_pointers(self):
        if self._c_pointers_cache is None:
            self._c_pointers_cache = self._arg.__c_pointers__()
        return self._c_pointers_cache

    def _tensor_adapter_get_mlir_types(self):
        if self._mlir_types_cache is None:
            self._mlir_types_cache = self._arg.__get_mlir_types__()
        return self._mlir_types_cache

    TensorAdapter.__init__ = _tensor_adapter_init_with_tvm_ffi
    TensorAdapter.__c_pointers__ = _tensor_adapter_c_pointers
    TensorAdapter.__get_mlir_types__ = _tensor_adapter_get_mlir_types
    TensorAdapter._optimus_tensor_adapter_patch = True
    return True


def patch_tvm_ffi_args_spec_converter() -> bool:
    """Handle tensor and compile-time aggregate arguments in TVM FFI."""
    try:
        import torch
        from cutlass.cute import _tvm_ffi_args_spec_converter as args_converter  # type: ignore
    except ModuleNotFoundError:
        return False

    if getattr(args_converter, "_optimus_tvm_ffi_args_converter_patch", False):
        return True

    original_convert_single_arg = args_converter._convert_single_arg
    convert_tensor = _get_runtime_tensor_converter()

    def _convert_single_arg_for_optimus(
        arg,
        arg_name: str,
        arg_type,
        ctx,
        *,
        is_constexpr: bool = False,
    ):
        if isinstance(arg, torch.Tensor):
            arg = convert_tensor(arg)
        if (
            isinstance(arg, tuple)
            and hasattr(type(arg), "_fields")
            and (arg_type is None or not hasattr(arg_type, "_fields"))
        ):
            arg_type = type(arg)
        return original_convert_single_arg(
            arg,
            arg_name,
            arg_type,
            ctx,
            is_constexpr=is_constexpr,
        )

    args_converter._convert_single_arg = _convert_single_arg_for_optimus
    args_converter._optimus_tvm_ffi_args_converter_patch = True
    return True


def _sanitize_mlir_symbol(symbol: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_$.]", "_", symbol)
    if not sanitized:
        return "_optimus_symbol"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def patch_tvm_ffi_provider_symbol_name() -> bool:
    """Sanitize generated LLVM global symbols for TVM-FFI call provider."""
    try:
        from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFICuteCallProvider
    except ModuleNotFoundError:
        return False

    if getattr(TVMFFICuteCallProvider, "_optimus_symbol_name_patch", False):
        return True

    original_init = TVMFFICuteCallProvider.__init__

    def _init_with_sanitized_symbol(self, target_func: str, *args, **kwargs):
        original_init(self, target_func, *args, **kwargs)
        self.cuda_global_state_symbol = _sanitize_mlir_symbol(self.cuda_global_state_symbol)

    TVMFFICuteCallProvider.__init__ = _init_with_sanitized_symbol  # type: ignore[assignment]
    TVMFFICuteCallProvider._optimus_symbol_name_patch = True
    return True


_OPTIMUS_PATCH_LOCK = threading.RLock()
_OPTIMUS_PATCHES_APPLIED = False
_OPTIMUS_PATCH_INITIALIZING = False
_OPTIMUS_PATCH_ERROR: Exception | None = None


def _require_patch(name: str, patcher: Callable[[], bool]) -> None:
    if not patcher():
        raise RuntimeError(
            f"Required Step5 CUTLASS compatibility step {name!r} could not "
            "be applied because its runtime dependency is unavailable."
        )


def apply_patches() -> None:
    """Apply all required Step5 compatibility patches once per process."""
    global _OPTIMUS_PATCH_ERROR, _OPTIMUS_PATCH_INITIALIZING
    global _OPTIMUS_PATCHES_APPLIED
    if _OPTIMUS_PATCHES_APPLIED:
        return

    with _OPTIMUS_PATCH_LOCK:
        if _OPTIMUS_PATCHES_APPLIED:
            return
        if _OPTIMUS_PATCH_ERROR is not None:
            raise RuntimeError(
                "Step5 CUTLASS compatibility initialization failed earlier "
                "in this process."
            ) from _OPTIMUS_PATCH_ERROR
        if _OPTIMUS_PATCH_INITIALIZING:
            raise RuntimeError(
                "Step5 CUTLASS compatibility initialization was re-entered "
                "in the same thread."
            )

        _OPTIMUS_PATCH_INITIALIZING = True
        try:
            required_steps = (
                ("torch dtype verification", verify_torch_dtype_compat),
                ("CUTLASS runtime verification", verify_cutlass_runtime_compat),
                ("CuTe arch enum aliases", patch_cute_arch_proxy_enums),
                ("CuTe core type aliases", patch_cute_core_type_aliases),
                ("fake tensor dynamic layout", patch_fake_tensor_dynamic_methods),
                ("tensor argument validation", patch_cutlass_tensor_arg_validation),
                ("tensor runtime adapter", patch_cutlass_tensor_adapter),
                ("TVM-FFI argument converter", patch_tvm_ffi_args_spec_converter),
                ("TVM-FFI symbol sanitization", patch_tvm_ffi_provider_symbol_name),
            )
            for name, patcher in required_steps:
                _require_patch(name, patcher)

        except Exception as error:
            _OPTIMUS_PATCH_ERROR = error
            raise RuntimeError(
                "Failed to initialize Step5 compatibility for "
                f"nvidia-cutlass-dsl {_EXPECTED_CUTLASS_DSL_VERSION}."
            ) from error
        finally:
            _OPTIMUS_PATCH_INITIALIZING = False

        _OPTIMUS_PATCHES_APPLIED = True


# NOTE:
# Do NOT apply patches at import time.
#
# This package is often imported in multi-process runtimes (e.g. vLLM). Import-time
# side effects that transitively import/initialize CUDA toolchains can break forked
# workers (CUDA init must typically happen after process creation).
#
# Call ``apply_patches()`` explicitly from the model-scoped lazy facades.
