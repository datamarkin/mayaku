"""TensorRT exporter — bonus deployment target on CUDA hosts.

Mirrors `BACKEND_PORTABILITY_REPORT.md` §6 ("If ONNX works,
``trtexec --onnx=...`` works for the backbone+heads body. Plugin only
needed for fused NMS — keep NMS out of TRT entirely."). Same Graph A
body export: backbone + FPN producing ``{p2..p6}``; the per-image
Python glue (RPN top-k + per-class NMS, mask paste, keypoint
sub-pixel decode) stays out of the engine entirely.

The path is **ONNX → TensorRT**:

1. Use the existing :class:`ONNXExporter` to write a temp ``.onnx``
   for the backbone+FPN body (already validated against eager — see
   `tests/unit/test_onnx_export.py`).
2. Parse that ONNX with TensorRT's :class:`tensorrt.OnnxParser` and
   build a :class:`tensorrt.IBuilderConfig` engine.
3. Serialise the engine to ``out_path`` (a ``.engine`` file).

The Python builder API is preferred over shelling out to ``trtexec``
because it's deterministic across PATH variations, gives us
structured error messages on parse failures, and lets us pin a
reproducible builder workspace size.

Parity check loads the engine via :class:`tensorrt.Runtime`, runs it
on CUDA via :mod:`tensorrt.cuda` allocations, and compares per-output
to eager. Default tolerance is loose (``atol=1e-2``) because TRT's
optimised kernel selection introduces small numerical drift on top of
fp32 cuBLAS / cuDNN — that's the deal you take for the
~3-10x speed-up.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from mayaku.inference.export.base import ExportResult, ParityResult
from mayaku.inference.export.onnx import ONNXExporter

__all__ = ["TensorRTBackbone", "TensorRTExporter"]

# Standard FPN output names (`Backbone.output_shape()` from Step 8).
_DEFAULT_OUT_NAMES: tuple[str, ...] = ("p2", "p3", "p4", "p5", "p6")

# Per-level FPN strides — kept in sync with `ONNXBackbone` (`onnx.py`).
_DEFAULT_STRIDES: dict[str, int] = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}

# 1 GiB workspace — enough for the ResNet-50 + FPN body without
# inflating the build memory footprint to multi-GB.
_DEFAULT_WORKSPACE_BYTES: int = 1 << 30


def _trt_install_hint() -> str:
    if sys.platform == "darwin":
        return "TensorRT is not supported on macOS."
    if not torch.cuda.is_available():
        return "TensorRT requires a CUDA-enabled GPU (Linux or Windows)."
    return "pip install mayaku[tensorrt]"


class TensorRTBackbone(nn.Module):
    """Runtime drop-in for ``model.backbone`` that delegates to a serialised
    TensorRT engine.

    Mirrors :class:`mayaku.inference.export.ONNXBackbone` but talks to the
    TRT runtime directly — no host↔device numpy round-trip, executes on
    the caller's current CUDA stream so downstream PyTorch ops natively
    pipeline behind it (no per-call ``synchronize()``).

    Args:
        engine_path: Path to a ``.engine`` file produced by
            :class:`TensorRTExporter`. Must have been built for the same
            GPU architecture (engines are not portable across SMs); use
            :func:`mayaku.utils.download.engine_cache_path` to locate /
            stash an engine keyed by ``(name, shape, precision, sm)``.
        pinned: ``(H, W)`` the engine was built at. Inputs smaller than
            this are zero-padded into the top-left corner; inputs
            larger raise.
        output_names: FPN feature names to bind. Defaults to
            ``("p2", "p3", "p4", "p5", "p6")``.
        strides: Per-level downsampling factor; used to crop the padded
            output back to the valid region. Defaults to the standard
            FPN strides (4, 8, 16, 32, 64).
        size_divisibility: Forwarded to detectron2-style preprocessing
            (the detector reads ``backbone.size_divisibility`` to decide
            how to pad each batch).
    """

    def __init__(
        self,
        engine_path: Path,
        *,
        pinned: tuple[int, int],
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        strides: dict[str, int] | None = None,
        size_divisibility: int = 32,
    ) -> None:
        super().__init__()
        try:
            import tensorrt as trt
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"TensorRTBackbone requires the tensorrt package. {_trt_install_hint()}"
            ) from e

        self.pinned: tuple[int, int] = (pinned[0], pinned[1])
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.strides: dict[str, int] = (
            dict(strides)
            if strides is not None
            else {n: _DEFAULT_STRIDES[n] for n in self.output_names}
        )
        self._size_divisibility = size_divisibility

        logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(logger)
        engine_path = Path(engine_path)
        with open(engine_path, "rb") as fh:
            self._engine = self._runtime.deserialize_cuda_engine(fh.read())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialise TRT engine at {engine_path}")
        self._context = self._engine.create_execution_context()
        self._context.set_input_shape("image", (1, 3, *self.pinned))

        # Lazy-allocated on first forward — we need to know the input
        # device, and we want a single allocation reused forever rather
        # than per-call ``torch.empty``.
        self._padded: Tensor | None = None
        self._bufs: dict[str, Tensor] = {}
        self._last_hw: tuple[int, int] | None = None

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def _allocate(self, device: torch.device) -> Tensor:
        target_h, target_w = self.pinned
        padded = torch.zeros(
            (1, 3, target_h, target_w), dtype=torch.float32, device=device
        )
        self._context.set_tensor_address("image", int(padded.data_ptr()))
        for name in self.output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=torch.float32, device=device)
            self._bufs[name] = buf
            self._context.set_tensor_address(name, int(buf.data_ptr()))
        self._padded = padded
        return padded

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"TensorRTBackbone.forward expects (1, 3, H, W); got {tuple(image.shape)}."
            )
        target_h, target_w = self.pinned
        _b, _c, h, w = image.shape
        if h > target_h or w > target_w:
            raise ValueError(
                f"Input shape ({h}, {w}) exceeds the engine's pinned shape "
                f"({target_h}, {target_w}). Re-build the engine at a larger size "
                "or constrain the input."
            )

        padded = self._padded if self._padded is not None else self._allocate(image.device)

        if (h, w) != self._last_hw:
            # Active region size changed — clear so the padding outside
            # the new (h, w) slice is zero rather than stale data from a
            # larger previous call.
            padded.zero_()
            self._last_hw = (h, w)
        padded[:, :, :h, :w].copy_(image)

        stream = torch.cuda.current_stream(device=image.device).cuda_stream
        if not self._context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TRT execute_async_v3 returned False")

        out: dict[str, Tensor] = {}
        for name in self.output_names:
            buf = self._bufs[name]
            stride = self.strides[name]
            crop_h = (h + stride - 1) // stride
            crop_w = (w + stride - 1) // stride
            if crop_h == buf.shape[-2] and crop_w == buf.shape[-1]:
                out[name] = buf
            else:
                out[name] = buf[:, :, :crop_h, :crop_w].contiguous()
        return out


class TensorRTExporter:
    """Export the backbone+FPN body to a TensorRT serialised engine.

    Args:
        output_names: FPN feature output names. Defaults to
            ``("p2", "p3", "p4", "p5", "p6")``.
        fp16: If ``True`` enable FP16 mode — ~2x throughput at small
            accuracy cost on most GPUs. Default ``False`` so parity
            against eager fp32 stays tight.
        workspace_bytes: TRT builder workspace cap. Default 1 GiB.
        opset: ONNX opset for the intermediate ``.onnx``. Default 17
            (matches :class:`ONNXExporter`).
    """

    name: str = "tensorrt"

    def __init__(
        self,
        *,
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        fp16: bool = False,
        workspace_bytes: int = _DEFAULT_WORKSPACE_BYTES,
        opset: int = 17,
    ) -> None:
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.fp16 = fp16
        self.workspace_bytes = workspace_bytes
        self.opset = opset

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        model: nn.Module,
        sample: Tensor,
        out_path: Path,
        **opts: object,
    ) -> ExportResult:
        """Build a serialised TRT engine for ``model.backbone``.

        Implementation: ONNXExporter → ``tempfile`` → TRT
        :class:`OnnxParser` + :class:`Builder` → ``out_path``.
        """
        del opts
        # Lazy import — tensorrt is CUDA-only and lives behind the
        # `[tensorrt]` extra.
        try:
            import tensorrt as trt
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(_trt_install_hint()) from e

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "body.onnx"
            # Trace on the model's own device — passing a CPU sample to a
            # CUDA model trips "weight type vs input type" at the first
            # conv. Tests deliberately put the model on CUDA so the
            # engine is built on the host that will run it.
            trace_device = next(iter(model.parameters())).device
            ONNXExporter(opset=self.opset, output_names=self.output_names).export(
                model, sample.to(trace_device), onnx_path
            )

            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, "rb") as fh:
                if not parser.parse(fh.read()):
                    msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                    raise RuntimeError(f"TensorRT ONNX parser failed:\n{msgs}")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_bytes)
            if self.fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                # TF32 is on by default in TRT 10.x and produces ~0.1-0.3
                # abs drift vs cuDNN fp32 on Ampere+ GPUs, breaking the
                # strict parity check (analogous to OpenVINO's silent
                # fp16 default — see openvino.py for the equivalent
                # `INFERENCE_PRECISION_HINT=f32` pin). Clear it on the
                # default fp32 path so the parity test pins real fp32
                # numerics; the fp16 path inherently expects drift and
                # users opt into it explicitly.
                config.clear_flag(trt.BuilderFlag.TF32)

            # ONNXExporter declares dynamic batch + spatial axes; TRT
            # 10.x refuses to build a dynamic-shape network without an
            # IOptimizationProfile. Pin a single point at sample.shape —
            # callers that want a real shape envelope can re-export
            # against a TensorRTExporter that takes min/opt/max args (a
            # natural follow-up once the basic CUDA path is green).
            profile = builder.create_optimization_profile()
            shape = tuple(sample.shape)
            profile.set_shape("image", min=shape, opt=shape, max=shape)
            config.add_optimization_profile(profile)

            serialised = builder.build_serialized_network(network, config)
            if serialised is None:
                raise RuntimeError(
                    "TensorRT builder.build_serialized_network returned None — "
                    "see TRT logger output for the failing layer"
                )
            with open(out_path, "wb") as fh:
                fh.write(bytes(serialised))

        return ExportResult(
            path=out_path,
            target=self.name,
            opset=self.opset,
            input_names=("image",),
            output_names=self.output_names,
            extras={
                "fp16": str(self.fp16),
                "workspace_bytes": str(self.workspace_bytes),
                "trt_version": getattr(trt, "__version__", "unknown"),
            },
        )

    # ------------------------------------------------------------------
    # Parity
    # ------------------------------------------------------------------

    def parity_check(
        self,
        model: nn.Module,
        exported_path: Path,
        sample: Tensor,
        *,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ) -> ParityResult:
        """Run eager + TRT engine on ``sample`` and compare per-output.

        Both halves run on CUDA — eager uses whatever device ``model``
        lives on (must be ``cuda:*`` for a meaningful comparison); TRT
        uses ``cuda:0`` by default. Default tolerance is loose
        (``atol=1e-2``) to absorb TRT's kernel-selection drift; pass
        a tighter value if you've validated a specific build.
        """
        try:
            import tensorrt as trt
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(_trt_install_hint()) from e

        backbone = _resolve_backbone(model)
        backbone.eval()
        device = next(iter(backbone.parameters())).device
        if device.type != "cuda":
            raise RuntimeError(f"TensorRT parity check requires a CUDA model; got device={device}")

        # Force strict fp32 on the eager half to match the TRT engine
        # built with `BuilderFlag.TF32` cleared. PyTorch enables TF32
        # for cuDNN convs and matmuls by default on Ampere+ GPUs, so
        # without this the eager forward uses TF32 and TRT uses fp32 —
        # the two disagree by ~0.25 abs even though both are nominally
        # "fp32 paths". Restore the user's settings on the way out so
        # the parity check is a no-op on global state.
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.no_grad():
                eager_out = backbone(sample.to(device))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(exported_path, "rb") as fh:
            engine = runtime.deserialize_cuda_engine(fh.read())
        if engine is None:
            raise RuntimeError(f"failed to deserialise TRT engine at {exported_path}")

        context = engine.create_execution_context()
        # Set the input shape (TRT 10.x wants this even for static
        # shapes, since networks are EXPLICIT_BATCH by default).
        context.set_input_shape("image", tuple(sample.shape))

        # Allocate device buffers per binding via torch (saves us
        # rolling our own pycuda / cuda-python).
        bindings: list[int] = []
        outputs_t: dict[str, Tensor] = {}
        # Inputs first.
        sample_cuda = sample.to(device).contiguous()
        bindings.append(int(sample_cuda.data_ptr()))
        # Outputs.
        for name in self.output_names:
            shape = tuple(context.get_tensor_shape(name))
            buf = torch.empty(shape, dtype=torch.float32, device=device)
            outputs_t[name] = buf
            bindings.append(int(buf.data_ptr()))

        # Bind tensor addresses (TRT 10.x API).
        context.set_tensor_address("image", int(sample_cuda.data_ptr()))
        for name, buf in outputs_t.items():
            context.set_tensor_address(name, int(buf.data_ptr()))

        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        with torch.cuda.stream(stream):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        stream.synchronize()

        per_output: dict[str, tuple[float, float]] = {}
        max_abs = 0.0
        max_rel = 0.0
        for name in self.output_names:
            trt_t = outputs_t[name].to("cpu")
            eager_t = eager_out[name].to("cpu")
            abs_err = float((trt_t - eager_t).abs().max().item())
            rel_err = float(((trt_t - eager_t).abs() / eager_t.abs().clamp(min=1e-8)).max().item())
            per_output[name] = (abs_err, rel_err)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)

        passed = max_abs <= atol or max_rel <= rtol
        return ParityResult(
            target=self.name,
            passed=passed,
            max_abs_error=max_abs,
            max_rel_error=max_rel,
            atol=atol,
            rtol=rtol,
            per_output=per_output,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        return model.backbone
    return model
