"""ONNX exporter for the in-scope detectors.

Mirrors `BACKEND_PORTABILITY_REPORT.md` §5: target opset 17, the
**Graph A** body (``backbone`` + FPN producing ``{p2..p6}``) is
exported; the per-image Python glue (RPN top-k + per-class NMS,
mask paste, keypoint sub-pixel decode) stays out of the exported
graph by design (BPR §5/§6: keeping NMS in Python is what makes the
ONNX export competitive vs Ultralytics' fused-NMS approach).

Why split the model: a full Faster/Mask/Keypoint R-CNN forward
includes the :class:`Instances` container (Python dict-of-fields),
per-image loops, and `batched_nms` — none of which trace cleanly into
ONNX. Exporting just the backbone+FPN body gives downstream consumers
a deterministic, dynamic-shape-friendly feature extractor that pairs
with the Python postprocess we already have (Step 16).

Inputs/outputs of the exported graph:

* Input: ``"image"`` shaped ``(N, 3, H, W)`` float32 RGB *normalised*
  (mean already subtracted, std already divided). The
  :class:`FasterRCNN._preprocess_image` step is *not* part of the
  graph because pixel mean/std are model parameters and bake in
  cleanly via constant folding; we still pass the model's
  ``pixel_mean`` / ``pixel_std`` so the user can apply the same
  normalisation outside the graph if they prefer.
* Outputs: ``"p2"`` … ``"p6"``, each shaped ``(N, C, H/stride,
  W/stride)`` with ``C = cfg.model.fpn.out_channels`` (256 by
  default). Strides are 4, 8, 16, 32, 64 respectively.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from mayaku.inference.export.base import ExportResult, ParityResult

__all__ = ["ONNXBackbone", "ONNXExporter"]

DEFAULT_OPSET: int = 17

# The standard FPN output names (`Backbone.output_shape()` from Step 8).
_DEFAULT_OUT_NAMES: tuple[str, ...] = ("p2", "p3", "p4", "p5", "p6")
# Strides associated with the default FPN levels above. p6 is
# `LastLevelMaxPool(stride=2)` of p5 → effective stride 64.
_DEFAULT_STRIDES: dict[str, int] = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}


class ONNXExporter:
    """Export the backbone+FPN body to an ONNX graph + parity-check it.

    Args:
        opset: ONNX opset version. Default 17 (`spec §5`: gives RoiAlign
            with ``coordinate_transformation_mode="half_pixel"`` and
            modern ``Resize`` semantics).
        output_names: Names of the FPN feature outputs. Defaults to
            ``("p2", "p3", "p4", "p5", "p6")``.
        dynamic_input_shape: When ``True`` (default), the exported
            graph supports any ``(N, 3, H, W)`` input where ``H`` and
            ``W`` are multiples of ``backbone.size_divisibility`` —
            one ``.onnx`` file deploys to images of any size. When
            ``False``, the graph is exported at the literal shape of
            ``sample`` and any other input shape is rejected at
            session.run() time.

            This trade-off matters for **TensorRT throughput**: TRT's
            engine builder produces dramatically faster kernels when
            input shapes are static (it can pre-compile per-layer
            algorithms for the exact shape). With dynamic shapes,
            TRT either falls back to a one-size-fits-all kernel set
            or rebuilds at runtime — both options are slower than
            PyTorch eager for R-CNN-class graphs. Mayaku's runtime
            hybrid path (``ONNXBackbone``) already pads each input
            up to a fixed shape, so the dynamic-shapes flexibility
            isn't actually needed for the eval pipeline; export with
            ``dynamic_input_shape=False`` when targeting TRT.
    """

    name: str = "onnx"

    def __init__(
        self,
        *,
        opset: int = DEFAULT_OPSET,
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        dynamic_input_shape: bool = True,
    ) -> None:
        self.opset = opset
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.dynamic_input_shape = dynamic_input_shape

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
        """Serialise ``model`` (must expose ``.backbone``) to ``out_path``.

        ``sample`` is a single normalised RGB batch ``(N, 3, H, W)``
        used as the tracing input. The exporter wraps ``model.backbone``
        in a small adapter that returns a *tuple* of feature tensors in
        the order configured by ``output_names`` so torch's ONNX exporter
        gets a deterministic output ordering (the model itself returns a
        ``dict[str, Tensor]``, which traces less cleanly).
        """
        del opts  # Reserved for future per-target options.
        backbone = _resolve_backbone(model)
        adapter = _DictToTupleBackbone(backbone, self.output_names).eval()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # When `dynamic_input_shape=True`: declare batch + spatial
        # axes as dynamic so the graph supports any input size that's
        # a multiple of `backbone.size_divisibility`. When `False`:
        # bake the literal `sample.shape` into the graph for TRT
        # throughput (see class docstring).
        dynamic_axes: dict[str, dict[int, str]] | None
        if self.dynamic_input_shape:
            dynamic_axes = {"image": {0: "batch", 2: "height", 3: "width"}}
            for name in self.output_names:
                dynamic_axes[name] = {0: "batch", 2: "height", 3: "width"}
        else:
            dynamic_axes = None

        with torch.no_grad():
            torch.onnx.export(
                adapter,
                (sample,),
                str(out_path),
                input_names=["image"],
                output_names=list(self.output_names),
                opset_version=self.opset,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                # Pin the legacy TorchScript-based exporter so the
                # only ONNX-side dependency is `onnx`/`onnxruntime`.
                # The new dynamo path drags `onnxscript` into the
                # required deps, which we'd rather defer.
                dynamo=False,
            )

        return ExportResult(
            path=out_path,
            target=self.name,
            opset=self.opset,
            input_names=("image",),
            output_names=self.output_names,
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
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> ParityResult:
        """Run eager + ONNX Runtime on ``sample`` and compare per-output.

        The ONNX Runtime session uses the CPU execution provider so the
        check is reproducible on every backend (the eager forward
        happens on whatever device ``model`` lives on; we move the
        sample to that device for the eager half and back to CPU for
        the comparison)."""
        # Lazy import: onnxruntime is an optional extra.
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as e:
            _ort_install_hint = (
                "pip install onnxruntime-gpu"
                if torch.cuda.is_available()
                else "pip install onnxruntime"
            )
            raise ModuleNotFoundError(
                f"ONNX verify requires onnxruntime. Install with: {_ort_install_hint}"
            ) from e

        backbone = _resolve_backbone(model)
        backbone.eval()
        device = next(iter(backbone.parameters())).device

        with torch.no_grad():
            eager_out = backbone(sample.to(device))
        eager_tensors = {name: eager_out[name].cpu() for name in self.output_names}

        sess = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        ort_inputs = {"image": sample.cpu().numpy()}
        ort_out = sess.run(list(self.output_names), ort_inputs)

        per_output: dict[str, tuple[float, float]] = {}
        max_abs = 0.0
        max_rel = 0.0
        for name, ort_arr in zip(self.output_names, ort_out, strict=True):
            ort_t = torch.from_numpy(ort_arr)
            eager_t = eager_tensors[name]
            abs_err = float((ort_t - eager_t).abs().max().item())
            rel_err = float(((ort_t - eager_t).abs() / eager_t.abs().clamp(min=1e-8)).max().item())
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


class _DictToTupleBackbone(nn.Module):
    """Wrap a dict-returning backbone so ONNX export sees a fixed tuple.

    ``torch.onnx.export`` understands tuple outputs much better than
    ``dict[str, Tensor]``; the adapter just gathers the configured
    output names and returns them in order.
    """

    def __init__(self, backbone: nn.Module, output_names: Sequence[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_names = tuple(output_names)

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        out = self.backbone(image)
        return tuple(out[name] for name in self.output_names)


class ONNXBackbone(nn.Module):
    """Run an ONNX-exported backbone+FPN as a drop-in replacement.

    Mirrors :class:`mayaku.inference.export.coreml.CoreMLBackbone` —
    same fixed-shape pad/crop pattern, same output dict contract. The
    only differences are the runtime (ONNX Runtime vs Core ML) and the
    ``providers`` knob, which selects the execution provider stack.

    Cross-platform — unlike CoreMLBackbone there is no macOS gate.
    `CoreMLExecutionProvider` is only available on macOS but
    `CPUExecutionProvider` (and on Linux/CUDA hosts the
    CUDA / TensorRT providers) work everywhere ONNX Runtime installs.
    """

    def __init__(
        self,
        onnx_path: Path,
        *,
        input_height: int = 800,
        input_width: int = 1344,
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        strides: dict[str, int] | None = None,
        size_divisibility: int = 32,
        providers: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        # Lazy import — onnxruntime is not a hard dependency.
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as e:
            _ort_install_hint = (
                "pip install onnxruntime-gpu"
                if torch.cuda.is_available()
                else "pip install onnxruntime"
            )
            raise ModuleNotFoundError(
                f"ONNXBackbone requires onnxruntime. Install with: {_ort_install_hint}"
            ) from e

        self.onnx_path = Path(onnx_path)
        self.input_shape: tuple[int, int] = (int(input_height), int(input_width))
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.strides: dict[str, int] = (
            dict(strides)
            if strides is not None
            else {n: _DEFAULT_STRIDES[n] for n in self.output_names}
        )
        self._size_divisibility = int(size_divisibility)
        # Default to CPUExecutionProvider so ORT never tries to load CUDA
        # shared libraries (.so/.dll) that may not match the host CUDA version.
        # Pass providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        # explicitly if you need GPU-accelerated ORT inference and have
        # installed onnxruntime-gpu built against your CUDA version.
        self.providers: tuple[str, ...] = (
            tuple(providers) if providers is not None else ("CPUExecutionProvider",)
        )
        self._session = ort.InferenceSession(str(self.onnx_path), providers=list(self.providers))
        # Cache the session's actual chosen providers (post-fallback).
        self.active_providers: tuple[str, ...] = tuple(self._session.get_providers())

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"ONNXBackbone.forward expects (1, 3, H, W); got {tuple(image.shape)}."
            )
        target_h, target_w = self.input_shape
        _b, c, h, w = image.shape
        if h > target_h or w > target_w:
            raise ValueError(
                f"Input shape ({h}, {w}) exceeds the exported ONNX "
                f"shape ({target_h}, {target_w}). Re-export at a "
                "larger size or constrain input."
            )

        device = image.device
        if h == target_h and w == target_w:
            padded = image
        else:
            padded = image.new_zeros((1, c, target_h, target_w))
            padded[:, :, :h, :w] = image

        ml_out = self._session.run(
            list(self.output_names), {"image": padded.detach().cpu().numpy()}
        )

        out: dict[str, Tensor] = {}
        for name, full_np in zip(self.output_names, ml_out, strict=True):
            full = torch.from_numpy(full_np).to(device)
            stride = self.strides[name]
            crop_h = (h + stride - 1) // stride
            crop_w = (w + stride - 1) // stride
            out[name] = full[:, :, :crop_h, :crop_w].contiguous()
        return out


def _resolve_backbone(model: nn.Module) -> nn.Module:
    """Return ``model.backbone`` if it exists, else ``model`` itself.

    The CLI passes a full :class:`FasterRCNN`; user scripts may pass a
    raw FPN. Either is fine — both expose the same forward contract
    (``image -> dict[str, Tensor]`` of FPN levels)."""
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        return model.backbone
    return model
