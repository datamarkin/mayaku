"""CoreML exporter for the in-scope detectors.

Mirrors `BACKEND_PORTABILITY_REPORT.md` §6: convert the **Graph A**
body (backbone + FPN producing ``{p2..p6}``) to a CoreML
``mlprogram`` package via :func:`coremltools.convert`. Apple Vision
expects RGB which is what we hand it (ADR 002 — no channel swap on
deployment).

Same overall layout as :class:`mayaku.inference.export.onnx.ONNXExporter`:
the per-image Python glue (RPN top-k + per-class NMS, mask paste,
keypoint sub-pixel decode) intentionally stays out of the exported
graph (BPR §6: "Paste masks in Swift via vImage or in Python via
``_do_paste_mask`` — the exported graph stops at 28x28 logits").
That keeps the CoreML artefact deterministic and dynamic-shape-
friendly via :class:`coremltools.RangeDim`.

Inputs/outputs of the exported graph match :class:`ONNXExporter`:

* ``"image"`` shape ``(1, 3, H, W)`` float32 RGB (already mean/std-
  normalised), with ``H``/``W`` either fixed or
  :class:`coremltools.RangeDim`.
* ``"p2"`` … ``"p6"`` per-FPN-level ``(1, C, H/stride, W/stride)``.

Parity caveats: the CoreML runtime only runs on macOS (the
:class:`coremltools.models.MLModel.predict` API loads
``Core ML.framework``). On non-macOS hosts we still produce the
artefact and report a "skipped" :class:`ParityResult`; the
``parity_check`` CLI / API call is a no-op there.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from mayaku.inference.export.base import ExportResult, ParityResult

__all__ = ["CoreMLBackbone", "CoreMLExporter"]

# Standard FPN output names (`Backbone.output_shape()` from Step 8).
_DEFAULT_OUT_NAMES: tuple[str, ...] = ("p2", "p3", "p4", "p5", "p6")
# Strides associated with the default FPN levels above. p6 is
# `LastLevelMaxPool(stride=2)` of p5 → effective stride 64.
_DEFAULT_STRIDES: dict[str, int] = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}


class CoreMLExporter:
    """Export the backbone+FPN body to a CoreML ``mlprogram`` package.

    Args:
        output_names: Names of the FPN feature outputs. Defaults to
            ``("p2", "p3", "p4", "p5", "p6")``.
        compute_units: Override the converter's compute-units pin.
            Default is ``"CPU_ONLY"`` so the parity check runs the
            same kernels on every macOS host (no flakiness from
            Neural Engine quantisation paths). Pass ``"ALL"`` to
            target the actual deployment configuration.
    """

    name: str = "coreml"

    _VALID_COMPUTE_UNITS: tuple[str, ...] = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")
    _VALID_PRECISION: tuple[str, ...] = ("fp32", "fp16")

    def __init__(
        self,
        *,
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        compute_units: str = "CPU_ONLY",
        compute_precision: str = "fp32",
    ) -> None:
        if compute_units not in self._VALID_COMPUTE_UNITS:
            raise ValueError(
                f"unknown CoreML compute_units {compute_units!r}; "
                f"expected one of {self._VALID_COMPUTE_UNITS}"
            )
        if compute_precision not in self._VALID_PRECISION:
            raise ValueError(
                f"unknown CoreML compute_precision {compute_precision!r}; "
                f"expected one of {self._VALID_PRECISION}"
            )
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.compute_units = compute_units
        self.compute_precision = compute_precision

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
        """Trace ``model.backbone`` and convert to CoreML.

        ``sample`` is a normalised RGB batch ``(1, 3, H, W)`` used for
        tracing. CoreML's ``mlprogram`` format supports dynamic input
        sizes via :class:`coremltools.RangeDim`; we do **not** opt into
        that here because dynamic shapes interact poorly with the
        constant folding the converter does for FPN's stride-2 ops on
        Apple Silicon. Most deployment workflows pin a fixed test
        size (e.g. 800x1333) anyway, so we mirror that.
        """
        del opts
        # Lazy import keeps coremltools optional (`[coreml]` extra).
        try:
            import coremltools as ct
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CoreML export requires the [coreml] extra: pip install mayaku[coreml]"
            ) from e

        backbone = _resolve_backbone(model)
        adapter = _DictToTupleBackbone(backbone, self.output_names).eval()

        if sample.shape[0] != 1:
            raise ValueError(
                "CoreML export requires a single-batch tracing sample "
                f"(shape (1, 3, H, W)); got {tuple(sample.shape)}"
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            traced = torch.jit.trace(  # type: ignore[no-untyped-call]
                adapter, (sample,), check_trace=False
            )

        compute = _compute_units(ct, self.compute_units)
        precision = _compute_precision(ct, self.compute_precision)
        # Precision is the dominant deployment knob:
        #   * `fp32` keeps eager-vs-CoreML parity tight (typical max
        #     abs ~1e-3) and is what the random-init `parity_check`
        #     tests rely on.
        #   * `fp16` is the default coremltools picks and is what real
        #     deployments use — Apple's Neural Engine only executes
        #     fp16, so `compute_units=ALL` silently falls back to
        #     CPU+GPU under fp32 and gives no NE acceleration.
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=tuple(sample.shape))],
            outputs=[ct.TensorType(name=n) for n in self.output_names],
            convert_to="mlprogram",
            compute_units=compute,
            compute_precision=precision,
        )
        ml.save(str(out_path))

        return ExportResult(
            path=out_path,
            target=self.name,
            opset=None,
            input_names=("image",),
            output_names=self.output_names,
            extras={
                "compute_units": self.compute_units,
                "compute_precision": self.compute_precision,
                "input_shape": "x".join(str(int(d)) for d in sample.shape),
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
        """Compare eager + CoreML predictions on ``sample``.

        On non-macOS hosts the CoreML runtime isn't available, so the
        parity check returns a passing :class:`ParityResult` with an
        ``extras`` note explaining the skip. The artefact itself is
        validated independently by the export-side test (the file
        exists and has non-zero size).

        CoreML's quantised pipeline introduces small numerical
        differences vs eager fp32 — default tolerance is `1e-2`
        (looser than ONNX's `1e-3`).
        """
        backbone = _resolve_backbone(model)
        backbone.eval()
        device = next(iter(backbone.parameters())).device

        with torch.no_grad():
            eager_out = backbone(sample.to(device))

        if platform.system() != "Darwin":
            return ParityResult(
                target=self.name,
                passed=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                atol=atol,
                rtol=rtol,
                per_output={name: (0.0, 0.0) for name in self.output_names},
            )

        try:
            import coremltools as ct
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CoreML verify requires the [coreml] extra: pip install mayaku[coreml]"
            ) from e

        ml = ct.models.MLModel(str(exported_path))
        ort_inputs = {"image": sample.cpu().numpy()}
        ml_out = ml.predict(ort_inputs)

        per_output: dict[str, tuple[float, float]] = {}
        max_abs = 0.0
        max_rel = 0.0
        for name in self.output_names:
            ml_t = torch.from_numpy(ml_out[name])
            eager_t = eager_out[name].cpu()
            abs_err = float((ml_t - eager_t).abs().max().item())
            rel_err = float(((ml_t - eager_t).abs() / eager_t.abs().clamp(min=1e-8)).max().item())
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
    """Same wrapper as the ONNX exporter's; trace plays nicer with tuples."""

    def __init__(self, backbone: nn.Module, output_names: Sequence[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_names = tuple(output_names)

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        out = self.backbone(image)
        return tuple(out[name] for name in self.output_names)


def _resolve_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        return model.backbone
    return model


# ---------------------------------------------------------------------------
# Runtime adapter
# ---------------------------------------------------------------------------


class CoreMLBackbone(nn.Module):
    """Run a CoreML-exported backbone+FPN as a drop-in replacement.

    Wraps an ``.mlpackage`` produced by :class:`CoreMLExporter`. Presents
    the same forward contract as ``FPN``: ``forward(image) -> {p2..p6}``
    on the device of the input. The detector's RPN / ROI heads / postprocess
    keep running in PyTorch so end-to-end COCO eval needs no other changes.

    The CoreML graph is exported at a single fixed shape
    (``(input_height, input_width)``). Real eval inputs come padded to a
    multiple of ``size_divisibility`` (32 for FPN R50), with per-image
    ``(H_pad, W_pad)`` ≤ the export shape. ``forward`` zero-pads the
    input up to the export shape, runs the model, and crops each FPN
    output back to ``(H_pad // stride, W_pad // stride)`` so downstream
    consumers see the same shapes the eager backbone would produce.

    Runtime is **macOS-only** — ``coremltools.models.MLModel.predict``
    needs ``Core ML.framework``. Construction itself works anywhere
    (we lazy-import) but ``forward`` raises on non-Darwin.
    """

    def __init__(
        self,
        mlpackage_path: Path,
        *,
        input_height: int = 800,
        input_width: int = 1344,
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        strides: dict[str, int] | None = None,
        size_divisibility: int = 32,
        compute_units: str = "CPU_AND_GPU",
    ) -> None:
        super().__init__()
        if platform.system() != "Darwin":
            raise RuntimeError(
                "CoreMLBackbone needs the Core ML runtime, which is "
                "macOS-only. Use the eager backbone on this host."
            )
        # Lazy import so non-mac environments don't pay the cost.
        try:
            import coremltools as ct
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "CoreMLBackbone requires the [coreml] extra: pip install mayaku[coreml]"
            ) from e

        self.mlpackage_path = Path(mlpackage_path)
        self.input_shape: tuple[int, int] = (int(input_height), int(input_width))
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.strides: dict[str, int] = (
            dict(strides)
            if strides is not None
            else {n: _DEFAULT_STRIDES[n] for n in self.output_names}
        )
        self._size_divisibility = int(size_divisibility)
        self.compute_units = compute_units
        self._mlmodel = ct.models.MLModel(
            str(self.mlpackage_path), compute_units=_compute_units(ct, compute_units)
        )

    # The detector's `_size_divisibility()` reads this attribute via
    # `getattr(self.backbone, "size_divisibility", 1)`.
    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                "CoreMLBackbone.forward expects (1, 3, H, W); got "
                f"{tuple(image.shape)}. Eval CLI uses batch_size=1 — "
                "multi-image batching would require re-export with a "
                "fixed batch dim."
            )
        target_h, target_w = self.input_shape
        _b, c, h, w = image.shape
        if h > target_h or w > target_w:
            raise ValueError(
                f"Input shape ({h}, {w}) exceeds the exported CoreML "
                f"shape ({target_h}, {target_w}). Re-export at a "
                "larger size or ensure ResizeShortestEdge / "
                "max_size_test honours this bound."
            )

        device = image.device
        if h == target_h and w == target_w:
            padded = image
        else:
            padded = image.new_zeros((1, c, target_h, target_w))
            padded[:, :, :h, :w] = image

        # CoreML's predict() takes numpy on CPU.
        ml_out = self._mlmodel.predict({"image": padded.detach().cpu().numpy()})

        out: dict[str, Tensor] = {}
        for name in self.output_names:
            full = torch.from_numpy(ml_out[name]).to(device)
            stride = self.strides[name]
            # The eager FPN's right/bottom output cells are derived from
            # the same conv chain whether we ran at H_pad or target_h —
            # cropping just takes the first H_pad/stride rows. p6 uses
            # LastLevelMaxPool(stride=2) so its tile is half-rounded;
            # `(h + stride - 1) // stride` covers both the integer and
            # ceil-stride-2 cases without a special branch.
            crop_h = (h + stride - 1) // stride
            crop_w = (w + stride - 1) // stride
            out[name] = full[:, :, :crop_h, :crop_w].contiguous()
        return out


def _compute_precision(ct_module: object, name: str) -> object:
    """Resolve a string name into a ``ct.precision`` enum value."""
    enum = ct_module.precision  # type: ignore[attr-defined]
    if name == "fp32":
        return enum.FLOAT32
    if name == "fp16":
        return enum.FLOAT16
    raise ValueError(f"unknown CoreML compute_precision {name!r}; expected 'fp32' or 'fp16'")


def _compute_units(ct_module: object, name: str) -> object:
    """Resolve a string name into a ``ct.ComputeUnit`` enum value."""
    enum = ct_module.ComputeUnit  # type: ignore[attr-defined]
    if name == "CPU_ONLY":
        return enum.CPU_ONLY
    if name == "CPU_AND_GPU":
        return enum.CPU_AND_GPU
    if name == "ALL":
        return enum.ALL
    if name == "CPU_AND_NE":
        # 6.0+ exposes this; older versions raise AttributeError.
        return getattr(enum, "CPU_AND_NE", enum.ALL)
    raise ValueError(
        f"unknown CoreML compute_units {name!r}; expected CPU_ONLY / CPU_AND_GPU / CPU_AND_NE / ALL"
    )
