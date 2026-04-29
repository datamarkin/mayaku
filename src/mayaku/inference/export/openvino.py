"""OpenVINO exporter for the in-scope detectors.

Mirrors `BACKEND_PORTABILITY_REPORT.md` §7 for the CPU/Intel
deployment target. Same strategy as the ONNX and CoreML exporters:
serialise the **Graph A** body (backbone + FPN producing
``{p2..p6}``); the per-image Python glue (RPN top-k + per-class NMS,
mask paste, keypoint sub-pixel decode) stays out of the exported
graph by design.

OpenVINO's Python API supports converting a PyTorch ``nn.Module``
directly via :func:`openvino.convert_model` — no intermediate ONNX
hop needed (though the ONNX route works as a fallback if OpenVINO
ever stops supporting direct PT conversion).

Inputs/outputs of the exported graph:

* ``"image"`` shape ``(N, 3, H, W)`` float32 RGB *normalised* (mean
  already subtracted, std already divided), with ``N``/``H``/``W``
  dynamic by default.
* ``"p2"`` … ``"p6"`` per-FPN-level ``(N, C, H/stride, W/stride)``.

Parity is checked via :class:`openvino.Core`'s ``CPU`` device, so the
test runs on every backend (Linux CUDA host included — OpenVINO's
CPU plugin works there).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from mayaku.inference.export.base import ExportResult, ParityResult

__all__ = ["OpenVINOExporter"]

# Standard FPN output names (`Backbone.output_shape()` from Step 8).
_DEFAULT_OUT_NAMES: tuple[str, ...] = ("p2", "p3", "p4", "p5", "p6")


class OpenVINOExporter:
    """Export the backbone+FPN body to OpenVINO IR (``.xml`` + ``.bin``).

    Args:
        output_names: Names of the FPN feature outputs. Defaults to
            ``("p2", "p3", "p4", "p5", "p6")``.
        compress_to_fp16: If ``True`` (the OpenVINO default), the IR's
            ``.bin`` is stored at fp16 — half the disk size at the
            cost of small numerical drift. Default ``False`` here so
            parity vs eager is tight (`atol=1e-3`); production
            deployments that want the smaller artefact should re-export.
    """

    name: str = "openvino"

    def __init__(
        self,
        *,
        output_names: Sequence[str] = _DEFAULT_OUT_NAMES,
        compress_to_fp16: bool = False,
    ) -> None:
        self.output_names: tuple[str, ...] = tuple(output_names)
        self.compress_to_fp16 = compress_to_fp16

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
        """Convert ``model.backbone`` to OpenVINO IR and save to ``out_path``.

        ``out_path`` is the path to the ``.xml`` file; OpenVINO writes
        a sibling ``.bin`` next to it. The wrapper sets the input name
        to ``"image"`` and the output names to ``self.output_names``
        so downstream callers don't have to discover them.
        """
        del opts
        # Lazy import keeps openvino optional (`[openvino]` extra).
        try:
            import openvino as ov
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "OpenVINO export requires the [openvino] extra: pip install mayaku[openvino]"
            ) from e

        backbone = _resolve_backbone(model)
        adapter = _DictToTupleBackbone(backbone, self.output_names).eval()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            ov_model = ov.convert_model(adapter, example_input=sample)

        # Tag inputs/outputs with friendly names.
        ov_model.inputs[0].set_names({"image"})
        for ov_output, name in zip(ov_model.outputs, self.output_names, strict=True):
            ov_output.set_names({name})

        ov.save_model(ov_model, str(out_path), compress_to_fp16=self.compress_to_fp16)

        bin_path = out_path.with_suffix(".bin")
        return ExportResult(
            path=out_path,
            target=self.name,
            opset=None,
            input_names=("image",),
            output_names=self.output_names,
            extras={
                "compress_to_fp16": str(self.compress_to_fp16),
                "bin_path": str(bin_path),
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
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> ParityResult:
        """Run eager + OpenVINO CPU inference on ``sample`` and compare.

        Always uses the ``CPU`` device so the check is reproducible on
        every backend (the eager forward happens on whatever device
        ``model`` lives on; we move the sample to that device for the
        eager half and back to CPU for the comparison).
        """
        try:
            import openvino as ov
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "OpenVINO verify requires the [openvino] extra: pip install mayaku[openvino]"
            ) from e

        backbone = _resolve_backbone(model)
        backbone.eval()
        device = next(iter(backbone.parameters())).device

        with torch.no_grad():
            eager_out = backbone(sample.to(device))
        eager_tensors = {name: eager_out[name].cpu() for name in self.output_names}

        core = ov.Core()
        # OpenVINO's CPU plugin defaults to fp16 inference precision
        # even when the IR is fp32, which produces ~3 abs error vs
        # eager on the random-init untrained backbones our tests
        # exercise. Force fp32 inference for the parity check; users
        # who deploy with the default fp16 path are inherently OK with
        # that drift (they get the speed/size benefits).
        core.set_property("CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        ov_model = core.read_model(str(exported_path))
        compiled = core.compile_model(ov_model, device_name="CPU")
        ov_results = compiled([sample.cpu().numpy()])

        # OpenVINO returns a dict keyed by ``ConstOutput`` port objects;
        # iteration order isn't guaranteed to match our declared
        # output_names. Index by the friendly name we tagged at export
        # time so a re-ordering inside the converter doesn't silently
        # compare the wrong tensors.
        ov_by_name: dict[str, torch.Tensor] = {}
        for port, arr in ov_results.items():
            for name in port.get_names():
                if name in self.output_names:
                    ov_by_name[name] = torch.from_numpy(arr)
        if set(ov_by_name) != set(self.output_names):
            missing = set(self.output_names) - set(ov_by_name)
            raise RuntimeError(f"OpenVINO compiled model is missing expected outputs: {missing}")

        per_output: dict[str, tuple[float, float]] = {}
        max_abs = 0.0
        max_rel = 0.0
        for name in self.output_names:
            ov_t = ov_by_name[name]
            eager_t = eager_tensors[name]
            abs_err = float((ov_t - eager_t).abs().max().item())
            rel_err = float(((ov_t - eager_t).abs() / eager_t.abs().clamp(min=1e-8)).max().item())
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
    """Same wrapper as the ONNX/CoreML exporters; converters trace
    cleaner with tuples than dicts."""

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
