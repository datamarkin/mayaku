"""``Exporter`` protocol shared by every deployment target.

Mirrors the ``Exporter`` API documented in
``03_d2_reimplementation_prompt.md`` ("Device + export abstractions"):

```python
class Exporter(Protocol):
    name: str
    def export(self, model, sample, out_path, **opts) -> ExportResult: ...
    def parity_check(self, model, exported_path, sample) -> ParityResult: ...
```

Every concrete target — :class:`mayaku.inference.export.onnx.ONNXExporter`,
:class:`mayaku.inference.export.coreml.CoreMLExporter`,
:class:`mayaku.inference.export.openvino.OpenVINOExporter`,
:class:`mayaku.inference.export.tensorrt.TensorRTExporter` — is one file
under ``mayaku.inference.export`` that satisfies this contract.

The two return dataclasses keep the call surface narrow but informative:
``ExportResult`` records *what was written* (so the CLI can emit a
machine-readable summary) and ``ParityResult`` carries the per-tensor
error magnitudes so a tolerance failure is easy to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from torch import Tensor, nn

__all__ = ["ExportResult", "Exporter", "ParityResult"]


@dataclass(frozen=True)
class ExportResult:
    """Description of what an :meth:`Exporter.export` call produced."""

    path: Path
    target: str  # e.g. "onnx"
    opset: int | None = None
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParityResult:
    """Eager-vs-runtime numerical comparison summary."""

    target: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    # Per-output errors when the model emits multiple tensors. Keys are
    # the output names from ``ExportResult.output_names``.
    per_output: dict[str, tuple[float, float]] = field(default_factory=dict)


class Exporter(Protocol):
    """Protocol every deployment-target exporter implements."""

    name: str

    def export(
        self,
        model: nn.Module,
        sample: Tensor,
        out_path: Path,
        **opts: object,
    ) -> ExportResult:
        """Serialise ``model`` to ``out_path`` using ``sample`` as a
        tracing input; return an :class:`ExportResult`."""
        ...

    def parity_check(
        self,
        model: nn.Module,
        exported_path: Path,
        sample: Tensor,
        *,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> ParityResult:
        """Run ``model`` and the exported artefact on ``sample`` and
        compare. Returns a :class:`ParityResult` (does not raise)."""
        ...
