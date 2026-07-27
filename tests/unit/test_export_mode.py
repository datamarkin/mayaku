"""Export mode is declared, not inferred from the capture mechanism.

If :func:`is_exporting` ever stops being true mid-export, the artifact silently
gets the eager graph — no error, just a backend miscompile at inference time.
See :mod:`mayaku.utils.export_mode` for why sniffing the capture mechanism is
not enough.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mayaku.inference.export import dispatch
from mayaku.inference.export.base import ExportResult
from mayaku.utils.export_mode import exporting, is_exporting


def test_is_exporting_is_false_by_default() -> None:
    assert not is_exporting()


def test_exporting_scope_sets_and_restores() -> None:
    with exporting():
        assert is_exporting()
    assert not is_exporting()


def test_exporting_scope_restores_on_exception() -> None:
    with pytest.raises(RuntimeError), exporting():
        raise RuntimeError("boom")
    assert not is_exporting()


@pytest.mark.parametrize("target", ["onnx", "coreml", "openvino", "tensorrt"])
def test_every_export_target_runs_in_export_mode(
    target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whatever each exporter uses to capture, the graph is built in export mode.

    Asserted at the dispatch seam with the real exporters stubbed out, so this
    keeps holding if an exporter switches front-end (torch.jit.trace ->
    torch.export, say) — which is precisely the change that would silently
    break a trace-sniffing check.
    """
    seen: list[bool] = []

    class _Spy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def export(self, *args: object, **kwargs: object) -> ExportResult:
            seen.append(is_exporting())
            out = tmp_path / f"{target}.bin"
            out.write_bytes(b"")
            return ExportResult(path=out, target=target)

    for name in ("ONNXExporter", "CoreMLExporter", "OpenVINOExporter", "TensorRTExporter"):
        monkeypatch.setattr(dispatch, name, _Spy)
    monkeypatch.setattr(dispatch, "embed_sidecar", lambda *a, **k: None)

    model = nn.Conv2d(3, 3, 1)
    dispatch.export_detector(
        model, target, tmp_path / f"{target}.out", sample=torch.zeros(1, 3, 8, 8)
    )

    assert seen == [True], f"{target} exporter did not run in export mode"
    assert not is_exporting(), "export mode leaked past export_detector"
