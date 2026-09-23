"""Export a trained detector to a deployable artifact with its sidecar embedded.

3.0 exports ONNX: the model's deploy graph (`Detector.for_deploy`) at its
canvas, contract checked on the file, sidecar in the metadata, and its raw
output maps compared against the Torch model. The formats a sidecar can be
read from are listed in `metadata.SUFFIX_TO_TARGET`; CoreML, OpenVINO and
TensorRT exporters come later.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from mayaku.inference.export.metadata import SUFFIX_TO_TARGET, embed_sidecar
from mayaku.model import Detector
from mayaku.model.contract import export_onnx

__all__ = ["TARGETS", "export", "onnx_parity"]

#: Writable targets and their artifact suffix.
TARGETS = {t: s for s, t in SUFFIX_TO_TARGET.items() if t == "onnx"}


def export(model: Detector, sidecar: Mapping[str, Any], target: str,
           path: str | Path) -> Path:
    """Write `model` (a trained `Detector`, left untouched) as a `target`
    artifact at `path`, with `sidecar` embedded. The artifact's raw maps must
    match the Torch model's, or this raises."""
    if target not in TARGETS:
        raise NotImplementedError(f"export target {target!r} is not available in this version; "
                                  f"available: {', '.join(TARGETS)}")
    path = Path(path)
    m = model.for_deploy().cpu()
    export_onnx(m, str(path), tuple(sidecar["canvas_hw"]))
    embed_sidecar(path, target, dict(sidecar))
    err, scale = onnx_parity(m, path)
    if err > 1e-3 * max(scale, 1.0):
        raise RuntimeError(f"{path}: ONNX outputs differ from the Torch model by {err:.3e}")
    return path


@torch.no_grad()
def onnx_parity(model: torch.nn.Module, path: Path, seed: int = 0) -> tuple[float, float]:
    """Max abs difference between the ONNX artifact's raw maps and a deploy
    model's (`Detector.for_deploy`) on a random canvas-sized input, and the
    maps' scale."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = torch.rand(*inp.shape, generator=torch.Generator().manual_seed(seed))
    got = sess.run(None, {inp.name: x.numpy()})
    ref = model.eval()(x)
    err = max(float(abs(r.numpy() - g).max()) for r, g in zip(ref, got, strict=True))
    return err, max(float(r.abs().max()) for r in ref)
