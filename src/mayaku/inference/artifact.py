"""Run an exported artifact with its own runtime: `ArtifactPredictor`.

The artifact carries everything in its embedded sidecar, so no checkpoint or
config is needed; the graph runs in the runtime and the decode on the host,
exactly as a deployment would.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mayaku.inference.export import open_session
from mayaku.inference.export.metadata import read_sidecar, target_from_suffix
from mayaku.inference.runner import Runner
from mayaku.utils.checkpoint import check_sidecar

__all__ = ["ArtifactPredictor"]


class ArtifactPredictor(Runner):
    """An exported artifact (ONNX, Core ML, OpenVINO or TensorRT) run by its
    runtime, called like a `Predictor`."""

    def __init__(self, path: str | Path, device: str = "auto"):
        path = Path(path)
        super().__init__(check_sidecar(read_sidecar(path, target_from_suffix(path)), str(path)),
                         str(path))
        self._session = open_session(path, self.sidecar, device)
        if self._session.input_hw != self.canvas:
            raise ValueError(f"{path}: graph input {self._session.input_hw} disagrees with the "
                             f"sidecar canvas {self.canvas}")

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        a = x.numpy().astype(np.float32)
        a /= 255
        step = self._session.batch or len(a)
        runs = [self._session(a[i:i + step]) for i in range(0, len(a), step)]
        maps = runs[0] if len(runs) == 1 else [np.concatenate(m) for m in zip(*runs, strict=True)]
        return [torch.from_numpy(np.ascontiguousarray(m)) for m in maps]
