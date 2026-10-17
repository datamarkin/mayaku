"""Run an exported artifact with its own runtime: `ArtifactPredictor`.

The artifact carries everything in its embedded sidecar, so no checkpoint or
config is needed; the graph runs in the runtime and the decode on the host,
exactly as a deployment would.
"""

from __future__ import annotations

from pathlib import Path

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
        step = self._session.batch or len(x)
        if step >= len(x):
            return self._session(x)
        # a session may hand back its own buffers, overwritten by the next run
        runs = [[m.clone() for m in self._session(x[i:i + step])] for i in range(0, len(x), step)]
        return [torch.cat(m) for m in zip(*runs, strict=True)]
