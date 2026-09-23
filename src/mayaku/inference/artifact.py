"""Run an exported artifact with its own runtime: `ArtifactPredictor`.

The artifact carries everything in its embedded sidecar, so no checkpoint or
config is needed; the graph runs in the runtime and the decode on the host,
exactly as a deployment would. 3.0 runs ONNX (onnxruntime).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mayaku.inference.export.metadata import read_sidecar, target_from_suffix
from mayaku.inference.runner import Runner
from mayaku.utils.checkpoint import check_sidecar

__all__ = ["ArtifactPredictor"]


class ArtifactPredictor(Runner):
    """An exported artifact run by its runtime, called like a `Predictor`."""

    def __init__(self, path: str | Path, device: str = "auto"):
        path = Path(path)
        target = target_from_suffix(path)
        if target != "onnx":
            raise NotImplementedError(f"{path}: running {target} artifacts is not available in "
                                      "this version; export to ONNX, or use the checkpoint")
        super().__init__(check_sidecar(read_sidecar(path, target), str(path)), str(path))
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device in ("cuda", "auto") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self._sess = ort.InferenceSession(str(path), providers=providers)
        inp = self._sess.get_inputs()[0]
        if tuple(inp.shape[-2:]) != self.canvas:
            raise ValueError(f"{path}: graph input {inp.shape} disagrees with the sidecar "
                             f"canvas {self.canvas}")
        self._input = inp.name
        # the graph is traced at a fixed batch size (1 unless exported otherwise)
        self._step = inp.shape[0] if isinstance(inp.shape[0], int) else None

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        a = x.numpy().astype(np.float32)
        a /= 255
        step = self._step or len(a)
        names = self.sidecar["outputs"]
        runs = [self._sess.run(names, {self._input: a[i:i + step]}) for i in range(0, len(a), step)]
        maps = runs[0] if len(runs) == 1 else [np.concatenate(m) for m in zip(*runs, strict=True)]
        return [torch.from_numpy(m) for m in maps]
