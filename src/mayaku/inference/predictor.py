"""Run a trained detector from its checkpoint: `from_pretrained` and `Predictor`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from mayaku.backends.device import Device
from mayaku.data.batch import batch_to
from mayaku.inference.export.metadata import SUFFIX_TO_TARGET
from mayaku.inference.runner import Runner
from mayaku.utils.checkpoint import read_deploy_checkpoint

if TYPE_CHECKING:
    from mayaku.model import Detector

__all__ = ["Predictor", "from_pretrained"]


class Predictor(Runner):
    """A trained detector's deploy graph (`Detector.for_deploy`) in Torch,
    with its sidecar: architecture, canvas, classes and decode all come from
    the checkpoint."""

    def __init__(self, model: Detector, sidecar: Mapping[str, Any], device: str = "auto"):
        super().__init__(sidecar, "Predictor")
        self.device = torch.device(Device.resolve(device))
        self.model = model.for_deploy().to(self.device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> Predictor:
        """Rebuild the model its sidecar describes (quantization-aware when it
        was trained so) and load the weights strictly."""
        sidecar, cfg, state = read_deploy_checkpoint(Path(path))
        model = cfg.model.build(cfg.input.canvas, len(sidecar["class_names"]))
        model.load_state_dict(state, strict=True)
        return cls(model, sidecar, device)

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self.model(batch_to(x, self.device)))

    def export(self, target: str = "onnx", output: str | Path | None = None,
               precision: str | None = None) -> Path:
        """Write a deployable artifact with this model's sidecar embedded;
        `precision` defaults to the target's first. See
        `mayaku.inference.export.export`."""
        from mayaku.inference.export import TARGETS, export

        if output is None and target in TARGETS:
            output = Path("model").with_suffix(TARGETS[target].suffix)
        return export(self.model, self.sidecar, target, output or "model", precision)

def from_pretrained(source: str | Path, device: str = "auto") -> Runner:
    """Load a deployable detector: a checkpoint path or model name gives a
    `Predictor`; an exported artifact gives an `ArtifactPredictor` run by its
    own runtime. Both are called with an image and return `Detections`."""
    if Path(source).suffix.lower() in SUFFIX_TO_TARGET:
        from mayaku.inference.artifact import ArtifactPredictor

        return ArtifactPredictor(source, device)
    from mayaku.utils.download import resolve_weights

    return Predictor.from_checkpoint(resolve_weights(source), device)
