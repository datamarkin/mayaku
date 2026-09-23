"""What every deployed detector shares: its sidecar, and images in ->
`Detections` out through the one preprocessing and the one decode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mayaku.inference.decode import Detections, decode_sidecar
from mayaku.inference.preprocess import ImageInput, letterbox_batch
from mayaku.utils.checkpoint import check_sidecar


class Runner:
    """Call with an image path or an (H, W, 3) RGB array and get `Detections`
    in that image's pixels; `batch` takes several. A subclass supplies
    `_forward`: a (B, 3, H, W) uint8 batch on the canvas -> the raw maps in
    `sidecar["outputs"]` order."""

    def __init__(self, sidecar: Mapping[str, Any], source: str):
        self.sidecar = dict(check_sidecar(sidecar, source))

    @property
    def canvas(self) -> tuple[int, int]:
        return tuple(self.sidecar["canvas_hw"])

    @property
    def class_names(self) -> list[str]:
        return list(self.sidecar["class_names"])

    def _forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def __call__(self, image: ImageInput, conf: float | None = None) -> Detections:
        return self.batch([image], conf)[0]

    @torch.no_grad()
    def batch(self, images: Sequence[ImageInput], conf: float | None = None) -> list[Detections]:
        """Several images in one pass. `conf` overrides the recorded score
        threshold."""
        x, metas = letterbox_batch(images, self.canvas)
        return decode_sidecar(self._forward(x), metas, self.sidecar, conf)
