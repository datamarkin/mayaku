"""Dataset statistics for the health report.

One vectorised pass over a loaded COCO annotation file
(`mayaku.data.coco.CocoLabels`). Box statistics are measured in the frame the
model sees: after the aspect-preserving letterbox onto the canvas when one is
given, else in original pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from mayaku.data.coco import CocoLabels

__all__ = ["DatasetStats", "analyze_dataset"]


@dataclass(frozen=True)
class DatasetStats:
    num_images: int
    num_classes: int
    class_counts: dict[int, int]            # class index -> images containing it
    sqrt_areas: npt.NDArray[np.floating[Any]]  # per box, in the measured frame
    aspect_ratios: npt.NDArray[np.floating[Any]]  # per box, w / h
    num_degenerate_boxes: int = 0           # dropped at load: a side of a pixel or less
    num_images_without_annotations: int = 0

    @property
    def num_boxes(self) -> int:
        return len(self.sqrt_areas)

    @property
    def class_imbalance(self) -> float:
        """Most- over least-common class image frequency; 1.0 with fewer
        than two classes present."""
        if len(self.class_counts) < 2:
            return 1.0
        counts = self.class_counts.values()
        return max(counts) / max(1, min(counts))


def analyze_dataset(coco: CocoLabels, canvas: tuple[int, int] | None = None) -> DatasetStats:
    """`DatasetStats` of a `mayaku.data.coco.CocoLabels`, box sizes measured
    after letterboxing onto `canvas` (H, W) when given."""
    labels = list(coco.labels)
    counts = np.array([len(x) for x in labels], np.int64)
    boxes = np.concatenate(labels) if labels else np.zeros((0, 5), np.float32)
    h, w = coco.shapes[:, 0].astype(np.float64), coco.shapes[:, 1].astype(np.float64)
    scale = np.minimum(canvas[0] / h, canvas[1] / w) if canvas else np.ones(len(h))
    s = np.repeat(scale, counts)
    bw, bh = (boxes[:, 3] - boxes[:, 1]) * s, (boxes[:, 4] - boxes[:, 2]) * s
    image = np.repeat(np.arange(len(counts)), counts)
    pairs = np.unique(np.stack((image, boxes[:, 0].astype(np.int64)), 1), axis=0)
    classes, images = np.unique(pairs[:, 1], return_counts=True) if len(pairs) else ((), ())
    return DatasetStats(
        num_images=len(counts),
        num_classes=len(coco.cat_ids),
        class_counts={int(c): int(n) for c, n in zip(classes, images, strict=True)},
        sqrt_areas=np.sqrt(bw * bh),
        aspect_ratios=bw / bh,
        num_degenerate_boxes=coco.num_degenerate,
        num_images_without_annotations=int((counts == 0).sum()),
    )
