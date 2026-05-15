"""Compute dataset statistics for auto-config.

Pure-function analyser: given the output of
:func:`mayaku.data.datasets.coco.load_coco_json` plus its
:class:`mayaku.data.catalog.Metadata`, returns a :class:`DatasetStats`
record capturing everything the recipe layer needs.

Box statistics are computed in *resized* image space — that is, after
the canonical short-edge resize that ``ResizeShortestEdge`` applies
during training. K-means clusters on raw input-pixel areas would produce
anchor scales that don't match the model's actual input distribution.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mayaku.data.transforms.augmentation import compute_resized_hw

__all__ = ["DatasetStats", "analyze_dataset"]


@dataclass(frozen=True)
class DatasetStats:
    """Summary of a COCO-format dataset for auto-config.

    All box-derived stats are computed in *resized* space (i.e. after
    short-edge resize to ``resize_short_edge``). Image-size stats are
    in original pixels.
    """

    num_images: int
    num_classes: int
    class_counts: dict[int, int]
    sqrt_areas: tuple[float, ...]
    aspect_ratios: tuple[float, ...]
    median_image_short_edge: int
    median_image_long_edge: int

    @property
    def num_boxes(self) -> int:
        return len(self.sqrt_areas)

    @property
    def class_imbalance(self) -> float:
        """Ratio of most-common to least-common class image-frequency.

        Returns 1.0 for empty / single-class datasets so callers can
        compare against a threshold without a special case.
        """
        if len(self.class_counts) < 2:
            return 1.0
        counts = list(self.class_counts.values())
        lo = max(1, min(counts))
        return max(counts) / lo


def analyze_dataset(
    dataset_dicts: Sequence[dict[str, Any]],
    *,
    num_classes: int,
    resize_short_edge: int = 800,
    resize_max_edge: int = 1333,
) -> DatasetStats:
    """Compute :class:`DatasetStats` from loaded dataset dicts.

    Args:
        dataset_dicts: Output of
            :func:`mayaku.data.datasets.coco.load_coco_json`.
        num_classes: Number of classes in the dataset (from metadata).
        resize_short_edge: Short-edge target of the canonical resize.
            Defaults to 800 (the COCO / Mayaku default).
        resize_max_edge: Max long-edge after resize. Defaults to 1333.

    Returns:
        A :class:`DatasetStats` with all per-image / per-box stats.
    """
    if not dataset_dicts:
        return DatasetStats(
            num_images=0,
            num_classes=num_classes,
            class_counts={},
            sqrt_areas=(),
            aspect_ratios=(),
            median_image_short_edge=resize_short_edge,
            median_image_long_edge=resize_max_edge,
        )

    short_edges: list[int] = []
    long_edges: list[int] = []
    sqrt_areas: list[float] = []
    aspect_ratios: list[float] = []
    # Image-level — same semantics as RepeatFactorTrainingSampler so a
    # downstream RFS toggle keys off identical numbers.
    class_image_count: Counter[int] = Counter()

    for d in dataset_dicts:
        h = int(d["height"])
        w = int(d["width"])
        short_edges.append(min(h, w))
        long_edges.append(max(h, w))

        # Match ResizeShortestEdge's target size exactly so box stats are
        # in the same space the model will see.
        new_h, _ = compute_resized_hw(h, w, resize_short_edge, resize_max_edge)
        scale = new_h / h

        seen_classes: set[int] = set()
        for ann in d.get("annotations", ()):
            if ann.get("iscrowd", 0):
                # Crowd annotations are excluded from detection loss,
                # so they shouldn't influence anchor design either.
                continue
            cat_id = ann["category_id"]
            seen_classes.add(cat_id)
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            # bbox is XYWH_ABS in original pixels.
            bw = float(bbox[2]) * scale
            bh = float(bbox[3]) * scale
            if bw <= 0 or bh <= 0:
                continue
            sqrt_areas.append((bw * bh) ** 0.5)
            aspect_ratios.append(bw / bh)

        for c in seen_classes:
            class_image_count[c] += 1

    return DatasetStats(
        num_images=len(dataset_dicts),
        num_classes=num_classes,
        class_counts=dict(class_image_count),
        sqrt_areas=tuple(sqrt_areas),
        aspect_ratios=tuple(aspect_ratios),
        median_image_short_edge=int(statistics.median(short_edges)),
        median_image_long_edge=int(statistics.median(long_edges)),
    )
