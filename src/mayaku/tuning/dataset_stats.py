"""Compute dataset statistics for auto-config.

Pure-function analyser: given the output of
:func:`mayaku.data.datasets.coco.load_coco_json` plus its
:class:`mayaku.data.catalog.Metadata`, returns a :class:`DatasetStats`
record capturing everything the recipe layer needs.

Box statistics are computed in *resized* image space — the frame the
pipeline actually produces: the aspect-preserving letterbox scale when a
``letterbox_canvas`` is given, else the canonical short-edge resize that
``ResizeShortestEdge`` applies. K-means clusters on any other frame
would produce anchor scales that don't match the model's actual input
distribution.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mayaku.data.transforms.augmentation import compute_resized_hw
from mayaku.data.transforms.geometry import letterbox_scale

__all__ = ["DatasetStats", "analyze_dataset", "dataset_aspect"]

# Robust aspect spread (p90/p10) at or below this → the dataset is "one aspect"
# and a fixed (H, W) canvas beats square letterbox. Consumed by the train-time
# canvas resolver via :func:`dataset_aspect`.
ASPECT_UNIFORMITY_THRESHOLD = 1.10


def _aspect_spread(aspects: Sequence[float]) -> float:
    """Robust aspect spread ``p90 / p10`` (1.0 for < 10 samples). The one place
    the percentile math lives — shared by ``dataset_aspect`` and ``DatasetStats``."""
    n = len(aspects)
    if n < 10:
        return 1.0
    s = sorted(aspects)
    return s[(n * 9) // 10] / max(s[n // 10], 1e-9)


def dataset_aspect(dataset_dicts: Sequence[dict[str, Any]]) -> tuple[float, bool]:
    """Median image aspect ``W / H`` + uniformity, from image dims only.

    A light dims-only pass (no box analysis) shared by the letterbox canvas
    resolver. Uniform = robust ``p90 / p10 <= ASPECT_UNIFORMITY_THRESHOLD`` so a
    few outliers never flip it. Returns ``(median_aspect, is_uniform)``.
    """
    aspects = [int(d["width"]) / int(d["height"]) for d in dataset_dicts]
    if not aspects:
        return 1.0, False
    return statistics.median(aspects), _aspect_spread(aspects) <= ASPECT_UNIFORMITY_THRESHOLD


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
    # Hygiene counts — boxes/images the analyser skipped, surfaced
    # instead of silently dropped so a health report can flag bad labels.
    num_degenerate_boxes: int = 0
    num_images_without_annotations: int = 0

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
    letterbox_canvas: tuple[int, int] | None = None,
) -> DatasetStats:
    """Compute :class:`DatasetStats` from loaded dataset dicts.

    Args:
        dataset_dicts: Output of
            :func:`mayaku.data.datasets.coco.load_coco_json`.
        num_classes: Number of classes in the dataset (from metadata).
        resize_short_edge: Short-edge target of the canonical resize.
            Defaults to 800 (the COCO / Mayaku default).
        resize_max_edge: Max long-edge after resize. Defaults to 1333.
        letterbox_canvas: The resolved deploy ``(H, W)`` canvas when the
            pipeline letterboxes. When given, box stats use the
            aspect-preserving letterbox scale ``min(H/h, W/w)`` instead
            of the short-edge rule, so they're measured in the frame
            the model actually sees.

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
    num_degenerate = 0
    num_images_without_annotations = 0

    for d in dataset_dicts:
        h = int(d["height"])
        w = int(d["width"])
        short_edges.append(min(h, w))
        long_edges.append(max(h, w))

        # Match the pipeline's actual resize exactly so box stats are in
        # the same space the model will see.
        if letterbox_canvas is not None:
            scale = letterbox_scale(h, w, *letterbox_canvas)
        else:
            new_h, _ = compute_resized_hw(h, w, resize_short_edge, resize_max_edge)
            scale = new_h / h

        annotations = d.get("annotations", ())
        if not annotations:
            num_images_without_annotations += 1

        seen_classes: set[int] = set()
        for ann in annotations:
            if ann.get("iscrowd", 0):
                # Crowd annotations are excluded from detection loss,
                # so they shouldn't influence anchor design either.
                continue
            cat_id = ann["category_id"]
            seen_classes.add(cat_id)
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                num_degenerate += 1
                continue
            # bbox is XYWH_ABS in original pixels.
            bw = float(bbox[2]) * scale
            bh = float(bbox[3]) * scale
            if bw <= 0 or bh <= 0:
                num_degenerate += 1
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
        num_degenerate_boxes=num_degenerate,
        num_images_without_annotations=num_images_without_annotations,
    )
