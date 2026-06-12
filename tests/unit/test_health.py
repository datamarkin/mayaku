"""Unit tests for :func:`mayaku.health_check`.

The health scan reads box stats from the COCO JSON (dims included), so
no real images are needed — an empty ``train/`` directory next to the
annotation file is enough.
"""

from __future__ import annotations

import json
from pathlib import Path

import mayaku


def _write_dataset(root: Path) -> None:
    train = root / "train"
    train.mkdir(parents=True)
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "height": 64, "width": 64},
            {"id": 2, "file_name": "b.jpg", "height": 64, "width": 64},
            {"id": 3, "file_name": "c.jpg", "height": 64, "width": 64},  # no annotations
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1, 1, 30, 30], "iscrowd": 0},
            # Degenerate: zero width.
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 0, 10], "iscrowd": 0},
        ],
    }
    (train / "_annotations.coco.json").write_text(json.dumps(coco))


def test_health_check_reports_counts_and_warnings(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    report = mayaku.health_check(tmp_path)

    assert set(report) == {"train"}
    train = report["train"]

    assert train["images"] == 3
    assert train["boxes"] == 2  # the degenerate box is excluded
    assert train["classes"] == 2
    assert train["class_counts"] == {"cat": 2, "dog": 1}  # keyed by name, not id

    # Warnings are factual hygiene flags only (no tuning thresholds).
    warnings = " ".join(train["warnings"])
    assert "degenerate" in warnings
    assert "no annotations" in warnings


def test_health_check_object_size_fractions_sum_to_one(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    sizes = mayaku.health_check(tmp_path)["train"]["object_size"]
    assert set(sizes) == {"small", "medium", "large"}
    assert round(sum(sizes.values()), 2) == 1.0
