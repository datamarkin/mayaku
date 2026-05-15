"""Tests for :mod:`mayaku.tuning.dataset_stats`."""

from __future__ import annotations

import pytest

from mayaku.tuning.dataset_stats import DatasetStats, analyze_dataset


def _img(image_id: int, h: int, w: int, anns: list[dict]) -> dict:
    return {
        "image_id": image_id,
        "file_name": f"img_{image_id}.png",
        "height": h,
        "width": w,
        "annotations": anns,
    }


def _box(category_id: int, x: float, y: float, w: float, h: float, iscrowd: int = 0) -> dict:
    return {
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "bbox_mode": 0,  # XYWH_ABS
        "iscrowd": iscrowd,
    }


def test_empty_dataset_returns_zero_counts() -> None:
    stats = analyze_dataset([], num_classes=10)
    assert stats.num_images == 0
    assert stats.num_classes == 10
    assert stats.class_counts == {}
    assert stats.sqrt_areas == ()
    assert stats.aspect_ratios == ()
    assert stats.num_boxes == 0
    # Image-edge defaults fall back to the resize targets so callers
    # have something sensible when probing an empty dataset.
    assert stats.median_image_short_edge == 800
    assert stats.median_image_long_edge == 1333
    assert stats.class_imbalance == 1.0


def test_class_counts_are_image_level_not_box_level() -> None:
    # Mirrors RepeatFactorTrainingSampler semantics: count images
    # containing each class, not annotations.
    dataset = [
        _img(1, 800, 800, [_box(0, 0, 0, 10, 10), _box(0, 5, 5, 10, 10), _box(1, 0, 0, 10, 10)]),
        _img(2, 800, 800, [_box(0, 0, 0, 10, 10)]),
    ]
    stats = analyze_dataset(dataset, num_classes=2)
    assert stats.class_counts == {0: 2, 1: 1}


def test_crowd_annotations_are_excluded_from_box_stats() -> None:
    dataset = [
        _img(
            1,
            800,
            800,
            [
                _box(0, 0, 0, 100, 100, iscrowd=1),  # excluded
                _box(0, 0, 0, 50, 100, iscrowd=0),  # included
            ],
        )
    ]
    stats = analyze_dataset(dataset, num_classes=1)
    # Only one box made it in.
    assert stats.num_boxes == 1
    # And it's the iscrowd=0 one (w=50, h=100 → AR=0.5).
    assert pytest.approx(stats.aspect_ratios[0]) == 0.5


def test_boxes_are_in_resized_space() -> None:
    # 400×400 image, short-edge target 800 → scale 2.0. A 50×50 box
    # in pixels becomes 100×100 in resized space; √area=100.
    dataset = [_img(1, 400, 400, [_box(0, 0, 0, 50, 50)])]
    stats = analyze_dataset(dataset, num_classes=1, resize_short_edge=800)
    assert pytest.approx(stats.sqrt_areas[0]) == 100.0


def test_max_edge_caps_resize_scale() -> None:
    # 200×1000 image; raw scale to short_edge=800 would be 4.0 → long
    # edge would land at 4000, exceeding max_edge=1333. Scale must
    # clamp to 1333/1000 = 1.333.
    dataset = [_img(1, 200, 1000, [_box(0, 0, 0, 30, 30)])]
    stats = analyze_dataset(dataset, num_classes=1, resize_short_edge=800, resize_max_edge=1333)
    # 30 px → 30 * 1.333 ≈ 40, area ≈ 1600, sqrt ≈ 40.
    assert pytest.approx(stats.sqrt_areas[0], abs=0.5) == 40.0


def test_class_imbalance_handles_single_class_dataset() -> None:
    dataset = [_img(1, 800, 800, [_box(0, 0, 0, 10, 10)])]
    stats = analyze_dataset(dataset, num_classes=1)
    # Single class → ratio is 1.0 by convention, not div-by-zero.
    assert stats.class_imbalance == 1.0


def test_class_imbalance_uses_max_over_min() -> None:
    # 9 images of class 0, 1 image of class 1 → 9× imbalance.
    dataset = [
        *[_img(i, 800, 800, [_box(0, 0, 0, 10, 10)]) for i in range(9)],
        _img(99, 800, 800, [_box(1, 0, 0, 10, 10)]),
    ]
    stats = analyze_dataset(dataset, num_classes=2)
    assert stats.class_imbalance == pytest.approx(9.0)


def test_zero_size_boxes_are_dropped() -> None:
    dataset = [_img(1, 800, 800, [_box(0, 0, 0, 0, 10), _box(0, 0, 0, 10, 10)])]
    stats = analyze_dataset(dataset, num_classes=1)
    assert stats.num_boxes == 1


def test_returns_immutable_dataclass() -> None:
    stats = analyze_dataset([_img(1, 800, 800, [_box(0, 0, 0, 10, 10)])], num_classes=1)
    assert isinstance(stats, DatasetStats)
    with pytest.raises(Exception):  # FrozenInstanceError subclass
        stats.num_images = 99  # type: ignore[misc]
