"""End-to-end test: COCO loader + DatasetMapper on a toy dataset.

Builds a 4-image COCO-format dataset on disk in ``tmp_path``:

* image 0: one bbox + one polygon segmentation (instance seg)
* image 1: one bbox + one keypoint annotation (keypoints — 4 KP for brevity)
* image 2: two bboxes, one with iscrowd=1 (mapper drops crowds at train)
* image 3: empty annotations list (passes through with empty Instances)

Then exercises:

* :func:`load_coco_json` — every image is read, ids are remapped,
  +0.5 keypoint shift is applied.
* :class:`DatasetMapper` (train + test mode) — image becomes
  ``(3, H, W)`` float RGB; ``Instances`` carries ``gt_boxes``,
  ``gt_classes``, ``gt_masks`` (PolygonMasks) and ``gt_keypoints``
  (Keypoints) on the right images; horizontal-flip swap honours the
  flip-pair permutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mayaku.data import (
    AugmentationList,
    DatasetMapper,
    Metadata,
    RandomFlip,
    ResizeShortestEdge,
    build_coco_metadata,
    load_coco_json,
)
from mayaku.structures.boxes import BoxMode
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import Keypoints
from mayaku.structures.masks import BitMasks, PolygonMasks

# ---------------------------------------------------------------------------
# Toy dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_coco(tmp_path: Path) -> tuple[Path, Path, Metadata]:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # Four small RGB images, each 32x40 (H=32, W=40).
    for i in range(4):
        arr = np.full((32, 40, 3), fill_value=10 * (i + 1), dtype=np.uint8)
        Image.fromarray(arr).save(images_dir / f"img_{i}.png")

    coco = {
        "images": [
            {"id": i, "file_name": f"img_{i}.png", "height": 32, "width": 40} for i in range(4)
        ],
        "categories": [
            {"id": 5, "name": "thing_a"},
            {"id": 7, "name": "thing_b"},
        ],
        "annotations": [
            # image 0 — bbox + polygon
            {
                "id": 100,
                "image_id": 0,
                "category_id": 5,
                "bbox": [4.0, 4.0, 12.0, 8.0],  # XYWH_ABS
                "iscrowd": 0,
                "segmentation": [[4, 4, 16, 4, 16, 12, 4, 12]],
                "area": 96.0,
            },
            # image 1 — bbox + keypoints (K=4, two pairs)
            {
                "id": 101,
                "image_id": 1,
                "category_id": 5,
                "bbox": [10.0, 10.0, 10.0, 10.0],
                "iscrowd": 0,
                # COCO triplets (x, y, v); we use K=4: nose + eye_l/eye_r + ear_l
                "keypoints": [12, 12, 2, 14, 12, 2, 16, 12, 2, 18, 12, 1],
                "num_keypoints": 4,
            },
            # image 2 — two bboxes, one crowd
            {
                "id": 102,
                "image_id": 2,
                "category_id": 7,
                "bbox": [0.0, 0.0, 8.0, 8.0],
                "iscrowd": 0,
            },
            {
                "id": 103,
                "image_id": 2,
                "category_id": 7,
                "bbox": [20.0, 0.0, 8.0, 8.0],
                "iscrowd": 1,
            },
            # image 3 — no annotations (intentional: just no entry)
        ],
    }
    json_path = tmp_path / "toy.json"
    json_path.write_text(json.dumps(coco))

    # Metadata with a 4-keypoint flip permutation (pair 1<->2; pair 3
    # alone — swap with itself for simplicity).
    md = Metadata(
        name="toy",
        thing_classes=("thing_a", "thing_b"),
        thing_dataset_id_to_contiguous_id={5: 0, 7: 1},
        keypoint_names=("nose", "eye_l", "eye_r", "ear_l"),
        keypoint_flip_indices=(0, 2, 1, 3),
    )
    return json_path, images_dir, md


# ---------------------------------------------------------------------------
# COCO loader
# ---------------------------------------------------------------------------


def test_load_coco_json_remaps_categories_and_shifts_keypoints(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    assert len(dicts) == 4

    # image 0
    assert Path(dicts[0]["file_name"]).name == "img_0.png"
    assert dicts[0]["height"] == 32 and dicts[0]["width"] == 40
    assert dicts[0]["annotations"][0]["category_id"] == 0  # 5 → 0
    assert dicts[0]["annotations"][0]["bbox_mode"] == BoxMode.XYWH_ABS
    assert "segmentation" in dicts[0]["annotations"][0]

    # image 1: keypoints shifted by +0.5
    kp = dicts[1]["annotations"][0]["keypoints"]
    assert kp[0:3] == [12.5, 12.5, 2]
    assert kp[3:6] == [14.5, 12.5, 2]
    assert kp[-3:] == [18.5, 12.5, 1]

    # image 2: both annotations preserved (mapper, not loader, drops crowds)
    assert len(dicts[2]["annotations"]) == 2
    assert dicts[2]["annotations"][1]["iscrowd"] == 1

    # image 3: empty
    assert dicts[3]["annotations"] == []


def test_load_coco_json_rejects_unknown_category(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    json_path, images_dir, _md = toy_coco
    md = Metadata(
        name="toy",
        thing_classes=("thing_a",),
        thing_dataset_id_to_contiguous_id={5: 0},  # 7 is missing
    )
    with pytest.raises(ValueError, match="category_id=7"):
        load_coco_json(json_path, images_dir, md)


def test_load_coco_json_skips_segmentation_when_disabled(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    """``keep_segmentation=False`` drops polygon lists from every record."""
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md, keep_segmentation=False)
    for dd in dicts:
        for ann in dd["annotations"]:
            assert "segmentation" not in ann


def test_load_coco_json_skips_keypoints_when_disabled(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    """``keep_keypoints=False`` drops keypoint arrays from every record."""
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md, keep_keypoints=False)
    for dd in dicts:
        for ann in dd["annotations"]:
            assert "keypoints" not in ann


def test_build_coco_metadata_infers_classes(toy_coco: tuple[Path, Path, Metadata]) -> None:
    json_path, _images_dir, _md = toy_coco
    md = build_coco_metadata("toy_inferred", json_path)
    assert md.thing_classes == ("thing_a", "thing_b")
    assert md.thing_dataset_id_to_contiguous_id == {5: 0, 7: 1}


# ---------------------------------------------------------------------------
# DatasetMapper
# ---------------------------------------------------------------------------


def _train_mapper(md: Metadata, *, keypoint_on: bool, flip_prob: float = 0.0) -> DatasetMapper:
    rng = np.random.default_rng(0)
    augs = AugmentationList(
        [
            ResizeShortestEdge((32,), max_size=200, sample_style="choice", rng=rng),
            RandomFlip(prob=flip_prob, rng=rng),
        ],
        flip_indices=md.keypoint_flip_indices,
    ).augmentations  # extract the underlying list — DatasetMapper composes again
    return DatasetMapper(
        augs,
        is_train=True,
        mask_format="polygon",
        keypoint_on=keypoint_on,
        metadata=md,
    )


def test_mapper_test_mode_drops_annotations(toy_coco: tuple[Path, Path, Metadata]) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    mapper = DatasetMapper(
        [ResizeShortestEdge((32,), max_size=200, sample_style="choice")],
        is_train=False,
        keypoint_on=False,
        metadata=md,
    )
    out = mapper(dicts[0])
    assert "instances" not in out
    assert "annotations" not in out
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 32, 40)
    assert out["image"].dtype == torch.float32


def test_mapper_builds_instances_with_boxes_classes_polygon_masks(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    mapper = _train_mapper(md, keypoint_on=False)
    out = mapper(dicts[0])
    assert isinstance(out["instances"], Instances)
    inst = out["instances"]
    assert len(inst) == 1
    assert inst.gt_boxes.shape == (1, 4)
    # bbox 4,4,12,8 (XYWH) → 4,4,16,12 XYXY
    torch.testing.assert_close(inst.gt_boxes, torch.tensor([[4.0, 4.0, 16.0, 12.0]]))
    assert inst.gt_classes.tolist() == [0]
    assert isinstance(inst.gt_masks, PolygonMasks)
    assert len(inst.gt_masks) == 1


def test_mapper_drops_crowd_annotations(toy_coco: tuple[Path, Path, Metadata]) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    mapper = _train_mapper(md, keypoint_on=False)
    out = mapper(dicts[2])
    # Two annotations on disk; one is iscrowd=1 → only one survives.
    assert len(out["instances"]) == 1
    torch.testing.assert_close(out["instances"].gt_boxes, torch.tensor([[0.0, 0.0, 8.0, 8.0]]))


def test_mapper_handles_empty_annotations(toy_coco: tuple[Path, Path, Metadata]) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    mapper = _train_mapper(md, keypoint_on=False)
    out = mapper(dicts[3])
    inst = out["instances"]
    assert len(inst) == 0
    assert inst.gt_boxes.shape == (0, 4)


def test_mapper_keypoint_path_with_flip_swap(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    mapper = _train_mapper(md, keypoint_on=True, flip_prob=1.0)
    out = mapper(dicts[1])
    inst = out["instances"]
    assert isinstance(inst.gt_keypoints, Keypoints)
    kp = inst.gt_keypoints.tensor
    # Original (post +0.5 shift): [(12.5,12.5,2), (14.5,12.5,2),
    #                              (16.5,12.5,2), (18.5,12.5,1)]
    # Image width = 40 (image is 32x40, after resize-short=32 it stays 32x40).
    # Horizontal flip → x' = 40 - x.
    # Then flip_indices (0, 2, 1, 3) reorders → final[i] = src[idx[i]].
    # Expected x positions: src indices 0,2,1,3 → flipped x = 27.5, 23.5, 25.5, 21.5.
    np.testing.assert_allclose(kp[0, :, 0].numpy(), [27.5, 23.5, 25.5, 21.5])
    # Visibilities follow the swap too: src vis = [2,2,2,1] → reordered → [2,2,2,1].
    np.testing.assert_array_equal(kp[0, :, 2].to(torch.long).numpy(), [2, 2, 2, 1])


def test_mapper_keypoint_requires_flip_indices_in_metadata(
    toy_coco: tuple[Path, Path, Metadata],
) -> None:
    json_path, _images_dir, md = toy_coco
    bad_md = Metadata(
        name="toy",
        thing_classes=md.thing_classes,
        thing_dataset_id_to_contiguous_id=md.thing_dataset_id_to_contiguous_id,
        keypoint_names=md.keypoint_names,
        keypoint_flip_indices=None,  # <-- missing
    )
    del json_path  # not needed
    with pytest.raises(ValueError, match="flip_indices"):
        DatasetMapper(
            [ResizeShortestEdge((32,), max_size=200)],
            is_train=True,
            keypoint_on=True,
            metadata=bad_md,
        )


def test_mapper_image_dtype_layout_and_dtype(toy_coco: tuple[Path, Path, Metadata]) -> None:
    json_path, images_dir, md = toy_coco
    dicts = load_coco_json(json_path, images_dir, md)
    mapper = _train_mapper(md, keypoint_on=False)
    out = mapper(dicts[0])
    img = out["image"]
    assert img.shape[0] == 3  # channels-first
    assert img.dtype == torch.float32
    # Pixel values sit in [0, 255] (no normalisation here — the model does it).
    assert img.max() >= 1.0


# ---------------------------------------------------------------------------
# Bitmask path: rasterisation must happen in the pre-augmentation frame
# ---------------------------------------------------------------------------


def _polygon_dict(image_h: int, image_w: int, box_xyxy: tuple[int, int, int, int]) -> dict:
    """Synthetic mapper input: solid-grey image + one rectangular polygon.

    Uses ``__image`` to hand the array straight to the mapper without
    going through ``read_image``; mirrors how the multi-sample augs pass
    a composed canvas to the mapper.
    """
    img = np.full((image_h, image_w, 3), 128, dtype=np.uint8)
    x0, y0, x1, y1 = box_xyxy
    polygon = [x0, y0, x1, y0, x1, y1, x0, y1]
    return {
        "file_name": "_synthetic_",
        "image_id": 0,
        "height": image_h,
        "width": image_w,
        "__image": img,
        "annotations": [
            {
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": [polygon],
            }
        ],
    }


def _bitmask_mapper(short_edge: int) -> DatasetMapper:
    return DatasetMapper(
        [ResizeShortestEdge((short_edge,), max_size=1333)],
        is_train=True,
        mask_format="bitmask",
    )


def test_mapper_bitmask_polygon_square_canvas_downscale() -> None:
    """Mosaic-style 1024×1024 canvas → ResizeShortestEdge((672,)) → 672×672.

    Regression for the case where a polygon segmentation in the canvas
    coordinate frame must be rasterised at canvas size, then resized
    through the transform list to the post-aug shape.
    """
    dd = _polygon_dict(1024, 1024, (100, 100, 300, 300))
    out = _bitmask_mapper(short_edge=672)(dd)
    inst = out["instances"]
    assert isinstance(inst.gt_masks, BitMasks)
    assert inst.gt_masks.tensor.shape == (1, 672, 672)
    assert inst.gt_masks.tensor.any()


def test_mapper_bitmask_polygon_nonsquare_upscale() -> None:
    """480×640 source → ResizeShortestEdge((672,)) → 672×896.

    Regression for the plain (no multi-sample aug) path: a real COCO
    polygon annotation in source-image coords must rasterise at source
    dims, not at post-aug dims.
    """
    dd = _polygon_dict(480, 640, (50, 50, 200, 200))
    out = _bitmask_mapper(short_edge=672)(dd)
    inst = out["instances"]
    assert isinstance(inst.gt_masks, BitMasks)
    assert inst.gt_masks.tensor.shape == (1, 672, 896)
    assert inst.gt_masks.tensor.any()
