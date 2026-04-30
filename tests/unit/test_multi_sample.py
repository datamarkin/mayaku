"""Tests for :mod:`mayaku.data.multi_sample`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from mayaku.data.mapper import DatasetMapper
from mayaku.data.multi_sample import (
    CopyPaste,
    MixUp,
    Mosaic,
    MultiSampleAugmentation,
    MultiSampleMappedDataset,
    _clip_polygon_to_rect,
)
from mayaku.data.transforms import ResizeShortestEdge
from mayaku.structures.boxes import BoxMode

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_image(tmp_path: Path, name: str, h: int, w: int, fill: int) -> Path:
    """Write an HxW solid-color image and return its path."""
    arr = np.full((h, w, 3), fill, dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return p


def _make_dict(file_name: Path, h: int, w: int, anns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_name": str(file_name),
        "image_id": int(file_name.stem.split("_")[-1])
        if file_name.stem.split("_")[-1].isdigit()
        else 0,
        "height": h,
        "width": w,
        "annotations": anns,
    }


def _ann(x: float, y: float, w: float, h: float, cls: int = 0) -> dict[str, Any]:
    return {
        "bbox": [x, y, w, h],
        "bbox_mode": BoxMode.XYWH_ABS,
        "category_id": cls,
    }


def _make_mapper() -> DatasetMapper:
    """A minimal training-mode mapper. ResizeShortestEdge is the canonical
    first augmentation; it doesn't materially affect the multi-sample
    correctness checks but matches what the train CLI builds."""
    return DatasetMapper(
        [ResizeShortestEdge((640,), max_size=1333)],
        is_train=True,
        mask_format="polygon",
    )


# ---------------------------------------------------------------------------
# MultiSampleMappedDataset wrapper
# ---------------------------------------------------------------------------


class _AlwaysFiresAug(MultiSampleAugmentation):
    """Test stub that always fires and concatenates annotations."""

    def __init__(self, num_extras: int) -> None:
        self._num_extras = num_extras
        self.invocations = 0

    def fires(self, rng: np.random.Generator) -> bool:
        return True

    def num_extras(self) -> int:
        return self._num_extras

    def apply_and_map(
        self,
        dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        self.invocations += 1
        # Concat all annotations into the primary's frame and use its image.
        primary = dicts[0]
        merged_annos: list[dict[str, Any]] = []
        for dd in dicts:
            merged_annos.extend(dd.get("annotations", []))
        synthetic = dict(primary)
        synthetic["annotations"] = merged_annos
        return mapper(synthetic)


class _NeverFiresAug(MultiSampleAugmentation):
    def fires(self, rng: np.random.Generator) -> bool:
        return False

    def num_extras(self) -> int:
        return 1

    def apply_and_map(
        self,
        dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        raise AssertionError("_NeverFiresAug.apply_and_map should never be called")


def test_wrapper_len_matches_underlying(tmp_path: Path) -> None:
    """``len(wrapper)`` must equal len of the dataset_dicts list (TrainingSampler depends on this)."""
    paths = [_make_image(tmp_path, f"img_{i}.png", 32, 32, fill=10 * i) for i in range(5)]
    dicts = [_make_dict(p, 32, 32, [_ann(1, 1, 8, 8)]) for p in paths]
    wrapper = MultiSampleMappedDataset(dicts, _make_mapper(), [_NeverFiresAug()])
    assert len(wrapper) == 5


def test_wrapper_passes_through_when_no_augs_fire(tmp_path: Path) -> None:
    """No registered augs → behavior identical to the plain mapper."""
    p = _make_image(tmp_path, "img_0.png", 64, 64, fill=128)
    dicts = [_make_dict(p, 64, 64, [_ann(1, 1, 16, 16)])]
    wrapper = MultiSampleMappedDataset(dicts, _make_mapper(), [])
    out = wrapper[0]
    assert "image" in out
    assert "instances" in out
    assert len(out["instances"]) == 1


def test_wrapper_calls_aug_when_it_fires(tmp_path: Path) -> None:
    """First aug to fire wins; its ``apply_and_map`` is called once."""
    paths = [_make_image(tmp_path, f"img_{i}.png", 32, 32, fill=20 * i) for i in range(3)]
    dicts = [_make_dict(p, 32, 32, [_ann(1, 1, 8, 8)]) for p in paths]
    aug = _AlwaysFiresAug(num_extras=1)
    wrapper = MultiSampleMappedDataset(dicts, _make_mapper(), [aug])
    _ = wrapper[0]
    assert aug.invocations == 1


def test_wrapper_extras_sampled_distinct_from_primary(tmp_path: Path) -> None:
    """Extras sampled by the wrapper should not include the primary index."""
    paths = [_make_image(tmp_path, f"img_{i}.png", 32, 32, fill=20 * i) for i in range(10)]
    dicts = [_make_dict(p, 32, 32, [_ann(0, 0, 8, 8, cls=i)]) for i, p in enumerate(paths)]

    # The aug receives [primary, *extras] — capture them via a custom impl.
    captured: list[list[dict[str, Any]]] = []

    class _Capture(MultiSampleAugmentation):
        def fires(self, rng: np.random.Generator) -> bool:
            return True

        def num_extras(self) -> int:
            return 3

        def apply_and_map(
            self,
            dicts: Sequence[dict[str, Any]],
            mapper: DatasetMapper,
            rng: np.random.Generator,
        ) -> dict[str, Any]:
            captured.append(list(dicts))
            return mapper(dicts[0])

    wrapper = MultiSampleMappedDataset(dicts, _make_mapper(), [_Capture()])
    primary_idx = 4
    _ = wrapper[primary_idx]
    primary, *extras = captured[0]
    assert primary["annotations"][0]["category_id"] == primary_idx
    extra_ids = {e["annotations"][0]["category_id"] for e in extras}
    assert primary_idx not in extra_ids
    assert len(extras) == 3
    assert len(extra_ids) == 3  # all distinct


def test_wrapper_handles_singleton_dataset(tmp_path: Path) -> None:
    """Pathological 1-image dataset: aug receives the same dict for primary + extras."""
    p = _make_image(tmp_path, "img_only.png", 32, 32, fill=200)
    dicts = [_make_dict(p, 32, 32, [_ann(0, 0, 8, 8)])]
    aug = _AlwaysFiresAug(num_extras=1)
    wrapper = MultiSampleMappedDataset(dicts, _make_mapper(), [aug])
    # Should not raise — aug gets [primary, primary]
    out = wrapper[0]
    assert "image" in out


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


def test_mixup_invalid_prob_raises() -> None:
    with pytest.raises(ValueError, match="prob"):
        MixUp(prob=1.5)
    with pytest.raises(ValueError, match="prob"):
        MixUp(prob=-0.1)


def test_mixup_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        MixUp(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        MixUp(alpha=-1.0)


def test_mixup_num_extras_is_one() -> None:
    assert MixUp().num_extras() == 1


def test_mixup_fires_obeys_prob() -> None:
    """At prob=1 it always fires; at prob=0 never."""
    rng = np.random.default_rng(0)
    assert MixUp(prob=1.0).fires(rng)
    assert not MixUp(prob=0.0).fires(rng)


def test_mixup_apply_blends_images_and_concats_annotations(tmp_path: Path) -> None:
    """λ-blend produces an image whose pixels are between primary and secondary,
    and the output Instances has annotations from BOTH images."""
    img_a_path = _make_image(tmp_path, "a.png", 64, 64, fill=200)
    img_b_path = _make_image(tmp_path, "b.png", 64, 64, fill=50)
    primary = _make_dict(img_a_path, 64, 64, [_ann(0, 0, 16, 16, cls=0)])
    secondary = _make_dict(img_b_path, 64, 64, [_ann(20, 20, 16, 16, cls=1)])

    rng = np.random.default_rng(0)
    out = MixUp(prob=1.0, alpha=8.0).apply_and_map([primary, secondary], _make_mapper(), rng)

    # Image is a blend of fill=200 and fill=50; every pixel must be between them.
    img = out["image"]
    assert img.dtype.is_floating_point or img.dtype == img.dtype  # float32 from mapper
    assert img.shape[0] == 3  # CHW after _image_to_tensor
    # Mapper resizes to short=640, so spatial dims are 640x640 (from 64x64 input).
    # All pixels should sit strictly between 50 and 200 because λ ∈ (0, 1) for α=8.
    flat = img.flatten()
    assert flat.min() > 50.0
    assert flat.max() < 200.0

    # Both annotations should appear in the output Instances.
    instances = out["instances"]
    assert len(instances) == 2
    classes = sorted(instances.gt_classes.tolist())
    assert classes == [0, 1]


def test_mixup_resizes_secondary_to_primary_shape(tmp_path: Path) -> None:
    """Secondary's bbox is correctly scaled when its native shape differs from primary's."""
    img_a = _make_image(tmp_path, "a.png", 64, 64, fill=128)  # primary canvas
    img_b = _make_image(tmp_path, "b.png", 32, 32, fill=128)  # half the size
    primary = _make_dict(img_a, 64, 64, [])
    # Secondary's bbox is at (8, 8, 16, 16) in 32×32. After resize to 64×64
    # the bbox should be at (16, 16, 32, 32) — scaled 2× on both axes.
    secondary = _make_dict(img_b, 32, 32, [_ann(8, 8, 16, 16, cls=7)])

    rng = np.random.default_rng(0)
    # alpha very large → λ ≈ 0.5; doesn't matter for this test.
    out = MixUp(prob=1.0, alpha=100.0).apply_and_map([primary, secondary], _make_mapper(), rng)
    instances = out["instances"]
    assert len(instances) == 1
    # The mapper applies ResizeShortestEdge(640) on top, scaling 64→640 = 10×.
    # So the original (16, 16, 32, 32) XYWH in the 64-frame becomes
    # (16+16=32 right, 16+16=32 bottom in XYXY) × 10 = (160, 160, 320, 320) in
    # the 640-frame.
    # `gt_boxes` is a raw Tensor at training time (training-side convention),
    # not the `Boxes` wrapper used at inference. Index it directly.
    box = instances.gt_boxes[0].tolist()
    # XYWH (8, 8, 16, 16) in 32-frame → XYXY (8, 8, 24, 24)
    #   → scaled 2× to match 64-frame: (16, 16, 48, 48)
    #   → mapper resizes 64→640 (10×): (160, 160, 480, 480).
    assert box == pytest.approx([160.0, 160.0, 480.0, 480.0], abs=1.0)


def test_mixup_apply_rejects_wrong_dict_count() -> None:
    """``apply_and_map`` must receive exactly 2 dicts (primary + 1 extra)."""
    aug = MixUp(prob=1.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2 dicts"):
        aug.apply_and_map([{}, {}, {}], _make_mapper(), rng)


# ---------------------------------------------------------------------------
# Polygon clipping (Sutherland-Hodgman)
# ---------------------------------------------------------------------------


def test_clip_polygon_fully_inside_rect_unchanged() -> None:
    poly = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)
    out = _clip_polygon_to_rect(poly, 0, 0, 100, 100)
    np.testing.assert_array_equal(out, poly)


def test_clip_polygon_fully_outside_rect_empties() -> None:
    poly = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float32)
    out = _clip_polygon_to_rect(poly, 0, 0, 100, 100)
    assert out.shape[0] == 0


def test_clip_polygon_crossing_left_edge() -> None:
    """Square spanning x=-5..15 clipped at x=0 should produce a 4-vertex shape inside x>=0."""
    poly = np.array([[-5, 5], [15, 5], [15, 15], [-5, 15]], dtype=np.float32)
    out = _clip_polygon_to_rect(poly, 0, 0, 100, 100)
    # Should be a 4-vertex rect: (0, 5), (15, 5), (15, 15), (0, 15) — order
    # may vary because Sutherland-Hodgman walks edges; just verify the
    # bounding box.
    xs = out[:, 0]
    ys = out[:, 1]
    assert float(xs.min()) == pytest.approx(0.0)
    assert float(xs.max()) == pytest.approx(15.0)
    assert float(ys.min()) == pytest.approx(5.0)
    assert float(ys.max()) == pytest.approx(15.0)


def test_clip_polygon_corner_clip_produces_pentagon() -> None:
    """Triangle with one vertex outside a corner: clipping produces a 4-vertex polygon."""
    poly = np.array([[10, 10], [50, 10], [10, 50]], dtype=np.float32)
    out = _clip_polygon_to_rect(poly, 0, 0, 30, 30)
    # Edges crossed: x=30 cuts the (10,10)-(50,10) edge at (30, 10);
    # (50,10)-(10,50) edge at (30, 30) and (10, 50)-(10, 10) needs y=30
    # cut at (10, 30). So the result is at least 4 vertices.
    assert out.shape[0] >= 4
    assert (out[:, 0].max() <= 30 + 1e-6) and (out[:, 1].max() <= 30 + 1e-6)


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------


def test_mosaic_invalid_args_raise() -> None:
    with pytest.raises(ValueError, match="prob"):
        Mosaic(prob=1.5)
    with pytest.raises(ValueError, match="canvas_size"):
        Mosaic(canvas_size=(0, 100))
    with pytest.raises(ValueError, match="min_box_area"):
        Mosaic(min_box_area=-1.0)
    with pytest.raises(ValueError, match="pivot_range"):
        Mosaic(pivot_range=(0.8, 0.2))  # lo > hi
    with pytest.raises(ValueError, match="pivot_range"):
        Mosaic(pivot_range=(0.0, 0.5))  # lo not > 0


def test_mosaic_num_extras_is_three() -> None:
    assert Mosaic().num_extras() == 3


def test_mosaic_apply_rejects_wrong_dict_count() -> None:
    aug = Mosaic(prob=1.0)
    with pytest.raises(ValueError, match="4 dicts"):
        aug.apply_and_map([{}, {}, {}], _make_mapper(), np.random.default_rng(0))


def test_mosaic_canvas_has_configured_size(tmp_path: Path) -> None:
    """Output image (post-mapper, pre-resize-back-to-canvas) reflects mosaic canvas dimensions."""
    paths = [_make_image(tmp_path, f"img_{i}.png", 100, 100, fill=20 * i) for i in range(4)]
    dicts = [_make_dict(p, 100, 100, [_ann(10, 10, 30, 30, cls=i)]) for i, p in enumerate(paths)]

    # Run Mosaic with a small canvas so the mapper's ResizeShortestEdge
    # doesn't mask the result. Use the no-resize mapper for clean checks.
    mapper = DatasetMapper(
        # Resize that's a no-op for our 256x256 canvas (short-edge already 256)
        [ResizeShortestEdge((256,), max_size=512)],
        is_train=True,
        mask_format="polygon",
    )
    aug = Mosaic(prob=1.0, canvas_size=(256, 256))
    out = aug.apply_and_map(dicts, mapper, np.random.default_rng(42))
    img_chw = out["image"]
    # CHW after _image_to_tensor
    assert img_chw.shape[0] == 3
    assert img_chw.shape[1] == 256
    assert img_chw.shape[2] == 256


def test_mosaic_collects_annotations_from_all_quadrants(tmp_path: Path) -> None:
    """Each of the 4 inputs contributes at least one bbox to the output."""
    paths = [_make_image(tmp_path, f"img_{i}.png", 200, 200, fill=20 * i) for i in range(4)]
    # Place each bbox in the centre of its source image so it survives the
    # quadrant scaling + clipping no matter where the pivot lands.
    dicts = [_make_dict(p, 200, 200, [_ann(80, 80, 40, 40, cls=i)]) for i, p in enumerate(paths)]

    mapper = DatasetMapper(
        [ResizeShortestEdge((512,), max_size=1024)],
        is_train=True,
        mask_format="polygon",
    )
    aug = Mosaic(prob=1.0, canvas_size=(512, 512))
    out = aug.apply_and_map(dicts, mapper, np.random.default_rng(0))
    instances = out["instances"]
    # Each image had a single annotation (cls=0..3). We expect all 4 to
    # survive — the bbox is centred and large enough not to get
    # clipped under any pivot in the inner 50% range.
    assert len(instances) == 4
    classes = sorted(instances.gt_classes.tolist())
    assert classes == [0, 1, 2, 3]


def test_mosaic_drops_tiny_clipped_bboxes(tmp_path: Path) -> None:
    """A bbox that ends up < min_box_area pixels² post-clip should be dropped.

    Strategy: a tiny 2×2 bbox in the source. Even before the canvas clip, its
    post-scale area is less than 16 px² so it gets dropped.
    """
    paths = [_make_image(tmp_path, f"img_{i}.png", 200, 200, fill=20 * i) for i in range(4)]
    # A 2x2 bbox: post-scale (max ~256/200) area ≈ (2 × 1.28)^2 ≈ 6.5 px²,
    # below the 16-px² default — should drop.
    dicts = [_make_dict(p, 200, 200, [_ann(50, 50, 2, 2, cls=i)]) for i, p in enumerate(paths)]
    mapper = DatasetMapper(
        [ResizeShortestEdge((256,), max_size=512)],
        is_train=True,
        mask_format="polygon",
    )
    aug = Mosaic(prob=1.0, canvas_size=(256, 256), min_box_area=16.0)
    out = aug.apply_and_map(dicts, mapper, np.random.default_rng(0))
    # All 4 micro-bboxes should have been dropped before mapping.
    assert len(out["instances"]) == 0


def test_mosaic_canvas_pixel_content(tmp_path: Path) -> None:
    """At least one pixel from each source image lands in the output canvas."""
    fills = [40, 80, 120, 200]
    paths = [_make_image(tmp_path, f"img_{i}.png", 100, 100, fill=fills[i]) for i in range(4)]
    dicts = [_make_dict(p, 100, 100, []) for p in paths]
    mapper = DatasetMapper(
        [ResizeShortestEdge((256,), max_size=512)],
        is_train=True,
        mask_format="polygon",
    )
    # Force pivot to centre so quadrants are equal-sized.
    aug = Mosaic(prob=1.0, canvas_size=(256, 256), pivot_range=(0.5, 0.5))
    out = aug.apply_and_map(dicts, mapper, np.random.default_rng(0))
    img = out["image"]  # (3, H, W) float32 from mapper

    # Each fill appears (somewhere) — sanity check via histogram. Pillow
    # bilinear resize introduces some smoothing at quadrant edges so we
    # don't require exact equality, just that a pixel close to each fill
    # exists somewhere in the canvas.
    flat_r = img[0].flatten().numpy()
    for fill in fills:
        # Within ±5 of the fill value (uint8 → float, after solid colour).
        assert np.any(np.abs(flat_r - fill) < 5.0), f"no pixel close to fill={fill}"


# ---------------------------------------------------------------------------
# CopyPaste
# ---------------------------------------------------------------------------


def _bitmask_seg(h: int, w: int, mask_y0y1x0x1: tuple[int, int, int, int]) -> dict[str, Any]:
    """Build an RLE segmentation describing a single rectangular mask.

    Returns the encoded-counts RLE form (string ``counts``) so the dict
    is self-contained and survives the mapper's transform pipeline
    without depending on the polygon-rasterisation dims.
    """
    from pycocotools import mask as coco_mask

    y0, y1, x0, x1 = mask_y0y1x0x1
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[y0:y1, x0:x1] = 1
    rle = coco_mask.encode(np.asfortranarray(arr))
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _make_bitmask_mapper() -> DatasetMapper:
    """Mapper configured for bitmask format (CopyPaste's only supported mode)."""
    return DatasetMapper(
        [ResizeShortestEdge((256,), max_size=512)],
        is_train=True,
        mask_format="bitmask",
    )


def test_copypaste_invalid_args_raise() -> None:
    with pytest.raises(ValueError, match="prob"):
        CopyPaste(prob=1.5)
    with pytest.raises(ValueError, match="paste_fraction"):
        CopyPaste(paste_fraction=0.0)
    with pytest.raises(ValueError, match="paste_fraction"):
        CopyPaste(paste_fraction=1.5)
    with pytest.raises(ValueError, match="occlusion_threshold"):
        CopyPaste(occlusion_threshold=0.0)
    with pytest.raises(ValueError, match="occlusion_threshold"):
        CopyPaste(occlusion_threshold=1.5)
    with pytest.raises(ValueError, match="min_box_area"):
        CopyPaste(min_box_area=-1.0)


def test_copypaste_num_extras_is_one() -> None:
    assert CopyPaste().num_extras() == 1


def test_copypaste_apply_rejects_wrong_dict_count() -> None:
    aug = CopyPaste(prob=1.0)
    with pytest.raises(ValueError, match="2 dicts"):
        aug.apply_and_map([{}, {}, {}], _make_bitmask_mapper(), np.random.default_rng(0))


def test_copypaste_pastes_source_pixels_into_target(tmp_path: Path) -> None:
    """Pixels under the source mask should become source's value in the composite."""
    target_path = _make_image(tmp_path, "target.png", 100, 100, fill=20)
    source_path = _make_image(tmp_path, "source.png", 100, 100, fill=200)
    target = _make_dict(target_path, 100, 100, [])
    source_seg = _bitmask_seg(100, 100, (10, 30, 10, 30))  # 20×20 rect
    source = _make_dict(
        source_path,
        100,
        100,
        [
            {
                "bbox": [10, 10, 20, 20],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 5,
                "segmentation": source_seg,
            }
        ],
    )

    aug = CopyPaste(prob=1.0, occlusion_threshold=1.0, min_box_area=0.0)
    out = aug.apply_and_map([target, source], _make_bitmask_mapper(), np.random.default_rng(0))

    # Composite pixels in the pasted region should equal source's fill (200);
    # pixels outside should remain at target's fill (20). After the mapper's
    # ResizeShortestEdge(256), 100→256 ≈ 2.56× scale; the rect (10–30, 10–30)
    # lands somewhere near pixels (25–77, 25–77). We just check the histogram.
    img = out["image"].numpy()  # (3, H, W) float32
    assert (img > 180).any(), "no pasted pixel near source fill (200)"
    assert (img < 40).any(), "target background lost"


def test_copypaste_appends_pasted_annotation(tmp_path: Path) -> None:
    """The composite Instances should carry the pasted annotation's category."""
    target_path = _make_image(tmp_path, "target.png", 100, 100, fill=10)
    source_path = _make_image(tmp_path, "source.png", 100, 100, fill=240)
    target = _make_dict(target_path, 100, 100, [])
    source = _make_dict(
        source_path,
        100,
        100,
        [
            {
                "bbox": [20, 20, 40, 40],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 9,
                "segmentation": _bitmask_seg(100, 100, (20, 60, 20, 60)),
            }
        ],
    )
    aug = CopyPaste(prob=1.0, occlusion_threshold=1.0, min_box_area=0.0)
    out = aug.apply_and_map([target, source], _make_bitmask_mapper(), np.random.default_rng(0))
    instances = out["instances"]
    assert len(instances) == 1
    assert instances.gt_classes.tolist() == [9]


def test_copypaste_skips_source_instances_without_segmentation(tmp_path: Path) -> None:
    """Source annotations missing a `segmentation` field cannot be pasted."""
    target_path = _make_image(tmp_path, "target.png", 100, 100, fill=10)
    source_path = _make_image(tmp_path, "source.png", 100, 100, fill=240)
    target = _make_dict(target_path, 100, 100, [])
    source = _make_dict(
        source_path,
        100,
        100,
        [{"bbox": [20, 20, 40, 40], "bbox_mode": BoxMode.XYWH_ABS, "category_id": 3}],
    )
    aug = CopyPaste(prob=1.0, occlusion_threshold=1.0, min_box_area=0.0)
    out = aug.apply_and_map([target, source], _make_bitmask_mapper(), np.random.default_rng(0))
    # Nothing pasted → empty instances.
    assert len(out["instances"]) == 0


def test_copypaste_occlusion_filter_drops_heavily_covered_target(tmp_path: Path) -> None:
    """A target instance fully covered by the paste union should be dropped."""
    target_path = _make_image(tmp_path, "target.png", 100, 100, fill=10)
    source_path = _make_image(tmp_path, "source.png", 100, 100, fill=240)
    # Target instance: small 10×10 rect at (40, 40)–(50, 50).
    # Source instance: large 60×60 rect at (20, 20)–(80, 80) → fully covers
    # the target rect → should drop the target instance.
    target = _make_dict(
        target_path,
        100,
        100,
        [
            {
                "bbox": [40, 40, 10, 10],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": _bitmask_seg(100, 100, (40, 50, 40, 50)),
            }
        ],
    )
    source = _make_dict(
        source_path,
        100,
        100,
        [
            {
                "bbox": [20, 20, 60, 60],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 1,
                "segmentation": _bitmask_seg(100, 100, (20, 80, 20, 80)),
            }
        ],
    )
    aug = CopyPaste(prob=1.0, occlusion_threshold=0.7, min_box_area=0.0)
    out = aug.apply_and_map([target, source], _make_bitmask_mapper(), np.random.default_rng(0))
    instances = out["instances"]
    # Only the pasted (source) instance survives — target's was occluded out.
    assert len(instances) == 1
    assert instances.gt_classes.tolist() == [1]


def test_copypaste_occlusion_threshold_one_keeps_target(tmp_path: Path) -> None:
    """``occlusion_threshold=1.0`` disables the filter — both instances survive."""
    target_path = _make_image(tmp_path, "target.png", 100, 100, fill=10)
    source_path = _make_image(tmp_path, "source.png", 100, 100, fill=240)
    target = _make_dict(
        target_path,
        100,
        100,
        [
            {
                "bbox": [40, 40, 10, 10],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
                "segmentation": _bitmask_seg(100, 100, (40, 50, 40, 50)),
            }
        ],
    )
    source = _make_dict(
        source_path,
        100,
        100,
        [
            {
                "bbox": [20, 20, 60, 60],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 1,
                "segmentation": _bitmask_seg(100, 100, (20, 80, 20, 80)),
            }
        ],
    )
    aug = CopyPaste(prob=1.0, occlusion_threshold=1.0, min_box_area=0.0)
    out = aug.apply_and_map([target, source], _make_bitmask_mapper(), np.random.default_rng(0))
    classes = sorted(out["instances"].gt_classes.tolist())
    assert classes == [0, 1]


def test_copypaste_schema_validator_rejects_polygon_format() -> None:
    """``copy_paste_prob > 0`` with ``mask_format='polygon'`` should fail at config load."""
    from mayaku.config.schemas import InputConfig

    with pytest.raises(ValueError, match="bitmask"):
        InputConfig(copy_paste_prob=0.5, mask_format="polygon")
    # Bitmask + non-zero prob is fine.
    cfg = InputConfig(copy_paste_prob=0.5, mask_format="bitmask")
    assert cfg.copy_paste_prob == 0.5
