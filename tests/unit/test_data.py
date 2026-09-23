"""The v3 data pipeline on square and rectangular canvases: letterbox round
trip, augmentation invariants, masks and keypoints riding along, batching."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest
import torch

from mayaku.data import CLEAN_AUG, DEFAULT_AUG, Augment, CocoDetection, batch_to, collate
from mayaku.data.batch import multiscale_sizes, rescale_batch, to_tensor
from mayaku.data.coco import COCO_FLIP_PAIRS, flip_pairs
from mayaku.data.geometry import keep, letterbox, unletterbox
from mayaku.data.polygons import Polys

from ._coco_fixture import SYNTH_KPT, fixture, synthetic_coco

STRONG_AUG = Augment(scale=(0.1, 2.0), mixup=0.15)
CANVASES = [320, (256, 384), (384, 256)]


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco")
    return str(root), synthetic_coco(str(root))


def _dataset(coco, **kw):
    return CocoDetection(coco[0], coco[1], **kw)


@pytest.mark.parametrize("canvas", CANVASES)
def test_letterbox_round_trip(coco, canvas) -> None:
    val = _dataset(coco, canvas=canvas)
    h, w = val.canvas
    worst = 0.0
    for i in range(len(val)):
        s = val[i]
        assert s["img"].shape == (3, h, w) and s["img"].dtype == torch.uint8
        m = s["meta"]
        back = unletterbox(s["labels"][:, 1:5], m["ratio"], m["pad"], m["shape"])
        if len(back):
            worst = max(worst, (back - torch.from_numpy(val.labels[i][:, 1:5])).abs().max().item())
    assert worst < 1.0


def test_letterbox_geometry_on_a_rectangle() -> None:
    img, r, pad = letterbox(np.zeros((300, 500, 3), np.uint8), (256, 384))
    assert img.shape == (256, 384, 3)
    assert r == min(256 / 300, 384 / 500) and pad == (0, (256 - 230) // 2)


@pytest.mark.parametrize("canvas", CANVASES)
@pytest.mark.parametrize("aug", [DEFAULT_AUG, STRONG_AUG], ids=["default", "strong"])
def test_training_samples_stay_on_the_canvas(coco, canvas, aug) -> None:
    ds = _dataset(coco, canvas=canvas, aug=aug, seed=1)
    h, w = ds.canvas
    assert ds[0]["meta"] is None
    objs = empty = 0
    for i in range(48):
        s = ds[i % len(ds)]
        b = s["labels"]
        assert s["img"].shape == (3, h, w)
        objs += len(b)
        empty += len(b) == 0
        if len(b):
            assert b[:, 1].min() >= 0 and b[:, 3].max() <= w
            assert b[:, 2].min() >= 0 and b[:, 4].max() <= h
            assert (b[:, 3] - b[:, 1]).min() > 0 and (b[:, 4] - b[:, 2]).min() > 0
    # strong reaches scale 0.1, which empties some samples; not a third of them
    assert empty < 48 // 3 and objs / 48 > 1.5


@pytest.mark.parametrize("canvas", CANVASES)
def test_quiet_clean_stage_is_a_plain_letterbox(coco, canvas) -> None:
    quiet = dataclasses.replace(CLEAN_AUG, hflip=0.0, hsv=(0.0, 0.0, 0.0))
    ds, val = _dataset(coco, canvas=canvas, aug=quiet), _dataset(coco, canvas=canvas)
    a = ds[0]
    img, r, pad = val.load(0)
    plain, _, _ = val.labels_at(0, r, pad)
    assert torch.equal(a["labels"], torch.from_numpy(plain))
    assert torch.equal(a["img"], to_tensor(img))


def test_keep_drops_slivers() -> None:
    moved = np.float32([[0, 0, 100, 100], [-90, 0, 10, 100], [0, 0, 1, 100]])
    assert list(keep(moved, moved.copy().clip(0, 320))) == [True, False, False]


def test_collate_and_batch_to(coco) -> None:
    val = _dataset(coco, canvas=320)
    imgs, targets, metas, masks = collate([val[i] for i in range(4)])
    assert imgs.shape == (4, 3, 320, 320) and targets.shape[1] == 6 and masks is None
    floats = batch_to(imgs, "cpu")
    assert floats.dtype == torch.float32 and 0.0 <= floats.min() and floats.max() <= 1.0
    assert set(targets[:, 0].tolist()) <= {0.0, 1.0, 2.0, 3.0}
    assert len(metas) == 4 and metas[0]["id"] == val.ids[0]


@pytest.mark.parametrize("canvas", CANVASES)
@pytest.mark.parametrize("aug", [None, DEFAULT_AUG, STRONG_AUG], ids=["val", "default", "strong"])
def test_masks_and_keypoints_follow_their_boxes(coco, canvas, aug) -> None:
    """Every synthetic mask is its box and its keypoints are the box's corners
    and centre, through every augmentation."""
    ds = _dataset(coco, canvas=canvas, aug=aug, seed=2, masks=True, kpt=SYNTH_KPT)
    h, w = ds.canvas
    worst_iou, worst_kp = 1.0, 0.0
    for i in range(32):
        s = ds[i % len(ds)]
        t, m = s["labels"].numpy(), s["masks"].numpy()
        assert t.shape[1] == 6 + 3 * SYNTH_KPT and m.shape == (h // 8, w // 8)
        for row in range(len(t)):
            if not t[row, 5]:
                continue
            x1, y1, x2, y2 = t[row, 1:5]
            box = np.zeros_like(m)
            corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            cv2.fillPoly(box, [np.round(corners).astype(np.int32)], 1, shift=3)
            inst = m == row + 1
            covered = box.astype(bool) & ((m == 0) | inst)   # minus nearer instances
            union = (inst | covered).sum()
            if union == 0:
                continue                                      # a sub-cell sliver
            worst_iou = min(worst_iou, (inst & covered).sum() / union)
            kp = t[row, 6:].reshape(SYNTH_KPT, 3)
            vis = kp[:, 2] > 0
            assert not vis.any() or ((kp[vis, 0] >= 0).all() and (kp[vis, 0] <= w).all()
                                     and (kp[vis, 1] >= 0).all() and (kp[vis, 1] <= h).all())
            interior = x1 > 1 and y1 > 1 and x2 < w - 1 and y2 < h - 1
            if interior and vis.all():
                ref = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2],
                                  [(x1 + x2) / 2, (y1 + y2) / 2]])
                worst_kp = max(worst_kp, np.abs(kp[:, None, :2] - ref[None]).max(-1).min(-1).max())
    assert worst_iou > 0.85
    assert worst_kp <= 8.0      # a clipped edge can move a corner by one cell


@pytest.mark.parametrize("canvas", CANVASES)
def test_rescale_batch_moves_boxes_keypoints_and_raster(coco, canvas) -> None:
    ds = _dataset(coco, canvas=canvas, aug=DEFAULT_AUG, seed=2, masks=True, kpt=SYNTH_KPT)
    imgs, t, _, m = collate([ds[i] for i in range(4)])
    imgs = imgs.float() / 255
    h, w = ds.canvas
    size = (h // 2, w // 2)
    ri, rt, rm = rescale_batch(imgs, t.clone(), m.clone(), size, seg=True, kpt=SYNTH_KPT)
    assert ri.shape == (4, 3, *size) and rm.shape == (4, size[0] // 8, size[1] // 8)
    rx, ry = size[1] / w, size[0] / h
    assert torch.allclose(rt[:, [2, 4]], t[:, [2, 4]] * rx, atol=1e-3)
    assert torch.allclose(rt[:, [3, 5]], t[:, [3, 5]] * ry, atol=1e-3)
    k0, k1 = t[:, 7:].view(-1, SYNTH_KPT, 3), rt[:, 7:].view(-1, SYNTH_KPT, 3)
    assert torch.allclose(k1[..., 0], k0[..., 0] * rx, atol=1e-3)
    assert torch.allclose(k1[..., 1], k0[..., 1] * ry, atol=1e-3)
    assert torch.equal(k1[..., 2], k0[..., 2])
    assert set(rm.unique().tolist()) <= set(m.unique().tolist())
    assert rescale_batch(imgs, t, m, (h, w), seg=True, kpt=SYNTH_KPT)[0] is imgs


def test_multiscale_sizes() -> None:
    assert multiscale_sizes(512, 800) == list(range(512, 801, 32))


def test_flip_pairs() -> None:
    names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    assert flip_pairs(names, 5) == ((1, 2), (3, 4))
    coco17 = ["nose"] + [s + p for p in ("eye", "ear", "shoulder", "elbow", "wrist",
                                         "hip", "knee", "ankle") for s in ("left_", "right_")]
    assert flip_pairs(coco17, 17) == COCO_FLIP_PAIRS
    assert flip_pairs([], 17) == COCO_FLIP_PAIRS
    assert flip_pairs(["tl", "ctr", "br"], 3) == ()


def test_rle_segmentation_becomes_a_polygon() -> None:
    from pycocotools import mask as mask_util

    m = np.zeros((40, 60), np.uint8)
    m[10:30, 20:50] = 1
    rle = mask_util.encode(np.asfortranarray(m))
    rle["counts"] = rle["counts"].decode("ascii")
    p = Polys.from_coco([rle, None])
    assert list(p.has) == [True, False]
    xy = p.parts(0)[0]
    assert xy[:, 0].min() == 20 and xy[:, 0].max() == 49
    assert xy[:, 1].min() == 10 and xy[:, 1].max() == 29


def test_unreadable_images_are_skipped_in_training(tmp_path) -> None:
    ds = fixture(tmp_path, n=6, canvas=160, aug=DEFAULT_AUG, seed=4)
    (tmp_path / "000.jpg").write_bytes(b"not a jpeg")
    for i in range(12):
        assert ds[i % len(ds)]["img"].shape == (3, 160, 160)
