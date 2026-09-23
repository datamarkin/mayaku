"""The COCO ruler and the reference host decode."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from mayaku.engine.evaluation import DEPLOY, coco_ap, evaluate, postprocess, to_coco, to_coco_segm
from mayaku.model import STRIDES, TINY, Detector

from ._coco_fixture import fixture


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return fixture(tmp_path_factory.mktemp("coco"))


def _as_detections(labels):
    """(n, 5) [class, x1, y1, x2, y2] -> (n, 6) detections scored 1.0."""
    return torch.cat((labels[:, 1:5], torch.ones(len(labels), 1), labels[:, :1]), 1)


def _original_frame(ds, i):
    return {"id": ds.ids[i], "ratio": 1.0, "pad": (0, 0), "shape": ds.shapes[i]}


def test_annotations_as_detections_score_ap_1(ds) -> None:
    raw = []
    for i in range(len(ds)):
        raw += to_coco(_as_detections(torch.from_numpy(ds.labels[i])), _original_frame(ds, i), ds.cat_ids)
    assert coco_ap(ds.ann_path, raw)["AP"] > 0.999


def test_rasterised_boxes_score_segm_ap_1(ds) -> None:
    res = []
    for i in range(len(ds)):
        h, w = ds.shapes[i]
        labels = torch.from_numpy(ds.labels[i])
        masks = torch.zeros(len(labels), h, w, dtype=torch.bool)
        for k, (x1, y1, x2, y2) in enumerate(labels[:, 1:5].int().tolist()):
            masks[k, y1:y2, x1:x2] = True
        res += to_coco_segm(_as_detections(labels), masks, _original_frame(ds, i), ds.cat_ids)
    assert coco_ap(ds.ann_path, res, "segm")["AP"] > 0.999


def test_letterbox_frame_round_trips_through_the_ruler(ds) -> None:
    """Catches a pad or ratio applied in the wrong direction, which would
    otherwise show up as a mysterious AP-S deficit."""
    boxed = []
    for i in range(len(ds)):
        s = ds[i]
        boxed += to_coco(_as_detections(s["labels"]), s["meta"], ds.cat_ids)
    assert coco_ap(ds.ann_path, boxed)["AP"] > 0.99
    # a known displacement costs AP without costing AP50
    nudged = [dict(r, bbox=[r["bbox"][0] + 0.12 * r["bbox"][2], r["bbox"][1] + 0.12 * r["bbox"][3],
                            r["bbox"][2], r["bbox"][3]]) for r in boxed]
    moved = coco_ap(ds.ann_path, nudged)
    assert moved["AP50"] > 0.999 and moved["AP"] < 0.9


def _planted(nc, reg_max, hw, cell, bin_):
    """Head outputs on an (H, W) canvas with one object lit at the finest
    stride: class 1 at `cell` (row, col), DFL mass on `bin_` for every side,
    so its box is the anchor centre plus and minus `bin_` strides."""
    (h, w), (row, col) = hw, cell
    preds = []
    for s in STRIDES:
        preds += [torch.full((1, nc, h // s, w // s), -10.0),
                  torch.zeros(1, 4 * reg_max, h // s, w // s)]
    preds[0][0, 1, row, col] = 10.0
    preds[1][0, bin_::reg_max, row, col] = 10.0
    cx, cy, reach = (col + 0.5) * STRIDES[0], (row + 0.5) * STRIDES[0], bin_ * STRIDES[0]
    return preds, torch.tensor([cx - reach, cy - reach, cx + reach, cy + reach, 1.0, 1.0])


@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize(("hw", "cell"), [((64, 64), (2, 3)), ((64, 128), (2, 11))])
def test_postprocess_decodes_a_planted_object(multi_label, hw, cell) -> None:
    """Anchor indices run row-major per level, so the lit cell is anchor
    row * (W / 8) + col, on square and rectangular canvases alike."""
    nc, reg_max = 4, 8
    preds, want = _planted(nc, reg_max, hw, cell, 2)
    d = dataclasses.replace(DEPLOY, conf=0.5, multi_label=multi_label)
    dets, idxs, _, _ = postprocess(preds, nc, reg_max, d)
    assert len(dets[0]) == 1 and idxs[0].tolist() == [cell[0] * (hw[1] // 8) + cell[1]]
    assert (dets[0][0] - want).abs().max().item() < 0.05


def test_evaluate_runs_end_to_end(ds) -> None:
    stats = evaluate(Detector(TINY, ds.nc), ds, batch=4, d=dataclasses.replace(DEPLOY, conf=0.05))
    assert stats["AP"] < 0.05     # untrained: what is under test is the plumbing


@pytest.mark.parametrize("canvas", [320, (256, 384)])
def test_evaluate_with_masks_and_keypoints(tmp_path, canvas) -> None:
    ev = fixture(tmp_path, canvas=canvas, masks=True, kpt=3)
    m = Detector(dataclasses.replace(TINY, seg=True, kpt=3), ev.nc).eval()
    out = evaluate(m, ev, batch=4)
    assert "segm_AP" in out and "kpt_AP" in out
