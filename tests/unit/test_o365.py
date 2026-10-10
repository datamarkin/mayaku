"""The Objects365 memory-mapped label cache against the COCO path it replaces."""

from __future__ import annotations

import json
import pickle

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("ijson")
from pycocotools import mask as mask_util

from mayaku.data import DEFAULT_AUG, CocoDetection
from mayaku.data.o365 import (
    build_cache,
    build_poly_cache,
    o365_train_labels,
    o365_val_labels,
)


def _tiny_o365(root):
    """Three images in the Objects365 layout: images under images/v1/patch0,
    a boxes JSON with sparse category ids, one SAM RLE JSON per image, and
    the same annotations with inline RLE as a COCO file."""
    img_root, sam = root / "train", root / "sam"
    (img_root / "images/v1/patch0").mkdir(parents=True)
    (sam / "patch0").mkdir(parents=True)
    images, anns, aid = [], [], 1
    for k in range(3):
        h, w = 400, 500
        img = np.full((h, w, 3), 40, np.uint8)
        rel = "images/v1/patch0/objects365_v1_%08d.jpg" % k
        segs = {}
        for j in range(2):
            x, y, bw, bh = 40 + 120 * j, 60 + 20 * k, 90, 110
            img[y:y + bh, x:x + bw] = 200
            m = np.zeros((h, w), np.uint8, order="F")
            m[y:y + bh, x:x + bw] = 1
            r = mask_util.encode(m)
            rle = {"size": [h, w], "counts": r["counts"].decode()}
            segs[str(aid)] = rle
            anns.append({"id": aid, "image_id": k, "category_id": 3 * j + 2, "iscrowd": 0,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "segmentation": rle})
            aid += 1
        cv2.imwrite(str(img_root / rel), img)
        images.append({"id": k, "height": h, "width": w, "file_name": rel})
        (sam / "patch0" / ("objects365_v1_%08d.json" % k)).write_text(
            json.dumps({"image_id": k, "segmentations": segs}))
    cats = [{"id": 2, "name": "a"}, {"id": 5, "name": "b"}]
    boxes = root / "boxes.json"
    boxes.write_text(json.dumps({"images": images, "categories": cats, "annotations":
                                 [{k: v for k, v in a.items() if k != "segmentation"} for a in anns]}))
    coco = root / "coco.json"
    coco.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return boxes, img_root, sam, coco


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("o365")
    boxes, img_root, sam, coco = _tiny_o365(root)
    cache = root / "cache"
    build_cache(str(boxes), str(img_root), str(cache), str(sam))
    build_poly_cache(str(cache), workers=1, chunk=2)
    return str(cache), str(img_root), str(coco)


def test_cache_matches_the_coco_path(tiny) -> None:
    cache, img_root, coco = tiny
    ref = CocoDetection(img_root, coco, 320, masks=True)
    lab = pickle.loads(pickle.dumps(o365_train_labels(cache, masks=True)))   # the spawn path
    ds = CocoDetection(None, None, 320, masks=True, coco=lab)
    assert len(ds) == len(ref) == 3 and ds.cat_ids == ref.cat_ids == [2, 5]
    for i in range(3):
        a, b = ref[i], ds[i]
        assert a["meta"] == b["meta"]
        for k in ("img", "labels", "masks"):
            assert torch.equal(a[k], b[k]), (i, k)


def test_cache_trains_and_limits(tiny) -> None:
    cache, _, _ = tiny
    lab = o365_train_labels(cache, masks=True, limit=2)
    ds = CocoDetection(None, None, 320, aug=DEFAULT_AUG, masks=True, coco=lab, seed=1)
    assert len(ds) == 2 and len(lab.shapes) == 2
    for i in range(4):
        s = ds[i % 2]
        assert s["labels"].shape[1] == 6 and s["masks"].shape == (320, 320)
    # samples are fresh copies: writing one never reaches the cache
    lab.labels[0][:] = -1
    assert (lab.labels[0] >= 0).all()


def test_val_labels_splice_the_version(tmp_path) -> None:
    (tmp_path / "v2/patch3").mkdir(parents=True)
    ann = tmp_path / "val.json"
    ann.write_text(json.dumps({
        "images": [{"id": 1, "height": 4, "width": 4, "file_name": "patch3/objects365_v2_1.jpg"}],
        "annotations": [], "categories": [{"id": 1, "name": "a"}]}))
    coco = o365_val_labels(ann, tmp_path)
    assert coco.files[0] == str(tmp_path / "v2/patch3/objects365_v2_1.jpg")
