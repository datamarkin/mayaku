"""A synthetic COCO directory small enough to build in a second.

Every object is a filled rectangle of its class colour, its mask is the
rectangle itself and its three keypoints are its top-left corner, centre and
bottom-right corner, so boxes, masks and keypoints that survive the pipeline
can all be checked against each other and against pixels.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np

from mayaku.data import CocoDetection

SYNTH_KPT = 3   # keypoints per synthetic object


def synthetic_coco(root, n=12, nc=4, seed=0):
    """Write `n` images and an instances.json under `root`; return its path."""
    rng = random.Random(seed)
    os.makedirs(root, exist_ok=True)
    images, anns = [], []
    for i in range(n):
        h, w = rng.choice([(360, 480), (480, 360), (400, 400), (300, 500)])
        img = np.full((h, w, 3), 40, np.uint8)
        # one object per image is under 32x32, so AP-S has a bucket to report on
        for k in range(rng.randint(2, 5)):
            bw, bh = ((12, 12) if k == 0 else
                      (rng.randint(20, w // 3), rng.randint(20, h // 3)))
            x, y = rng.randint(0, w - bw), rng.randint(0, h - bh)
            c = rng.randrange(nc)
            img[y:y + bh, x:x + bw] = [60 * c + 40, 255 - 50 * c, 30 * c]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": c + 1,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
                         "segmentation": [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]],
                         "keypoints": [x, y, 2, x + bw / 2, y + bh / 2, 2, x + bw, y + bh, 2],
                         "num_keypoints": 3})
        cv2.imwrite(os.path.join(root, "%03d.jpg" % i), img)
        images.append({"id": i, "file_name": "%03d.jpg" % i, "height": h, "width": w})
    path = os.path.join(root, "instances.json")
    with open(path, "w") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": c + 1, "name": "c%d" % c,
                                   "keypoints": ["tl", "ctr", "br"],
                                   "skeleton": [[1, 2], [2, 3]]}
                                  for c in range(nc)]}, f)
    return path


def fixture(root, n=12, canvas=320, aug=None, seed=0, masks=False, kpt=0):
    """A `CocoDetection` over a freshly written synthetic set in `root`."""
    ann = synthetic_coco(str(root), n=n, seed=seed)
    return CocoDetection(str(root), ann, canvas=canvas, aug=aug, seed=seed, masks=masks, kpt=kpt)
