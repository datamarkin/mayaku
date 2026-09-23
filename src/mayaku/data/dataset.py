"""The COCO detection dataset: letterboxed samples on an (H, W) canvas, with
mosaic, affine, mixup, colour and flip augmentation, and optional instance
masks and keypoints carried through every step.

`aug=None` is the evaluation path: letterbox only, plus the meta needed to map
detections back to original coordinates. Any `Augment` turns on the training
path, which has no meta because four source images may share one sample.
"""

import functools
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from mayaku.data.augment import (
    apply_hsv,
    mixup,
    mosaic_tile,
    sample_affine,
    sample_hsv,
    sample_mosaic_centre,
    warp_image,
)
from mayaku.data.batch import to_tensor
from mayaku.data.coco import load_coco
from mayaku.data.geometry import (
    PAD,
    as_canvas,
    fit,
    flip_boxes,
    flip_kpts,
    letterbox,
    place,
    scale_shift_kpts,
    warp_boxes,
    warp_kpts,
)
from mayaku.data.polygons import Polys


class CocoDetection(Dataset):
    """Images plus boxes (and optionally masks / keypoints) from a COCO JSON.

    `canvas` is the network input: a side, or (H, W). Labels come out as the
    [class, x1, y1, x2, y2, has_mask?, keypoints...] rows the loss reads, in
    canvas pixels.
    """

    def __init__(self, images, annotations, canvas=640, aug=None, seed=0,
                 masks=False, kpt=0, kpt_annotations=None):
        self.canvas = as_canvas(canvas)
        self.aug = aug
        self.rng = random.Random(seed)
        self.masks, self.kpt = masks, kpt
        self.coco = load_coco(images, annotations, masks=masks, kpt=kpt,
                              kpt_annotations=kpt_annotations)

    # the fields the trainer, the evaluator and the tests read
    ann_path = property(lambda self: self.coco.ann_path)
    kpt_ann_path = property(lambda self: self.coco.kpt_ann_path)
    cat_ids = property(lambda self: self.coco.cat_ids)
    ids = property(lambda self: self.coco.ids)
    shapes = property(lambda self: self.coco.shapes)
    labels = property(lambda self: self.coco.labels)

    @property
    def nc(self):
        return len(self.coco.cat_ids)

    def __len__(self):
        return len(self.coco.ids)

    # ---------------------------------------------------------------- loading

    def read(self, i):
        """The image file, BGR, or None when it is missing or corrupt."""
        return cv2.imread(self.coco.files[i])

    def _read_any(self, i):
        """`read(i)`, resampling at random past unreadable files. Returns the
        index actually used and its image."""
        img = self.read(i)
        while img is None:
            i = self.rng.randrange(len(self))
            img = self.read(i)
        return i, img

    def load(self, i):
        """Image i letterboxed onto the canvas: (img, ratio, pad), or all None
        for an unreadable file."""
        img = self.read(i)
        if img is None:
            return None, None, None
        return letterbox(img, self.canvas)

    def labels_at(self, i, r, offset):
        """Image i's boxes, polygons and keypoints scaled by `r` and shifted by
        `offset`: where its resized image was pasted."""
        boxes = place(self.coco.labels[i], r, offset)
        polys = self.coco.polys[i].scaled(r, offset) if self.masks else None
        kp = scale_shift_kpts(self.coco.kpts[i], self.kpt, r, offset) if self.kpt else None
        return boxes, polys, kp

    # ----------------------------------------------------------- augmentation

    def mosaic(self, i):
        """Four images on a (2H, 2W) canvas around a jittered centre (Mosaic,
        Bochkovskiy et al. 2020, arXiv 2004.10934).

        Each image is resized to fit the canvas and pasted bare, the
        neighbour filling the rest of the quadrant. It is not letterboxed
        first: letterboxing every source image fills much of each sample with
        dead grey and delivers fewer objects per sample.
        """
        ch, cw = self.canvas
        centre = sample_mosaic_centre(self.rng, self.canvas)
        canvas = np.full((2 * ch, 2 * cw, 3), PAD, np.uint8)
        picks = [i] + [self.rng.randrange(len(self)) for _ in range(3)]
        parts = []
        for k, j in enumerate(picks):
            j, img = self._read_any(j)
            img, r = fit(img, self.canvas)
            nh, nw = img.shape[:2]
            (xa, xb, ya, yb), (sx, sy) = mosaic_tile(k, centre, (nw, nh), self.canvas)
            canvas[ya:yb, xa:xb] = img[sy:sy + (yb - ya), sx:sx + (xb - xa)]
            parts.append(self.labels_at(j, r, (xa - sx, ya - sy)))
        boxes, polys, kps = zip(*parts, strict=True)
        return (canvas, np.concatenate(boxes),
                functools.reduce(Polys.cat, polys) if self.masks else None,
                np.concatenate(kps) if self.kpt else None)

    def _single(self, i):
        """One image letterboxed, skipping unreadable files, with its labels."""
        i, img = self._read_any(i)
        img, r, pad = letterbox(img, self.canvas)
        return (img, *self.labels_at(i, r, pad))

    def _affine(self, img, boxes, polys, kp):
        m = sample_affine(self.rng, self.aug, self.canvas, img.shape[:2])
        img = warp_image(img, m, self.canvas)
        boxes, ok = warp_boxes(boxes, m, self.canvas)
        if self.masks:
            polys = polys.affine(m).select(ok)
        if self.kpt:
            kp = warp_kpts(kp, self.kpt, m, self.canvas)[ok]
        return img, boxes, polys, kp

    def train_item(self, i):
        """Mosaic or a single image, then crop / scale, mixup, colour, flip."""
        rng, aug = self.rng, self.aug
        sample = self.mosaic(i) if rng.random() < aug.mosaic else self._single(i)
        img, boxes, polys, kp = self._affine(*sample)
        if aug.mixup and rng.random() < aug.mixup:
            # the partner is one image, not a second mosaic, which would
            # double the decode cost that dominates this pipeline
            other, ob, op, okp = self._affine(*self._single(rng.randrange(len(self))))
            # lambda ~ Beta(32, 32): close to an even blend
            img = mixup(img, other, rng.betavariate(32.0, 32.0))
            boxes = np.concatenate((boxes, ob))
            polys = polys.cat(op) if self.masks else None
            kp = np.concatenate((kp, okp)) if self.kpt else None
        img = apply_hsv(img, sample_hsv(rng, aug.hsv))
        if rng.random() < aug.hflip:
            w = self.canvas[1]
            img = img[:, ::-1]
            flip_boxes(boxes, w)
            if self.masks:
                polys.flip(w)
            if self.kpt:
                kp = flip_kpts(kp, self.kpt, w, self.coco.kpt_flip_pairs)
        return img, self._table(boxes, polys, kp)

    def _table(self, boxes, polys, kp):
        """The sample's label table and its stride-8 instance raster: columns
        [class, x1, y1, x2, y2] then, when carried, has_mask and the 3K
        keypoint values (`mayaku.data.batch.split_extras` order). The
        raster's value at a cell is row + 1."""
        cols, raster = [boxes], None
        if self.masks:
            raster, has = polys.raster(boxes, self.canvas)
            cols.append(has[:, None].astype(np.float32))
        if self.kpt:
            cols.append(kp)
        return (np.concatenate(cols, 1) if len(cols) > 1 else boxes), raster

    def __getitem__(self, i):
        if self.aug is None:
            # the same letterbox as training, so objects are learned and
            # scored at one scale
            img, r, pad = self.load(i)
            assert img is not None, "eval image missing: %s" % self.coco.files[i]
            meta = {"id": self.coco.ids[i], "ratio": r, "pad": pad, "shape": self.coco.shapes[i]}
            boxes, raster = self._table(*self.labels_at(i, r, pad))
        else:
            img, (boxes, raster) = self.train_item(i)
            meta = None
        out = {"img": to_tensor(img), "labels": torch.from_numpy(boxes), "meta": meta}
        if raster is not None:
            out["masks"] = torch.from_numpy(raster)
        return out
