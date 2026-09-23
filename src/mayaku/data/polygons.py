"""Instance polygons that travel with a box table, and their stride-8 raster.

The mask pipeline is polygon-native: points survive mosaic, affine and flip
as one vectorised op each, and are rasterised once, at the end, onto the
stride-8 grid the mask head predicts on.
"""

import cv2
import numpy as np

from mayaku.data.geometry import apply_affine
from mayaku.model.aux import MASK_STRIDE

# `Polys.raster` uses fillPoly's `shift=3`, which is log2 of this stride
assert MASK_STRIDE == 8


def rle_to_polygons(rle):
    """COCO RLE ({'size': [h, w], 'counts': str | list | bytes}) -> list of
    flat [x, y, x, y, ...] polygons, one per external contour.

    Uncompressed counts (a list) are packed to compressed RLE first; a
    compressed string is encoded to bytes, which is what pycocotools decodes.
    """
    from pycocotools import mask as mask_util

    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, list):
        r = mask_util.frPyObjects(rle, h, w)
    elif isinstance(counts, str):
        r = {"size": rle["size"], "counts": counts.encode()}
    else:
        r = rle
    m = mask_util.decode(r)
    if m.ndim == 3:
        m = m[..., 0]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(np.float32) for c in cnts if len(c) >= 3]


class Polys:
    """Instance polygons for one sample.

    One flat (P, 2) point array plus two offset tables: part `p` is
    `xy[part_off[p]:part_off[p+1]]`, instance `i` owns parts
    `inst_off[i]:inst_off[i+1]`. `has` marks instances that carry a mask at
    all (crowd regions and box-only data do not). Every geometric change a
    box table goes through has a method here that takes the same numbers, so
    a mask cannot drift from its box.
    """

    __slots__ = ("has", "inst_off", "part_off", "xy")

    def __init__(self, xy, part_off, inst_off, has):
        self.xy, self.part_off, self.inst_off, self.has = xy, part_off, inst_off, has

    @classmethod
    def from_coco(cls, segs):
        """A list (one per instance) of COCO segmentations, or None for an
        instance without one. A segmentation is a polygon list
        ([[x, y, ...], ...]) or an RLE dict; RLE is decoded to polygons so
        both flow through the same point pipeline."""
        xy, part_off, inst_off, has = [], [0], [0], []
        for seg in segs:
            if isinstance(seg, dict):
                seg = rle_to_polygons(seg)
            parts = [np.asarray(p, np.float32).reshape(-1, 2) for p in (seg or []) if len(p) >= 6]
            for p in parts:
                xy.append(p)
                part_off.append(part_off[-1] + len(p))
            inst_off.append(inst_off[-1] + len(parts))
            has.append(bool(parts))
        return cls(np.concatenate(xy) if xy else np.zeros((0, 2), np.float32),
                   np.asarray(part_off, np.int32), np.asarray(inst_off, np.int32),
                   np.asarray(has, bool))

    def __len__(self):
        return len(self.has)

    def scaled(self, r, offset):
        """A new Polys scaled by `r` and shifted by `offset` (x, y); the
        offset tables are shared, they are never written."""
        return Polys(self.xy * r + np.float32(tuple(offset)), self.part_off, self.inst_off, self.has)

    def affine(self, m):
        self.xy = apply_affine(self.xy, m)
        return self

    def flip(self, w):
        self.xy[:, 0] = w - self.xy[:, 0]
        return self

    def select(self, keep):
        """Instance subset, by the same boolean mask applied to the boxes."""
        keep = np.asarray(keep, bool)
        n_parts = np.diff(self.inst_off)           # parts per instance
        n_pts = np.diff(self.part_off)             # points per part
        keep_part = np.repeat(keep, n_parts)
        xy = self.xy[np.repeat(keep_part, n_pts)]
        part_off = np.concatenate(([0], np.cumsum(n_pts[keep_part]))).astype(np.int32)
        inst_off = np.concatenate(([0], np.cumsum(n_parts[keep]))).astype(np.int32)
        return Polys(xy, part_off, inst_off, self.has[keep])

    def cat(self, other):
        return Polys(np.concatenate((self.xy, other.xy)),
                     np.concatenate((self.part_off, other.part_off[1:] + self.part_off[-1])),
                     np.concatenate((self.inst_off, other.inst_off[1:] + self.inst_off[-1])),
                     np.concatenate((self.has, other.has)))

    def parts(self, i):
        return [self.xy[self.part_off[p]:self.part_off[p + 1]]
                for p in range(self.inst_off[i], self.inst_off[i + 1])]

    def raster(self, boxes, canvas):
        """Instance index map at stride 8 over an (H, W) canvas: cell value =
        row + 1 of the instance whose polygon covers the cell, 0 for none.

        Larger instances are painted first so a small object in front keeps
        its own cells (painter's order; the object behind loses them).
        `shift=3` makes fillPoly read integer pixel coordinates as eighths of
        a cell, so cell (i, j) covers pixels [8j, 8j+8): the stride-8 anchor
        convention. Rows above 254 cannot be encoded in uint8 and are marked
        mask-less. Returns (map, has).
        """
        h, w = canvas
        m = np.zeros((h // MASK_STRIDE, w // MASK_STRIDE), np.uint8)
        has = self.has.copy()
        if len(boxes) == 0 or not has.any():
            return m, has
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        for i in np.argsort(-area):
            if not has[i]:
                continue
            if i >= 255:
                has[i] = False
                continue
            pts = [np.round(p).astype(np.int32) for p in self.parts(i)]
            cv2.fillPoly(m, pts, int(i) + 1, shift=3)
        return m, has
