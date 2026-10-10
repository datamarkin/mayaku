"""Instance polygons that travel with a box table, and their instance map.

The mask pipeline is polygon-native: points survive mosaic, affine and flip
as one vectorised op each, and are painted once, at the end, into an
instance-index map at the full canvas resolution. The loss averages that map
over each mask cell, so a target is the exact share of the cell an instance
covers, at whatever stride the mask head predicts on.

Coordinates are continuous pixels: pixel (r, c) is the square
[c, c + 1) x [r, r + 1) with its centre at (c + 0.5, r + 0.5), the convention
boxes, flips and resizes already use.
"""

import cv2
import numpy as np

from mayaku.data.geometry import apply_affine


def rle_to_polygons(rle):
    """COCO RLE ({'size': [h, w], 'counts': str | list | bytes}) -> list of
    flat [x, y, x, y, ...] polygons, one per external contour, around the
    outer edges of the mask's boundary pixels (holes are dropped), in
    continuous coordinates (`outline`).

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
    parts = [c.reshape(-1, 2) for c in cnts if len(c) >= 3]
    if not parts:
        return []
    off = np.concatenate(([0], np.cumsum([len(q) for q in parts])))
    return [q.reshape(-1) for q in np.split(outline(np.concatenate(parts), off), off[1:-1])]


def _rings(n, part_off):
    """Next and previous point index within each closed part of a CSR
    polygon table: (n,) and (n,) int arrays."""
    nxt, prv = np.arange(1, n + 1), np.arange(-1, n - 1)
    nxt[part_off[1:] - 1] = part_off[:-1]
    prv[part_off[:-1]] = part_off[1:] - 1
    return nxt, prv


def outline(idx, part_off):
    """Contours through boundary-pixel indices (`cv2.findContours`) -> the
    polygons around those pixels' outer edges, in continuous coordinates,
    as one (P, 2) float32 array with the same point count.

    Each vertex moves to its pixel centre (+0.5; findContours reports pixel
    indices) and then half a pixel outward along the bisector of its two
    edges' normals (a miter join, limited to the pixel corner's distance,
    half a diagonal), so painting the result (`fill_spans`) gives back the
    pixels the contour traced rather than the region inside their centres,
    which is half a pixel smaller all round. Where a contour turns back on
    itself (a one-pixel-wide spur) the vertex moves half a pixel along it.
    """
    part_off = np.asarray(part_off, np.int64)
    v = np.asarray(idx, np.float64) + 0.5
    if len(v) == 0:
        return v.astype(np.float32)
    nxt, prv = _rings(len(v), part_off)
    # orientation per part (shoelace), so "outward" is the same side for all
    cross = v[:, 0] * v[nxt, 1] - v[nxt, 0] * v[:, 1]
    side = np.sign(np.add.reduceat(cross, part_off[:-1]))
    side = np.repeat(np.where(side == 0, 1.0, side), np.diff(part_off))[:, None]

    def unit(d):
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    d_in, d_out = unit(v - v[prv]), unit(v[nxt] - v)
    n_in = side * np.stack((d_in[:, 1], -d_in[:, 0]), 1)
    n_out = side * np.stack((d_out[:, 1], -d_out[:, 0]), 1)
    b = n_in + n_out
    norm = np.linalg.norm(b, axis=1, keepdims=True)
    dot = (n_in * n_out).sum(1, keepdims=True)
    # miter length h * sqrt(2 / (1 + dot)), capped at the pixel corner, h * sqrt(2)
    length = 0.5 * np.minimum(np.sqrt(2 / np.maximum(1 + dot, 1e-12)), np.sqrt(2))
    move = np.where(norm > 1e-6, b / np.maximum(norm, 1e-12) * length, 0.5 * d_in)
    return (v + move).astype(np.float32)


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

    def paint(self, boxes, canvas):
        """Instance index map over the (H, W) canvas, at its full resolution:
        pixel value = row + 1 of the instance whose polygon covers the
        pixel's centre, 0 for none (`spans`).

        Larger instances are painted first so a small object in front keeps
        its own pixels (painter's order; the object behind loses them). Rows
        above 254 cannot be encoded in uint8 and are marked mask-less.
        Returns (map, has).
        """
        h, w = canvas
        m = np.zeros((h, w), np.uint8)
        has = self.has.copy()
        has[255:] = False
        if len(boxes) == 0 or not has.any():
            return m, has
        part_inst = np.repeat(np.arange(len(has)), np.diff(self.inst_off))
        inst, rows, c0, c1 = spans(self.xy, self.part_off, part_inst, (h, w))
        keep = has[inst]
        inst, rows, c0, c1 = inst[keep], rows[keep], c0[keep], c1[keep]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        rank = np.empty(len(area), np.int64)
        rank[np.argsort(-area, kind="stable")] = np.arange(len(area))
        order = np.argsort(rank[inst], kind="stable")
        inst, rows, c0, c1 = inst[order], rows[order], c0[order], c1[order]
        cut = np.flatnonzero(np.diff(inst)) + 1
        for g in np.split(np.arange(len(inst)), cut):
            if len(g):
                _paint_spans(m, rows[g], c0[g], c1[g], int(inst[g[0]]) + 1)
        return m, has


def spans(xy, part_off, part_inst, canvas):
    """The pixel runs whose centres lie inside polygon parts, exactly:
    even-odd inside each part. -> (inst, row, c0, c1) arrays, one run of
    pixels [c0, c1) on `row` per entry, tagged with its part's instance
    (`part_inst`), clipped to the (H, W) canvas.

    A scanline fill, vectorised over every part at once: each edge crosses
    the pixel-centre rows y = r + 0.5 with ymin <= y < ymax (half-open, so a
    vertex shared by two edges counts once); a row's crossings, sorted, pair
    into runs; and a run [xa, xb) holds the pixels whose centres c + 0.5
    fall inside it. `cv2.fillPoly` is not used because it also fills pixels
    its edges merely touch, about a pixel of dilation all round.
    """
    h, w = canvas
    a = np.asarray(xy, np.float64)
    empty = np.zeros(0, np.int64)
    if len(a) == 0:
        return empty, empty, empty, empty
    part_off = np.asarray(part_off, np.int64)
    b = a[_rings(len(a), part_off)[0]]
    part = np.repeat(np.arange(len(part_off) - 1), np.diff(part_off))
    y0, y1 = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    r0 = np.ceil(y0 - 0.5).astype(np.int64)        # first row whose centre is >= y0
    n = np.maximum(np.ceil(y1 - 0.5).astype(np.int64) - r0, 0)   # horizontal edges: none
    if not n.any():
        return empty, empty, empty, empty
    e = np.repeat(np.arange(len(a)), n)
    row = r0[e] + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    x = a[e, 0] + (row + 0.5 - a[e, 1]) * (b[e, 0] - a[e, 0]) / (b[e, 1] - a[e, 1])
    order = np.lexsort((x, row, part[e]))          # (part, row) groups hold even counts
    row, x, pe = row[order], x[order], part[e][order]
    rs, xa, xb = row[0::2], x[0::2], x[1::2]
    c0 = np.clip(np.ceil(xa - 0.5).astype(np.int64), 0, w)
    c1 = np.clip(np.ceil(xb - 0.5).astype(np.int64), 0, w)
    ok = (rs >= 0) & (rs < h) & (c1 > c0)
    return np.asarray(part_inst, np.int64)[pe[0::2][ok]], rs[ok], c0[ok], c1[ok]


def _paint_spans(m, rows, c0, c1, value):
    """Set the runs' pixels of `m` to `value` (overlapping runs: union)."""
    # +1 where a run starts, -1 where it ends, over the runs' bounding box
    # only; a running sum along each row is then > 0 inside any run
    top, left = rows.min(), c0.min()
    nr, nc = rows.max() - top + 1, c1.max() - left + 1
    at = (rows - top) * nc
    size = nr * nc
    diff = np.bincount(at + c0 - left, minlength=size) - np.bincount(at + c1 - left, minlength=size)
    inside = np.cumsum(diff.reshape(nr, nc)[:, :-1], 1) > 0
    m[top:top + nr, left:left + nc - 1][inside] = value


def fill_spans(m, xy, part_off, value):
    """Set every pixel of `m` whose centre lies inside the polygon parts to
    `value`: even-odd inside each part, union across parts (`spans`)."""
    part_off = np.asarray(part_off, np.int64)
    _, rows, c0, c1 = spans(xy, part_off, np.zeros(len(part_off) - 1, np.int64), m.shape)
    if len(rows):
        _paint_spans(m, rows, c0, c1, value)
    return m
