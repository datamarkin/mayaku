"""Mask targets: the exact pixel-centre painter, the RLE round trip, soft
coverage and multi-scale alignment."""

from __future__ import annotations

import numpy as np
import torch
from pycocotools import mask as mask_util

from mayaku.data.batch import rescale_batch
from mayaku.data.polygons import Polys, fill_spans, rle_to_polygons
from mayaku.model.mask import coverage


def _inside(h, w, parts):
    """Brute force: pixel centres inside, even-odd per part, union of parts."""
    yy, xx = np.mgrid[:h, :w]
    px, py = xx + 0.5, yy + 0.5
    out = np.zeros((h, w), bool)
    for p in parts:
        ins = np.zeros((h, w), bool)
        for (x0, y0), (x1, y1) in zip(p, np.roll(p, -1, 0), strict=True):
            if y0 == y1:
                continue
            cross = ((y0 <= py) & (py < y1)) | ((y1 <= py) & (py < y0))
            ins ^= cross & (px < x0 + (py - y0) * (x1 - x0) / (y1 - y0))
        out |= ins
    return out


def _polygons(rng, h, w, n_parts):
    parts = []
    for _ in range(n_parts):
        k = int(rng.integers(3, 10))
        c, r = rng.uniform(-5, max(h, w) + 5, 2), rng.uniform(0.3, 30)
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rr = r * rng.uniform(0.2, 1, k)
        p = np.stack([c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang)], 1)
        if rng.random() < 0.3:
            p = np.round(p * 2) / 2           # vertices on pixel edges and centres: ties
        parts.append(p.astype(np.float32))
    return parts


def test_fill_spans_paints_exactly_the_pixel_centres_inside() -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        h, w = (int(v) for v in rng.integers(5, 60, 2))
        parts = _polygons(rng, h, w, int(rng.integers(1, 3)))
        m = np.zeros((h, w), np.uint8)
        fill_spans(m, np.concatenate(parts),
                   np.concatenate(([0], np.cumsum([len(p) for p in parts]))).astype(np.int32), 7)
        assert np.array_equal(m == 7, _inside(h, w, [p.astype(np.float64) for p in parts]))


def test_rle_polygon_paint_round_trip() -> None:
    """A mask without holes comes back at its own resolution: pixel for pixel
    almost always, a diagonal staircase step off at worst. (Through the
    boundary pixels' centres instead, every mask lost half a pixel all round.)"""
    rng = np.random.default_rng(1)
    exact = 0
    for _ in range(100):
        m = np.zeros((48, 64), np.uint8)
        for _ in range(int(rng.integers(1, 4))):    # unions of rectangles and discs
            y, x = (int(v) for v in rng.integers(4, 44, 2))
            if rng.random() < 0.5:
                m[y - 3:y + int(rng.integers(1, 6)), x - 4:x + int(rng.integers(1, 9))] = 1
            else:
                yy, xx = np.mgrid[:48, :64]
                m[(yy - y) ** 2 + (xx - x) ** 2 <= int(rng.integers(2, 40))] = 1
        rle = mask_util.encode(np.asfortranarray(m))
        p = Polys.from_coco([rle_to_polygons(rle)])
        boxes = np.float32([[0, 0, 0, 64, 48]])
        got, has = p.paint(boxes, (48, 64))
        g, t = got == 1, m == 1
        assert has.all() and (g & t).sum() / (g | t).sum() > 0.99
        exact += np.array_equal(g, t)
    assert exact >= 97


def test_paint_keeps_the_front_instance_and_caps_rows() -> None:
    big = [[0, 0, 32, 0, 32, 32, 0, 32]]
    small = [[8, 8, 16, 8, 16, 16, 8, 16]]
    p = Polys.from_coco([[big[0]], [small[0]]])
    m, _ = p.paint(np.float32([[0, 0, 0, 32, 32], [0, 8, 8, 16, 16]]), (32, 32))
    assert (m[8:16, 8:16] == 2).all() and (m == 1).sum() == 32 * 32 - 64


def test_coverage_is_each_instances_share_of_the_cell() -> None:
    g = torch.Generator().manual_seed(0)
    inst = torch.randint(0, 4, (2, 32, 48), generator=g, dtype=torch.uint8)
    img = torch.tensor([0, 0, 1, 1, 1])
    ids = torch.tensor([1, 2, 1, 3, 0])
    for s in (4, 8):
        got = coverage(inst, img, ids, s, chunk=2)
        ref = torch.stack([(inst[i] == v).float().view(32 // s, s, 48 // s, s).mean((1, 3)).flatten()
                           for i, v in zip(img.tolist(), ids.tolist(), strict=True)])
        assert torch.allclose(got, ref)
    # two instances in one cell each get their own fraction
    shared = torch.zeros(1, 8, 8, dtype=torch.uint8)
    shared[0, :, :3], shared[0, :, 3:] = 1, 2
    two = coverage(shared, torch.tensor([0, 0]), torch.tensor([1, 2]), 8)
    assert torch.allclose(two[:, 0], torch.tensor([3 / 8, 5 / 8]))


def test_multi_scale_resize_keeps_the_map_centred() -> None:
    """Downscaling the painted map lands where painting the scaled polygons
    would: the centroid agrees within a twentieth of a cell."""
    rng = np.random.default_rng(2)
    errs = []
    for _ in range(40):
        c, r = rng.uniform(150, 650, 2), rng.uniform(30, 120)
        ang = np.sort(rng.uniform(0, 2 * np.pi, 7))
        xy = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], 1).astype(np.float32)
        box = np.float32([[0, *xy.min(0), *xy.max(0)]])
        p = Polys(xy, np.int32([0, 7]), np.int32([0, 1]), np.array([True]))
        big, _ = p.paint(box, (800, 800))
        small, _ = p.scaled(0.64, (0, 0)).paint(box * 0.64, (512, 512))
        _, _, got = rescale_batch(torch.zeros(1, 3, 800, 800), torch.zeros(0, 7),
                                  torch.from_numpy(big)[None], (512, 512), seg=True)
        cov = [coverage(t, torch.tensor([0]), torch.tensor([1]), 8).view(64, 64)
               for t in (got, torch.from_numpy(small)[None])]
        yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
        cen = [torch.stack(((m * xx).sum(), (m * yy).sum())) / m.sum() for m in cov]
        errs.append((cen[0] - cen[1]).abs().max().item())
    assert max(errs) < 0.05

