"""Coordinate geometry on an (H, W) canvas: letterbox, its inverse, and the
box and keypoint side of every warp.

Labels are xyxy in pixels of the letterboxed canvas, which is what the
assigner and the loss take. The only frame change between an annotation and
the network is `letterbox`, and `unletterbox` is its exact inverse; the two
live next to each other so a sign error in one is visible against the other.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PAD = 114  # mid-grey padding


def as_canvas(canvas):
    """An int side or an (H, W) pair -> (H, W)."""
    return (canvas, canvas) if isinstance(canvas, int) else tuple(canvas)


def fitted(hw, r):
    """The (h, w) an image of shape `hw` has after scaling by `r`."""
    return round(hw[0] * r), round(hw[1] * r)


def fit(img, canvas):
    """Resize preserving aspect so the image just fits the (H, W) canvas.
    Returns img, ratio."""
    ch, cw = canvas
    h, w = img.shape[:2]
    r = min(ch / h, cw / w)
    nh, nw = fitted((h, w), r)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return img, r


def letterbox(img, canvas):
    """Fit to the (H, W) canvas and pad the rest, centred. Returns img,
    ratio, pad (left, top)."""
    ch, cw = canvas
    img, r = fit(img, canvas)
    nh, nw = img.shape[:2]
    left, top = (cw - nw) // 2, (ch - nh) // 2
    img = cv2.copyMakeBorder(img, top, ch - nh - top, left, cw - nw - left,
                             cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return img, r, (left, top)


def clip_xyxy(boxes, w, h):
    """Clamp xyxy boxes into a w by h frame, in place."""
    boxes[..., 0::2] = boxes[..., 0::2].clip(0, w)
    boxes[..., 1::2] = boxes[..., 1::2].clip(0, h)
    return boxes


def unletterbox(boxes, ratio, pad, shape):
    """xyxy (or x, y pairs) in letterboxed pixels -> the original image,
    clipped to it. The exact inverse of `letterbox`."""
    boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[..., 0::2] = (boxes[..., 0::2] - pad[0]) / ratio
    boxes[..., 1::2] = (boxes[..., 1::2] - pad[1]) / ratio
    h, w = shape
    return clip_xyxy(boxes, w, h)


def unletterbox_maps(maps, meta):
    """(N, C, H, W) maps over the whole canvas -> (N, C, h, w) over the
    original image: cut the padding, resize bilinearly to the source shape.
    The dense-map counterpart of `unletterbox`; `meta` is the {"ratio",
    "pad", "shape"} a letterboxed sample carries."""
    left, top = meta["pad"]
    nh, nw = fitted(meta["shape"], meta["ratio"])
    maps = maps[:, :, top:top + nh, left:left + nw]
    return F.interpolate(maps, size=tuple(meta["shape"]), mode="bilinear", align_corners=False)


def apply_affine(xy, m):
    """(..., 2) points under the 3x3 affine `m`."""
    return xy @ m[:2, :2].T + m[:2, 2]


def place(boxes, ratio, offset):
    """Scale a [class, x1, y1, x2, y2] table by `ratio` and shift it by
    `offset` (x, y), in place: where a resized image was pasted."""
    boxes[:, 1:5] = boxes[:, 1:5] * ratio + np.float32(tuple(offset) * 2)
    return boxes


def keep(moved, clipped, min_side=2.0, min_area=0.10, max_ratio=100.0):
    """Which boxes survive a crop.

    A box mostly outside the canvas becomes a sliver of an object that is no
    longer there, and training on it teaches the detector to fire on edges.
    Drop anything under two pixels a side, holding less than a tenth of the
    area it had before clipping, or stretched past a hundred to one.
    """
    wh_before = (moved[:, 2:] - moved[:, :2]).clip(1e-6)
    wh_after = (clipped[:, 2:] - clipped[:, :2]).clip(0)
    area = wh_after.prod(1) / wh_before.prod(1)
    ratio = np.maximum(wh_after[:, 0] / (wh_after[:, 1] + 1e-16),
                       wh_after[:, 1] / (wh_after[:, 0] + 1e-16))
    return (wh_after.min(1) > min_side) & (area > min_area) & (ratio < max_ratio)


def warp_boxes(boxes, m, canvas):
    """A [class, x1, y1, x2, y2] table under the affine `m`, onto the (H, W)
    canvas: the axis-aligned hull of the moved corners, clipped. Returns the
    surviving rows and the boolean `keep` mask over the input rows."""
    ch, cw = canvas
    xy = boxes[:, 1:5]
    corners = np.stack([xy[:, [0, 1]], xy[:, [2, 1]], xy[:, [2, 3]], xy[:, [0, 3]]], 1)
    corners = apply_affine(corners, m)
    moved = np.concatenate((corners.min(1), corners.max(1)), 1)
    clipped = clip_xyxy(moved.copy(), cw, ch)
    ok = keep(moved, clipped)
    return np.concatenate((boxes[:, :1], clipped), 1)[ok], ok


def flip_boxes(boxes, w):
    """A [class, x1, y1, x2, y2] table under a horizontal flip of width w, in place."""
    x1 = boxes[:, 1].copy()
    boxes[:, 1] = w - boxes[:, 3]
    boxes[:, 3] = w - x1
    return boxes


def scale_shift_kpts(kp, k, r, offset):
    """(n, 3K) keypoints scaled by `r` and shifted by `offset` (x, y)."""
    kp = kp.copy().reshape(-1, k, 3)
    kp[:, :, :2] = kp[:, :, :2] * r + np.float32(tuple(offset))
    return kp.reshape(-1, 3 * k)


def warp_kpts(kp, k, m, canvas):
    """(n, 3K) keypoints under the affine `m`; points that leave the (H, W)
    canvas become unlabelled (v = 0)."""
    ch, cw = canvas
    kp = kp.reshape(-1, k, 3).copy()
    xy = apply_affine(kp[:, :, :2], m)
    out = (xy[..., 0] < 0) | (xy[..., 0] >= cw) | (xy[..., 1] < 0) | (xy[..., 1] >= ch)
    kp[:, :, :2] = xy
    kp[:, :, 2] = np.where(out, 0.0, kp[:, :, 2])
    return kp.reshape(-1, 3 * k)


def flip_kpts(kp, k, w, pairs):
    """(n, 3K) keypoints under a horizontal flip of width w: mirror x and
    swap each left/right pair."""
    kp = kp.reshape(-1, k, 3).copy()
    kp[:, :, 0] = np.where(kp[:, :, 2] > 0, w - kp[:, :, 0], kp[:, :, 0])
    for a, b in pairs:
        kp[:, [a, b]] = kp[:, [b, a]]
    return kp.reshape(-1, 3 * k)
