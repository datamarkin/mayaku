"""Anchor points, distance-to-box decoding, and IoU.

None of this is in the deploy graph. The head emits raw class logits and raw
DFL bin logits; the expectation, the sigmoid and the box decode run on the
host. What is here is what the loss and the assigner need during training, and
the reference every host decoder must reproduce.

Boxes are xyxy in pixels unless a name says otherwise. Distances are ltrb in
units of the anchor's own stride, which is what the network regresses.
"""

import functools
import math

import torch

from mayaku.model.blocks import STRIDES


def anchor_grid(shapes, strides, offset=0.5, device="cpu", dtype=torch.float32):
    """Cell centres for a list of (H, W) feature shapes.

    Returns points (N, 2) in pixels and stride (N, 1), concatenated over
    levels in the order the head emits them: fine to coarse.

    Cached, because the grid depends on nothing but the feature shapes and
    would otherwise be rebuilt on every step. The result is shared between
    callers, so treat it as read-only.
    """
    return _grid(tuple(tuple(s) for s in shapes), tuple(strides), offset,
                 torch.device(device), dtype)


@functools.lru_cache(maxsize=32)
def _grid(shapes, strides, offset, device, dtype):
    points, strides_out = [], []
    for (h, w), s in zip(shapes, strides, strict=True):
        sx = (torch.arange(w, device=device, dtype=dtype) + offset) * s
        sy = (torch.arange(h, device=device, dtype=dtype) + offset) * s
        y, x = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((x.reshape(-1), y.reshape(-1)), -1))
        strides_out.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(strides_out)


def flatten_levels(maps, c):
    """Per-level maps [(B, c, h, w), ...] -> (B, N, c), N fine to coarse,
    matching `anchor_grid`, so an anchor index means the same thing to the
    assigner, the loss and the decoder."""
    b = maps[0].shape[0]
    return torch.cat([m.reshape(b, c, -1) for m in maps], 2).permute(0, 2, 1)


def flatten_head(preds, nc, reg_max):
    """[cls0, box0, cls1, box1, ...] -> (B, N, nc), (B, N, 4R), level shapes."""
    shapes = [tuple(c.shape[2:]) for c in preds[0::2]]
    return (flatten_levels(preds[0::2], nc).contiguous(),
            flatten_levels(preds[1::2], 4 * reg_max).contiguous(), shapes)


def decode_head(preds, nc, reg_max):
    """The head's six tensors -> everything downstream needs from them: flat
    logits, flat bin logits, decoded boxes in pixels, and the anchor grid they
    were decoded against. One owner, so the training decode and the deployed
    decode cannot drift apart."""
    # the model may append auxiliary tensors (masks, keypoints) after the
    # head's own; the detection decode reads exactly two per level
    cls, dist, shapes = flatten_head(preds[:2 * len(STRIDES)], nc, reg_max)
    points, stride = anchor_grid(shapes, STRIDES, device=cls.device, dtype=cls.dtype)
    return cls, dist, ltrb_to_xyxy(dfl_expectation(dist, reg_max), points,
                                   stride), points, stride, shapes


def dfl_expectation(logits, reg_max):
    """(B, N, 4*reg_max) bin logits -> (B, N, 4) expected ltrb distances.

    The head predicts a distribution over `reg_max` integer distances per side
    rather than one number, which lets the box loss express uncertainty. The
    expectation is the softmax mean, GFL (Li et al. 2020, arXiv 2006.04388)
    Eq. 5 with unit bin spacing. The bins here are 0..reg_max-1.
    """
    b, n, _ = logits.shape
    p = logits.view(b, n, 4, reg_max).softmax(-1)
    bins = torch.arange(reg_max, device=logits.device, dtype=logits.dtype)
    return (p * bins).sum(-1)


def ltrb_to_xyxy(dist, points, stride=None):
    """ltrb distances -> xyxy. `dist` in stride units unless stride is None."""
    if stride is not None:
        dist = dist * stride
    lt, rb = dist.chunk(2, -1)
    return torch.cat((points - lt, points + rb), -1)


def xyxy_to_ltrb(boxes, points, stride, reg_max):
    """xyxy -> ltrb in stride units, clamped into the representable range.

    The clamp is a real ceiling: a box side further than `reg_max` strides
    away cannot be represented, so the target is truncated. That bounds how
    large an object one level can describe, and it is why the assigner must
    not hand a large object to a fine level.
    """
    lt = (points - boxes[..., :2]) / stride
    rb = (boxes[..., 2:] - points) / stride
    return torch.cat((lt, rb), -1).clamp_(0, reg_max - 1 - 0.01)


def iou_xyxy(a, b, ciou=False, eps=1e-7):
    """IoU between broadcastable xyxy tensors, optionally complete IoU.

    CIoU as Zheng et al. 2020 (arXiv 1911.08287) Eqs. 9-11:
    IoU - rho^2/c^2 - alpha*v, v = 4/pi^2 (atan(w_gt/h_gt) - atan(w/h))^2,
    alpha = v / ((1 - IoU) + v). The paper differentiates v only, so alpha is
    held constant under autograd. `b` is the ground-truth side of v.

    Hand-written rather than torchvision's, because torchvision computes these
    only as a pairwise (N, M) matrix and cannot broadcast over a batch.
    """
    a1, a2 = a[..., :2], a[..., 2:]
    b1, b2 = b[..., :2], b[..., 2:]
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(-1)
    area_a = (a2 - a1).clamp_(0).prod(-1)
    area_b = (b2 - b1).clamp_(0).prod(-1)
    union = area_a + area_b - inter + eps
    iou = inter / union
    if not ciou:
        return iou
    cw, ch = (torch.max(a2, b2) - torch.min(a1, b1)).clamp_(eps).unbind(-1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    centre = ((b1 + b2) - (a1 + a2)).pow(2).sum(-1) / 4
    wa, ha = (a2 - a1).clamp_(eps).unbind(-1)
    wb, hb = (b2 - b1).clamp_(eps).unbind(-1)
    v = (4 / math.pi ** 2) * (torch.atan(wb / hb) - torch.atan(wa / ha)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (centre / c2 + alpha * v)
