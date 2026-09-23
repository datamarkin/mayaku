"""Keypoint encoding, decoding and heatmap targets, owned once.

Per anchor the graph emits, for each of K keypoints, an (x, y) offset from the
anchor point in units of the anchor's stride and a visibility logit; on the
shared stride-8 trunk it emits K heatmaps and two offset channels. The host
decodes the regression, then snaps each point to the nearest heatmap peak
inside a small window when that peak is confident (CenterNet's rule, Zhou et
al. 2019, arXiv 1904.07850). Training uses the regression through an OKS loss,
the visibility through BCE, the heatmaps through a penalty-reduced focal loss
on Gaussian targets and the offsets through L1 at the peak cells.

Cell convention as in `mayaku.model.mask`: cell (i, j) of the stride-8 map has
its centre at ((j + 0.5) * 8, (i + 0.5) * 8).
"""

import torch

from mayaku.model.aux import MASK_STRIDE
from mayaku.model.box import flatten_levels

# COCO's per-keypoint falloff constants (nose ... ankles). Any other K gets a
# flat 0.05, the middle of this range.
COCO_SIGMAS = (.026, .025, .025, .035, .035, .079, .079, .072, .072, .062,
               .062, .107, .107, .087, .087, .089, .089)
# half-width, in cells, of the window each keypoint's Gaussian is painted
# into: 3 sigma with the sigma clamp at 3 cells (see gaussian_targets)
HEAT_RADIUS = 9


def sigma_values(k):
    """The OKS falloff constants for K keypoints: COCO's person ones for 17,
    a flat 0.05 otherwise. Plain floats, as recorded in the sidecar."""
    return COCO_SIGMAS if k == 17 else (0.05,) * k


def sigmas(k):
    return torch.tensor(sigma_values(k))


def flatten_kpt(preds, k):
    """[kpt0, kpt1, ...] (B, 3K, h, w) -> (B, N, K, 3), N fine to coarse."""
    return flatten_levels(preds, 3 * k).reshape(preds[0].shape[0], -1, k, 3)


def decode(kp, points, stride):
    """(.., K, 3) raw -> xy in pixels (.., K, 2) and visibility logits (.., K)."""
    xy = points[..., None, :] + kp[..., :2] * stride[..., None, :]
    return xy, kp[..., 2]


def encode(xy, points, stride):
    """The inverse: pixel keypoints -> offsets in strides."""
    return (xy - points[..., None, :]) / stride[..., None, :]


def cells(xy):
    """Pixel points -> (col, row) of the stride-8 cell holding them, and the
    sub-cell offset in [-0.5, 0.5) from that cell's centre."""
    c = xy / MASK_STRIDE - 0.5
    idx = c.round()
    return idx.long(), c - idx


def gaussian_targets(xy, vis, boxes, k, h, w):
    """Heatmap and offset targets for one batch.

    xy (B, M, K, 2) pixels, vis (B, M, K) > 0 where labelled, boxes (B, M, 4)
    xyxy -> heat (B, K, h, w) in [0, 1] with exactly 1 at each keypoint's
    cell, offsets (B, 2, h, w), and a (B, h, w) bool of cells that hold a
    keypoint (where the offset loss applies).
    """
    B = xy.shape[0]
    device = xy.device
    heat = torch.zeros(B * k * h * w, device=device)
    off = torch.zeros(B, 2, h, w, device=device)
    at = torch.zeros(B, h, w, dtype=torch.bool, device=device)
    b, m, kk = vis.nonzero(as_tuple=True)
    if len(b) == 0:
        return heat.view(B, k, h, w), off, at
    p = xy[b, m, kk]                                                   # (n, 2)
    idx, sub = cells(p)
    inside = (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
    b, m, kk, p, idx, sub = (t[inside] for t in (b, m, kk, p, idx, sub))
    bw = boxes[b, m]                                                    # (n, 4)
    side = ((bw[:, 2] - bw[:, 0]) * (bw[:, 3] - bw[:, 1])).clamp(min=1).sqrt()
    # spread in cells: a tenth of the object's side, at least most of a cell
    sig = (0.1 * side / MASK_STRIDE).clamp(0.75, 3.0)                   # (n,)
    # The Gaussian is negligible past ~3 sigma and sigma <= 3 cells, so paint
    # it in a fixed (2R+1) window around each keypoint's cell rather than over
    # the whole map (CenterNet's radius trick).
    R = HEAT_RADIUS
    cx, cy = (p[:, 0] / MASK_STRIDE - 0.5), (p[:, 1] / MASK_STRIDE - 0.5)
    d = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
    wy, wx = torch.meshgrid(d, d, indexing="ij")                       # (W, W)
    ax = idx[:, 0, None, None] + wx                                    # (n, W, W)
    ay = idx[:, 1, None, None] + wy
    g = torch.exp(-((ax - cx[:, None, None]) ** 2 + (ay - cy[:, None, None]) ** 2)
                  / (2 * sig[:, None, None] ** 2))
    ok = (ax >= 0) & (ax < w) & (ay >= 0) & (ay < h)
    g = torch.where(ok, g, torch.zeros_like(g))                        # amax with 0 is a no-op
    axc, ayc = ax.long().clamp(0, w - 1), ay.long().clamp(0, h - 1)
    flat = (b * k + kk)[:, None, None] * (h * w) + ayc * w + axc
    heat.scatter_reduce_(0, flat.reshape(-1), g.reshape(-1), reduce="amax")
    heat = heat.view(B, k, h, w)
    heat[b, kk, idx[:, 1], idx[:, 0]] = 1.0
    off[b, 0, idx[:, 1], idx[:, 0]] = sub[:, 0]
    off[b, 1, idx[:, 1], idx[:, 0]] = sub[:, 1]
    at[b, idx[:, 1], idx[:, 0]] = True
    return heat, off, at


def focal(logits, target, alpha=2.0, beta=4.0):
    """CenterNet's penalty-reduced pixel-wise focal loss, summed and divided
    by the number of peaks."""
    p = logits.float().sigmoid().clamp(1e-6, 1 - 1e-6)
    pos = target.eq(1.0)
    pos_loss = -((1 - p) ** alpha * p.log())[pos].sum()
    neg_loss = -(((1 - target) ** beta) * p ** alpha * (1 - p).log())[~pos].sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1)


def oks_loss(pred_xy, gt_xy, valid, area, sig):
    """1 - OKS per labelled keypoint, averaged. pred/gt (P, K, 2), valid
    (P, K) bool, area (P,) box area in px^2, sig (K,)."""
    d2 = ((pred_xy - gt_xy) ** 2).sum(-1)
    e = d2 / (2 * area[:, None] * (2 * sig[None]) ** 2 + 1e-9)
    return ((1 - torch.exp(-e)) * valid).sum() / valid.sum().clamp(min=1)


@torch.no_grad()
def snap(xy, vis, heat, off, boxes, radius=2, tau=0.1):
    """Move each regressed keypoint to the nearest confident heatmap peak.

    xy (n, K, 2) pixels, vis (n, K) probabilities, heat (K, h, w) logits,
    off (2, h, w), boxes (n, 4). Within a (2r+1)^2 cell window around the
    regressed point, take the cell with the highest heatmap value; if its
    probability exceeds `tau` and it lies inside the box, the keypoint becomes
    that cell's centre plus the predicted offset. Returns the new xy.
    """
    k, h, w = heat.shape
    n = len(xy)
    if n == 0:
        return xy
    prob = heat.sigmoid()
    idx, _ = cells(xy)                                                # (n, K, 2)
    d = torch.arange(-radius, radius + 1, device=xy.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    wx = (idx[..., 0, None] + dx.reshape(-1)).clamp(0, w - 1)       # (n, K, W)
    wy = (idx[..., 1, None] + dy.reshape(-1)).clamp(0, h - 1)
    kk = torch.arange(k, device=xy.device)[None, :, None].expand_as(wx)
    vals = prob[kk, wy, wx]                                            # (n, K, W)
    best = vals.argmax(-1, keepdim=True)
    bx, by = wx.gather(-1, best)[..., 0], wy.gather(-1, best)[..., 0]
    pk = vals.gather(-1, best)[..., 0]
    px = (bx.float() + 0.5 + off[0, by, bx]) * MASK_STRIDE
    py = (by.float() + 0.5 + off[1, by, bx]) * MASK_STRIDE
    inside = ((px >= boxes[:, None, 0]) & (px <= boxes[:, None, 2])
              & (py >= boxes[:, None, 1]) & (py <= boxes[:, None, 3]))
    use = (pk > tau) & inside
    out = xy.clone()
    out[..., 0] = torch.where(use, px, xy[..., 0])
    out[..., 1] = torch.where(use, py, xy[..., 1])
    return out
