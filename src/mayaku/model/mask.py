"""The per-instance mask head, owned once.

The graph emits an 8-channel stride-8 mask feature and, per anchor, 169
numbers that are the weights and biases of a three-layer 1x1 network
(8+2 -> 8 -> 8 -> 1, ReLU between). The host prepends two relative-coordinate
channels and runs that network over the detection's box. Nothing here is in
the deploy graph; this file is what the loss, the evaluator and any host
decoder must agree on.

Coordinates: cell (i, j) of the stride-8 map covers image pixels
[8j, 8j+8) x [8i, 8i+8) and its centre is the stride-8 anchor point, the same
convention as `mayaku.model.box.anchor_grid`.
"""

import torch
import torch.nn.functional as F

from mayaku.model.aux import KERNEL_LAYOUT, KERNEL_PARAMS, MASK_CH, MASK_STRIDE
from mayaku.model.box import anchor_grid, flatten_levels

# Relative coordinates are scaled by this many strides of the anchor's level,
# so a stride-8 anchor sees +-1 at 64 px and a stride-32 anchor at 256 px: the
# size range each level owns (CondInst's size of interest).
COORD_SCALE = 8


def flatten_ker(kers):
    """[ker0, ker1, ...] (B, 169, h, w) -> (B, N, 169), N fine to coarse."""
    return flatten_levels(kers, KERNEL_PARAMS)


def split_kernels(k):
    """(P, 169) -> [(W1 (P,8,10), b1 (P,8)), (W2 (P,8,8), b2), (W3 (P,1,8), b3)]."""
    assert k.shape[-1] == KERNEL_PARAMS, k.shape
    out, o = [], 0
    for cin, cout in KERNEL_LAYOUT:
        w = k[:, o:o + cin * cout].reshape(-1, cout, cin)
        o += cin * cout
        b = k[:, o:o + cout]
        o += cout
        out.append((w, b))
    return out


def rel_coords(points, stride, h, w, dtype=None):
    """(P, 2) anchor points, (P, 1) strides -> (P, 2, h*w) relative coordinates
    of every stride-8 cell centre, in units of COORD_SCALE * stride."""
    dtype = dtype or points.dtype
    grid = anchor_grid([(h, w)], [MASK_STRIDE], device=points.device, dtype=dtype)[0].T  # (2, Q)
    scale = (COORD_SCALE * stride).to(dtype)                       # (P, 1)
    return (grid[None] - points.to(dtype)[:, :, None]) / scale[:, :, None]


def dyn_conv(feat, coords, kernels):
    """Apply each instance's kernels to its feature.

    feat (P, 8, Q), coords (P, 2, Q), kernels (P, 169) -> logits (P, Q). Every
    layer is 1x1, so Q can be a full map or any subset of points; the
    arithmetic is the same.
    """
    x = torch.cat((coords.to(feat.dtype), feat), 1)                # (P, 10, Q)
    layers = split_kernels(kernels.to(feat.dtype))
    for i, (w, b) in enumerate(layers):
        x = torch.baddbmm(b[:, :, None], w, x)                     # (P, cout, Q)
        if i + 1 < len(layers):
            x = F.relu(x)
    return x[:, 0]


def dice(logits, target, eps=5e-6):
    """Per-instance Dice loss on logits, in fp32. (P, Q) -> (P,)."""
    p = logits.float().sigmoid()
    t = target.float()
    inter = (p * t).sum(1)
    return 1 - (2 * inter + eps) / ((p * p).sum(1) + (t * t).sum(1) + eps)


def crop_cells(boxes, h, w):
    """xyxy pixel boxes -> inclusive-exclusive cell rectangles on the stride-8
    map, as (x0, y0, x1, y1) long tensors clamped to the map."""
    x0 = (boxes[:, 0] / MASK_STRIDE).floor().long().clamp(0, w - 1)
    y0 = (boxes[:, 1] / MASK_STRIDE).floor().long().clamp(0, h - 1)
    x1 = torch.maximum((boxes[:, 2] / MASK_STRIDE).ceil().long(), x0 + 1).clamp(max=w)
    y1 = torch.maximum((boxes[:, 3] / MASK_STRIDE).ceil().long(), y0 + 1).clamp(max=h)
    return x0, y0, x1, y1


@torch.no_grad()
def assemble(mask_feat, kernels, points, stride, boxes):
    """Full-map logits per detection, -inf outside the detection's box
    (cropped at cell granularity), so a threshold at 0 never fires there.

    mask_feat (8, H8, W8), kernels (n, 169), points (n, 2), stride (n, 1),
    boxes (n, 4) xyxy letterbox pixels -> logits (n, H8, W8).
    """
    n = len(kernels)
    _, h, w = mask_feat.shape
    if n == 0:
        return mask_feat.new_zeros(0, h, w)
    coords = rel_coords(points, stride, h, w, dtype=mask_feat.dtype)
    logits = dyn_conv(mask_feat[None].expand(n, -1, -1, -1).reshape(n, MASK_CH, -1),
                      coords, kernels).view(n, h, w)
    x0, y0, x1, y1 = crop_cells(boxes, h, w)
    ys = torch.arange(h, device=logits.device)[None, :, None]
    xs = torch.arange(w, device=logits.device)[None, None, :]
    inside = ((ys >= y0[:, None, None]) & (ys < y1[:, None, None])
              & (xs >= x0[:, None, None]) & (xs < x1[:, None, None]))
    return logits.masked_fill(~inside, float("-inf"))


def to_canvas(logits):
    """(n, H8, W8) crop logits -> (n, 1, H, W) logits over the whole canvas,
    upsampled x8 bilinearly, so the boundary is interpolated rather than
    blocky. Undo the letterbox with `mayaku.data.geometry.unletterbox_maps`
    and threshold at 0 for masks in the original image."""
    canvas = (logits.shape[-2] * MASK_STRIDE, logits.shape[-1] * MASK_STRIDE)
    return F.interpolate(logits[:, None].float(), size=canvas, mode="bilinear", align_corners=False)
