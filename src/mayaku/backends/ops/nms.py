"""Backend-portable non-maximum suppression.

Wraps :func:`torchvision.ops.nms` and :func:`torchvision.ops.batched_nms`
(CPU + CUDA + MPS in our pinned versions; see
``BACKEND_PORTABILITY_REPORT.md`` Appendix A) with a pure-PyTorch
fallback for backends where the torchvision kernel is unavailable.

The ``.float()`` cast on both ``boxes`` and ``scores`` is intentional.
On boxes it matches the Detectron2 reference
(``DETECTRON2_TECHNICAL_SPEC.md`` §7.2): the per-class offset trick used
by ``batched_nms`` adds ``idxs * (max_coord + 1)`` to coordinates, which
overflows ``float16`` on realistic image sizes. The matching scores cast
is required by torchvision CPU NMS, which rejects mismatched
``boxes``/``scores`` dtypes with ``RuntimeError("dets should have the
same type as scores")``. Promoting both to ``float32`` is cheap.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import batched_nms as _tv_batched_nms
from torchvision.ops import nms as _tv_nms

__all__ = ["batched_nms", "nms"]


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Single-class NMS.

    Args:
        boxes: ``(N, 4)`` in ``(x0, y0, x1, y1)`` format.
        scores: ``(N,)`` per-box scores.
        iou_threshold: Boxes with IoU above this are suppressed.

    Returns:
        ``(K,)`` ``int64`` indices into ``boxes``, sorted by descending
        score, of the kept boxes.
    """
    boxes_f = boxes.float()
    scores_f = scores.float()
    try:
        out: Tensor = _tv_nms(boxes_f, scores_f, iou_threshold)
        return out
    except NotImplementedError:
        return _nms_fallback(boxes_f, scores_f, iou_threshold)


def batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float) -> Tensor:
    """Per-class NMS via the coordinate-offset trick.

    Args:
        boxes: ``(N, 4)`` in ``(x0, y0, x1, y1)`` format.
        scores: ``(N,)`` per-box scores.
        idxs: ``(N,)`` integer class indices. Boxes from different
            classes are NMS-independent.
        iou_threshold: Boxes with IoU above this within the same class
            are suppressed.

    Returns:
        ``(K,)`` ``int64`` indices into ``boxes``, sorted by descending
        score across all classes.
    """
    boxes_f = boxes.float()
    scores_f = scores.float()
    try:
        out: Tensor = _tv_batched_nms(boxes_f, scores_f, idxs, iou_threshold)
        return out
    except NotImplementedError:
        return _batched_nms_fallback(boxes_f, scores_f, idxs, iou_threshold)


def _nms_fallback(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Pure-PyTorch NMS. ``O(N^2)`` worst case; only used when the
    torchvision kernel does not cover this backend."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    order = scores.argsort(descending=True)
    boxes_s = boxes[order]
    x0, y0, x1, y1 = boxes_s.unbind(dim=1)
    areas = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)

    keep_mask = torch.ones(boxes_s.shape[0], dtype=torch.bool, device=boxes.device)
    n = boxes_s.shape[0]
    for i in range(n):
        if not bool(keep_mask[i]):
            continue
        # Vectorize over the remaining candidates.
        if i + 1 >= n:
            break
        rest = slice(i + 1, n)
        ix0 = torch.maximum(x0[i], x0[rest])
        iy0 = torch.maximum(y0[i], y0[rest])
        ix1 = torch.minimum(x1[i], x1[rest])
        iy1 = torch.minimum(y1[i], y1[rest])
        inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
        suppressed = iou > iou_threshold
        keep_mask[rest] &= ~suppressed

    return order[keep_mask]


def _batched_nms_fallback(
    boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float
) -> Tensor:
    """Per-class NMS using the same offset trick torchvision uses
    internally."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coord = boxes.max()
    offsets = idxs.to(boxes) * (max_coord + 1)
    shifted = boxes + offsets[:, None]
    return _nms_fallback(shifted, scores, iou_threshold)
