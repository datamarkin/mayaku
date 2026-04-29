"""Box predictor + per-image post-processing for Faster R-CNN.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.5 (`fast_rcnn.py:174-569`):

* :class:`FastRCNNOutputLayers` — two linear heads on top of the box
  head's ``(R, fc_dim)`` features: a ``(K + 1)``-way classification
  (``+1`` for the background class) and a ``K * 4`` (or ``1 * 4`` if
  class-agnostic) bounding-box regression. Initialised with
  ``std=0.01`` for ``cls_score`` and ``std=0.001`` for ``bbox_pred``.
* :func:`fast_rcnn_inference` — per-image post-processing: clip to the
  image, apply the score threshold, run per-class NMS, take the
  top-``k``.

The training losses live alongside :class:`FastRCNNOutputLayers` so the
predictor and its loss travel together — this matches Detectron2's
file layout and makes the per-class bbox indexing easy to verify.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mayaku.backends.ops.nms import batched_nms
from mayaku.models.proposals.box_regression import Box2BoxTransform
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances

__all__ = [
    "FastRCNNOutputLayers",
    "fast_rcnn_inference",
    "fast_rcnn_inference_single_image",
]


class FastRCNNOutputLayers(nn.Module):
    """Linear cls + bbox heads + their losses.

    Args:
        input_dim: Dim of the feature vector from the box head.
        num_classes: Foreground class count (``K``); the cls head emits
            ``K + 1`` logits (the extra is background).
        box_transform: Shared :class:`Box2BoxTransform` (the box head
            uses ``weights=(10, 10, 5, 5)`` per spec).
        smooth_l1_beta: Smooth-L1 transition point. ``0`` (the default
            for detection) makes it equivalent to L1.
        cls_agnostic_bbox_reg: If ``True``, predict a single 4-vector
            per RoI; otherwise predict ``K * 4`` (one set per class).
        score_thresh_test: Per-class score floor at inference.
        nms_thresh_test: Per-class NMS IoU at inference.
        topk_per_image: Cap on detections per image.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        box_transform: Box2BoxTransform,
        *,
        smooth_l1_beta: float = 0.0,
        cls_agnostic_bbox_reg: bool = False,
        score_thresh_test: float = 0.05,
        nms_thresh_test: float = 0.5,
        topk_per_image: int = 100,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.box_transform = box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.score_thresh_test = score_thresh_test
        self.nms_thresh_test = nms_thresh_test
        self.topk_per_image = topk_per_image

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.cls_score = nn.Linear(input_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(input_dim, num_bbox_reg_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in (self.cls_score, self.bbox_pred):
            nn.init.constant_(layer.bias, 0.0)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.cls_score(x)  # (R, K+1)
        deltas = self.bbox_pred(x)  # (R, K*4) or (R, 4) if cls-agnostic
        return scores, deltas

    # ------------------------------------------------------------------
    # losses (training)
    # ------------------------------------------------------------------

    def losses(
        self,
        predictions: tuple[Tensor, Tensor],
        proposals: Sequence[Instances],
    ) -> dict[str, Tensor]:
        scores, deltas = predictions
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        # Foreground mask: classes in [0, num_classes); background == num_classes.
        fg_mask = (gt_classes >= 0) & (gt_classes < self.num_classes)
        num_total = max(gt_classes.numel(), 1)

        # Cross-entropy ignores the -1 ignore index automatically.
        loss_cls = F.cross_entropy(scores, gt_classes, reduction="mean")

        if not bool(fg_mask.any()):
            loss_loc = deltas.sum() * 0.0
        else:
            gt_classes_fg = gt_classes[fg_mask]
            gt_boxes = torch.cat([_gt_boxes_tensor(p) for p in proposals], dim=0)
            gt_deltas = self.box_transform.get_deltas(proposal_boxes[fg_mask], gt_boxes[fg_mask])
            if self.cls_agnostic_bbox_reg:
                pred_deltas_fg = deltas[fg_mask]
            else:
                # Index per-class deltas: each foreground RoI's gt class
                # selects 4 contiguous columns from the K*4-wide tensor.
                fg_idx = fg_mask.nonzero(as_tuple=False).squeeze(1)
                col_starts = gt_classes_fg * 4
                cols = col_starts[:, None] + torch.arange(4, device=deltas.device)
                pred_deltas_fg = deltas[fg_idx[:, None], cols]
            loss_loc = (
                F.smooth_l1_loss(
                    pred_deltas_fg, gt_deltas, beta=self.smooth_l1_beta, reduction="sum"
                )
                / num_total
            )
        return {"loss_cls": loss_cls, "loss_box_reg": loss_loc}

    # ------------------------------------------------------------------
    # inference (eval)
    # ------------------------------------------------------------------

    def predict_boxes(
        self, predictions: tuple[Tensor, Tensor], proposals: Sequence[Instances]
    ) -> list[Tensor]:
        _scores, deltas = predictions
        proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        decoded = self.box_transform.apply_deltas(deltas, proposal_boxes)
        # Split back per image.
        sizes = [len(p) for p in proposals]
        return list(torch.split(decoded, sizes, dim=0))

    def predict_probs(
        self, predictions: tuple[Tensor, Tensor], proposals: Sequence[Instances]
    ) -> list[Tensor]:
        scores, _deltas = predictions
        probs = F.softmax(scores, dim=-1)
        sizes = [len(p) for p in proposals]
        return list(torch.split(probs, sizes, dim=0))

    def inference(
        self, predictions: tuple[Tensor, Tensor], proposals: Sequence[Instances]
    ) -> list[Instances]:
        boxes = self.predict_boxes(predictions, proposals)
        probs = self.predict_probs(predictions, proposals)
        image_shapes = [p.image_size for p in proposals]
        return fast_rcnn_inference(
            boxes,
            probs,
            image_shapes,
            score_thresh=self.score_thresh_test,
            nms_thresh=self.nms_thresh_test,
            topk_per_image=self.topk_per_image,
        )


# ---------------------------------------------------------------------------
# Per-image post-processing
# ---------------------------------------------------------------------------


def fast_rcnn_inference(
    boxes: Sequence[Tensor],
    probs: Sequence[Tensor],
    image_shapes: Sequence[tuple[int, int]],
    *,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
) -> list[Instances]:
    """Per-image NMS + top-k. See :func:`fast_rcnn_inference_single_image`."""
    return [
        fast_rcnn_inference_single_image(
            b,
            p,
            shape,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            topk_per_image=topk_per_image,
        )
        for b, p, shape in zip(boxes, probs, image_shapes, strict=True)
    ]


def fast_rcnn_inference_single_image(
    boxes: Tensor,
    probs: Tensor,
    image_shape: tuple[int, int],
    *,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
) -> Instances:
    """Implements `spec §2.5` `fast_rcnn_inference_single_image`."""
    valid = torch.isfinite(boxes).all(dim=1) & torch.isfinite(probs).all(dim=1)
    if not bool(valid.all()):
        boxes = boxes[valid]
        probs = probs[valid]

    # Drop the background class from probs.
    probs = probs[:, :-1]
    num_classes = probs.shape[1]
    if boxes.shape[1] == 4:
        # Class-agnostic: replicate the 4-tuple per class so the rest of
        # the pipeline can address by (roi, cls) indices uniformly.
        boxes = boxes[:, None, :].expand(-1, num_classes, -1)  # (R, K, 4)
    else:
        boxes = boxes.view(-1, num_classes, 4)

    # Clip to the image canvas before threshold so out-of-image dets
    # don't leak through (`spec §2.5`).
    boxes_t = boxes.reshape(-1, 4).clone()
    Boxes(boxes_t).clip(image_shape)
    boxes = boxes_t.view(-1, num_classes, 4)

    filter_mask = probs > score_thresh
    filter_inds = filter_mask.nonzero(as_tuple=False)  # (R', 2): (roi_idx, cls_idx)
    if filter_inds.numel() == 0:
        return _empty_detection(image_shape, boxes.device)

    boxes_kept = boxes[filter_mask]
    scores_kept = probs[filter_mask]
    cls_kept = filter_inds[:, 1]

    keep = batched_nms(
        boxes_kept, scores_kept, cls_kept.to(boxes_kept.dtype), iou_threshold=nms_thresh
    )
    keep = keep[:topk_per_image]

    inst = Instances(image_size=image_shape)
    inst.pred_boxes = Boxes(boxes_kept[keep])
    inst.scores = scores_kept[keep]
    inst.pred_classes = cls_kept[keep]
    return inst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gt_boxes_tensor(inst: Instances) -> Tensor:
    boxes = inst.gt_boxes
    if isinstance(boxes, Boxes):
        return boxes.tensor
    assert isinstance(boxes, Tensor)
    return boxes


def _empty_detection(image_shape: tuple[int, int], device: torch.device) -> Instances:
    inst = Instances(image_size=image_shape)
    inst.pred_boxes = Boxes(torch.zeros(0, 4, device=device))
    inst.scores = torch.zeros(0, device=device)
    inst.pred_classes = torch.zeros(0, dtype=torch.long, device=device)
    return inst
