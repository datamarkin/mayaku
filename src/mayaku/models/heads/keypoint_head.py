"""Keypoint R-CNN head + losses + inference.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.5 (`keypoint_head.py`):

* :class:`KRCNNConvDeconvUpsampleHead` — eight 3x3-512-ReLU stages
  (the spec default, configurable via ``conv_dims``), then a
  ``ConvTranspose2d(channels → num_keypoints, k=4, s=2, padding=1)``
  to double the spatial size to ``28``, then a bilinear ``F.interpolate``
  by 2 to land on the canonical ``56x56`` heatmap. Final logits:
  ``(R, K, 56, 56)``. The forward intentionally does **not** apply
  softmax — the loss is a spatial cross-entropy over the flattened
  heatmap, which expects raw logits.

* :func:`keypoint_rcnn_loss` — masked spatial CE per spec §2.5. The
  default normaliser is "visible" (one element per visible keypoint
  in the batch); ``"static"`` falls back to the upstream alternative
  ``num_images * num_kp * batch_size_per_image * positive_fraction``.

* :func:`keypoint_rcnn_inference` — calls
  :func:`mayaku.structures.keypoints.heatmaps_to_keypoints` to decode
  the per-ROI heatmap to ``(x, y, logit, score)`` columns, attaches
  ``pred_keypoints`` (columns ``[0, 1, 3]``) and the raw heatmap to
  the corresponding :class:`Instances`. The decode itself is the
  postprocess-side bicubic + half-pixel correction documented in
  spec §2.6 — and explicitly kept out of any exported graph
  (`BACKEND_PORTABILITY_REPORT.md` §6).

* :func:`select_proposals_with_visible_keypoints` — Detectron2's
  additional filter on top of foreground selection: drop any
  foreground proposal whose ``gt_keypoints`` rows are all visibility-0
  (no useful supervision signal).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mayaku.config.schemas import ROIKeypointHeadConfig
from mayaku.models.backbones._base import ShapeSpec
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import (
    Keypoints,
    heatmaps_to_keypoints,
    keypoints_to_heatmap,
)

__all__ = [
    "KRCNNConvDeconvUpsampleHead",
    "build_keypoint_head",
    "keypoint_rcnn_inference",
    "keypoint_rcnn_loss",
    "select_proposals_with_visible_keypoints",
]

LossNormalizer = Literal["visible", "static"]

# Spec default: 56x56 heatmap. Achieved by 14 (pooler) → 28 (deconv stride 2) → 56 (bilinear x2).
_HEATMAP_SIDE: int = 56


class KRCNNConvDeconvUpsampleHead(nn.Module):
    """Conv stack + 2x deconv + 2x bilinear upsample → ``(R, K, 56, 56)``.

    Args:
        input_shape: Channels of the pooler output.
        num_keypoints: ``K`` — number of keypoint channels (17 for
            COCO Person).
        conv_dims: Per-stage conv channel counts. Detectron2's default
            is ``(512,) * 8`` (`spec §6.1`).
    """

    def __init__(
        self,
        input_shape: ShapeSpec,
        num_keypoints: int,
        conv_dims: Sequence[int] = (512,) * 8,
    ) -> None:
        super().__init__()
        if len(conv_dims) < 1:
            raise ValueError("KRCNNConvDeconvUpsampleHead requires at least one conv stage")
        self.num_keypoints = num_keypoints

        self.convs = nn.ModuleList()
        ch = input_shape.channels
        for c in conv_dims:
            self.convs.append(nn.Conv2d(ch, c, kernel_size=3, padding=1))
            ch = c
        # Spec: ConvTranspose2d(k=4, s=2, padding=1) is the upstream
        # convention here. With pooler resolution 14 the deconv yields 28.
        self.deconv = nn.ConvTranspose2d(ch, num_keypoints, kernel_size=4, stride=2, padding=1)

        # Init follows upstream: MSRA on convs, Caffe2-xavier-equivalent
        # (kaiming_normal with relu nonlinearity) on the deconv.
        for m in self.convs:
            assert isinstance(m, nn.Conv2d)
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        nn.init.kaiming_normal_(self.deconv.weight, mode="fan_out", nonlinearity="relu")
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))
        x = self.deconv(x)
        # Double again to land on the canonical 56x56. ``align_corners=False``
        # matches the bilinear convention used everywhere else in the codebase.
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return x


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def keypoint_rcnn_loss(
    pred_keypoint_logits: Tensor,
    fg_proposals: Sequence[Instances],
    *,
    normalizer: LossNormalizer = "visible",
    static_normalizer_constant: float = 0.0,
    loss_weight: float = 1.0,
) -> Tensor:
    """Spatial cross-entropy over the flattened heatmap.

    Args:
        pred_keypoint_logits: ``(R, K, S, S)`` raw logits from
            :class:`KRCNNConvDeconvUpsampleHead`. ``R`` is the
            *foreground-with-visible-keypoints* proposal count.
        fg_proposals: Per-image :class:`Instances` carrying
            ``proposal_boxes`` and ``gt_keypoints``.
        normalizer: ``"visible"`` divides by the number of visible
            keypoints (the canonical Mask R-CNN setting,
            `spec §2.5`). ``"static"`` divides by
            ``num_images * static_normalizer_constant`` — used when
            the visible-keypoint count fluctuates across batches and
            you want a stable scale.
        static_normalizer_constant: Used only with ``normalizer="static"``.
            The upstream value is ``num_kp * batch_size_per_image *
            positive_fraction`` (e.g. ``17 * 512 * 0.25 = 2176``).
        loss_weight: Final scalar multiplier (`spec §6.1`'s
            ``ROI_KEYPOINT_HEAD.LOSS_WEIGHT``, default 1.0).

    Returns:
        Scalar loss; returns a real ``0.0`` (with grad) when there are
        no foreground proposals so the backward pass is safe.
    """
    if pred_keypoint_logits.shape[0] == 0:
        return pred_keypoint_logits.sum() * 0.0

    n, k, h, w = pred_keypoint_logits.shape
    if h != w:
        raise ValueError(f"keypoint heatmap must be square; got ({h}, {w})")
    heatmap_side = h

    targets: list[Tensor] = []
    valids: list[Tensor] = []
    for inst in fg_proposals:
        if len(inst) == 0:
            continue
        boxes = inst.proposal_boxes.tensor
        kp = inst.gt_keypoints
        assert isinstance(kp, Keypoints), f"gt_keypoints must be Keypoints; got {type(kp).__name__}"
        ind, valid = keypoints_to_heatmap(kp.tensor, boxes, heatmap_side)
        targets.append(ind.flatten())
        valids.append(valid.flatten())
    if not targets:
        return pred_keypoint_logits.sum() * 0.0

    flat_targets = torch.cat(targets, dim=0)  # (N*K,)
    flat_valid = torch.cat(valids, dim=0)
    valid_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return pred_keypoint_logits.sum() * 0.0

    flat_logits = pred_keypoint_logits.view(n * k, h * w)
    loss = F.cross_entropy(flat_logits[valid_idx], flat_targets[valid_idx].long(), reduction="sum")

    if normalizer == "visible":
        denom = float(valid_idx.numel())
    else:
        denom = max(len(fg_proposals), 1) * static_normalizer_constant
        if denom <= 0:
            raise ValueError("normalizer='static' requires static_normalizer_constant > 0")
    return loss / denom * loss_weight


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def keypoint_rcnn_inference(
    pred_keypoint_logits: Tensor, pred_instances: Sequence[Instances]
) -> None:
    """Decode the heatmap and attach per-instance keypoint predictions.

    Mutates ``pred_instances`` in place. Sets:

    * ``pred_keypoints``: ``(R, K, 3)`` of ``(x, y, score)`` in
      image-pixel coordinates (per spec §2.6 columns ``[0, 1, 3]``).
    * ``pred_keypoint_heatmaps``: the raw ``(R, K, S, S)`` logits, useful
      for downstream visualisation / further refinement.
    """
    if pred_keypoint_logits.shape[0] == 0:
        for inst in pred_instances:
            shape = pred_keypoint_logits.shape
            inst.pred_keypoints = pred_keypoint_logits.new_zeros((0, shape[1], 3))
            inst.pred_keypoint_heatmaps = pred_keypoint_logits.new_zeros(
                (0, shape[1], shape[2], shape[3])
            )
        return

    pred_boxes_t = torch.cat([i.pred_boxes.tensor for i in pred_instances], dim=0)
    decoded = heatmaps_to_keypoints(pred_keypoint_logits.detach(), pred_boxes_t)
    # decoded: (R, K, 4) → keep (x, y, score) per spec §2.6.
    xy_score = decoded[:, :, [0, 1, 3]]

    sizes = [len(i) for i in pred_instances]
    for inst, kp_chunk, hm_chunk in zip(
        pred_instances,
        torch.split(xy_score, sizes, dim=0),
        torch.split(pred_keypoint_logits, sizes, dim=0),
        strict=True,
    ):
        inst.pred_keypoints = kp_chunk
        inst.pred_keypoint_heatmaps = hm_chunk


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def select_proposals_with_visible_keypoints(
    proposals: Sequence[Instances],
) -> tuple[list[Instances], list[Tensor]]:
    """Drop foreground proposals with no visible keypoints.

    Detectron2's keypoint training applies this filter on top of
    :func:`select_foreground_proposals` — keypoint heatmap targets
    that contain only invisible (``v=0``) annotations carry no
    gradient signal and risk inflating the loss denominator.
    """
    out_proposals: list[Instances] = []
    out_masks: list[Tensor] = []
    for inst in proposals:
        if len(inst) == 0 or not inst.has("gt_keypoints"):
            out_proposals.append(inst)
            out_masks.append(
                torch.zeros(len(inst), dtype=torch.bool, device=inst.gt_classes.device)
                if inst.has("gt_classes")
                else torch.zeros(0, dtype=torch.bool)
            )
            continue
        kp = inst.gt_keypoints
        assert isinstance(kp, Keypoints)
        # Visibility is column 2 of the (N, K, 3) tensor.
        has_visible = (kp.tensor[..., 2] > 0).any(dim=1)
        out_proposals.append(inst[has_visible])
        out_masks.append(has_visible)
    return out_proposals, out_masks


def build_keypoint_head(
    cfg: ROIKeypointHeadConfig, input_shape: ShapeSpec
) -> KRCNNConvDeconvUpsampleHead:
    """Construct a :class:`KRCNNConvDeconvUpsampleHead` from a typed config."""
    return KRCNNConvDeconvUpsampleHead(
        input_shape=input_shape,
        num_keypoints=cfg.num_keypoints,
        conv_dims=cfg.conv_dims,
    )


# Reference the spec heatmap side so future readers know where 56 came from.
_ = _HEATMAP_SIDE
