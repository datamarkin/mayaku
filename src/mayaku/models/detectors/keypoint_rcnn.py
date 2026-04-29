"""Keypoint R-CNN detector — Faster R-CNN with the keypoint head wired in.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.4: the same `GeneralizedRCNN`
shell with :class:`StandardROIHeads` extended by a keypoint pooler +
:class:`KRCNNConvDeconvUpsampleHead` (Step 11/12 wiring). We don't
subclass :class:`FasterRCNN`; just call the build factory with a
populated ``cfg.model.roi_keypoint_head``.

Inference output mirrors Mask R-CNN — each :class:`Instances` carries
``pred_boxes``, ``scores``, ``pred_classes`` plus ``pred_keypoints``
(``(R, K, 3)`` of ``(x, y, score)``) and ``pred_keypoint_heatmaps``
(the raw ``(R, K, 56, 56)`` logits, kept for downstream / visualisation
use). The bicubic decode is intentionally Python-side per
`BACKEND_PORTABILITY_REPORT.md` §6.
"""

from __future__ import annotations

from mayaku.config.schemas import MayakuConfig
from mayaku.models.backbones.resnet import ResNetBackbone
from mayaku.models.detectors.faster_rcnn import FasterRCNN
from mayaku.models.necks import FPN, LastLevelMaxPool
from mayaku.models.proposals.rpn import build_rpn
from mayaku.models.roi_heads.standard import build_standard_roi_heads

__all__ = ["build_keypoint_rcnn"]


def build_keypoint_rcnn(cfg: MayakuConfig, *, backbone_weights: str | None = None) -> FasterRCNN:
    """Build a Keypoint R-CNN model from a top-level :class:`MayakuConfig`.

    The schema validator in Step 5 enforces that
    ``cfg.model.meta_architecture == "keypoint_rcnn"`` agrees with
    ``cfg.model.keypoint_on`` and that ``cfg.model.roi_keypoint_head``
    is populated.
    """
    if cfg.model.meta_architecture != "keypoint_rcnn":
        raise ValueError(
            f"build_keypoint_rcnn requires meta_architecture='keypoint_rcnn'; got "
            f"{cfg.model.meta_architecture!r}"
        )
    assert cfg.model.roi_keypoint_head is not None  # enforced by ModelConfig validator

    bottom_up = ResNetBackbone(
        name=cfg.model.backbone.name,
        norm=cfg.model.backbone.norm,
        freeze_at=cfg.model.backbone.freeze_at,
        weights=backbone_weights,  # type: ignore[arg-type]
        stride_in_1x1=cfg.model.backbone.stride_in_1x1,
    )
    fpn = FPN(
        bottom_up=bottom_up,
        in_features=cfg.model.fpn.in_features,
        out_channels=cfg.model.fpn.out_channels,
        norm=cfg.model.fpn.norm,  # type: ignore[arg-type]
        fuse_type=cfg.model.fpn.fuse_type,
        top_block=LastLevelMaxPool(),
    )
    in_shapes = fpn.output_shape()
    rpn = build_rpn(cfg.model.rpn, cfg.model.anchor_generator, in_shapes)
    roi_heads = build_standard_roi_heads(
        cfg.model.roi_heads,
        cfg.model.roi_box_head,
        in_shapes,
        keypoint_head_cfg=cfg.model.roi_keypoint_head,
        test_detections_per_image=cfg.test.detections_per_image,
    )
    return FasterRCNN(
        backbone=fpn,
        rpn=rpn,
        roi_heads=roi_heads,
        pixel_mean=cfg.model.pixel_mean,
        pixel_std=cfg.model.pixel_std,
    )
