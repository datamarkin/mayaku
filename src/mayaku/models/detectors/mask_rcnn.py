"""Mask R-CNN detector — Faster R-CNN with the mask head wired in.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.4: the same `GeneralizedRCNN`
shell with the box head extended by a mask pooler + mask head living
inside :class:`StandardROIHeads`. We don't subclass
:class:`FasterRCNN` here — the only difference is which ROI heads get
constructed, so a separate build factory is sufficient and keeps the
class hierarchy flat.

Inference returns the same per-image dict format as Faster R-CNN, with
each ``Instances`` carrying ``pred_boxes``, ``scores``, ``pred_classes``
plus a ``pred_masks`` field of shape ``(R, 1, M, M)``. The 28x28 soft
masks are pasted back to image resolution by
:meth:`mayaku.structures.masks.ROIMasks.to_bitmasks` from the
postprocess stage (Step 16); the head's output stays in mask-RoI
coords so the model graph is export-friendly.
"""

from __future__ import annotations

from mayaku.config.schemas import MayakuConfig
from mayaku.models.backbones import build_bottom_up
from mayaku.models.detectors.faster_rcnn import FasterRCNN
from mayaku.models.necks import FPN, LastLevelMaxPool
from mayaku.models.proposals.rpn import build_rpn
from mayaku.models.roi_heads.standard import build_standard_roi_heads

__all__ = ["build_mask_rcnn"]


def build_mask_rcnn(cfg: MayakuConfig, *, backbone_weights: str | None = None) -> FasterRCNN:
    """Build a Mask R-CNN model from a top-level :class:`MayakuConfig`.

    The returned object is a :class:`FasterRCNN` instance whose ROI
    heads have the mask path enabled. ``cfg.model.meta_architecture``
    must be ``"mask_rcnn"`` and ``cfg.model.roi_mask_head`` must be
    populated (the schema validator in Step 5 enforces both jointly).
    """
    if cfg.model.meta_architecture != "mask_rcnn":
        raise ValueError(
            f"build_mask_rcnn requires meta_architecture='mask_rcnn'; got "
            f"{cfg.model.meta_architecture!r}"
        )
    assert cfg.model.roi_mask_head is not None  # enforced by ModelConfig validator

    bottom_up = build_bottom_up(
        cfg.model.backbone,
        weights=backbone_weights,  # type: ignore[arg-type]
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
        mask_head_cfg=cfg.model.roi_mask_head,
        test_detections_per_image=cfg.test.detections_per_image,
    )
    return FasterRCNN(
        backbone=fpn,
        rpn=rpn,
        roi_heads=roi_heads,
        pixel_mean=cfg.model.pixel_mean,
        pixel_std=cfg.model.pixel_std,
    )
