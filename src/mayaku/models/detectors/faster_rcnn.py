"""Faster R-CNN detector — backbone + FPN + RPN + StandardROIHeads.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.1's `GeneralizedRCNN`
(`modeling/meta_arch/rcnn.py`) for the box-only path. Mask and
keypoint variants subclass / replace the ROI heads in Steps 11 / 12.

Input contract: ``forward(batched_inputs)`` accepts a list of dicts in
the format produced by :class:`mayaku.data.DatasetMapper`:

```
{
    "image":     Tensor[3, H, W] float32 RGB (unnormalised),
    "instances": Instances(gt_boxes, gt_classes, ...),  # train only
    "image_id":  int,                                    # optional
    ...
}
```

Output contract:

* training: returns the loss dict ``{"loss_rpn_cls", "loss_rpn_loc",
  "loss_cls", "loss_box_reg"}``.
* inference: returns ``list[dict]``, one per image, with the
  detector's :class:`Instances` under ``"instances"``.

Pixel-mean/std normalisation happens here — the data layer
intentionally hands off raw RGB so the mapper has no model dependency
(see Step 6 notes).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from mayaku.config.schemas import MayakuConfig
from mayaku.models.backbones.resnet import ResNetBackbone
from mayaku.models.necks import FPN, LastLevelMaxPool
from mayaku.models.proposals.rpn import RPN, build_rpn
from mayaku.models.roi_heads.standard import (
    StandardROIHeads,
    build_standard_roi_heads,
)
from mayaku.structures.image_list import ImageList
from mayaku.structures.instances import Instances

__all__ = ["FasterRCNN", "build_faster_rcnn"]


class FasterRCNN(nn.Module):
    """Box-only detector. Inherits the contract of `GeneralizedRCNN`."""

    def __init__(
        self,
        backbone: nn.Module,  # outputs FPN-style {p2..p6} (FPN itself is a Backbone)
        rpn: RPN,
        roi_heads: StandardROIHeads,
        *,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # Buffers so .to(device) carries them along; (3, 1, 1) for broadcast.
        mean_t = torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
        std_t = torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)
        self.register_buffer("pixel_mean", mean_t, persistent=False)
        self.register_buffer("pixel_std", std_t, persistent=False)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self, batched_inputs: Sequence[dict[str, Any]]
    ) -> dict[str, Tensor] | list[dict[str, Any]]:
        images = self._preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.training:
            gt_instances = [
                self._move_instances(x["instances"], images.device) for x in batched_inputs
            ]
            proposals, rpn_losses = self.rpn(images.image_sizes, features, gt_instances)
            _instances, head_losses = self.roi_heads(features, proposals, gt_instances)
            return {**rpn_losses, **head_losses}

        proposals, _ = self.rpn(images.image_sizes, features, gt_instances=None)
        results, _ = self.roi_heads(features, proposals, targets=None)
        # Return per-image dicts so downstream postprocessors / evaluators
        # can rescale to the original image size (Step 16).
        return [{"instances": inst} for inst in results]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_image(self, batched_inputs: Sequence[dict[str, Any]]) -> ImageList:
        device = self.pixel_mean.device
        images = [
            (x["image"].to(device).to(torch.float32) - self.pixel_mean) / self.pixel_std
            for x in batched_inputs
        ]
        size_divisibility = self._size_divisibility()
        return ImageList.from_tensors(images, size_divisibility=size_divisibility, pad_value=0.0)

    def _size_divisibility(self) -> int:
        sd = getattr(self.backbone, "size_divisibility", 1)
        # nn.Module's __getattr__ widens to Tensor | Module — coerce
        # explicitly so mypy strict accepts the int() call.
        if isinstance(sd, int):
            return sd
        if isinstance(sd, Tensor):
            return int(sd.item())
        return 1

    @staticmethod
    def _move_instances(inst: Instances, device: torch.device) -> Instances:
        return inst.to(device)


# ---------------------------------------------------------------------------
# Build factory
# ---------------------------------------------------------------------------


def build_faster_rcnn(cfg: MayakuConfig, *, backbone_weights: str | None = None) -> FasterRCNN:
    """Build a Faster R-CNN model from a top-level :class:`MayakuConfig`.

    ``backbone_weights="DEFAULT"`` loads torchvision's IMAGENET1K_V2
    pretrained weights into the bottom-up ResNet (ADR 002 RGB-native);
    the default ``None`` initialises everything fresh, which is what the
    test suite uses.
    """
    if cfg.model.meta_architecture != "faster_rcnn":
        raise ValueError(
            f"build_faster_rcnn requires meta_architecture='faster_rcnn'; got "
            f"{cfg.model.meta_architecture!r}"
        )
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
        test_detections_per_image=cfg.test.detections_per_image,
    )
    return FasterRCNN(
        backbone=fpn,
        rpn=rpn,
        roi_heads=roi_heads,
        pixel_mean=cfg.model.pixel_mean,
        pixel_std=cfg.model.pixel_std,
    )
