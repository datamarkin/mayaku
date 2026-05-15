"""Backbones (R-50, R-101, ResNeXt-101 32x8d, DINOv3 ConvNeXt) + protocol.

Two architecture families ship today:

* :class:`ResNetBackbone` — torchvision ResNet/ResNeXt with FrozenBN
  defaults, used by the bundled Detectron2-parity recipes.
* :class:`ConvNeXtBackbone` — torchvision ConvNeXt wired to accept
  DINOv3-format pretrained weights (LVD-1689M distillation from a
  frozen ViT-7B teacher). License-gated; users supply the weight file.

:func:`build_bottom_up` is the single typed dispatch from
:class:`BackboneConfig` to the right concrete class — detector builders
(``build_faster_rcnn``/``build_mask_rcnn``/``build_keypoint_rcnn``) call
it so they don't have to know which family they're working with.
"""

from __future__ import annotations

from mayaku.config.schemas import BackboneConfig
from mayaku.models.backbones._base import Backbone, ShapeSpec
from mayaku.models.backbones._frozen_bn import (
    FrozenBatchNorm2d,
    convert_frozen_batchnorm,
)
from mayaku.models.backbones.convnext import (
    ConvNeXtBackbone,
    ConvNeXtVariant,
    build_convnext,
    is_convnext_variant,
)
from mayaku.models.backbones.resnet import (
    ResNetBackbone,
    WeightsChoice,
    build_backbone,
)

__all__ = [
    "Backbone",
    "ConvNeXtBackbone",
    "ConvNeXtVariant",
    "FrozenBatchNorm2d",
    "ResNetBackbone",
    "ShapeSpec",
    "WeightsChoice",
    "build_backbone",
    "build_bottom_up",
    "build_convnext",
    "convert_frozen_batchnorm",
    "is_convnext_variant",
]


def build_bottom_up(
    cfg: BackboneConfig,
    *,
    weights: WeightsChoice = None,
    out_features: tuple[str, ...] = ("res2", "res3", "res4", "res5"),
) -> Backbone:
    """Typed dispatch from :class:`BackboneConfig` to a concrete backbone.

    Args:
        cfg: Validated backbone config — ``cfg.name`` selects the family.
        weights: ``"DEFAULT"`` requests the canonical ImageNet-pretrained
            weights for the family (torchvision IMAGENET1K_V2 for ResNet,
            IMAGENET1K_V1 for ConvNeXt). ``None`` initialises randomly.
            For ConvNeXt, ``cfg.weights_path`` (DINOv3 checkpoint) takes
            precedence over this flag.
        out_features: Which stage outputs to materialise; defaults to
            all four FPN-feeding levels.
    """
    if is_convnext_variant(cfg.name):
        return build_convnext(cfg, weights=weights, out_features=out_features)
    return build_backbone(cfg, weights=weights, out_features=out_features)
