"""ResNet-50 / ResNet-101 / ResNeXt-101 32x8d backbones.

Wraps torchvision's reference implementations rather than reimplementing
``BottleneckBlock`` from scratch:

* The non-goal "bit-for-bit reproduction of Detectron2 internals" plus
  the non-goal "loading Detectron2 checkpoints directly" both apply
  here — the master plan explicitly rules out distorting the model
  code to match either, and torchvision's implementation is the same
  bottleneck design with FAIR-public ImageNet-trained weights.
* ADR 002 (RGB-native) means torchvision's RGB-trained weights load
  directly with no channel swap.
* ``BACKEND_PORTABILITY_REPORT.md`` §3 confirms the ResNet/ResNeXt
  building blocks are pure PyTorch and run unchanged on CUDA, MPS, and
  CPU; no in-scope config enables deformable convolution (ADR 001), so
  we do not need a custom ``BottleneckBlock`` subclass to gate that path.

The wrapper:

1. Builds the underlying torchvision model (optionally with pretrained
   weights via ``torchvision.models.<arch>(weights=...)``).
2. Discards the ``avgpool``/``fc`` classification head.
3. Implements a hand-rolled forward that calls
   ``conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4``
   and returns the four intermediate feature maps as
   ``{res2, res3, res4, res5}`` to match the FPN naming convention
   (`DETECTRON2_TECHNICAL_SPEC.md` §2.1).
4. Optionally swaps every ``BatchNorm2d`` for :class:`FrozenBatchNorm2d`
   when ``norm="FrozenBN"`` (the spec default), and freezes parameters
   up to ``freeze_at`` (default 2 = stem + ``res2``).

Channel/stride table (R-50 / R-101 / X-101_32x8d, identical):

| name  | channels | stride |
|-------|---------:|-------:|
| res2  |      256 |      4 |
| res3  |      512 |      8 |
| res4  |     1024 |     16 |
| res5  |     2048 |     32 |
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torchvision.models as tv
from torch import Tensor, nn

from mayaku.config.schemas import BackboneConfig, BackboneName
from mayaku.models.backbones._base import Backbone
from mayaku.models.backbones._frozen_bn import (
    convert_frozen_batchnorm,
)

__all__ = ["ResNetBackbone", "build_backbone"]

NormChoice = Literal["FrozenBN", "BN", "GN", "SyncBN"]
WeightsChoice = Literal["DEFAULT"] | None


_OUT_CHANNELS = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
_OUT_STRIDES = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
_DEFAULT_OUT_FEATURES: tuple[str, ...] = ("res2", "res3", "res4", "res5")


class ResNetBackbone(Backbone):
    """Bottom-up ResNet/ResNeXt feature extractor.

    Args:
        name: Architecture choice. ``"resnet50"`` and ``"resnet101"`` map
            to torchvision's same-named factories; ``"resnext101_32x8d"``
            maps to ``torchvision.models.resnext101_32x8d``.
        norm: ``"FrozenBN"`` (spec default for in-scope configs),
            ``"BN"`` (trainable BN), or ``"GN"`` (group norm). The
            distributed-only norms (``SyncBN``) get the same FrozenBN
            treatment under the hood since FrozenBN's forward
            doesn't depend on world size.
        freeze_at: Number of stages to freeze, counting stem as 1
            (`DETECTRON2_TECHNICAL_SPEC.md` §2.1). 0 = nothing,
            2 = stem + res2 (default), 5 = everything.
        weights: ``"DEFAULT"`` to load torchvision's published
            IMAGENET1K_V2 (ResNet) / IMAGENET1K_V2 (ResNeXt) weights;
            ``None`` for random init. Pretrained weights are
            **RGB-trained** and load directly without channel swap
            (ADR 002).
        out_features: Subset of ``("res2","res3","res4","res5")`` to
            return; defaults to all four.
    """

    def __init__(
        self,
        name: BackboneName = "resnet50",
        *,
        norm: NormChoice = "FrozenBN",
        freeze_at: int = 2,
        weights: WeightsChoice = None,
        out_features: tuple[str, ...] = _DEFAULT_OUT_FEATURES,
        stride_in_1x1: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= freeze_at <= 5:
            raise ValueError(f"freeze_at must be in [0, 5]; got {freeze_at}")
        for f in out_features:
            if f not in _OUT_CHANNELS:
                raise ValueError(
                    f"unknown out_feature {f!r}; expected one of {tuple(_OUT_CHANNELS)}"
                )
        self.name = name
        self.norm: NormChoice = norm
        self.freeze_at = freeze_at
        self._out_features = tuple(out_features)
        self._out_feature_channels = {f: _OUT_CHANNELS[f] for f in self._out_features}
        self._out_feature_strides = {f: _OUT_STRIDES[f] for f in self._out_features}

        tv_weights = _resolve_weights(name, weights)
        tv_model = _construct_torchvision(name, tv_weights)
        # Carve up the torchvision module into our four-stage layout. We
        # keep the original BatchNorm2d objects in place; norm/freeze
        # conversion happens at the end so pretrained weight loading is
        # the source of truth for any running stats.
        self.stem = nn.Sequential(
            tv_model.conv1,
            tv_model.bn1,
            tv_model.relu,
            tv_model.maxpool,
        )
        self.res2 = tv_model.layer1
        self.res3 = tv_model.layer2
        self.res4 = tv_model.layer3
        self.res5 = tv_model.layer4

        if stride_in_1x1:
            # torchvision builds Bottleneck with stride on the 3x3 conv.
            # MSRA-pretrained checkpoints (D2 model-zoo) expect stride on
            # the 1x1. The kernels are identical between the two layouts;
            # only the per-conv stride differs. Patch the first block of
            # res3/res4/res5 (the only stride-2 blocks).
            for stage in (self.res3, self.res4, self.res5):
                first_block = stage[0]
                spatial_stride = first_block.conv2.stride
                if spatial_stride == (1, 1):
                    continue  # dilated stage, no downsample to relocate
                first_block.conv1.stride = spatial_stride
                first_block.conv2.stride = (1, 1)

        if norm in ("FrozenBN", "SyncBN"):
            # SyncBN collapses to FrozenBN for our purposes — we don't
            # ship a SyncBN training story at v1, see Step 14 (DDP).
            convert_frozen_batchnorm(self.stem)
            convert_frozen_batchnorm(self.res2)
            convert_frozen_batchnorm(self.res3)
            convert_frozen_batchnorm(self.res4)
            convert_frozen_batchnorm(self.res5)
        elif norm == "GN":
            raise NotImplementedError(
                "norm='GN' for ResNet backbones is not implemented yet — "
                "torchvision ships BN-trained weights only. Open an issue "
                "if you need GN; the conversion is straightforward but "
                "irrelevant for in-scope FPN configs."
            )
        # norm == "BN" leaves the BN layers as torchvision built them.

        self._apply_freeze(freeze_at)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        x = self.stem(x)
        x = self.res2(x)
        if "res2" in self._out_features:
            out["res2"] = x
        x = self.res3(x)
        if "res3" in self._out_features:
            out["res3"] = x
        x = self.res4(x)
        if "res4" in self._out_features:
            out["res4"] = x
        x = self.res5(x)
        if "res5" in self._out_features:
            out["res5"] = x
        return out

    # ------------------------------------------------------------------
    # freeze
    # ------------------------------------------------------------------

    def _apply_freeze(self, freeze_at: int) -> None:
        # freeze_at semantics (DETECTRON2_TECHNICAL_SPEC §2.1, step 6):
        #   0 = nothing frozen
        #   1 = stem
        #   2 = stem + res2
        #   3 = stem + res2 + res3
        #   4 = stem + res2 + res3 + res4
        #   5 = stem + res2..res5 (everything)
        stages: list[nn.Module] = [self.stem, self.res2, self.res3, self.res4, self.res5]
        for i, mod in enumerate(stages, start=1):
            if i > freeze_at:
                break
            for p in mod.parameters():
                p.requires_grad_(False)
            # Convert to FrozenBN in case norm="BN" — frozen stages
            # cannot have BN running stats updated by the optimiser
            # without producing a mismatch between train and eval.
            convert_frozen_batchnorm(mod)


# ---------------------------------------------------------------------------
# torchvision construction
# ---------------------------------------------------------------------------


def _resolve_weights(name: BackboneName, weights: WeightsChoice) -> tv.WeightsEnum | None:
    if weights is None:
        return None
    if weights != "DEFAULT":
        raise ValueError(f"weights must be None or 'DEFAULT'; got {weights!r}")
    return _DEFAULT_WEIGHTS[name]


_DEFAULT_WEIGHTS: dict[BackboneName, tv.WeightsEnum] = {
    "resnet50": tv.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101": tv.ResNet101_Weights.IMAGENET1K_V2,
    "resnext101_32x8d": tv.ResNeXt101_32X8D_Weights.IMAGENET1K_V2,
}


def _construct_torchvision(name: BackboneName, weights: tv.WeightsEnum | None) -> tv.ResNet:
    factory: dict[BackboneName, Callable[..., tv.ResNet]] = {
        "resnet50": tv.resnet50,
        "resnet101": tv.resnet101,
        "resnext101_32x8d": tv.resnext101_32x8d,
    }
    if name not in factory:
        raise ValueError(f"unknown backbone name {name!r}")
    model: tv.ResNet = factory[name](weights=weights)
    return model


# ---------------------------------------------------------------------------
# Factory from BackboneConfig (Step 5)
# ---------------------------------------------------------------------------


def build_backbone(
    cfg: BackboneConfig,
    *,
    weights: WeightsChoice = None,
    out_features: tuple[str, ...] = _DEFAULT_OUT_FEATURES,
) -> ResNetBackbone:
    """Construct a :class:`ResNetBackbone` from a typed config.

    ``weights`` is *not* a config field because it's a deployment
    concern (whether to download pretrained ImageNet checkpoints) and
    in tests we always want the random-init path. The ``ModelConfig``-
    level ``weights`` field (Step 5) is reserved for full-model
    checkpoints, not backbone-only.
    """
    return ResNetBackbone(
        name=cfg.name,
        norm=cfg.norm,
        freeze_at=cfg.freeze_at,
        weights=weights,
        out_features=out_features,
        stride_in_1x1=cfg.stride_in_1x1,
    )
