"""ConvNeXt-Tiny / Small / Base / Large backbones.

Wraps torchvision's reference ConvNeXt rather than reimplementing the
``CNBlock``/``LayerNorm2d`` building blocks from scratch:

* The four variants are exactly torchvision's
  ``convnext_{tiny,small,base,large}`` — same depths ``[3, 3, {9 or
  27}, 3]``, same dims ``[96/96/128/192, ...]``, same 4×4 stride-4
  stem, same 7×7 depthwise + ``LayerNorm`` + MLP block with
  per-channel layer-scale.
* Architecture only — random init. Trained weights arrive by loading a mayaku
  checkpoint on top of the built model; the backbone never fetches or loads
  weights itself.
* RGB-native (ADR 002): ConvNeXt uses standard ImageNet RGB
  normalisation, matching :class:`ModelConfig.pixel_mean/std`.
* ``BACKEND_PORTABILITY_REPORT.md`` §3: the ConvNeXt primitives
  (depthwise conv, LayerNorm, GELU, Linear) are pure PyTorch and run
  unchanged on CUDA / MPS / CPU; ONNX / TensorRT / CoreML exports
  cover these ops natively.

Channel/stride table per variant:

| variant | res2 (s=4) | res3 (s=8) | res4 (s=16) | res5 (s=32) |
|---------|-----------:|-----------:|------------:|------------:|
| tiny    |         96 |        192 |         384 |         768 |
| small   |         96 |        192 |         384 |         768 |
| base    |        128 |        256 |         512 |        1024 |
| large   |        192 |        384 |         768 |        1536 |
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torchvision.models as tv
from torch import Tensor, nn

from mayaku.config.schemas import BackboneConfig
from mayaku.models.backbones._base import Backbone

__all__ = ["ConvNeXtBackbone", "ConvNeXtVariant"]

ConvNeXtVariant = Literal[
    "convnext_atto",
    "convnext_femto",
    "convnext_pico",
    "convnext_nano",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]
WeightsChoice = Literal["DEFAULT"] | None


# Per-variant stage-output channel tables. Strides are identical across
# variants (4 / 8 / 16 / 32) so they live in a single mapping.
_VARIANT_CHANNELS: dict[ConvNeXtVariant, dict[str, int]] = {
    "convnext_atto": {"res2": 40, "res3": 80, "res4": 160, "res5": 320},
    "convnext_femto": {"res2": 48, "res3": 96, "res4": 192, "res5": 384},
    "convnext_pico": {"res2": 64, "res3": 128, "res4": 256, "res5": 512},
    "convnext_nano": {"res2": 80, "res3": 160, "res4": 320, "res5": 640},
    "convnext_tiny": {"res2": 96, "res3": 192, "res4": 384, "res5": 768},
    "convnext_small": {"res2": 96, "res3": 192, "res4": 384, "res5": 768},
    "convnext_base": {"res2": 128, "res3": 256, "res4": 512, "res5": 1024},
    "convnext_large": {"res2": 192, "res3": 384, "res4": 768, "res5": 1536},
}
_OUT_STRIDES = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
_DEFAULT_OUT_FEATURES: tuple[str, ...] = ("res2", "res3", "res4", "res5")

# torchvision's Sequential indexes alternate downsample / stage:
#   features[0] = stem (Conv 4×4 s4 + LayerNorm2d)
#   features[1] = stage 0 blocks    → stride 4  → "res2"
#   features[2] = downsample (LN + Conv 2×2 s2)
#   features[3] = stage 1 blocks    → stride 8  → "res3"
#   features[4] = downsample
#   features[5] = stage 2 blocks    → stride 16 → "res4"
#   features[6] = downsample
#   features[7] = stage 3 blocks    → stride 32 → "res5"
_STAGE_FEATURE_INDEX: dict[str, tuple[int, int]] = {
    # name → (downsample_idx, stage_idx); downsample_idx is None-equivalent
    # for the stem (handled separately).
    "res2": (0, 1),
    "res3": (2, 3),
    "res4": (4, 5),
    "res5": (6, 7),
}

_TORCHVISION_FACTORIES: dict[str, Callable[..., tv.ConvNeXt]] = {
    "convnext_tiny": tv.convnext_tiny,
    "convnext_small": tv.convnext_small,
    "convnext_base": tv.convnext_base,
    "convnext_large": tv.convnext_large,
}

# Custom variants not in torchvision (V2 size configs, V1 block style)
_CUSTOM_BLOCK_SETTINGS: dict[str, list[tuple[int, int | None, int]]] = {
    "convnext_atto": [(40, 80, 2), (80, 160, 2), (160, 320, 6), (320, None, 2)],
    "convnext_femto": [(48, 96, 2), (96, 192, 2), (192, 384, 6), (384, None, 2)],
    "convnext_pico": [(64, 128, 2), (128, 256, 2), (256, 512, 6), (512, None, 2)],
    "convnext_nano": [(80, 160, 2), (160, 320, 2), (320, 640, 8), (640, None, 2)],
}


def _build_custom_convnext(name: str) -> tv.ConvNeXt:
    from torchvision.models.convnext import CNBlockConfig

    settings = _CUSTOM_BLOCK_SETTINGS[name]
    block_setting = [CNBlockConfig(c_in, c_out, depth) for c_in, c_out, depth in settings]
    return tv.ConvNeXt(block_setting=block_setting, stochastic_depth_prob=0.1)


class ConvNeXtBackbone(Backbone):
    """Bottom-up ConvNeXt feature extractor.

    Args:
        name: Variant choice. One of ``"convnext_{tiny,small,base,large}"``.
        freeze_at: Number of stages to freeze, counting stem as 1
            (mirrors :class:`ResNetBackbone`). 0 = nothing, 1 = stem,
            2 = stem + res2, ..., 5 = everything.
        out_features: Subset of ``("res2","res3","res4","res5")`` to
            return; defaults to all four.
    """

    def __init__(
        self,
        name: ConvNeXtVariant = "convnext_tiny",
        *,
        freeze_at: int = 0,
        out_features: tuple[str, ...] = _DEFAULT_OUT_FEATURES,
    ) -> None:
        super().__init__()
        if name not in _VARIANT_CHANNELS:
            raise ValueError(
                f"unknown ConvNeXt variant {name!r}; expected one of {tuple(_VARIANT_CHANNELS)}"
            )
        if not 0 <= freeze_at <= 5:
            raise ValueError(f"freeze_at must be in [0, 5]; got {freeze_at}")
        for f in out_features:
            if f not in _OUT_STRIDES:
                raise ValueError(
                    f"unknown out_feature {f!r}; expected one of {tuple(_OUT_STRIDES)}"
                )
        self.name = name
        self.freeze_at = freeze_at
        self._out_features = tuple(out_features)
        self._out_feature_channels = {f: _VARIANT_CHANNELS[name][f] for f in self._out_features}
        self._out_feature_strides = {f: _OUT_STRIDES[f] for f in self._out_features}

        # Architecture only — random init. Trained weights come from a mayaku
        # checkpoint loaded on top by the caller; the backbone never fetches or
        # loads weights itself.
        if name in _TORCHVISION_FACTORIES:
            tv_model = _TORCHVISION_FACTORIES[name](weights=None)
        else:
            tv_model = _build_custom_convnext(name)

        # torchvision ConvNeXt's ``features`` is a Sequential of 8
        # children alternating downsample/stage. The stem (features[0])
        # already does stride-4, so res2 has no preceding downsample —
        # store an ``Identity`` sentinel in ``_res_downs["res2"]`` to keep
        # the freeze and forward loops uniform across all four stages.
        feats = tv_model.features
        self.stem = feats[0]
        self._res_downs = nn.ModuleDict(
            {
                "res2": nn.Identity(),
                "res3": feats[2],
                "res4": feats[4],
                "res5": feats[6],
            }
        )
        self._res_stages = nn.ModuleDict(
            {
                "res2": feats[1],
                "res3": feats[3],
                "res4": feats[5],
                "res5": feats[7],
            }
        )

        self._apply_freeze(freeze_at)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        x = self.stem(x)
        for name in ("res2", "res3", "res4", "res5"):
            x = self._res_downs[name](x)
            x = self._res_stages[name](x)
            if name in self._out_features:
                out[name] = x
        return out

    # ------------------------------------------------------------------
    # freeze
    # ------------------------------------------------------------------

    def _apply_freeze(self, freeze_at: int) -> None:
        # freeze_at semantics (mirrors ResNetBackbone for cross-arch
        # consistency):
        #   0 = nothing
        #   1 = stem
        #   2 = stem + res2
        #   3 = stem + res2 + res3 (and the res3 downsample)
        #   4 = stem + res2 + res3 + res4
        #   5 = everything
        stages: list[nn.Module] = [
            self.stem,
            nn.ModuleList([self._res_downs["res2"], self._res_stages["res2"]]),
            nn.ModuleList([self._res_downs["res3"], self._res_stages["res3"]]),
            nn.ModuleList([self._res_downs["res4"], self._res_stages["res4"]]),
            nn.ModuleList([self._res_downs["res5"], self._res_stages["res5"]]),
        ]
        for i, mod in enumerate(stages, start=1):
            if i > freeze_at:
                break
            for p in mod.parameters():
                p.requires_grad_(False)


def build_convnext(
    cfg: BackboneConfig,
    *,
    out_features: tuple[str, ...] = _DEFAULT_OUT_FEATURES,
) -> ConvNeXtBackbone:
    """Construct a :class:`ConvNeXtBackbone` (architecture only) from a typed config."""
    name = cfg.name
    if name not in _VARIANT_CHANNELS:
        raise ValueError(f"build_convnext requires a ConvNeXt variant; got {name!r}")
    return ConvNeXtBackbone(name=name, freeze_at=cfg.freeze_at, out_features=out_features)


def is_convnext_variant(name: str) -> bool:
    """True iff ``name`` is in the ConvNeXt family of :class:`BackboneName`.

    Identifies family by the ``convnext_`` prefix — the same predicate
    the :class:`BackboneConfig` validator uses. The closed set of valid
    names is enforced separately by the :class:`BackboneName` Literal;
    this function takes ``str`` so test fixtures and CLI callers handling
    user-supplied strings can use it without casting first.
    """
    return name.startswith("convnext_")
