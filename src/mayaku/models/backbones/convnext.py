"""ConvNeXt-Tiny / Small / Base / Large backbones.

Wraps torchvision's reference ConvNeXt rather than reimplementing the
``CNBlock``/``LayerNorm2d`` building blocks from scratch:

* The four variants are exactly torchvision's
  ``convnext_{tiny,small,base,large}`` — same depths ``[3, 3, {9 or
  27}, 3]``, same dims ``[96/96/128/192, ...]``, same 4×4 stride-4
  stem, same 7×7 depthwise + ``LayerNorm`` + MLP block with
  per-channel layer-scale.
* Pretrained weights are supplied via :attr:`BackboneConfig.weights_path`.
  The loader accepts both checkpoint key-naming conventions:
    - **torchvision**: ``features.k.j.block.{0,2,3,5}`` / ``layer_scale``
      (shape ``(C, 1, 1)``)
    - **facebookresearch / Liu et al.**: ``downsample_layers.k`` /
      ``stages.k.j.{dwconv,norm,pwconv1,pwconv2,gamma}`` with ``gamma``
      at shape ``(C,)`` — the format used by the original ConvNeXt
      release and downstream forks (DINOv3 LVD-1689M distillation,
      user fine-tunes, etc.)
  The latter is handled by :func:`_remap_facebook_convnext_state_dict`,
  which renames keys and reshapes ``gamma`` → ``(C, 1, 1)`` before load.
  Mayaku ships no checkpoint URLs and no auto-download — supply a local
  ``.pth`` / ``.safetensors`` path. Some sources (DINOv3) are
  license-gated and must be downloaded manually after accepting the
  upstream license.
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

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torchvision.models as tv
from torch import Tensor, nn

from mayaku.config.schemas import BackboneConfig
from mayaku.models.backbones._base import Backbone

__all__ = ["ConvNeXtBackbone", "ConvNeXtVariant"]

ConvNeXtVariant = Literal[
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]
WeightsChoice = Literal["DEFAULT"] | None


# Per-variant stage-output channel tables. Strides are identical across
# variants (4 / 8 / 16 / 32) so they live in a single mapping.
_VARIANT_CHANNELS: dict[ConvNeXtVariant, dict[str, int]] = {
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

_TORCHVISION_FACTORIES: dict[ConvNeXtVariant, Callable[..., tv.ConvNeXt]] = {
    "convnext_tiny": tv.convnext_tiny,
    "convnext_small": tv.convnext_small,
    "convnext_base": tv.convnext_base,
    "convnext_large": tv.convnext_large,
}

_TORCHVISION_DEFAULT_WEIGHTS: dict[ConvNeXtVariant, tv.WeightsEnum] = {
    "convnext_tiny": tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    "convnext_small": tv.ConvNeXt_Small_Weights.IMAGENET1K_V1,
    "convnext_base": tv.ConvNeXt_Base_Weights.IMAGENET1K_V1,
    "convnext_large": tv.ConvNeXt_Large_Weights.IMAGENET1K_V1,
}


class ConvNeXtBackbone(Backbone):
    """Bottom-up ConvNeXt feature extractor.

    Args:
        name: Variant choice. One of ``"convnext_{tiny,small,base,large}"``.
        freeze_at: Number of stages to freeze, counting stem as 1
            (mirrors :class:`ResNetBackbone`). 0 = nothing, 1 = stem,
            2 = stem + res2, ..., 5 = everything.
        weights: ``"DEFAULT"`` to load torchvision's published
            IMAGENET1K_V1 weights; ``None`` for random init. Ignored
            when ``weights_path`` is set (no need to download
            torchvision weights just to overwrite them).
        weights_path: Path to a local ConvNeXt checkpoint, accepted in
            either torchvision key naming (``features.*.block.*.layer_scale``)
            or facebookresearch / Liu et al. key naming
            (``stages.*.{dwconv,norm,pwconv1,pwconv2,gamma}``,
            ``downsample_layers.*``). The latter covers DINOv3
            LVD-1689M, the original ConvNeXt release, and downstream
            user fine-tunes — it is renamed and ``gamma`` reshaped to
            ``(C, 1, 1)`` before load. Accepts ``.pth`` / ``.pt`` /
            ``.bin`` (PyTorch pickle) and ``.safetensors``. When set,
            takes precedence over ``weights="DEFAULT"``.
        out_features: Subset of ``("res2","res3","res4","res5")`` to
            return; defaults to all four.
    """

    def __init__(
        self,
        name: ConvNeXtVariant = "convnext_tiny",
        *,
        freeze_at: int = 0,
        weights: WeightsChoice = None,
        weights_path: str | os.PathLike[str] | None = None,
        out_features: tuple[str, ...] = _DEFAULT_OUT_FEATURES,
    ) -> None:
        super().__init__()
        if name not in _VARIANT_CHANNELS:
            raise ValueError(
                f"unknown ConvNeXt variant {name!r}; expected one of "
                f"{tuple(_VARIANT_CHANNELS)}"
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
        self._out_feature_channels = {
            f: _VARIANT_CHANNELS[name][f] for f in self._out_features
        }
        self._out_feature_strides = {f: _OUT_STRIDES[f] for f in self._out_features}

        # Choose: random init, or torchvision-pretrained "DEFAULT". The
        # local-checkpoint path is a *separate* override that runs after
        # construction, so we don't accidentally trigger a network
        # download for the torchvision weights when the caller already
        # has a local checkpoint in hand.
        tv_weights = (
            _TORCHVISION_DEFAULT_WEIGHTS[name]
            if weights == "DEFAULT" and weights_path is None
            else None
        )
        if weights is not None and weights != "DEFAULT":
            raise ValueError(f"weights must be None or 'DEFAULT'; got {weights!r}")
        tv_model = _TORCHVISION_FACTORIES[name](weights=tv_weights)

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

        if weights_path is not None:
            _load_pretrained_weights(self, Path(weights_path))

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


# ---------------------------------------------------------------------------
# Pretrained-weights loading (facebookresearch-style ConvNeXt state-dict)
# ---------------------------------------------------------------------------


def _load_pretrained_weights(model: ConvNeXtBackbone, path: Path) -> None:
    """Load a ConvNeXt checkpoint into ``model``.

    Accepts ``.pth`` / ``.pt`` / ``.bin`` (PyTorch pickle) and
    ``.safetensors`` (HuggingFace). The checkpoint may be:

    * A bare state-dict in the original Liu et al. / facebookresearch
      ConvNeXt key naming (this is what the DINOv3 LVD-1689M release,
      the original ConvNeXt release, and most downstream forks ship), or
    * A dict with a top-level ``"model"`` / ``"state_dict"`` /
      ``"teacher"`` / ``"student"`` key (some upstream tooling wraps it).

    Keys are remapped from the source naming (``stages.k.j.dwconv``,
    ``downsample_layers.k.``) to torchvision's
    (``features.{2k+1}.j.block.0``, ``features.{2k}.``). Final ``norm``
    and per-stage ``norms`` parameters (the source's classification-head
    bits) are discarded; we only need backbone params.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"ConvNeXt checkpoint not found at {path!s}. Provide a checkpoint "
            "in either torchvision or facebookresearch key naming — e.g., the "
            "DINOv3 LVD-1689M release on HuggingFace "
            "(https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m "
            "and siblings; license-gated, manual download), the original "
            "Liu et al. ConvNeXt release, or a user fine-tune."
        )
    raw = _read_checkpoint(path)
    state = _unwrap_checkpoint(raw)
    remapped = _remap_facebook_convnext_state_dict(state)

    target_keys = set(model.state_dict().keys())
    # Filter out keys that don't belong in this backbone (the checkpoint's
    # final ``norm.*`` and ``norms.*``, which we intentionally drop).
    filtered = {k: v for k, v in remapped.items() if k in target_keys}
    missing = target_keys - set(filtered)
    unexpected = set(remapped) - target_keys
    if missing:
        raise RuntimeError(
            f"ConvNeXt checkpoint at {path!s} is missing keys after remap: "
            f"{sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}"
        )
    # Strict load: every backbone parameter must be filled by the
    # checkpoint. Unexpected source keys (norm / norms / register tokens
    # if any) are silently dropped via the filter above.
    incompatible = model.load_state_dict(filtered, strict=False)
    del unexpected, incompatible  # checked above; load_state_dict echoes them


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Read a ConvNeXt checkpoint file into a top-level dict.

    The return type is intentionally ``dict[str, Any]`` because wrapped
    checkpoints (``{"model": state_dict, "epoch": 42, ...}``) have
    non-tensor leaves at the top level; :func:`_unwrap_checkpoint`
    narrows to the actual state-dict afterwards.
    """
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                f"Loading .safetensors requires the 'safetensors' package; "
                f"install with `pip install safetensors`, or download the "
                f"`.pth` variant of {path.name} instead."
            ) from exc
        return cast(dict[str, Any], load_file(str(path), device="cpu"))
    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    return cast(dict[str, Any], loaded)


def _unwrap_checkpoint(raw: dict[str, Any]) -> dict[str, Tensor]:
    # Common wrappers: {"model": state_dict}, {"state_dict": state_dict},
    # {"teacher": state_dict} (DINOv3 SSL checkpoints sometimes nest under
    # "teacher" or "student" for the EMA copy). Detect by absence of
    # tensor leaves at the top level.
    if not raw:
        return cast(dict[str, Tensor], raw)
    sample_value = next(iter(raw.values()))
    if isinstance(sample_value, torch.Tensor):
        return cast(dict[str, Tensor], raw)
    for key in ("model", "state_dict", "teacher", "student"):
        if key in raw and isinstance(raw[key], dict):
            return cast(dict[str, Tensor], raw[key])
    raise RuntimeError(
        "ConvNeXt checkpoint has a non-tensor top level but no known "
        f"wrapper key (model/state_dict/teacher/student). Got top-level "
        f"keys: {sorted(raw.keys())[:8]}"
    )


def _remap_facebook_convnext_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename facebookresearch-style keys → torchvision keys, reshape ``gamma`` → ``layer_scale``.

    Mapping:

    * ``downsample_layers.0.{0,1}.*`` → ``features.0.{0,1}.*`` (stem)
    * ``downsample_layers.k.{0,1}.*`` (k=1..3) → ``features.{2k}.{0,1}.*``
    * ``stages.k.j.dwconv.*``  → ``features.{2k+1}.j.block.0.*``
    * ``stages.k.j.norm.*``    → ``features.{2k+1}.j.block.2.*``
    * ``stages.k.j.pwconv1.*`` → ``features.{2k+1}.j.block.3.*``
    * ``stages.k.j.pwconv2.*`` → ``features.{2k+1}.j.block.5.*``
    * ``stages.k.j.gamma``     → ``features.{2k+1}.j.layer_scale``
      (reshape ``(C,) → (C, 1, 1)``)

    Source-only keys (``norm.*``, ``norms.*``, register tokens, etc.)
    pass through unchanged so the caller's strict-load filter can drop
    them.

    The torchvision prefix is also injected: keys land under
    ``stem.*`` / ``_res_downs.{res2..res5}.*`` / ``_res_stages.{res2..res5}.*``
    to match :class:`ConvNeXtBackbone`'s carved structure, **not** under
    ``features.*``. The two-step rename (source → tv features.*, then
    features.* → mayaku carved-name) keeps the per-step logic readable.
    """
    out: dict[str, Tensor] = {}
    # Block sub-module index inside the torchvision CNBlock.block
    # Sequential: dwconv=0, Permute=1, LayerNorm=2, pwconv1=3, GELU=4,
    # pwconv2=5, Permute=6. Only the parametric layers appear here.
    _BLOCK_SUB = {"dwconv": 0, "norm": 2, "pwconv1": 3, "pwconv2": 5}
    # k → carved-stage name. ``downsample_layers.0`` is the stem; the
    # other three downsamples sit before res3/res4/res5 in our layout.
    _STAGE_NAMES = ("res2", "res3", "res4", "res5")

    for key, value in state.items():
        parts = key.split(".")
        # ---- downsample_layers.k.{0,1}.{weight,bias} ---------------------
        if parts[0] == "downsample_layers":
            k = int(parts[1])  # 0 = stem, 1..3 = pre-res{3,4,5} downsamples
            if k == 0:
                # stem (Conv 4×4 + LayerNorm2d), lives at self.stem.{0,1}.*
                new_key = "stem." + ".".join(parts[2:])
            else:
                # 1 → res3, 2 → res4, 3 → res5
                stage = _STAGE_NAMES[k]
                # parts[2] ∈ {"0","1"} (LayerNorm2d, Conv2d) — preserve.
                new_key = f"_res_downs.{stage}." + ".".join(parts[2:])
            out[new_key] = value
            continue

        # ---- stages.k.j.{dwconv,norm,pwconv1,pwconv2,gamma}.{w,b} --------
        if parts[0] == "stages":
            k = int(parts[1])
            j = int(parts[2])
            sub = parts[3]
            stage = _STAGE_NAMES[k]
            if sub == "gamma":
                # The source format stores layer-scale as a length-C
                # vector applied in NHWC space; torchvision stores it as
                # ``(C, 1, 1)`` applied in NCHW space. Reshape here so
                # the runtime forward doesn't broadcast — keeps the
                # post-load model bitwise-identical to a torchvision init.
                channels = value.shape[0]
                out[f"_res_stages.{stage}.{j}.layer_scale"] = value.view(
                    channels, 1, 1
                )
                continue
            if sub not in _BLOCK_SUB:
                # Unexpected sub-key — let the strict-load filter drop it.
                out[key] = value
                continue
            tail = ".".join(parts[4:])  # "weight" / "bias"
            out[f"_res_stages.{stage}.{j}.block.{_BLOCK_SUB[sub]}.{tail}"] = value
            continue

        # ---- everything else (norm, norms, mask_token, ...) --------------
        # Pass through; caller filters against the carved model's keys.
        out[key] = value

    return out


# ---------------------------------------------------------------------------
# Factory from BackboneConfig
# ---------------------------------------------------------------------------


def build_convnext(
    cfg: BackboneConfig,
    *,
    weights: WeightsChoice = None,
    out_features: tuple[str, ...] = _DEFAULT_OUT_FEATURES,
) -> ConvNeXtBackbone:
    """Construct a :class:`ConvNeXtBackbone` from a typed config.

    ``weights`` selects torchvision's ImageNet baseline (mirrors the
    :class:`ResNetBackbone` API for parity tests). ``cfg.weights_path``,
    when set, points at a local ConvNeXt checkpoint in torchvision or
    facebookresearch key naming and takes precedence over
    ``weights="DEFAULT"``.
    """
    name = cfg.name
    if name not in _VARIANT_CHANNELS:
        raise ValueError(
            f"build_convnext requires a ConvNeXt variant; got {name!r}"
        )
    return ConvNeXtBackbone(
        name=name,
        freeze_at=cfg.freeze_at,
        weights=weights,
        weights_path=cfg.weights_path,
        out_features=out_features,
    )


def is_convnext_variant(name: str) -> bool:
    """True iff ``name`` is in the ConvNeXt family of :class:`BackboneName`.

    Identifies family by the ``convnext_`` prefix — the same predicate
    the :class:`BackboneConfig` validator uses. The closed set of valid
    names is enforced separately by the :class:`BackboneName` Literal;
    this function takes ``str`` so test fixtures and CLI callers handling
    user-supplied strings can use it without casting first.
    """
    return name.startswith("convnext_")
