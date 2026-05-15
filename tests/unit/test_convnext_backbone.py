"""Tests for :mod:`mayaku.models.backbones.convnext`.

The four DINOv3 ConvNeXt variants share one code path; tests
parametrise across them for the shape contract and stick to the
smallest (``tiny``) for the heavier checks. No DINOv3 weights are
downloaded — pretrained-load tests synthesise a DINOv3-format
state-dict on the fly from a torchvision random init, so the suite
runs offline and respects the license gating.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
import torch
import torchvision.models as tv
from pydantic import ValidationError
from torch import nn

from mayaku.config.schemas import BackboneConfig
from mayaku.models.backbones import (
    Backbone,
    ConvNeXtBackbone,
    ResNetBackbone,
    ShapeSpec,
    build_bottom_up,
)
from mayaku.models.backbones.convnext import (
    _remap_dinov3_state_dict,
    build_convnext,
)

_VARIANTS = (
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "dinov3_convnext_tiny",
    "dinov3_convnext_small",
    "dinov3_convnext_base",
    "dinov3_convnext_large",
)
# Architecturally the ``convnext_X`` and ``dinov3_convnext_X`` pairs are
# identical — both keys point at the same channel and depth tables.
_EXPECTED_CHANNELS: dict[str, tuple[int, int, int, int]] = {
    "convnext_tiny": (96, 192, 384, 768),
    "convnext_small": (96, 192, 384, 768),
    "convnext_base": (128, 256, 512, 1024),
    "convnext_large": (192, 384, 768, 1536),
    "dinov3_convnext_tiny": (96, 192, 384, 768),
    "dinov3_convnext_small": (96, 192, 384, 768),
    "dinov3_convnext_base": (128, 256, 512, 1024),
    "dinov3_convnext_large": (192, 384, 768, 1536),
}
_DEPTHS: dict[str, list[int]] = {
    "convnext_tiny": [3, 3, 9, 3],
    "convnext_small": [3, 3, 27, 3],
    "convnext_base": [3, 3, 27, 3],
    "convnext_large": [3, 3, 27, 3],
    "dinov3_convnext_tiny": [3, 3, 9, 3],
    "dinov3_convnext_small": [3, 3, 27, 3],
    "dinov3_convnext_base": [3, 3, 27, 3],
    "dinov3_convnext_large": [3, 3, 27, 3],
}

# ``(plain_name, dinov3_name)`` pairs — used by the alias-parity test
# below to confirm the two naming families share architecture.
_ALIAS_PAIRS = (
    ("convnext_tiny", "dinov3_convnext_tiny"),
    ("convnext_small", "dinov3_convnext_small"),
    ("convnext_base", "dinov3_convnext_base"),
    ("convnext_large", "dinov3_convnext_large"),
)


# ---------------------------------------------------------------------------
# Shape contract — parametrised across all four variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _VARIANTS)
def test_output_shape_contract(name: str) -> None:
    bb = ConvNeXtBackbone(name=name)  # type: ignore[arg-type]
    spec = bb.output_shape()
    assert set(spec) == {"res2", "res3", "res4", "res5"}
    c2, c3, c4, c5 = _EXPECTED_CHANNELS[name]
    expected = {
        "res2": ShapeSpec(channels=c2, stride=4),
        "res3": ShapeSpec(channels=c3, stride=8),
        "res4": ShapeSpec(channels=c4, stride=16),
        "res5": ShapeSpec(channels=c5, stride=32),
    }
    assert spec == expected
    assert bb.size_divisibility == 32


def test_forward_returns_named_feature_map(device: torch.device) -> None:
    bb = ConvNeXtBackbone(name="dinov3_convnext_tiny").to(device).eval()
    x = torch.zeros(1, 3, 64, 64, device=device)
    with torch.no_grad():
        out = bb(x)
    assert set(out) == {"res2", "res3", "res4", "res5"}
    assert out["res2"].shape == (1, 96, 16, 16)
    assert out["res3"].shape == (1, 192, 8, 8)
    assert out["res4"].shape == (1, 384, 4, 4)
    assert out["res5"].shape == (1, 768, 2, 2)
    for v in out.values():
        assert v.dtype == torch.float32
        assert v.device.type == device.type


def test_subset_of_out_features_emits_only_those_keys() -> None:
    bb = ConvNeXtBackbone(
        name="dinov3_convnext_tiny",
        out_features=("res4", "res5"),
    )
    out = bb(torch.zeros(1, 3, 64, 64))
    assert set(out) == {"res4", "res5"}
    assert set(bb.output_shape()) == {"res4", "res5"}
    assert bb.size_divisibility == 32


def test_unknown_variant_rejected() -> None:
    with pytest.raises(ValueError, match="unknown ConvNeXt variant"):
        ConvNeXtBackbone(name="dinov3_convnext_giant")  # type: ignore[arg-type]


def test_unknown_out_feature_rejected() -> None:
    with pytest.raises(ValueError, match="unknown out_feature"):
        ConvNeXtBackbone(out_features=("res2", "res99"))


# ---------------------------------------------------------------------------
# Freeze semantics — mirror ResNetBackbone for cross-arch parity
# ---------------------------------------------------------------------------


def _trainable(stage: nn.Module) -> bool:
    return any(p.requires_grad for p in stage.parameters())


@pytest.mark.parametrize("freeze_at", [0, 1, 2, 3, 4, 5])
def test_freeze_at_freezes_first_k_stages(freeze_at: int) -> None:
    bb = ConvNeXtBackbone(name="dinov3_convnext_tiny", freeze_at=freeze_at)
    stages: list[nn.Module] = [
        bb.stem,
        nn.ModuleList([bb._res_downs["res2"], bb._res_stages["res2"]]),
        nn.ModuleList([bb._res_downs["res3"], bb._res_stages["res3"]]),
        nn.ModuleList([bb._res_downs["res4"], bb._res_stages["res4"]]),
        nn.ModuleList([bb._res_downs["res5"], bb._res_stages["res5"]]),
    ]
    for i, stage in enumerate(stages, start=1):
        if i <= freeze_at:
            assert not _trainable(stage), f"stage {i} should be frozen"
        else:
            assert _trainable(stage), f"stage {i} should be trainable"


def test_freeze_at_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="freeze_at"):
        ConvNeXtBackbone(freeze_at=6)
    with pytest.raises(ValueError, match="freeze_at"):
        ConvNeXtBackbone(freeze_at=-1)


# ---------------------------------------------------------------------------
# DINOv3 weight loading — synthesise the official key-naming on the fly
# ---------------------------------------------------------------------------


def _make_dinov3_state_dict(name: str) -> tuple[dict[str, torch.Tensor], tv.ConvNeXt]:
    """Build a state_dict that mirrors DINOv3's official key naming.

    We start from a torchvision-initialised model (so the parameter
    shapes are exactly what the carved backbone expects), then rename
    every key into DINOv3 form. The returned ``tv.ConvNeXt`` is the
    *reference* — a model whose ``features.*`` values are bitwise
    identical to what the DINOv3 checkpoint claims to hold.
    """
    factory = {
        "convnext_tiny": tv.convnext_tiny,
        "convnext_small": tv.convnext_small,
        "convnext_base": tv.convnext_base,
        "convnext_large": tv.convnext_large,
        "dinov3_convnext_tiny": tv.convnext_tiny,
        "dinov3_convnext_small": tv.convnext_small,
        "dinov3_convnext_base": tv.convnext_base,
        "dinov3_convnext_large": tv.convnext_large,
    }[name]
    tv_model = factory(weights=None)
    tv_state = {
        k: v for k, v in tv_model.state_dict().items() if not k.startswith("classifier")
    }

    out: dict[str, torch.Tensor] = {}
    depths = _DEPTHS[name]
    dims = _EXPECTED_CHANNELS[name]
    # downsample_layers.0 == stem.
    out["downsample_layers.0.0.weight"] = tv_state["features.0.0.weight"].clone()
    out["downsample_layers.0.0.bias"] = tv_state["features.0.0.bias"].clone()
    out["downsample_layers.0.1.weight"] = tv_state["features.0.1.weight"].clone()
    out["downsample_layers.0.1.bias"] = tv_state["features.0.1.bias"].clone()
    # downsample_layers.k for k=1..3 → features.{2k}.
    for k in (1, 2, 3):
        idx = 2 * k
        out[f"downsample_layers.{k}.0.weight"] = tv_state[f"features.{idx}.0.weight"].clone()
        out[f"downsample_layers.{k}.0.bias"] = tv_state[f"features.{idx}.0.bias"].clone()
        out[f"downsample_layers.{k}.1.weight"] = tv_state[f"features.{idx}.1.weight"].clone()
        out[f"downsample_layers.{k}.1.bias"] = tv_state[f"features.{idx}.1.bias"].clone()
    # stages.k.j → features.{2k+1}.j.
    sub_inv = {"dwconv": 0, "norm": 2, "pwconv1": 3, "pwconv2": 5}
    for k, (depth, dim) in enumerate(zip(depths, dims, strict=True)):
        idx = 2 * k + 1
        for j in range(depth):
            for sub, tv_sub in sub_inv.items():
                out[f"stages.{k}.{j}.{sub}.weight"] = tv_state[
                    f"features.{idx}.{j}.block.{tv_sub}.weight"
                ].clone()
                out[f"stages.{k}.{j}.{sub}.bias"] = tv_state[
                    f"features.{idx}.{j}.block.{tv_sub}.bias"
                ].clone()
            # DINOv3 gamma is (C,), torchvision layer_scale is (C, 1, 1).
            out[f"stages.{k}.{j}.gamma"] = (
                tv_state[f"features.{idx}.{j}.layer_scale"].clone().view(dim)
            )
    # DINOv3-only extras the loader must drop without complaint.
    out["norm.weight"] = torch.randn(dims[-1])
    out["norm.bias"] = torch.randn(dims[-1])
    for i, d in enumerate(dims):
        out[f"norms.{i}.weight"] = torch.randn(d)
        out[f"norms.{i}.bias"] = torch.randn(d)
    return out, tv_model


@pytest.mark.parametrize("name", _VARIANTS)
def test_dinov3_remap_keys_into_carved_layout(name: str) -> None:
    """Synthetic DINOv3 state_dict remaps to exactly the keys
    :class:`ConvNeXtBackbone` declares — no missing/extra params
    after the strict-load filter."""
    state, _ = _make_dinov3_state_dict(name)
    remapped = _remap_dinov3_state_dict(state)

    bb = ConvNeXtBackbone(name=name)  # type: ignore[arg-type]
    target_keys = set(bb.state_dict().keys())
    mapped_keys = set(remapped.keys())

    # Every backbone parameter must have a remapped source.
    missing = target_keys - mapped_keys
    assert not missing, f"{name}: missing remapped keys {sorted(missing)[:5]}"

    # Remapped keys that don't belong (norm/norms) are tolerated by the
    # loader, but they must be a strict subset of {target_keys ∪ "norm.*"
    # ∪ "norms.*"} — i.e. no surprise leakage from a renaming bug.
    extras = mapped_keys - target_keys
    for k in extras:
        assert k.startswith("norm.") or k.startswith("norms."), f"unexpected extra key: {k}"


@pytest.mark.parametrize("name", _VARIANTS)
def test_dinov3_load_end_to_end_matches_torchvision_forward(name: str) -> None:
    """Loading a DINOv3-format checkpoint via ``dinov3_weights=PATH``
    yields a model whose forward is identical to a torchvision-init
    model with the same underlying parameters."""
    state, tv_model = _make_dinov3_state_dict(name)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"{name}.pth"
        torch.save(state, path)
        bb = ConvNeXtBackbone(
            name=name,  # type: ignore[arg-type]
            dinov3_weights=path,
            freeze_at=0,
        ).eval()

    # Reference model: directly copy tv_model.features.* into a fresh
    # ConvNeXtBackbone's carved children. This bypasses the DINOv3
    # remap path entirely; the two should still produce identical
    # outputs.
    ref = ConvNeXtBackbone(name=name, freeze_at=0).eval()  # type: ignore[arg-type]
    ref.stem.load_state_dict(tv_model.features[0].state_dict())
    ref._res_downs["res3"].load_state_dict(tv_model.features[2].state_dict())
    ref._res_downs["res4"].load_state_dict(tv_model.features[4].state_dict())
    ref._res_downs["res5"].load_state_dict(tv_model.features[6].state_dict())
    ref._res_stages["res2"].load_state_dict(tv_model.features[1].state_dict())
    ref._res_stages["res3"].load_state_dict(tv_model.features[3].state_dict())
    ref._res_stages["res4"].load_state_dict(tv_model.features[5].state_dict())
    ref._res_stages["res5"].load_state_dict(tv_model.features[7].state_dict())

    # Use a 64-divisible input so every level has integer spatial size.
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        a = bb(x)
        b = ref(x)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5), f"forward differs at {k}"


def test_dinov3_load_accepts_wrapped_checkpoint() -> None:
    """A checkpoint wrapped as ``{"model": state_dict}`` (some upstream
    tooling format) must unwrap automatically."""
    state, _ = _make_dinov3_state_dict("dinov3_convnext_tiny")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "wrapped.pth"
        torch.save({"model": state, "epoch": 42}, path)
        bb = ConvNeXtBackbone(name="dinov3_convnext_tiny", dinov3_weights=path)
    # Sanity: forward still runs.
    with torch.no_grad():
        bb.eval()(torch.zeros(1, 3, 64, 64))


def test_dinov3_load_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="DINOv3 weights not found"):
        ConvNeXtBackbone(
            name="dinov3_convnext_tiny",
            dinov3_weights="/nonexistent/path/to/dinov3.pth",
        )


def test_dinov3_load_rejects_incomplete_checkpoint() -> None:
    """A checkpoint missing essential ConvNeXt keys must error loudly
    rather than silently load a partially-initialised model."""
    # Build a real state dict, then strip the entire stage-0.
    state, _ = _make_dinov3_state_dict("dinov3_convnext_tiny")
    stripped = {k: v for k, v in state.items() if not k.startswith("stages.0.")}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "incomplete.pth"
        torch.save(stripped, path)
        with pytest.raises(RuntimeError, match="missing keys"):
            ConvNeXtBackbone(
                name="dinov3_convnext_tiny",
                dinov3_weights=path,
            )


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_build_convnext_from_config() -> None:
    cfg = BackboneConfig(name="dinov3_convnext_tiny", freeze_at=0)
    bb = build_convnext(cfg)
    assert isinstance(bb, ConvNeXtBackbone)
    assert bb.name == "dinov3_convnext_tiny"


def test_build_convnext_rejects_resnet_name() -> None:
    cfg = BackboneConfig(name="resnet50")
    with pytest.raises(ValueError, match="ConvNeXt variant"):
        build_convnext(cfg)


def test_build_bottom_up_dispatches_to_convnext() -> None:
    cfg = BackboneConfig(name="dinov3_convnext_base", freeze_at=0)
    bb = build_bottom_up(cfg)
    assert isinstance(bb, ConvNeXtBackbone)
    assert bb.name == "dinov3_convnext_base"


def test_build_bottom_up_still_returns_resnet_for_resnet_name() -> None:
    cfg = BackboneConfig(name="resnet50")
    bb = build_bottom_up(cfg)
    assert isinstance(bb, ResNetBackbone)


def test_build_bottom_up_threads_dinov3_weights_path() -> None:
    """``cfg.weights_path`` must reach the constructor's
    ``dinov3_weights`` argument so the checkpoint actually loads."""
    state, _ = _make_dinov3_state_dict("dinov3_convnext_tiny")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.pth"
        torch.save(state, path)
        cfg = BackboneConfig(
            name="dinov3_convnext_tiny",
            freeze_at=0,
            weights_path=str(path),
        )
        bb = build_bottom_up(cfg)
    assert isinstance(bb, ConvNeXtBackbone)
    # If the weights had failed to load, the first stem conv would
    # match the random init that ConvNeXtBackbone(name=...) produces.
    # Easy check: the stem conv weight must equal the file's value.
    expected = state["downsample_layers.0.0.weight"]
    got = bb.stem[0].weight
    assert torch.allclose(got, expected)


# ---------------------------------------------------------------------------
# Schema validation around ConvNeXt-specific fields
# ---------------------------------------------------------------------------


def test_schema_rejects_norm_choice_on_convnext() -> None:
    with pytest.raises(ValidationError, match=re.compile(r"norm.*ResNet", re.S)):
        BackboneConfig(name="dinov3_convnext_tiny", norm="BN")


def test_schema_rejects_stride_in_1x1_on_convnext() -> None:
    with pytest.raises(ValidationError, match=re.compile(r"stride_in_1x1", re.S)):
        BackboneConfig(name="dinov3_convnext_tiny", stride_in_1x1=True)


def test_schema_rejects_res5_dilation_on_convnext() -> None:
    with pytest.raises(ValidationError, match=re.compile(r"res5_dilation", re.S)):
        BackboneConfig(name="dinov3_convnext_tiny", res5_dilation=2)


def test_schema_rejects_weights_path_on_resnet() -> None:
    with pytest.raises(ValidationError, match=re.compile(r"weights_path", re.S)):
        BackboneConfig(name="resnet50", weights_path="/some/where.pth")


def test_schema_accepts_weights_path_on_convnext() -> None:
    cfg = BackboneConfig(
        name="dinov3_convnext_large",
        weights_path="/data/dinov3.safetensors",
    )
    assert cfg.weights_path == "/data/dinov3.safetensors"


# ---------------------------------------------------------------------------
# Backbone protocol parity
# ---------------------------------------------------------------------------


def test_is_a_backbone_subclass() -> None:
    bb = ConvNeXtBackbone(name="dinov3_convnext_tiny")
    assert isinstance(bb, Backbone)


def test_dummy_input_lives_on_backbone_device(device: torch.device) -> None:
    bb = ConvNeXtBackbone(name="dinov3_convnext_tiny").to(device)
    x = bb.dummy_input(batch=2, h=64, w=64)
    assert x.shape == (2, 3, 64, 64)
    assert x.device.type == device.type


# ---------------------------------------------------------------------------
# Plain ``convnext_*`` aliases — architecturally identical to the
# ``dinov3_convnext_*`` family; the naming distinction is intent (which
# pretrained weights you plan to load) rather than arch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("plain", "dinov3"), _ALIAS_PAIRS)
def test_alias_shares_architecture_with_dinov3_sibling(plain: str, dinov3: str) -> None:
    """``convnext_X`` and ``dinov3_convnext_X`` must declare identical
    output shapes — same channels, strides, and size-divisibility — so
    swapping the name in a config never changes downstream FPN/RPN/ROI
    wiring."""
    bb_plain = ConvNeXtBackbone(name=plain)  # type: ignore[arg-type]
    bb_dinov3 = ConvNeXtBackbone(name=dinov3)  # type: ignore[arg-type]
    assert bb_plain.output_shape() == bb_dinov3.output_shape()
    assert bb_plain.size_divisibility == bb_dinov3.size_divisibility


@pytest.mark.parametrize(("plain", "dinov3"), _ALIAS_PAIRS)
def test_alias_forward_produces_same_shaped_features(
    plain: str, dinov3: str
) -> None:
    """Independent random inits produce different *values* but the
    *shapes* of every stage output must match — confirms the alias
    routes to the same underlying torchvision factory."""
    bb_plain = ConvNeXtBackbone(name=plain).eval()  # type: ignore[arg-type]
    bb_dinov3 = ConvNeXtBackbone(name=dinov3).eval()  # type: ignore[arg-type]
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out_plain = bb_plain(x)
        out_dinov3 = bb_dinov3(x)
    assert set(out_plain) == set(out_dinov3) == {"res2", "res3", "res4", "res5"}
    for k in out_plain:
        assert out_plain[k].shape == out_dinov3[k].shape, k


def test_schema_accepts_plain_convnext_alias() -> None:
    """``BackboneConfig(name="convnext_tiny", weights_path=...)`` must
    validate — the alias is a member of the ConvNeXt family, so the
    weights_path field applies to it identically."""
    cfg = BackboneConfig(name="convnext_tiny", weights_path="/data/x.pth")
    assert cfg.name == "convnext_tiny"
    assert cfg.weights_path == "/data/x.pth"


def test_build_bottom_up_dispatches_alias_to_convnext() -> None:
    cfg = BackboneConfig(name="convnext_base", freeze_at=0)
    bb = build_bottom_up(cfg)
    assert isinstance(bb, ConvNeXtBackbone)
    assert bb.name == "convnext_base"
