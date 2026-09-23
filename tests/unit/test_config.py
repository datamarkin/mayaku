"""`MayakuConfig`: defaults, validation, YAML round-trip and override merging."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from mayaku.config import (
    InputConfig,
    KeypointConfig,
    MayakuConfig,
    ModelConfig,
    dump_yaml,
    load_yaml,
    merge_overrides,
    to_yaml_string,
)
from mayaku.data import DEFAULT_AUG
from mayaku.engine.trainer import BASE
from mayaku.model import TIERS
from mayaku.model.quant import is_qat


def test_defaults() -> None:
    c = MayakuConfig()
    assert c.model.tier == "n" and c.model.num_classes is None
    assert c.input.size_budget == 800 and c.input.canvas_hw is None
    assert c.train == BASE and c.train.aug == DEFAULT_AUG
    assert c.auto_config.enabled


@pytest.mark.parametrize(("tier", "qat"), [("n", True), ("s", True), ("m", False), ("l", False)])
def test_qat_defaults_per_tier_and_explicit_wins(tier, qat) -> None:
    assert ModelConfig(tier=tier).qat_enabled is qat
    assert ModelConfig(tier=tier, qat=not qat).qat_enabled is (not qat)


@pytest.mark.parametrize(("tier", "qat"), [("n", True), ("m", False)])
def test_build_makes_the_described_detector(tier, qat) -> None:
    m = ModelConfig(tier=tier, num_classes=3).build(640)
    assert m.cfg == TIERS[tier] and m.nc == 3 and is_qat(m) is qat
    assert ModelConfig(tier=tier).build(640, num_classes=5).nc == 5
    with pytest.raises(ValueError, match="num_classes"):
        ModelConfig(tier=tier).build(640)


def test_to_tier_carries_the_heads() -> None:
    m = ModelConfig(tier="s", seg=True, keypoints=KeypointConfig(num=17), aux_arm="tower")
    assert m.to_tier() == dataclasses.replace(TIERS["s"], seg=True, kpt=17, aux_arm="tower")
    assert ModelConfig().to_tier() == TIERS["n"]


def test_canvas_must_be_on_the_32_grid() -> None:
    assert InputConfig(size_budget=640, canvas_hw=(576, 1024)).canvas_hw == (576, 1024)
    for bad in ({"size_budget": 810}, {"canvas_hw": (576, 1000)}, {"canvas_hw": (0, 64)}):
        with pytest.raises(ValidationError, match="multiple"):
            InputConfig(**bad)


def test_keypoint_metadata_must_match_the_count() -> None:
    KeypointConfig(num=3, names=("a", "b", "c"), flip_pairs=((0, 2),))
    for bad in ({"names": ("a",)}, {"flip_pairs": ((0, 3),)}):
        with pytest.raises(ValidationError):
            KeypointConfig(num=3, **bad)


def test_unknown_fields_are_rejected_at_every_level() -> None:
    for bad in ({"modle": {}}, {"model": {"tierr": "n"}}, {"train": {"epochz": 3}},
                {"train": {"aug": {"mosaicc": 0.5}}}):
        with pytest.raises(ValidationError):
            MayakuConfig.model_validate(bad)


def test_recipe_invariants_surface_as_validation_errors() -> None:
    with pytest.raises(ValidationError):
        MayakuConfig.model_validate({"train": {"optimizer": "rmsprop"}})
    with pytest.raises(ValidationError):
        MayakuConfig.model_validate({"train": {"aug": {"mosaic": 2.0}}})


def test_yaml_round_trip(tmp_path: Path) -> None:
    src = MayakuConfig(
        model=ModelConfig(tier="m", num_classes=5, seg=True,
                          keypoints=KeypointConfig(num=3, names=("a", "b", "c"))),
        input=InputConfig(canvas_hw=(576, 1024)),
        train=dataclasses.replace(BASE, epochs=30, aug=dataclasses.replace(DEFAULT_AUG, mixup=0.1)),
    )
    p = tmp_path / "config.yaml"
    dump_yaml(src, p)
    assert load_yaml(p) == src
    assert "mixup: 0.1" in to_yaml_string(src)


def test_merge_overrides_reaches_into_the_recipe() -> None:
    c = merge_overrides(MayakuConfig(), {"model": {"tier": "l"},
                                         "train": {"epochs": 40, "aug": {"mixup": 0.2}}})
    assert c.model.tier == "l" and c.train.epochs == 40
    assert c.train.aug == dataclasses.replace(DEFAULT_AUG, mixup=0.2)
    assert c.train.final_aug == BASE.final_aug


def test_load_yaml_rejects_a_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)
