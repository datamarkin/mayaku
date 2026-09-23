"""Auto-config: dataset statistics, structural facts, the fine-tune recipe,
and the rule that explicit user values always win."""

from __future__ import annotations

import pytest

from mayaku.config import KeypointConfig, MayakuConfig, ModelConfig
from mayaku.data.coco import load_coco
from mayaku.health import health_check
from mayaku.tuning import AUTO_PATHS, analyze_dataset, apply_auto_config, collect_set_paths
from mayaku.tuning.recipe import MAX_FINETUNE_EPOCHS, MIN_FINETUNE_EPOCHS, _finetune_epochs

from ._coco_fixture import synthetic_coco


@pytest.fixture(scope="module")
def coco(tmp_path_factory):
    root = tmp_path_factory.mktemp("coco")
    return load_coco(str(root), synthetic_coco(str(root), n=24), kpt=3)


def test_stats_measure_boxes_on_the_canvas(coco) -> None:
    raw = analyze_dataset(coco)
    on_canvas = analyze_dataset(coco, (800, 800))
    assert raw.num_images == 24 and raw.num_classes == 4 and raw.num_boxes > 24
    assert sum(on_canvas.sqrt_areas) > sum(raw.sqrt_areas)   # images are upscaled onto 800
    assert raw.aspect_ratios == pytest.approx(on_canvas.aspect_ratios, rel=1e-5)
    assert raw.class_imbalance >= 1.0 and raw.num_degenerate_boxes == 0


def test_epoch_taper() -> None:
    assert _finetune_epochs(100) == MAX_FINETUNE_EPOCHS
    assert _finetune_epochs(100_000) == MIN_FINETUNE_EPOCHS
    assert MIN_FINETUNE_EPOCHS < _finetune_epochs(5_000) < MAX_FINETUNE_EPOCHS


def test_auto_config_fills_structure_and_the_finetune_recipe(coco) -> None:
    cfg, _ = apply_auto_config(MayakuConfig(), coco)
    assert cfg.model.num_classes == 4 and cfg.input.canvas_hw is not None
    assert cfg.train.epochs == MAX_FINETUNE_EPOCHS and cfg.train.final_epochs == 6
    assert cfg.train.assigner_warmup == 0


@pytest.mark.parametrize("model", [ModelConfig(), ModelConfig(keypoints=KeypointConfig(num=3))],
                         ids=["det", "kpt"])
def test_every_change_is_an_allowed_path(coco, model) -> None:
    _, changes = apply_auto_config(MayakuConfig(model=model), coco)
    assert changes and {p for p, _, _ in changes} <= AUTO_PATHS


def test_from_scratch_keeps_its_recipe(coco) -> None:
    cfg, changes = apply_auto_config(MayakuConfig(), coco, finetune=False)
    assert cfg.train == MayakuConfig().train
    assert {p for p, _, _ in changes} == {"model.num_classes", "input.canvas_hw"}


def test_keypoint_names_and_flip_pairs_come_from_the_data(coco) -> None:
    cfg = MayakuConfig(model=ModelConfig(keypoints=KeypointConfig(num=3)))
    cfg, _ = apply_auto_config(cfg, coco)
    assert cfg.model.keypoints.names == ("tl", "ctr", "br")
    assert cfg.model.keypoints.flip_pairs == ()


def test_user_values_always_win(coco) -> None:
    raw = {"model": {"num_classes": 7}, "input": {"canvas_hw": [576, 1024]},
           "train": {"epochs": 5, "aug": {"mosaic": 0.9}}}
    cfg = MayakuConfig.model_validate(raw)
    cfg, changes = apply_auto_config(cfg, coco, collect_set_paths(raw))
    assert cfg.model.num_classes == 7 and cfg.input.canvas_hw == (576, 1024)
    assert cfg.train.epochs == 5 and cfg.train.aug.mosaic == 0.9
    # the clean final stage is a share of whatever run the user pinned
    assert cfg.train.final_epochs == 1
    assert not {p for p, _, _ in changes} & collect_set_paths(raw)


def test_a_pinned_clean_stage_survives_a_derived_run_length(coco) -> None:
    raw = {"train": {"final_frac": 0.5}}
    cfg, _ = apply_auto_config(MayakuConfig.model_validate(raw), coco, collect_set_paths(raw))
    assert cfg.train.epochs == MAX_FINETUNE_EPOCHS and cfg.train.final_epochs == 15


def test_disabled_changes_nothing(coco) -> None:
    cfg = MayakuConfig.model_validate({"auto_config": {"enabled": False}})
    assert apply_auto_config(cfg, coco) == (cfg, [])


def test_tiny_datasets_get_structure_only(tmp_path) -> None:
    small = load_coco(str(tmp_path), synthetic_coco(str(tmp_path), n=5))
    cfg, _ = apply_auto_config(MayakuConfig(), small)
    assert cfg.model.num_classes == 4 and cfg.train == MayakuConfig().train


def test_health_report(tmp_path) -> None:
    ann = synthetic_coco(str(tmp_path), n=12)
    r = health_check(ann, tmp_path)
    assert r["images"] == 12 and r["classes"] == 4 and len(r["canvas"]) == 2
    pinned = MayakuConfig.model_validate({"input": {"canvas_hw": [256, 384]}})
    assert health_check(ann, tmp_path, pinned)["canvas"] == [256, 384]
    assert abs(sum(r["object_size"].values()) - 1.0) < 0.02
    assert set(r["class_counts"]) <= {"c0", "c1", "c2", "c3"} and r["warnings"] == []
