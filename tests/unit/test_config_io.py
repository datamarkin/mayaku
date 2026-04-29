"""Tests for :mod:`mayaku.config.io` — YAML round-trip + override merge."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mayaku.config import (
    MayakuConfig,
    ModelConfig,
    ROIMaskHeadConfig,
    dump_yaml,
    load_yaml,
    merge_overrides,
    to_yaml_string,
)


def test_round_trip_default_config(tmp_path: Path) -> None:
    src = MayakuConfig()
    p = tmp_path / "default.yaml"
    dump_yaml(src, p)
    loaded = load_yaml(p)
    assert loaded == src


def test_round_trip_mask_rcnn(tmp_path: Path) -> None:
    src = MayakuConfig(
        model=ModelConfig(
            meta_architecture="mask_rcnn",
            mask_on=True,
            roi_mask_head=ROIMaskHeadConfig(num_conv=2, conv_dim=128),
        ),
    )
    p = tmp_path / "mask.yaml"
    dump_yaml(src, p)
    loaded = load_yaml(p)
    assert loaded == src


def test_to_yaml_string_preserves_field_order() -> None:
    text = to_yaml_string(MayakuConfig())
    # Top-level field order from MayakuConfig declaration:
    # input, model, solver, test, dataloader.
    order = [text.find(f"\n{k}:") for k in ("input", "model", "solver", "test", "dataloader")]
    # First field appears at offset 0 — find returns -1 if missing, so a
    # leading "input" check goes through index() instead of find().
    assert text.startswith("input:")
    assert order[1] > 0 < order[2] < order[3] < order[4]
    assert order[1] < order[2] < order[3] < order[4]


def test_load_yaml_fills_missing_sections_with_defaults(tmp_path: Path) -> None:
    p = tmp_path / "partial.yaml"
    p.write_text("solver:\n  base_lr: 0.005\n", encoding="utf-8")
    cfg = load_yaml(p)
    assert cfg.solver.base_lr == 0.005
    # The other sections fell back to their schema defaults.
    assert cfg.input.max_size_train == 1333
    assert cfg.model.pixel_mean == (123.675, 116.280, 103.530)


def test_load_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("solver:\n  base_lr: 0.005\n  bogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="bogus_key"):
        load_yaml(p)


def test_load_yaml_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


def test_load_yaml_rejects_yaml_format_field(tmp_path: Path) -> None:
    # ADR 002 sentinel: the legacy INPUT.FORMAT knob must not be a
    # silent passthrough — it should fail loudly so users with old
    # configs get a clear signal to migrate.
    p = tmp_path / "legacy.yaml"
    p.write_text("input:\n  format: BGR\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="format"):
        load_yaml(p)


def test_load_yaml_empty_file_yields_defaults(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert load_yaml(p) == MayakuConfig()


# ---------------------------------------------------------------------------
# Override merge
# ---------------------------------------------------------------------------


def test_merge_overrides_replaces_leaf_values() -> None:
    cfg = MayakuConfig()
    out = merge_overrides(cfg, {"solver": {"base_lr": 0.001}})
    assert out.solver.base_lr == 0.001
    # Other solver fields preserved.
    assert out.solver.max_iter == cfg.solver.max_iter
    # Other top-level sections untouched.
    assert out.input == cfg.input


def test_merge_overrides_does_not_mutate_input() -> None:
    cfg = MayakuConfig()
    snapshot_lr = cfg.solver.base_lr
    _ = merge_overrides(cfg, {"solver": {"base_lr": 0.001}})
    assert cfg.solver.base_lr == snapshot_lr


def test_merge_overrides_runs_validation() -> None:
    cfg = MayakuConfig()
    # Override that would violate the warmup<max_iter invariant must raise.
    with pytest.raises(ValidationError, match="warmup_iters"):
        merge_overrides(cfg, {"solver": {"max_iter": 100, "steps": (50,)}})


def test_merge_overrides_can_change_meta_architecture() -> None:
    cfg = MayakuConfig()
    out = merge_overrides(
        cfg,
        {
            "model": {
                "meta_architecture": "mask_rcnn",
                "mask_on": True,
                "roi_mask_head": {"num_conv": 2},
            }
        },
    )
    assert out.model.meta_architecture == "mask_rcnn"
    assert out.model.roi_mask_head is not None
    assert out.model.roi_mask_head.num_conv == 2
