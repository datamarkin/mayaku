"""The ``mayaku-{n,s,m,l,xl,xxl}`` family is generator-owned — guard 0 drift.

These tests are the CI side of ``tools/gen_configs.py``: the committed YAMLs must
be byte-identical to what the generator renders, every member must load, and the
derived invariants (``out_channels == hidden_dim`` etc.) must hold across tiers.
If this fails, someone hand-edited a family config — run ``python
tools/gen_configs.py`` instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_generator():
    """Import ``tools/gen_configs.py`` (not a package) as a module."""
    spec = importlib.util.spec_from_file_location("gen_configs", _REPO / "tools" / "gen_configs.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_configs"] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()

# Render+validate the family once; reused for IDs and parametrization.
_TARGETS = gen._targets()
_IDS = [f"{p.parent.name}/{p.stem}" for p, _ in _TARGETS]


@pytest.mark.parametrize(("path", "want"), _TARGETS, ids=_IDS)
def test_committed_config_matches_generator(path: Path, want: str) -> None:
    """0-drift gate: the committed tree equals ``gen_configs.py`` output."""
    assert path.exists(), f"{path} missing — run `python tools/gen_configs.py`"
    have = path.read_text(encoding="utf-8")
    assert have == want, f"{path} drifted from generator — run `python tools/gen_configs.py`"


@pytest.mark.parametrize("tier", gen.TIERS, ids=[t.name for t in gen.TIERS])
@pytest.mark.parametrize("task", ["detection", "segmentation", "keypoints"])
def test_derived_invariants_hold(task: str, tier) -> None:
    from mayaku import configs

    cfg = configs.load(f"{task}/mayaku-{tier.name}")
    head = cfg.model.uniquery_head
    assert cfg.model.fpn.out_channels == head.hidden_dim == tier.hidden_dim
    # Head capacity is ABSOLUTE (Sparse R-CNN), not scaled with hidden_dim — the
    # 4*hidden / hidden//4 ratio starved the h128 head (see gen_configs.render).
    assert head.dim_feedforward == 2048
    assert head.dim_dynamic == 64
    assert head.pooler_sampling_ratio == (1 if tier.realtime else 2)
    assert cfg.input.size_budget == tier.size_budget
    # COCO keypoints is person-only; detection/segmentation default to 80.
    assert cfg.model.roi_heads.num_classes == (1 if task == "keypoints" else 80)
    if task == "segmentation":
        assert cfg.model.uniquery_mask.conv_dim == head.hidden_dim
    if task == "keypoints":
        assert cfg.model.uniquery_keypoint.num_keypoints == 17
