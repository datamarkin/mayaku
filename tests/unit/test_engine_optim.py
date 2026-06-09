"""Tests for :mod:`mayaku.engine.optim`."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from torch import nn

from mayaku.config.schemas import SolverConfig
from mayaku.engine.optim import (
    _layer_id_for_convnext_param,
    _layer_id_for_resnet_param,
    _resolve_llrd_num_layers,
    build_lr_scheduler,
    build_optimizer,
)
from mayaku.models.backbones._frozen_bn import FrozenBatchNorm2d


def _toy_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 4, 1),
        FrozenBatchNorm2d(4),
        nn.Linear(4, 2),
    )


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------


def _solver_cfg(**overrides: object) -> SolverConfig:
    """Tiny SolverConfig that satisfies the schema's cross-field validators."""
    base: dict[str, object] = {
        "max_iter": 1000,
        "warmup_iters": 10,
        "steps": (500, 800),
    }
    base.update(overrides)
    return SolverConfig(**base)  # type: ignore[arg-type]


def test_optimizer_splits_norm_and_other_groups() -> None:
    model = _toy_model()
    opt = build_optimizer(model, _solver_cfg(weight_decay=1e-4, weight_decay_norm=0.0))
    # Two parameter groups: non-norm (wd=1e-4) and norm (wd=0).
    assert len(opt.param_groups) == 2
    weights = sorted(g["weight_decay"] for g in opt.param_groups)
    assert weights == [0.0, 1e-4]


def test_optimizer_skips_frozen_parameters() -> None:
    model = _toy_model()
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model, _solver_cfg())


def test_optimizer_uses_base_lr_and_momentum() -> None:
    model = _toy_model()
    opt = build_optimizer(model, _solver_cfg(base_lr=0.05, momentum=0.5))
    for g in opt.param_groups:
        assert g["lr"] == 0.05
        assert g["momentum"] == 0.5


def test_optimizer_default_is_sgd() -> None:
    """Defaults preserve the SGD path bit-identically."""
    model = _toy_model()
    opt = build_optimizer(model, _solver_cfg())
    assert isinstance(opt, torch.optim.SGD)


def test_optimizer_adamw_factory() -> None:
    """`optimizer_name="AdamW"` returns AdamW with the configured betas / eps / weight_decay."""
    model = _toy_model()
    opt = build_optimizer(
        model,
        _solver_cfg(
            optimizer_name="AdamW",
            base_lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-7,
            weight_decay=0.05,
            weight_decay_norm=0.0,
        ),
    )
    assert isinstance(opt, torch.optim.AdamW)
    # Same norm-vs-other split as SGD.
    assert len(opt.param_groups) == 2
    weights = sorted(g["weight_decay"] for g in opt.param_groups)
    assert weights == [0.0, 0.05]
    for g in opt.param_groups:
        assert g["lr"] == 1e-4
        assert g["betas"] == (0.9, 0.95)
        assert g["eps"] == 1e-7


# ---------------------------------------------------------------------------
# build_lr_scheduler — multistep
# ---------------------------------------------------------------------------


def _make_opt_and_cfg(**overrides: object) -> tuple[torch.optim.SGD, SolverConfig]:
    model = nn.Linear(2, 2)
    cfg_kwargs: dict[str, object] = {
        "max_iter": 1000,
        "warmup_iters": 100,
        "warmup_factor": 0.001,
        "base_lr": 1.0,
        "steps": (500, 800),
        "gamma": 0.1,
    }
    cfg_kwargs.update(overrides)
    cfg = SolverConfig(**cfg_kwargs)  # type: ignore[arg-type]
    opt = torch.optim.SGD(model.parameters(), lr=cfg.base_lr)
    return opt, cfg


def test_warmup_multistep_warmup_grows_linearly_from_factor_to_one() -> None:
    opt, cfg = _make_opt_and_cfg()
    sched = build_lr_scheduler(opt, cfg)
    # Iter 0 → factor; iter warmup_iters → 1.0.
    assert math.isclose(opt.param_groups[0]["lr"], 0.001, abs_tol=1e-6)
    for _ in range(50):
        sched.step()
    # Half-way through warmup → ~0.5005 (lerp from 0.001 to 1.0)
    expected_mid = 0.001 * (1 - 0.5) + 0.5
    assert math.isclose(opt.param_groups[0]["lr"], expected_mid, abs_tol=1e-3)
    for _ in range(50):
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, abs_tol=1e-6)


def test_warmup_multistep_decays_at_each_step() -> None:
    opt, cfg = _make_opt_and_cfg()
    sched = build_lr_scheduler(opt, cfg)
    for _ in range(500):
        sched.step()
    # After the first decay step (500): lr = 1.0 * 0.1
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, abs_tol=1e-6)
    for _ in range(300):
        sched.step()
    # After both decay steps (800): lr = 1.0 * 0.1 * 0.1
    assert math.isclose(opt.param_groups[0]["lr"], 0.01, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# build_lr_scheduler — cosine
# ---------------------------------------------------------------------------


def test_warmup_cosine_decays_from_one_to_zero() -> None:
    opt, cfg = _make_opt_and_cfg(
        lr_scheduler_name="WarmupCosineLR",
        warmup_iters=10,
        max_iter=110,
        # Cosine doesn't need step milestones, but SolverConfig still
        # validates that every entry is < max_iter, so override here.
        steps=(50,),
    )
    sched = build_lr_scheduler(opt, cfg)
    for _ in range(10):
        sched.step()
    # At end of warmup, multiplier = 1.0.
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, abs_tol=1e-6)
    # At max_iter, cosine = 0.
    for _ in range(100):
        sched.step()
    assert opt.param_groups[0]["lr"] < 1e-5


def test_warmup_constant_method_holds_factor_through_warmup() -> None:
    opt, cfg = _make_opt_and_cfg(warmup_method="constant", warmup_iters=10)
    sched = build_lr_scheduler(opt, cfg)
    for _ in range(5):
        sched.step()
    # With constant method, lr stays at base_lr * warmup_factor through warmup.
    assert math.isclose(opt.param_groups[0]["lr"], 0.001, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# LLRD (layer-wise learning rate decay)
# ---------------------------------------------------------------------------
#
# Design note on test isolation: LLRD's name → layer_id adapter and the
# num_layers derivation depend only on parameter NAMES and on one ModuleDict
# attribute (``_res_stages["res4"]``'s length) respectively. The end-to-end
# scale-formula / freeze-composition tests legitimately need a real backbone
# to walk ``model.named_parameters()`` — but those are the only tests that do.
# Everything else uses synthetic names or 5-line stubs, so the optim unit
# tests don't pay for torchvision construction (and don't break if the
# backbone's internal module names refactor).


# --- no-regression baseline ------------------------------------------------


def test_llrd_disabled_matches_legacy_two_group_split() -> None:
    """Default config (llrd_enabled=False) must reproduce the prior splitter exactly."""
    model = _toy_model()
    opt = build_optimizer(
        model,
        _solver_cfg(weight_decay=1e-4, weight_decay_norm=0.0, llrd_enabled=False),
    )
    assert len(opt.param_groups) == 2
    weights = sorted(g["weight_decay"] for g in opt.param_groups)
    assert weights == [0.0, 1e-4]
    # And the legacy path doesn't tag groups with LLRD bookkeeping.
    for g in opt.param_groups:
        assert "layer_id" not in g
        assert "lr_scale" not in g


# --- num_layers derivation (stub-based, no torchvision) -------------------


class _StubConvNeXtForNumLayers(nn.Module):
    """Minimal stand-in exposing the one attribute ``_resolve_llrd_num_layers`` reads.

    ``_resolve_llrd_num_layers`` only inspects ``backbone._res_stages["res4"]``
    (via ``len(...)``) to apply the ConvNeXt bucketing rule. A real
    ConvNeXtBackbone allocates 29M+ params just to expose that ModuleDict;
    this stub gives the function exactly what it needs in ~3 lines.
    """

    def __init__(self, stage2_blocks: int) -> None:
        super().__init__()
        self._res_stages = nn.ModuleDict(
            {"res4": nn.ModuleList([nn.Identity() for _ in range(stage2_blocks)])}
        )


@pytest.mark.parametrize(
    "stage2_blocks,expected_num_layers",
    [
        (9, 6),  # ConvNeXt-Tiny: buckets {0,1,2} → max_layer_id 6
        (27, 12),  # ConvNeXt-S/B/L: buckets {0..8} → max_layer_id 12
    ],
)
def test_resolve_llrd_num_layers_convnext(stage2_blocks: int, expected_num_layers: int) -> None:
    """``num_layers`` falls out mechanically from stage-2's block count via MMDet's bucketing."""
    model = nn.Sequential(_StubConvNeXtForNumLayers(stage2_blocks))
    assert _resolve_llrd_num_layers(model, "0", "convnext") == expected_num_layers


def test_resolve_llrd_num_layers_resnet_is_four() -> None:
    """ResNet family short-circuits — no backbone introspection needed."""
    # The function only branches on ``family``; the model/prefix are unused.
    assert _resolve_llrd_num_layers(nn.Module(), "", "resnet") == 4


# --- GOLDEN-VALUE TEST (synthetic names, hand-derived expected) -----------

# Hand-written enumeration of every ConvNeXt adapter branch. Each row is a
# parameter name (carved-name style, no ``backbone.bottom_up.`` prefix —
# our adapter expects the local name) paired with the expected layer_id
# under MMDet's ``get_layer_id_for_convnext`` rule applied directly:
#
#   open-mmlab/mmdetection
#   mmdet/engine/optimizers/layer_decay_optimizer_constructor.py
#   commit 73a12e6508d4ba0331b84b1313027a511ba26fe3 (2022-08-24)
#
#   downsample_layers.0  → 0          stages.0.{block} → 1
#   downsample_layers.1  → 2          stages.1.{block} → 2
#   downsample_layers.2  → 3          stages.2.{block} → 3 + block_id // 3
#   downsample_layers.3  → max_layer  stages.3.{block} → max_layer
#
# Coverage requirements baked in:
#   * stem (downsample_layers.0)
#   * every _res_downs.res{3,4,5} (downsample_layers.{1,2,3})
#   * every _res_stages.res{2,3,4,5} (stages.{0,1,2,3})
#   * stage-2 bucketing edge cases: block_ids at bucket starts (0, 3, 6, 9, ...)
#     and at the Base-variant maximum (26)
#
# The ``expected`` column IS the source of truth. The previous version of
# this test also called a vendored copy of MMDet's function on a translated
# parameter name — that became redundant once the expected column was
# hand-derived directly from the upstream rules above.
_GOLDEN_CASES_TINY: tuple[tuple[str, int], ...] = (
    ("stem.0.weight", 0),
    ("stem.0.bias", 0),
    ("stem.1.weight", 0),
    ("stem.1.bias", 0),
    ("_res_downs.res3.0.weight", 2),
    ("_res_downs.res3.1.weight", 2),
    ("_res_downs.res4.0.weight", 3),
    ("_res_downs.res5.0.weight", 6),  # num_layers
    ("_res_stages.res2.0.block.0.weight", 1),
    ("_res_stages.res2.2.block.5.weight", 1),
    ("_res_stages.res3.0.block.0.weight", 2),
    ("_res_stages.res4.0.block.0.weight", 3),  # bucket 0
    ("_res_stages.res4.3.block.0.weight", 4),  # bucket 1 (boundary)
    ("_res_stages.res4.6.block.0.weight", 5),  # bucket 2 (boundary)
    ("_res_stages.res4.8.block.0.weight", 5),  # Tiny stage-2 max
    ("_res_stages.res5.0.block.0.weight", 6),  # num_layers
    ("_res_stages.res5.2.block.5.weight", 6),
)
_GOLDEN_CASES_BASE: tuple[tuple[str, int], ...] = (
    ("stem.0.weight", 0),
    ("_res_downs.res3.0.weight", 2),
    ("_res_downs.res4.0.weight", 3),
    ("_res_downs.res5.0.weight", 12),
    ("_res_stages.res2.0.block.0.weight", 1),
    ("_res_stages.res3.0.block.0.weight", 2),
    ("_res_stages.res4.0.block.0.weight", 3),
    ("_res_stages.res4.3.block.0.weight", 4),
    ("_res_stages.res4.6.block.0.weight", 5),
    ("_res_stages.res4.8.block.0.weight", 5),  # boundary: still bucket 2
    ("_res_stages.res4.9.block.0.weight", 6),  # boundary: next bucket
    ("_res_stages.res4.26.block.0.weight", 11),  # Base stage-2 max
    ("_res_stages.res5.0.block.0.weight", 12),
)


@pytest.mark.parametrize(
    "num_layers,cases",
    [(6, _GOLDEN_CASES_TINY), (12, _GOLDEN_CASES_BASE)],
    ids=["tiny", "base"],
)
def test_layer_id_for_convnext_matches_mmdet_rule(
    num_layers: int, cases: tuple[tuple[str, int], ...]
) -> None:
    """Adapter output must equal the hand-derived MMDet layer_id for every case."""
    for local_name, expected in cases:
        ours = _layer_id_for_convnext_param(local_name, num_layers=num_layers)
        assert ours == expected, f"adapter wrong: {local_name!r} → {ours}, expected {expected}"


def test_layer_id_adapter_rejects_unknown_convnext_param() -> None:
    """The adapter raises rather than silently bucketing unknown names into layer 0."""
    with pytest.raises(AssertionError, match="unrecognised ConvNeXt"):
        _layer_id_for_convnext_param("unexpected.path.weight", num_layers=6)


def test_layer_id_adapter_rejects_param_under_res_downs_res2() -> None:
    """``_res_downs.res2`` is Identity in our model — any param there is a contract violation."""
    with pytest.raises(AssertionError, match=r"_res_downs\.res2 must be parameterless"):
        _layer_id_for_convnext_param("_res_downs.res2.0.weight", num_layers=6)


# --- end-to-end optimizer-construction tests --------------------------------
#
# These three tests are the ONLY ones in this file that construct real
# torchvision-backed backbones. They legitimately need a real model because
# they exercise ``build_optimizer(model, cfg)`` end-to-end, which walks
# ``model.named_parameters()``. We pin them to the smallest variant of each
# family (ConvNeXt-Tiny, ResNet-50) — larger variants exercise the same code
# path and add no behavioral coverage.


def _build_convnext_tiny(**kwargs):
    """Construct a ConvNeXt-Tiny for end-to-end optimizer tests.

    Imported lazily so test-collection in environments without torchvision
    available (CI configs that only run schema tests) doesn't fail.
    """
    from mayaku.models.backbones.convnext import ConvNeXtBackbone

    return ConvNeXtBackbone(name="convnext_tiny", weights=None, weights_path=None, **kwargs)


def test_llrd_scale_formula_on_convnext_tiny() -> None:
    """End-to-end: build_optimizer assigns the right LR at every layer_id.

    ConvNeXt-Tiny: num_layers=6, internal=8. With llrd_decay=0.7:
      stem (0): 0.7^7;  res2 stage (1): 0.7^6;  pre-res3 downsample (2): 0.7^5;
      res3 stage / stage-2 bucket 0 (3): 0.7^4;  bucket 1 (4): 0.7^3;
      bucket 2 (5): 0.7^2;  res5 (6): 0.7^1;  head (7): 1.0.
    """
    model = nn.ModuleDict({"backbone": _build_convnext_tiny(), "head": nn.Linear(8, 4)})
    base_lr, decay = 1e-4, 0.7
    cfg = _solver_cfg(
        optimizer_name="AdamW",
        base_lr=base_lr,
        weight_decay=0.05,
        weight_decay_norm=0.0,
        llrd_enabled=True,
        llrd_decay=decay,
    )
    opt = build_optimizer(model, cfg)

    lrs_by_layer: dict[int, float] = {}
    for g in opt.param_groups:
        lrs_by_layer.setdefault(int(g["layer_id"]), float(g["lr"]))

    expected = {i: base_lr * decay ** (7 - i) for i in range(8)}
    expected[7] = base_lr  # head/neck explicitly == base_lr (scale = decay^0)
    for layer_id, lr in lrs_by_layer.items():
        assert math.isclose(lr, expected[layer_id], rel_tol=1e-12), (
            f"layer_id {layer_id}: lr={lr} expected {expected[layer_id]}"
        )

    ordered = sorted(lrs_by_layer.items())
    for (_a, lr_a), (_b, lr_b) in itertools.pairwise(ordered):
        assert lr_b >= lr_a - 1e-12


def test_llrd_composes_with_freeze_at() -> None:
    """End-to-end: frozen stages contribute zero groups; remaining LRs stay monotonic."""
    model = nn.ModuleDict({"backbone": _build_convnext_tiny(freeze_at=2), "head": nn.Linear(8, 4)})
    cfg = _solver_cfg(optimizer_name="AdamW", base_lr=1e-4, llrd_enabled=True, llrd_decay=0.7)
    opt = build_optimizer(model, cfg)

    layer_ids = {int(g["layer_id"]) for g in opt.param_groups}
    # freeze_at=2 → stem + res2 frozen. layer_ids 0 and 1 must be absent.
    assert 0 not in layer_ids
    assert 1 not in layer_ids
    # Head still at exactly base_lr.
    head_groups = [g for g in opt.param_groups if int(g["layer_id"]) == 7]
    assert head_groups
    for g in head_groups:
        assert math.isclose(g["lr"], 1e-4, rel_tol=1e-12)

    lrs_by_layer: dict[int, float] = {}
    for g in opt.param_groups:
        lrs_by_layer.setdefault(int(g["layer_id"]), float(g["lr"]))
    ordered = sorted(lrs_by_layer.items())
    for (_a, lr_a), (_b, lr_b) in itertools.pairwise(ordered):
        assert lr_b >= lr_a - 1e-12


# --- ResNet adapter (synthetic) + one end-to-end -----------------------------


def test_layer_id_for_resnet_param_per_stage() -> None:
    """Synthetic-name coverage of every ResNet adapter branch.

    Includes the known limitation: ``res4`` collapses ALL blocks (23 for
    R-101) to a single layer_id. If that becomes a measurable problem,
    see the v2 note in ``src/mayaku/engine/optim.py`` about adding
    ``block_id // K`` bucketing for res4.
    """
    assert _layer_id_for_resnet_param("stem.0.weight", num_layers=4) == 0
    assert _layer_id_for_resnet_param("res2.0.conv1.weight", num_layers=4) == 1
    assert _layer_id_for_resnet_param("res3.2.conv1.weight", num_layers=4) == 2
    assert _layer_id_for_resnet_param("res4.0.conv1.weight", num_layers=4) == 3
    assert _layer_id_for_resnet_param("res4.22.conv1.weight", num_layers=4) == 3
    assert _layer_id_for_resnet_param("res5.0.conv1.weight", num_layers=4) == 4


def test_layer_id_for_resnet_param_rejects_unknown() -> None:
    with pytest.raises(AssertionError, match="unrecognised ResNet"):
        _layer_id_for_resnet_param("unexpected.0.weight", num_layers=4)


def test_llrd_resnet50_end_to_end_monotonic() -> None:
    """End-to-end ResNet path: every depth gets a group, head at base_lr, monotonic."""
    from mayaku.models.backbones.resnet import ResNetBackbone

    # norm="BN" exposes real BatchNorm params so we exercise the
    # norm-vs-non-norm × layer_id composition (FrozenBN has no requires_grad
    # params after freeze conversion).
    backbone = ResNetBackbone(name="resnet50", norm="BN", freeze_at=0, weights=None)
    model = nn.ModuleDict({"backbone": backbone, "head": nn.Linear(8, 4)})
    cfg = _solver_cfg(base_lr=0.02, llrd_enabled=True, llrd_decay=0.75)
    opt = build_optimizer(model, cfg)

    layer_ids = sorted({int(g["layer_id"]) for g in opt.param_groups})
    assert layer_ids == [0, 1, 2, 3, 4, 5]

    lrs_by_layer: dict[int, float] = {}
    for g in opt.param_groups:
        lrs_by_layer.setdefault(int(g["layer_id"]), float(g["lr"]))
    assert math.isclose(lrs_by_layer[5], 0.02, rel_tol=1e-12)  # head == base_lr
    # Stem: base_lr * decay^5  (num_layers=4 → internal=6 → exponent 5).
    assert math.isclose(lrs_by_layer[0], 0.02 * 0.75**5, rel_tol=1e-12)


# --- error surface ---------------------------------------------------------


def test_llrd_without_supported_backbone_raises() -> None:
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(8, 2))
    cfg = _solver_cfg(llrd_enabled=True)
    with pytest.raises(ValueError, match="no supported backbone"):
        build_optimizer(model, cfg)
