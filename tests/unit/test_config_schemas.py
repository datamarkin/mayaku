"""Tests for :mod:`mayaku.config.schemas`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mayaku.config.schemas import (
    BackboneConfig,
    InputConfig,
    MayakuConfig,
    ModelConfig,
    ROIHeadsConfig,
    ROIKeypointHeadConfig,
    ROIMaskHeadConfig,
    RPNConfig,
    SolverConfig,
)

# ---------------------------------------------------------------------------
# Defaults / contract
# ---------------------------------------------------------------------------


def test_defaults_match_3x_schedule_and_rgb_pixel_stats() -> None:
    cfg = MayakuConfig()
    # ADR 002: pixel mean/std are RGB-order.
    assert cfg.model.pixel_mean == (123.675, 116.280, 103.530)
    assert cfg.model.pixel_std == (58.395, 57.120, 57.375)
    # Spec §6.3: 3x schedule numbers.
    assert cfg.solver.max_iter == 270_000
    assert cfg.solver.steps == (210_000, 250_000)
    assert cfg.solver.base_lr == 0.02
    assert cfg.solver.ims_per_batch == 16
    # Device default is "auto", not "cuda" (BPR §4).
    assert cfg.model.device == "auto"


def test_models_are_frozen() -> None:
    cfg = MayakuConfig()
    with pytest.raises(ValidationError):
        cfg.model.pixel_mean = (1.0, 2.0, 3.0)  # type: ignore[misc]


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        BackboneConfig(name="resnet50", oops=True)  # type: ignore[call-arg]


def test_input_config_has_no_format_field() -> None:
    # ADR 002: channel order is a contract, not a setting.
    assert "format" not in InputConfig.model_fields
    assert "channel_format" not in InputConfig.model_fields
    assert "FORMAT" not in InputConfig.model_fields


# ---------------------------------------------------------------------------
# Backbone validation
# ---------------------------------------------------------------------------


def test_backbone_name_restricted_to_in_scope_set() -> None:
    BackboneConfig(name="resnet50")
    BackboneConfig(name="resnet101")
    BackboneConfig(name="resnext101_32x8d")
    with pytest.raises(ValidationError):
        BackboneConfig(name="vit_b")  # type: ignore[arg-type]


def test_backbone_freeze_at_bounds() -> None:
    BackboneConfig(name="resnet50", freeze_at=0)
    BackboneConfig(name="resnet50", freeze_at=5)
    with pytest.raises(ValidationError):
        BackboneConfig(name="resnet50", freeze_at=-1)
    with pytest.raises(ValidationError):
        BackboneConfig(name="resnet50", freeze_at=6)


# ---------------------------------------------------------------------------
# RPN / ROI heads validation
# ---------------------------------------------------------------------------


def test_rpn_post_nms_topk_train_default_matches_d2_fpn() -> None:
    # detectron2's `Base-RCNN-FPN.yaml` sets POST_NMS_TOPK_TRAIN: 1000.
    # The 2000 in detectron2's `defaults.py` is the legacy non-FPN value.
    # Mayaku's targeted reference is the FPN variant — keeping this default
    # at 1000 means feeding the ROI heads the same proposal count as D2.
    cfg = RPNConfig()
    assert cfg.post_nms_topk_train == 1000
    assert cfg.post_nms_topk_test == 1000
    assert cfg.pre_nms_topk_train == 2000
    assert cfg.pre_nms_topk_test == 1000


def test_rpn_iou_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="iou_thresholds"):
        RPNConfig(iou_thresholds=(0.7, 0.3))
    with pytest.raises(ValidationError, match="iou_thresholds"):
        RPNConfig(iou_thresholds=(0.5, 0.5))
    RPNConfig(iou_thresholds=(0.3, 0.7))  # OK


def test_roi_heads_iou_label_arity_invariant() -> None:
    # len(iou_labels) must equal len(iou_thresholds) + 1.
    ROIHeadsConfig(iou_thresholds=(0.5,), iou_labels=(0, 1))
    with pytest.raises(ValidationError, match="iou_labels"):
        ROIHeadsConfig(iou_thresholds=(0.5,), iou_labels=(0,))
    with pytest.raises(ValidationError, match="iou_labels"):
        ROIHeadsConfig(iou_thresholds=(0.3, 0.7), iou_labels=(0, 1))


# ---------------------------------------------------------------------------
# Keypoint head + flip indices (Step 4 hand-off)
# ---------------------------------------------------------------------------


def test_keypoint_flip_indices_must_be_permutation() -> None:
    # Length mismatch
    with pytest.raises(ValidationError, match="length num_keypoints"):
        ROIKeypointHeadConfig(num_keypoints=4, flip_indices=(0, 1, 2))
    # Not a permutation
    with pytest.raises(ValidationError, match="permutation"):
        ROIKeypointHeadConfig(num_keypoints=4, flip_indices=(0, 0, 1, 2))
    # Valid
    ROIKeypointHeadConfig(num_keypoints=4, flip_indices=(1, 0, 3, 2))


def test_coco_keypoint_factory_carries_canonical_pairs() -> None:
    cfg = ROIKeypointHeadConfig.with_coco_person_keypoints()
    assert cfg.num_keypoints == 17
    assert cfg.flip_indices is not None
    # nose is unpaired (maps to itself), eye_l <-> eye_r, etc.
    assert cfg.flip_indices[0] == 0
    assert cfg.flip_indices[1] == 2 and cfg.flip_indices[2] == 1
    assert cfg.flip_indices[3] == 4 and cfg.flip_indices[4] == 3
    # Permutation invariant.
    assert sorted(cfg.flip_indices) == list(range(17))


# ---------------------------------------------------------------------------
# Model consistency
# ---------------------------------------------------------------------------


def test_model_meta_architecture_requires_matching_heads() -> None:
    # mask_rcnn requires roi_mask_head + mask_on=True
    with pytest.raises(ValidationError, match="mask_on"):
        ModelConfig(meta_architecture="mask_rcnn", mask_on=False)
    with pytest.raises(ValidationError, match="roi_mask_head"):
        ModelConfig(meta_architecture="mask_rcnn", mask_on=True)
    # Valid mask_rcnn
    ModelConfig(
        meta_architecture="mask_rcnn",
        mask_on=True,
        roi_mask_head=ROIMaskHeadConfig(),
    )
    # Valid keypoint_rcnn
    ModelConfig(
        meta_architecture="keypoint_rcnn",
        keypoint_on=True,
        roi_keypoint_head=ROIKeypointHeadConfig.with_coco_person_keypoints(),
        roi_heads=ROIHeadsConfig(num_classes=1),
    )


def test_model_rejects_orphan_head_subconfigs() -> None:
    # roi_mask_head set but meta_architecture is faster_rcnn → error
    with pytest.raises(ValidationError, match="roi_mask_head is set"):
        ModelConfig(
            meta_architecture="faster_rcnn",
            roi_mask_head=ROIMaskHeadConfig(),
        )


def test_resolved_device_passes_through_explicit_kinds() -> None:
    cfg = ModelConfig(device="cpu")
    assert cfg.resolved_device() == "cpu"


def test_resolved_device_auto_returns_a_concrete_kind() -> None:
    cfg = ModelConfig(device="auto")
    assert cfg.resolved_device() in ("cpu", "mps", "cuda")


# ---------------------------------------------------------------------------
# Solver validation
# ---------------------------------------------------------------------------


def test_solver_warmup_must_fit_inside_max_iter() -> None:
    with pytest.raises(ValidationError, match="warmup_iters"):
        SolverConfig(max_iter=500, warmup_iters=500, steps=(100,))


def test_solver_steps_must_be_strictly_ascending_and_below_max_iter() -> None:
    # Use small warmup so max_iter=1000 doesn't trip the warmup<max_iter check.
    with pytest.raises(ValidationError, match="steps"):
        SolverConfig(max_iter=1000, warmup_iters=10, steps=(500, 500))
    with pytest.raises(ValidationError, match="steps"):
        SolverConfig(max_iter=1000, warmup_iters=10, steps=(2000,))
    SolverConfig(max_iter=1000, warmup_iters=10, steps=(500, 800))  # OK


def test_solver_momentum_and_gamma_bounds() -> None:
    with pytest.raises(ValidationError):
        SolverConfig(momentum=1.0)
    with pytest.raises(ValidationError):
        SolverConfig(gamma=1.0)
    with pytest.raises(ValidationError):
        SolverConfig(gamma=0.0)
