"""Typed configuration schemas for Mayaku.

Pydantic v2 models replace Detectron2's `CfgNode` (`config/defaults.py`)
and `LazyConfig` machinery. The design follows the recommendation from
`DETECTRON2_TECHNICAL_SPEC.md` §9.1: defaults live with the type, no
600-line monolithic ``defaults.py``, no string-based registry indirection
in the schema (architecture choice is a typed ``Literal``, not the
``"build_resnet_fpn_backbone"`` factory-name string the legacy YAML used).

What this file pins down explicitly relative to the spec / portability
report:

* **RGB pixel mean and std** (per ADR 002,
  ``docs/decisions/002-rgb-native-image-ingestion.md``). Detectron2's
  defaults are BGR; we are not inheriting them. There is no
  ``INPUT.FORMAT`` knob — channel order is a contract, not a setting.
* **No rotated boxes**, **no deformable convolution**. Both are out of
  scope for v1 (`BACKEND_PORTABILITY_REPORT.md` §3, ADR 001).
* **Device default is ``"auto"``**, not ``"cuda"``. The legacy
  ``MODEL.DEVICE = "cuda"`` (see ``BACKEND_PORTABILITY_REPORT.md`` §4)
  would break anyone running on MPS; ``"auto"`` resolves through
  :meth:`mayaku.backends.device.Device.auto` at construction time.
* **3x is the default schedule.** ``SolverConfig`` ships with the
  spec §6.3 numbers (``max_iter=270000``, ``steps=(210000, 250000)``,
  ``base_lr=0.02`` from ``Base-RCNN-FPN.yaml``). 1x and other
  schedules are constructed via :mod:`mayaku.config.schedules`.

All models are frozen and reject unknown fields. Use
``model.model_copy(update={...})`` to derive a variant.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mayaku.backends.device import DeviceKind

__all__ = [
    "AnchorGeneratorConfig",
    "AutoConfig",
    "BackboneConfig",
    "BackboneName",
    "DataLoaderConfig",
    "DeviceSetting",
    "FPNConfig",
    "InputConfig",
    "MayakuConfig",
    "MetaArchitecture",
    "ModelConfig",
    "ROIBoxHeadConfig",
    "ROIHeadsConfig",
    "ROIKeypointHeadConfig",
    "ROIMaskHeadConfig",
    "RPNConfig",
    "SolverConfig",
    "TestConfig",
]

BackboneName = Literal[
    "resnet50",
    "resnet101",
    "resnext101_32x8d",
    # ConvNeXt variants (Tiny / Small / Base / Large). Standard ConvNeXt
    # architecture (torchvision's reference). Weight provenance is
    # carried by :attr:`BackboneConfig.weights_path`, not the name —
    # the same `convnext_small` works with random init, torchvision
    # ImageNet-1k (via ``--pretrained-backbone``), or any local
    # checkpoint supplied as a path (the original Liu et al. release,
    # the DINOv3 LVD-1689M distillation, user fine-tunes, …).
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]
MetaArchitecture = Literal["faster_rcnn", "mask_rcnn", "keypoint_rcnn"]
DeviceSetting = Literal["cpu", "mps", "cuda", "auto"]


class _BaseModel(BaseModel):
    """Shared pydantic config: immutable, strict, validate defaults."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
    )


# ---------------------------------------------------------------------------
# Backbone + neck
# ---------------------------------------------------------------------------


class BackboneConfig(_BaseModel):
    """Backbone selection + stem/freezing knobs.

    Architecture-specific shape (e.g. ResNeXt's ``num_groups=32`` /
    ``width_per_group=8`` / ``stride_in_1x1=False`` from the FAIR C2
    pre-trained checkpoint, see
    `DETECTRON2_TECHNICAL_SPEC.md` §2.1) is keyed off ``name`` rather
    than carried as separate fields — Step 7 will translate this into
    the actual module construction.

    The same config covers ResNet/ResNeXt and ConvNeXt variants;
    a model-validator rejects field combinations that don't apply to
    the chosen architecture (e.g. ``stride_in_1x1`` only makes sense
    for ResNets, ``weights_path`` only for ConvNeXts).
    """

    name: BackboneName = "resnet50"
    norm: Literal["FrozenBN", "BN", "GN", "SyncBN"] = "FrozenBN"
    freeze_at: Annotated[int, Field(ge=0, le=5)] = 2
    stem_out_channels: Annotated[int, Field(gt=0)] = 64
    res5_dilation: Literal[1, 2] = 1
    # Where the stride-2 sits inside the first bottleneck of res3/res4/res5.
    # torchvision (and Mayaku's default) puts it on the 3x3 conv. Detectron2's
    # MSRA-pretrained model zoo (e.g. faster_rcnn_R_50_FPN_3x) puts it on the
    # 1x1 conv — loading those weights with the default placement silently
    # produces wrong activations (same shapes, identical kernels, different
    # downsampling step). Flip this to True when loading D2 model-zoo weights.
    stride_in_1x1: bool = False

    # Local path to a pretrained ConvNeXt checkpoint. Accepts both
    # torchvision key naming (``features.*.block.*.layer_scale``) and
    # facebookresearch / Liu et al. key naming (``stages.*.{dwconv,
    # norm,pwconv1,pwconv2,gamma}``, ``downsample_layers.*``). File
    # formats: ``.pth`` / ``.pt`` / ``.bin`` (PyTorch pickle) and
    # ``.safetensors`` (HuggingFace).
    #
    # Mayaku ships no URLs and no auto-download — the user supplies the
    # file. Common sources: the original ConvNeXt release, the DINOv3
    # LVD-1689M distillation
    # (``facebook/dinov3-convnext-{tiny,small,base,large}-pretrain-lvd1689m``
    # on HuggingFace, license-gated; accept the upstream license and
    # download manually), or a user fine-tune. Only valid for
    # ``convnext_*`` variants.
    weights_path: str | None = None

    @model_validator(mode="after")
    def _check_arch_specific_fields(self) -> BackboneConfig:
        # Naming convention: ConvNeXt variants are exactly those whose
        # name starts with ``convnext_`` (BackboneName Literal enforces
        # the closed set). Same predicate as ``is_convnext_variant`` —
        # they're kept in sync by the convention, not a shared table.
        is_convnext = self.name.startswith("convnext_")
        if is_convnext:
            # ConvNeXt uses LayerNorm exclusively — the BN-family knobs are
            # nonsense for it. We don't silently ignore them because that
            # masks user error; we reject any non-default value.
            if self.norm != "FrozenBN":
                raise ValueError(
                    f"backbone.norm={self.norm!r} only applies to ResNet variants; "
                    f"ConvNeXt uses LayerNorm intrinsically. Remove the field or "
                    f"leave it at the default 'FrozenBN'."
                )
            if self.stride_in_1x1:
                raise ValueError(
                    "backbone.stride_in_1x1=True only applies to ResNet variants "
                    "loading Detectron2 MSRA-pretrained weights; ConvNeXt has no "
                    "bottleneck to relocate the stride within."
                )
            if self.res5_dilation != 1:
                raise ValueError("backbone.res5_dilation only applies to ResNet variants.")
        else:
            if self.weights_path is not None:
                raise ValueError(
                    f"backbone.weights_path is only valid for ConvNeXt variants "
                    f"(got name={self.name!r}). ResNet/ResNeXt use torchvision's "
                    f"published ImageNet weights via the --pretrained-backbone CLI "
                    f"flag (``weights='DEFAULT'``)."
                )
        return self


class FPNConfig(_BaseModel):
    in_features: tuple[str, ...] = ("res2", "res3", "res4", "res5")
    out_channels: Annotated[int, Field(gt=0)] = 256
    norm: str = ""  # empty string == no norm, matching upstream defaults
    fuse_type: Literal["sum", "avg"] = "sum"


# ---------------------------------------------------------------------------
# Anchors / RPN
# ---------------------------------------------------------------------------


class AnchorGeneratorConfig(_BaseModel):
    """Per-FPN-level anchor sizes and shared aspect ratios.

    The default ``sizes`` ladder ``((32,), (64,), (128,), (256,), (512,))``
    is the FPN convention from `DETECTRON2_TECHNICAL_SPEC.md` §2.3 — one
    anchor scale per level so each level handles a single object size
    band. Aspect ratios are shared across levels.
    """

    sizes: tuple[tuple[int, ...], ...] = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios: tuple[tuple[float, ...], ...] = ((0.5, 1.0, 2.0),)
    offset: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0


class RPNConfig(_BaseModel):
    """Region proposal network knobs (`DETECTRON2_TECHNICAL_SPEC.md` §2.3)."""

    in_features: tuple[str, ...] = ("p2", "p3", "p4", "p5", "p6")
    pre_nms_topk_train: Annotated[int, Field(gt=0)] = 2000
    pre_nms_topk_test: Annotated[int, Field(gt=0)] = 1000
    # 1000 matches detectron2's `Base-RCNN-FPN.yaml` (POST_NMS_TOPK_TRAIN: 1000).
    # The 2000 in detectron2's `defaults.py` is the legacy non-FPN value;
    # the FPN base config overrides it. Feeding 2× as many proposals to
    # the ROI heads doubles the low-IoU clutter in the negative-sample
    # pool and biases the ROI cls head toward learning "background" from
    # noisy proposals.
    post_nms_topk_train: Annotated[int, Field(gt=0)] = 1000
    post_nms_topk_test: Annotated[int, Field(gt=0)] = 1000
    nms_thresh: Annotated[float, Field(gt=0.0, le=1.0)] = 0.7
    min_box_size: Annotated[float, Field(ge=0.0)] = 1e-5
    iou_thresholds: tuple[float, float] = (0.3, 0.7)
    iou_labels: tuple[int, int, int] = (0, -1, 1)
    batch_size_per_image: Annotated[int, Field(gt=0)] = 256
    positive_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    bbox_reg_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    smooth_l1_beta: Annotated[float, Field(ge=0.0)] = 0.0
    loss_weight: Annotated[float, Field(gt=0.0)] = 1.0
    box_reg_loss_type: Literal["smooth_l1", "giou"] = "smooth_l1"

    @model_validator(mode="after")
    def _check_iou_thresholds(self) -> RPNConfig:
        lo, hi = self.iou_thresholds
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError(f"RPN.iou_thresholds must satisfy 0 <= lo < hi <= 1; got ({lo}, {hi})")
        return self


# ---------------------------------------------------------------------------
# ROI heads
# ---------------------------------------------------------------------------


class ROIBoxHeadConfig(_BaseModel):
    """FastRCNNConvFCHead + box predictor (`DETECTRON2_TECHNICAL_SPEC.md` §3.4)."""

    pooler_resolution: Annotated[int, Field(gt=0)] = 7
    pooler_sampling_ratio: Annotated[int, Field(ge=0)] = 0
    num_conv: Annotated[int, Field(ge=0)] = 0
    conv_dim: Annotated[int, Field(gt=0)] = 256
    num_fc: Annotated[int, Field(ge=0)] = 2
    fc_dim: Annotated[int, Field(gt=0)] = 1024
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)
    smooth_l1_beta: Annotated[float, Field(ge=0.0)] = 0.0
    box_reg_loss_type: Literal["smooth_l1", "giou"] = "smooth_l1"
    cls_agnostic_bbox_reg: bool = False


class ROIMaskHeadConfig(_BaseModel):
    """MaskRCNNConvUpsampleHead (`DETECTRON2_TECHNICAL_SPEC.md` §3.5)."""

    pooler_resolution: Annotated[int, Field(gt=0)] = 14
    pooler_sampling_ratio: Annotated[int, Field(ge=0)] = 0
    num_conv: Annotated[int, Field(gt=0)] = 4
    conv_dim: Annotated[int, Field(gt=0)] = 256
    cls_agnostic_mask: bool = False
    loss_weight: Annotated[float, Field(gt=0.0)] = 1.0


class ROIKeypointHeadConfig(_BaseModel):
    """KRCNNConvDeconvUpsampleHead (`DETECTRON2_TECHNICAL_SPEC.md` §3.6).

    ``flip_indices`` is the permutation used by horizontal-flip
    augmentation (Step 4 / Step 6) so left/right keypoints swap
    semantics correctly. It is dataset-specific; the COCO 17-keypoint
    convention is the upstream default and is set by
    :meth:`with_coco_person_keypoints`.
    """

    pooler_resolution: Annotated[int, Field(gt=0)] = 14
    pooler_sampling_ratio: Annotated[int, Field(ge=0)] = 0
    conv_dims: tuple[int, ...] = (512,) * 8
    num_keypoints: Annotated[int, Field(gt=0)] = 17
    min_keypoints_per_image: Annotated[int, Field(ge=0)] = 1
    normalize_loss_by_visible_keypoints: bool = True
    loss_weight: Annotated[float, Field(gt=0.0)] = 1.0
    flip_indices: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _check_flip_indices(self) -> ROIKeypointHeadConfig:
        if self.flip_indices is not None:
            k = self.num_keypoints
            if len(self.flip_indices) != k:
                raise ValueError(
                    f"flip_indices must have length num_keypoints={k}; "
                    f"got length {len(self.flip_indices)}"
                )
            if sorted(self.flip_indices) != list(range(k)):
                raise ValueError("flip_indices must be a permutation of range(num_keypoints)")
        return self

    @classmethod
    def with_coco_person_keypoints(cls) -> ROIKeypointHeadConfig:
        """COCO Person Keypoints flip-pair convention (17 keypoints).

        Order: nose, eye_l, eye_r, ear_l, ear_r, shoulder_l, shoulder_r,
        elbow_l, elbow_r, wrist_l, wrist_r, hip_l, hip_r, knee_l, knee_r,
        ankle_l, ankle_r. Pairs are swapped on horizontal flip.
        """
        return cls(
            num_keypoints=17,
            flip_indices=(0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15),
        )


class ROIHeadsConfig(_BaseModel):
    """StandardROIHeads dispatcher knobs (`DETECTRON2_TECHNICAL_SPEC.md` §3.3)."""

    in_features: tuple[str, ...] = ("p2", "p3", "p4", "p5")
    num_classes: Annotated[int, Field(gt=0)] = 80
    batch_size_per_image: Annotated[int, Field(gt=0)] = 512
    positive_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    iou_thresholds: tuple[float, ...] = (0.5,)
    iou_labels: tuple[int, ...] = (0, 1)
    score_thresh_test: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    nms_thresh_test: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    proposal_append_gt: bool = True

    @model_validator(mode="after")
    def _check_iou_label_arity(self) -> ROIHeadsConfig:
        # Detectron2 invariant (matcher.py): len(iou_labels) == len(iou_thresholds) + 1.
        if len(self.iou_labels) != len(self.iou_thresholds) + 1:
            raise ValueError(
                "ROIHeads.iou_labels must have one more element than "
                f"iou_thresholds; got {len(self.iou_labels)} labels for "
                f"{len(self.iou_thresholds)} thresholds"
            )
        return self


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ModelConfig(_BaseModel):
    """Top-level model knobs + per-component sub-configs.

    ``pixel_mean`` and ``pixel_std`` are RGB-order per ADR 002. The
    legacy BGR order from `DETECTRON2_TECHNICAL_SPEC.md` §2.1 is not
    inherited; channel order is a contract, not a setting (no
    ``INPUT.FORMAT`` knob).
    """

    meta_architecture: MetaArchitecture = "faster_rcnn"
    mask_on: bool = False
    keypoint_on: bool = False
    # ImageNet RGB normalisation (ADR 002). Mean is in [0, 255]; std is
    # the per-channel std x 255. Matches torchvision's pretrained ResNet
    # contract end-to-end so backbone weights load directly without a
    # channel swap.
    pixel_mean: tuple[float, float, float] = (123.675, 116.280, 103.530)
    pixel_std: tuple[float, float, float] = (58.395, 57.120, 57.375)
    weights: str | None = None
    device: DeviceSetting = "auto"

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    fpn: FPNConfig = Field(default_factory=FPNConfig)
    anchor_generator: AnchorGeneratorConfig = Field(default_factory=AnchorGeneratorConfig)
    rpn: RPNConfig = Field(default_factory=RPNConfig)
    roi_heads: ROIHeadsConfig = Field(default_factory=ROIHeadsConfig)
    roi_box_head: ROIBoxHeadConfig = Field(default_factory=ROIBoxHeadConfig)
    roi_mask_head: ROIMaskHeadConfig | None = None
    roi_keypoint_head: ROIKeypointHeadConfig | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelConfig:
        # meta_architecture, mask_on, keypoint_on, and the head sub-configs
        # must agree. We treat meta_architecture as the single source of
        # truth for which heads are required and accept matching booleans
        # as a redundant convenience for callers who forget either side.
        wants_mask = self.meta_architecture == "mask_rcnn"
        wants_kpt = self.meta_architecture == "keypoint_rcnn"
        if self.mask_on != wants_mask:
            raise ValueError(
                f"mask_on={self.mask_on} disagrees with "
                f"meta_architecture={self.meta_architecture!r}"
            )
        if self.keypoint_on != wants_kpt:
            raise ValueError(
                f"keypoint_on={self.keypoint_on} disagrees with "
                f"meta_architecture={self.meta_architecture!r}"
            )
        if wants_mask and self.roi_mask_head is None:
            raise ValueError("mask_rcnn requires roi_mask_head to be set")
        if wants_kpt and self.roi_keypoint_head is None:
            raise ValueError("keypoint_rcnn requires roi_keypoint_head to be set")
        if not wants_mask and self.roi_mask_head is not None:
            raise ValueError("roi_mask_head is set but meta_architecture is not mask_rcnn")
        if not wants_kpt and self.roi_keypoint_head is not None:
            raise ValueError("roi_keypoint_head is set but meta_architecture is not keypoint_rcnn")
        return self

    def resolved_device(self) -> DeviceKind:
        """Translate the configured device into a concrete backend kind.

        ``"auto"`` resolves through :meth:`mayaku.backends.device.Device.auto`;
        the literal kinds pass through unchanged.
        """
        if self.device == "auto":
            from mayaku.backends.device import Device

            return Device.auto().kind
        return self.device


# ---------------------------------------------------------------------------
# Input + dataloader + solver + test
# ---------------------------------------------------------------------------


class InputConfig(_BaseModel):
    """Image-pipeline knobs.

    ``min_size_train`` defaults to the ``Base-RCNN-FPN.yaml`` jitter
    range ``(640, 672, 704, 736, 768, 800)`` (`DETECTRON2_TECHNICAL_SPEC.md`
    §6.1). Pure-train scaling resamples one of these short edges per
    iteration when ``min_size_train_sampling="choice"``; ``"range"``
    samples uniformly between the min and max of the tuple.

    There is **no FORMAT field** — ADR 002 fixes channel order to RGB.
    """

    min_size_train: tuple[int, ...] = (640, 672, 704, 736, 768, 800)
    max_size_train: Annotated[int, Field(gt=0)] = 1333
    min_size_train_sampling: Literal["choice", "range"] = "choice"
    min_size_test: Annotated[int, Field(gt=0)] = 800
    max_size_test: Annotated[int, Field(gt=0)] = 1333
    mask_format: Literal["polygon", "bitmask"] = "polygon"
    random_flip: Literal["none", "horizontal", "vertical"] = "horizontal"

    # Photometric augmentation (Phase 1 modernization). Each delta is the
    # max deviation from no-op; a delta of 0 disables that component. The
    # defaults below match HSV-V/S/H knobs translated
    # into mayaku's (brightness, contrast, saturation, hue) parameterisation.
    # Disabled by default to preserve D2-replication; enable in modernized
    # configs.
    color_jitter_enabled: bool = False
    color_jitter_brightness: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
    color_jitter_contrast: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
    color_jitter_saturation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    color_jitter_hue: Annotated[float, Field(ge=0.0, le=0.5)] = 0.015
    color_jitter_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5

    # RandAugment (Cubuk et al. 2019), photometric-only pool. Two knobs:
    # ``num_ops`` (paper's N, ops applied per image) and ``magnitude``
    # (paper's M, intensity in [0, 30]). Replaces per-op probability /
    # range tuning with a single intensity dial — useful for users who
    # don't want to tune brightness/contrast/etc. separately. Compatible
    # with ``color_jitter_enabled``; usually you'd pick one or the other.
    randaugment_enabled: bool = False
    randaugment_num_ops: Annotated[int, Field(ge=0, le=9)] = 2
    randaugment_magnitude: Annotated[float, Field(ge=0.0, le=30.0)] = 9.0

    # Multi-sample augmentation (Phase 1b modernization). Each
    # ``*_prob`` is the chance the augmentation fires for any given
    # training sample; default 0.0 disables. ``copy_paste_prob > 0``
    # additionally requires ``mask_format='bitmask'`` (validator below).
    mosaic_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    mosaic_canvas_size: tuple[int, int] = (1024, 1024)
    mixup_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    mixup_alpha: Annotated[float, Field(gt=0.0)] = 8.0
    copy_paste_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @model_validator(mode="after")
    def _check_copy_paste_needs_bitmask(self) -> InputConfig:
        if self.copy_paste_prob > 0.0 and self.mask_format != "bitmask":
            raise ValueError(
                "copy_paste_prob > 0 requires mask_format='bitmask' "
                "(polygon-mask paste requires a lossy raster→polygon round-trip; "
                "switch the format or disable CopyPaste)."
            )
        return self


class SolverConfig(_BaseModel):
    """Optimizer and LR schedule.

    Defaults are the **3x schedule** from `DETECTRON2_TECHNICAL_SPEC.md`
    §6.3: ``ims_per_batch=16``, ``base_lr=0.02``, ``max_iter=270000``,
    ``steps=(210000, 250000)``, ``warmup_iters=1000``,
    ``warmup_factor=1/1000``. Use :func:`mayaku.config.schedules.schedule_1x`
    or ``schedule_3x`` to construct the canonical variants.
    """

    ims_per_batch: Annotated[int, Field(gt=0)] = 16
    # Gradient accumulation: divide effective batch into ``grad_accum_steps``
    # micro-batches of ``ims_per_batch``. Memory scales with the micro-batch
    # only, so this is the standard knob for fitting a large effective batch
    # into a small GPU. Effective batch = ``ims_per_batch * grad_accum_steps``.
    # ``base_lr`` should be tuned against the effective batch, not the micro.
    grad_accum_steps: Annotated[int, Field(ge=1)] = 1
    base_lr: Annotated[float, Field(gt=0.0)] = 0.02
    lr_scheduler_name: Literal["WarmupMultiStepLR", "WarmupCosineLR"] = "WarmupMultiStepLR"
    steps: tuple[int, ...] = (210_000, 250_000)
    max_iter: Annotated[int, Field(gt=0)] = 270_000
    warmup_iters: Annotated[int, Field(ge=0)] = 1000
    warmup_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0 / 1000.0
    warmup_method: Literal["linear", "constant"] = "linear"
    momentum: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.9
    nesterov: bool = False
    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-4
    weight_decay_norm: Annotated[float, Field(ge=0.0)] = 0.0
    gamma: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.1
    amp_enabled: bool = False
    # Resolved at Step 13 (engine). ``"fp16"`` is the safe cross-backend
    # default; ``"bf16"`` is recommended on CUDA Ampere+ where it gives a
    # wider dynamic range and lets us drop GradScaler. MPS does not yet
    # support bf16 autocast (verify per-PT-version before flipping).
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    checkpoint_period: Annotated[int, Field(gt=0)] = 5000
    clip_gradients_enabled: bool = False
    # When clipping is enabled, the defaults below give the standard
    # global-L2-norm safety net at 5.0 — wide enough to catch genuine
    # gradient blow-ups without throttling normal training, narrow enough
    # to keep an accidental NaN from poisoning the rest of the run.
    # ``"value"`` element-wise clamping is also supported but rarely
    # what people want.
    clip_gradients_value: Annotated[float, Field(gt=0.0)] = 5.0
    clip_gradients_type: Literal["value", "norm"] = "norm"

    # Diagnostic: when true, every training step computes the global L2
    # gradient norm AND per-module-group sub-norms (RPN cls / RPN loc /
    # ROI cls / ROI loc / ROI box-head / backbone / FPN) before clipping
    # and records them on ``trainer.storage`` so MetricsPrinter logs
    # them. Used to localise gradient blow-ups when training without
    # clipping. Adds one element-wise op per parameter per step (cheap;
    # ~ms-level on COCO scale). Default off.
    grad_norm_log_enabled: bool = False

    # Exponential moving average of model weights (Phase 1 modernization).
    # When enabled, an EMA shadow tracks the live weights at every step
    # and a parallel EMA checkpoint is saved alongside the live one.
    # Default off so D2-replication training runs are bit-identical to
    # the existing 40.2-AP baseline.
    ema_enabled: bool = False
    ema_decay: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9999
    ema_tau: Annotated[float, Field(gt=0.0)] = 2000.0

    @model_validator(mode="after")
    def _check_schedule(self) -> SolverConfig:
        if self.warmup_iters >= self.max_iter:
            raise ValueError(
                f"warmup_iters ({self.warmup_iters}) must be < max_iter ({self.max_iter})"
            )
        for s in self.steps:
            if not 0 < s < self.max_iter:
                raise ValueError(
                    f"steps entries must satisfy 0 < step < max_iter; got {s} "
                    f"with max_iter={self.max_iter}"
                )
        if any(b <= a for a, b in zip(self.steps, self.steps[1:], strict=False)):
            raise ValueError(f"steps must be strictly ascending; got {self.steps}")
        return self


class TestConfig(_BaseModel):
    detections_per_image: Annotated[int, Field(gt=0)] = 100
    eval_period: Annotated[int, Field(ge=0)] = 0  # 0 == disabled
    precise_bn_enabled: bool = False
    precise_bn_num_iter: Annotated[int, Field(gt=0)] = 200


class DataLoaderConfig(_BaseModel):
    num_workers: Annotated[int, Field(ge=0)] = 4
    aspect_ratio_grouping: bool = True
    sampler_train: Literal["TrainingSampler", "RepeatFactorTrainingSampler"] = "TrainingSampler"
    filter_empty_annotations: bool = True
    # RepeatFactorTrainingSampler threshold ``t`` (Gupta et al. LVIS 2019).
    # Per-class repeat factor is ``max(1, sqrt(t / f_c))`` where ``f_c`` is
    # the fraction of training images containing class ``c``. ``t=0.001``
    # is the LVIS default; raise it (e.g. 0.01) for aggressive balancing
    # on extremely imbalanced custom datasets, lower it for milder
    # oversampling. Ignored unless ``sampler_train="RepeatFactorTrainingSampler"``.
    repeat_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.001


class AutoConfig(_BaseModel):
    """Dataset-aware auto-tuning of fine-tune fields at ``mayaku train`` start.

    When ``enabled`` is true (the default), ``mayaku train`` runs a
    single read-only pass over the COCO dataset *before* model
    construction and overrides fine-tune-relevant fields that the user
    did NOT explicitly set in the source YAML:

    * ``model.roi_heads.num_classes`` — from the dataset's category count
    * ``model.anchor_generator.sizes`` / ``aspect_ratios`` — k-means on
      GT box √area and w/h (skipped if <50 boxes)
    * ``model.backbone.freeze_at`` — from dataset size bucket
    * ``solver.base_lr`` / ``max_iter`` / ``steps`` / ``warmup_iters`` /
      ``lr_scheduler_name`` / ``ema_enabled`` / ``ema_decay`` / ``ema_tau``
      — from dataset size bucket, anchored to the D2 scratch recipe and
      applying both batch-scaling and the 10× fine-tune drop
    * ``input.mosaic_prob`` / ``mixup_prob`` / ``copy_paste_prob`` — from
      dataset size bucket
    * ``dataloader.sampler_train`` / ``repeat_threshold`` — switched to
      ``RepeatFactorTrainingSampler`` when class-imbalance ratio > 10

    Explicit user values always win — auto-config only fills gaps. The
    resolved config is dumped to ``output_dir/config.yaml`` for
    reproducibility.

    Set ``enabled: false`` for replication runs where the config's
    defaults are intentional (e.g. the bundled COCO 1x/3x recipes), or
    to make the train run bit-identical to a hand-written recipe.

    Tiny datasets (<10 images) are skipped automatically — there isn't
    enough signal to derive a sensible recipe.
    """

    enabled: bool = True


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class MayakuConfig(_BaseModel):
    """The whole configuration tree, with every section defaulting to the
    Detectron2 3x convention (modulo ADR-driven changes — RGB channel
    order, no rotated boxes, no deformable conv, ``device="auto"``)."""

    input: InputConfig = Field(default_factory=InputConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    test: TestConfig = Field(default_factory=TestConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    auto_config: AutoConfig = Field(default_factory=AutoConfig)
