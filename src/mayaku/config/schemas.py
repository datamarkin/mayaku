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
* **Training length is in epochs.** ``SolverConfig.num_epochs`` sets the
  number of passes over the dataset; the engine resolves it to an iteration
  count at train time, and the LR follows a single warmup→cosine decay.

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
    "UniQueryHeadConfig",
    "UniQueryKeypointConfig",
    "UniQueryMaskConfig",
]

BackboneName = Literal[
    "resnet50",
    "resnet101",
    "resnext101_32x8d",
    # ConvNeXt variants. Atto/Femto/Pico/Nano use V2 size configs with
    # V1-style blocks (no GRN). Tiny/Small/Base/Large are torchvision's
    # reference. Weight provenance is carried by weights_path, not name.
    "convnext_atto",
    "convnext_femto",
    "convnext_pico",
    "convnext_nano",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]
MetaArchitecture = Literal["faster_rcnn", "mask_rcnn", "keypoint_rcnn", "uniquery"]
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
# UniQuery head
# ---------------------------------------------------------------------------


class UniQueryHeadConfig(_BaseModel):
    """UniQuery iterative dynamic head configuration.

    Implements a set-prediction detector with learned proposals that
    iteratively refine via self-attention and dynamic convolution. No RPN,
    no NMS — fixed output of ``num_proposals`` predictions.
    """

    num_proposals: Annotated[int, Field(gt=0)] = 300
    hidden_dim: Annotated[int, Field(gt=0)] = 256
    num_heads: Annotated[int, Field(gt=0)] = 8
    num_stages: Annotated[int, Field(gt=0)] = 6
    dim_feedforward: Annotated[int, Field(gt=0)] = 2048
    dim_dynamic: Annotated[int, Field(gt=0)] = 64
    pooler_resolution: Annotated[int, Field(gt=0)] = 7
    # Samples per ROI bin (fixed; same value in train and deploy — the export
    # one-pass can't do per-box adaptive). 1 = one sample/bin (faster, no
    # averaging op → TensorRT-fp16 safe); <=0 resolves to 2. Real-time tiers
    # (n/s/m) use 1, accuracy tiers (l/xl/xxl) use 2.
    pooler_sampling_ratio: Annotated[int, Field(ge=0)] = 0
    dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    # Hungarian matching cost weights
    cost_class: Annotated[float, Field(gt=0.0)] = 2.0
    cost_bbox: Annotated[float, Field(gt=0.0)] = 5.0
    cost_giou: Annotated[float, Field(gt=0.0)] = 2.0

    # QGN (Featurized Query R-CNN, arXiv 2206.06258): image-conditioned
    # query initialization from a light dense scorer on FPN, replacing
    # the blind learned embeddings. Enables strong AP at fewer stages.
    uniquery_generator: bool = False
    qgn_quality_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    qgn_obj_weight: Annotated[float, Field(gt=0.0)] = 1.0
    qgn_giou_weight: Annotated[float, Field(gt=0.0)] = 2.0

    # Add conv-based P6/P7 (strides 64/128) to the FPN so the QGN proposes
    # and the head pools large objects at a coarse pyramid (the paper's QGN
    # runs P3-P7). Targets the large-object (APl) deficit of a P3-P5 head.
    fpn_p6p7: bool = False

    # DN-DETR-style query denoising (box-only): feed noised GT boxes as
    # auxiliary queries trained to reconstruct the clean GT. Stabilizes the
    # early Hungarian matching -> faster convergence, most useful at few
    # stages. Training-only: DN queries are not generated at inference, so
    # zero deployment/export impact.
    denoising: bool = False
    dn_groups: Annotated[int, Field(gt=0)] = 5
    dn_box_noise_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 0.4
    dn_loss_weight: Annotated[float, Field(gt=0.0)] = 1.0

    # Cascade-IoU: per-stage minimum IoU floor for Hungarian matching.
    # Empty tuple = disabled (vanilla flat matching). When set, length
    # must equal num_stages. Predictions below the floor cannot match a
    # GT in that stage. Training-only — zero inference/export impact.
    # Recommended for 6 stages: (0.0, 0.0, 0.4, 0.5, 0.6, 0.7)
    cascade_iou_thresholds: tuple[float, ...] = ()

    # Inference-time knobs: use fewer stages or proposals at test time
    # for speed without retraining. None = use training values.
    inference_num_stages: Annotated[int, Field(gt=0)] | None = None
    inference_num_proposals: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _check_cascade_iou(self) -> UniQueryHeadConfig:
        t = self.cascade_iou_thresholds
        if t and len(t) != self.num_stages:
            raise ValueError(
                f"cascade_iou_thresholds length ({len(t)}) must equal "
                f"num_stages ({self.num_stages}) or be empty"
            )
        if any(v < 0.0 or v > 1.0 for v in t):
            raise ValueError("cascade_iou_thresholds values must be in [0.0, 1.0]")
        if self.inference_num_stages is not None and self.inference_num_stages > self.num_stages:
            raise ValueError(
                f"inference_num_stages ({self.inference_num_stages}) cannot exceed "
                f"num_stages ({self.num_stages})"
            )
        if (
            self.inference_num_proposals is not None
            and self.inference_num_proposals > self.num_proposals
        ):
            raise ValueError(
                f"inference_num_proposals ({self.inference_num_proposals}) cannot exceed "
                f"num_proposals ({self.num_proposals})"
            )
        return self


class UniQueryMaskConfig(_BaseModel):
    """UniQuery dynamic mask head (instance segmentation).

    Built by :func:`build_uniquery` when ``model.uniquery_mask`` is set; drives
    ``configs/segmentation/mayaku-*``.
    """

    pooler_resolution: Annotated[int, Field(gt=0)] = 14
    mask_resolution: Annotated[int, Field(gt=0)] = 28
    num_conv: Annotated[int, Field(gt=0)] = 4
    conv_dim: Annotated[int, Field(gt=0)] = 256
    loss_weight: Annotated[float, Field(gt=0.0)] = 1.0


class UniQueryKeypointConfig(_BaseModel):
    """UniQuery keypoint head (person-pose).

    Built by :func:`build_uniquery` when ``model.uniquery_keypoint`` is set;
    drives ``configs/keypoints/mayaku-*``. ``heatmap_resolution`` is currently
    advisory — the head derives its output size internally.
    """

    pooler_resolution: Annotated[int, Field(gt=0)] = 14
    num_keypoints: Annotated[int, Field(gt=0)] = 17
    heatmap_resolution: Annotated[int, Field(gt=0)] = 56
    loss_weight: Annotated[float, Field(gt=0.0)] = 1.0


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

    # UniQuery head configs (only used when meta_architecture == "uniquery")
    uniquery_head: UniQueryHeadConfig | None = None
    uniquery_mask: UniQueryMaskConfig | None = None
    uniquery_keypoint: UniQueryKeypointConfig | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelConfig:
        # meta_architecture, mask_on, keypoint_on, and the head sub-configs
        # must agree. We treat meta_architecture as the single source of
        # truth for which heads are required and accept matching booleans
        # as a redundant convenience for callers who forget either side.
        is_query = self.meta_architecture == "uniquery"
        wants_mask = self.meta_architecture == "mask_rcnn"
        wants_kpt = self.meta_architecture == "keypoint_rcnn"

        # UniQuery has its own head configs and doesn't use mask_on/keypoint_on flags
        if is_query:
            if self.mask_on or self.keypoint_on:
                raise ValueError(
                    "uniquery uses uniquery_mask/uniquery_keypoint configs, "
                    "not mask_on/keypoint_on flags"
                )
            if self.uniquery_head is None:
                raise ValueError("uniquery requires uniquery_head to be set")
            if self.fpn.out_channels != self.uniquery_head.hidden_dim:
                raise ValueError(
                    f"fpn.out_channels ({self.fpn.out_channels}) must equal "
                    f"uniquery_head.hidden_dim ({self.uniquery_head.hidden_dim}): "
                    "the FPN feeds the head at this width"
                )
            return self

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
        if self.uniquery_head is not None:
            raise ValueError("uniquery_head is set but meta_architecture is not uniquery")
        if self.uniquery_mask is not None:
            raise ValueError("uniquery_mask is only valid with meta_architecture='uniquery'")
        if self.uniquery_keypoint is not None:
            raise ValueError("uniquery_keypoint is only valid with meta_architecture='uniquery'")
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
    # The COMPUTE BUDGET DIAL for fixed-size letterbox inference: the compute
    # budget is ``size_budget ** 2`` pixels. The actual canvas is *derived* — the
    # largest 128-aligned ``(H, W)`` under that budget at the data's native aspect
    # (square for diverse data). Raise/lower it to trade speed for resolution.
    size_budget: Annotated[int, Field(gt=0)] = 640
    # Resolved letterbox canvas ``(H, W)`` — a DEPLOY ARTIFACT, not a user input:
    # training re-resolves it from *this* run's data every time (so fine-tuning a
    # 1:1 base on 16:9 data adapts to a 16:9 canvas) and bakes it into the sidecar
    # so deploy reads the exact shape with no dataset. Any inbound value is
    # overwritten at train. ``None`` → deploy falls back to the largest aligned
    # square in budget. Both dims are FPN-stride (32) multiples. See
    # ``mayaku.tuning.sizing``.
    canvas_hw: tuple[int, int] | None = None
    # How inference/eval resize an image to the network input:
    #   "shortest_edge" — variable resize (ResizeShortestEdge), the legacy path.
    #   "letterbox"     — aspect-preserving resize + pad to the resolved canvas,
    #                     the fixed-size deploy geometry (host un-letterboxes preds).
    # The mayaku-* family uses "letterbox"; kept switchable as the proven fallback.
    resize_mode: Literal["shortest_edge", "letterbox"] = "shortest_edge"
    # Multi-scale letterbox training: the smallest budget fraction (one canvas
    # drawn per iteration, from this fraction up to the full deploy canvas). The
    # deploy canvas is always the top — train geometry == deploy. Derived from the
    # budget; replaces the old explicit ``train_sizes`` list.
    train_scale_min: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
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

    # The FPN samples down to stride 32 (res5) for every in-scope backbone, so
    # the fixed deploy size must tile evenly — else letterbox padding wouldn't be
    # stride-aligned and the static export shape would be off. (32 is hard-coded
    # because the config can't see the model's stride; it holds for all current
    # ResNet/ConvNeXt FPN backbones.)
    @model_validator(mode="after")
    def _check_size_budget_stride(self) -> InputConfig:
        if self.size_budget % 32 != 0:
            raise ValueError(
                f"size_budget must be a multiple of 32 (the FPN max stride); got "
                f"{self.size_budget}. Use e.g. 640, 672, …, 800."
            )
        if self.canvas_hw is not None and any(d % 32 != 0 for d in self.canvas_hw):
            raise ValueError(
                f"canvas_hw dims must each be a multiple of 32 (the FPN max stride); "
                f"got {self.canvas_hw}. Resolve it via mayaku.tuning.snap_max_content "
                "(128-aligned), or use 32-multiples."
            )
        return self


class SolverConfig(_BaseModel):
    """Optimizer and LR schedule.

    Training length is set in epochs (:attr:`num_epochs`) and the LR follows a
    single warmup→cosine decay; the engine resolves epochs to an iteration
    count at train time from the dataset size and effective batch.
    """

    ims_per_batch: Annotated[int, Field(gt=0)] = 16
    # Gradient accumulation: divide effective batch into ``grad_accum_steps``
    # micro-batches of ``ims_per_batch``. Memory scales with the micro-batch
    # only, so this is the standard knob for fitting a large effective batch
    # into a small GPU. Effective batch = ``ims_per_batch * grad_accum_steps``.
    # ``base_lr`` should be tuned against the effective batch, not the micro.
    grad_accum_steps: Annotated[int, Field(ge=1)] = 1
    base_lr: Annotated[float, Field(gt=0.0)] = 0.02
    # Training length is expressed in epochs (passes over the dataset). The
    # engine resolves it to an iteration count at train time from the dataset
    # size and effective batch (``num_images / (ims_per_batch * grad_accum *
    # world_size)`` iters per epoch). The LR follows a single warmup→cosine
    # decay over the full run; ``warmup_fraction`` is the share of total
    # iterations spent warming up from ``warmup_factor * base_lr`` to
    # ``base_lr``. Default 16 epochs is a sane fine-tune length for any dataset;
    # auto-config picks a dataset-adaptive value when enabled.
    num_epochs: Annotated[int, Field(gt=0)] = 16
    warmup_fraction: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.03
    warmup_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0 / 1000.0

    # Optimizer choice. ``"SGD"`` (default) is the D2-replication path
    # and pairs with ``momentum`` / ``nesterov`` below. ``"AdamW"`` is
    # the published-validated path for ConvNeXt / Swin / ViT backbones
    # and pairs with ``betas`` / ``eps``. The unused pair is silently
    # ignored by the optimizer builder, matching torch's own behaviour.
    optimizer_name: Literal["SGD", "AdamW"] = "SGD"
    momentum: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.9
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: Annotated[float, Field(gt=0.0)] = 1.0e-8

    weight_decay: Annotated[float, Field(ge=0.0)] = 1e-4
    weight_decay_norm: Annotated[float, Field(ge=0.0)] = 0.0

    # Layer-wise learning rate decay (LLRD). When enabled, each backbone
    # parameter's LR is scaled by ``llrd_decay ** ((num_layers + 2) - layer_id - 1)``
    # where ``layer_id`` is assigned input→output along the backbone depth
    # and ``num_layers`` is derived from the backbone variant (ConvNeXt-T:
    # 6; ConvNeXt-S/B/L: 12; ResNet: 4). Detector neck/heads (FPN/RPN/ROI)
    # are treated as the top layer and keep ``base_lr``. Default off so
    # all prior runs are reproducible bit-for-bit. The ConvNeXt scheme is
    # MMDet's ``get_layer_id_for_convnext`` (with the stage-2 ``block_id //
    # 3`` bucketing); ResNet is per-stage. ``llrd_decay`` is a recipe
    # sweep knob — no canonical per-variant value exists; see the
    # ``*_llrd.yaml`` recipes for cited starting points.
    llrd_enabled: bool = False
    llrd_decay: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
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

    def effective_batch(self, world_size: int = 1) -> int:
        """Images per optimizer step: ``ims_per_batch * grad_accum_steps``,
        times ``world_size`` for the cross-rank total. ``base_lr`` should be
        tuned against this, and one epoch is ``ceil(num_images / this)`` steps.
        """
        return self.ims_per_batch * self.grad_accum_steps * world_size


class TestConfig(_BaseModel):
    detections_per_image: Annotated[int, Field(gt=0)] = 100
    eval_period: Annotated[int, Field(ge=0)] = 0  # 0 == disabled
    precise_bn_enabled: bool = False
    precise_bn_num_iter: Annotated[int, Field(gt=0)] = 200


class DataLoaderConfig(_BaseModel):
    num_workers: Annotated[int, Field(ge=0)] = 4
    # Batches each worker pre-builds ahead of demand (PyTorch DataLoader
    # ``prefetch_factor``; total buffered samples = num_workers x this).
    # The default of 2 buffers only ~one batch, which starves a fast GPU
    # because AspectRatioGroupedDataset drains ~1.5x batch_size samples per
    # step (two aspect buckets) — every step empties the buffer and the GPU
    # waits while workers rebuild it. Raise it (4-6) for small/fast models
    # on big-image datasets to give the workers runway to stay ahead.
    # Ignored when ``num_workers == 0`` (no worker processes to prefetch).
    prefetch_factor: Annotated[int, Field(ge=1)] = 2
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
    * ``solver.base_lr`` / ``num_epochs`` / ``ema_enabled`` / ``ema_decay`` /
      ``ema_tau`` — from dataset size bucket, anchored to the scratch recipe
      and applying both batch-scaling and the 10× fine-tune drop
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
