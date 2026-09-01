"""Import a Detectron2 checkpoint as a Mayaku model.

    mayaku convert-d2 model_final.pth --d2-config cfg.yaml -o model.pth --class-names wing

Covers Faster / Mask / Keypoint R-CNN on R-50, R-101 and X-101_32x8d FPN, from a
``.pkl`` (model zoo) or a ``.pth`` (your own training run). Needs no ``detectron2``
install. The output is deploy-ready — it carries the embedded sidecar, so::

    from mayaku import from_pretrained
    predictor = from_pretrained("model.pth", device="auto")

**Why ``--d2-config`` is required.** Two settings change predictions without
changing any tensor shape, so ``load_state_dict(strict=True)`` cannot catch them
and hand-writing the config is how people get silently wrong models:

* ``MODEL.PIXEL_STD`` — caffe2-pretrained checkpoints use ``[1, 1, 1]``, not
  Mayaku's ``[58.395, 57.12, 57.375]``. Wrong by ~58x, with no error.
* ``MODEL.RESNETS.STRIDE_IN_1X1`` — caffe2 puts the stride on the 1x1 conv.
  Both layouts have identical parameter shapes.

Reading them from the config is the only reliable fix. ``INPUT.FORMAT`` is read
too, so the BGR->RGB swap is folded into the stem conv's weights (ADR 002) and
there is no runtime flag to get backwards.

Detectron2 is supported here for the community migrating off it; it is not part of
Mayaku's own architecture. Everything lives in this one module; ``mayaku convert-d2``
is a thin wrapper over :func:`convert_d2`.

Safety: a real ``cfg.yaml`` is a *pickled* ``CfgNode``, so ``yaml.unsafe_load`` /
``merge_from_file(allow_unsafe=True)`` would execute arbitrary code from the file.
:func:`load_d2_config` uses a restricted loader that reads those tags as inert
data instead.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

from mayaku.config.schemas import MayakuConfig
from mayaku.tuning.recipe import walk_leaves
from mayaku.utils.checkpoint import build_sidecar, load_checkpoint

__all__ = ["D2ConversionError", "config_from_d2", "convert_d2", "load_d2_config"]


class D2ConversionError(ValueError):
    """A Detectron2 model describes something Mayaku cannot build."""


# ---------------------------------------------------------------------------
# Reading cfg.yaml without executing it
# ---------------------------------------------------------------------------


class _CfgNodeLoader(yaml.SafeLoader):
    """``SafeLoader`` that reads Detectron2's pickled tags as plain data."""


def _inert(loader: yaml.Loader, suffix: str, node: yaml.Node) -> Any:
    """Construct a ``python/object*`` tag as data, never as an object.

    D2 dumps a ``CfgNode`` as ``!!python/object/new:...CfgNode`` with the real
    mapping under ``dictitems`` and yacs bookkeeping under ``state``; tuples
    arrive as ``!!python/tuple``. Returning the payload keeps every value we
    need while the class named in the tag is never imported or instantiated.
    """
    if isinstance(node, yaml.MappingNode):
        mapping = loader.construct_mapping(node, deep=True)
        return mapping.get("dictitems", mapping)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise D2ConversionError(f"unexpected {type(node).__name__} under tag {suffix!r}")


for _tag in (
    "tag:yaml.org,2002:python/object/new:",
    "tag:yaml.org,2002:python/object/apply:",
    "tag:yaml.org,2002:python/object:",
    "tag:yaml.org,2002:python/name:",
    "tag:yaml.org,2002:python/tuple",
):
    _CfgNodeLoader.add_multi_constructor(_tag, _inert)


def load_d2_config(path: str | Path) -> dict[str, Any]:
    """Parse a Detectron2 ``cfg.yaml`` into nested dicts, executing no code."""
    with Path(path).open(encoding="utf-8") as fh:
        loaded = yaml.load(fh, Loader=_CfgNodeLoader)
    if not isinstance(loaded, dict):
        raise D2ConversionError(
            f"{path}: expected a Detectron2 config mapping, got {type(loaded).__name__}"
        )
    return loaded


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted ``MODEL.RESNETS.DEPTH``-style key, or ``default``."""
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# cfg.yaml -> MayakuConfig
# ---------------------------------------------------------------------------

# Settings Mayaku has no field for. Ignoring one would produce a model that
# loads cleanly and predicts differently, so a mismatch is refused instead.
# A key absent from the source config is treated as agreeing.
_MUST_BE: dict[str, Any] = {
    "MODEL.PROPOSAL_GENERATOR.NAME": "RPN",
    "MODEL.RPN.HEAD_NAME": "StandardRPNHead",
    "MODEL.ANCHOR_GENERATOR.NAME": "DefaultAnchorGenerator",
    "MODEL.ROI_HEADS.NAME": "StandardROIHeads",
    # Mayaku's StandardRPNHead is a single shared 3x3 conv; [-1] is D2's
    # "one conv, same width as the input".
    "MODEL.RPN.CONV_DIMS": [-1],
    # ROIAlignV2 (aligned=True) is the only pooler Mayaku implements.
    "MODEL.ROI_BOX_HEAD.POOLER_TYPE": "ROIAlignV2",
    "MODEL.ROI_MASK_HEAD.POOLER_TYPE": "ROIAlignV2",
    "MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE": "ROIAlignV2",
    # Mayaku's box/mask heads have no `norm` field; "" is D2's no-norm default.
    "MODEL.ROI_BOX_HEAD.NORM": "",
    "MODEL.ROI_MASK_HEAD.NORM": "",
    # Mayaku's ResNet is torchvision's; res2 width is fixed at 256.
    "MODEL.RESNETS.RES2_OUT_CHANNELS": 256,
    # Alternative classification losses change the scores inference reads.
    "MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE": False,
    "MODEL.ROI_BOX_HEAD.USE_FED_LOSS": False,
}


def _backbone_name(cfg: dict[str, Any]) -> str:
    """Map ``RESNETS.{DEPTH,NUM_GROUPS,WIDTH_PER_GROUP}`` to a Mayaku backbone."""
    depth = _get(cfg, "MODEL.RESNETS.DEPTH", 50)
    groups = _get(cfg, "MODEL.RESNETS.NUM_GROUPS", 1)
    width = _get(cfg, "MODEL.RESNETS.WIDTH_PER_GROUP", 64)
    if (groups, width) == (32, 8):
        if depth != 101:
            raise D2ConversionError(f"ResNeXt depth {depth}; Mayaku ships resnext101_32x8d")
        return "resnext101_32x8d"
    if (groups, width) != (1, 64):
        raise D2ConversionError(
            f"MODEL.RESNETS.NUM_GROUPS={groups} / WIDTH_PER_GROUP={width} is not supported"
        )
    if depth not in (50, 101):
        raise D2ConversionError(f"MODEL.RESNETS.DEPTH={depth}; Mayaku ships 50 and 101")
    return f"resnet{depth}"


def config_from_d2(d2: dict[str, Any]) -> tuple[MayakuConfig, str]:
    """Build ``(config, input_format)`` from a parsed Detectron2 config.

    ``input_format`` is the source checkpoint's channel order, which decides
    whether the stem conv's input channels get reversed.
    """
    # Structural: is this a Detectron2 R-CNN config at all? Absence is fatal.
    for key, required in (
        ("MODEL.META_ARCHITECTURE", "GeneralizedRCNN"),
        ("MODEL.BACKBONE.NAME", "build_resnet_fpn_backbone"),
    ):
        if _get(d2, key) != required:
            raise D2ConversionError(f"{key}={_get(d2, key)!r}; Mayaku needs {required!r}")
    for key, required in _MUST_BE.items():
        got = _get(d2, key, required)
        if got != required:
            raise D2ConversionError(f"{key}={got!r} is not supported (Mayaku needs {required!r})")
    if any(_get(d2, "MODEL.RESNETS.DEFORM_ON_PER_STAGE", []) or []):
        raise D2ConversionError("deformable convolution is not implemented in Mayaku (ADR 001)")

    mask_on = bool(_get(d2, "MODEL.MASK_ON", False))
    keypoint_on = bool(_get(d2, "MODEL.KEYPOINT_ON", False))
    if mask_on and keypoint_on:
        raise D2ConversionError("MASK_ON and KEYPOINT_ON together has no Mayaku meta_architecture")
    meta = "keypoint_rcnn" if keypoint_on else "mask_rcnn" if mask_on else "faster_rcnn"

    # D2 lists PIXEL_MEAN/STD in INPUT.FORMAT order; Mayaku always stores RGB.
    input_format = str(_get(d2, "INPUT.FORMAT", "BGR")).upper()
    if input_format not in ("BGR", "RGB"):
        raise D2ConversionError(f"INPUT.FORMAT={input_format!r} is not BGR or RGB")
    mean = list(_get(d2, "MODEL.PIXEL_MEAN", [103.53, 116.28, 123.675]))
    std = list(_get(d2, "MODEL.PIXEL_STD", [1.0, 1.0, 1.0]))
    if input_format == "BGR":
        mean, std = mean[::-1], std[::-1]

    model: dict[str, Any] = {
        "meta_architecture": meta,
        "mask_on": mask_on,
        "keypoint_on": keypoint_on,
        "pixel_mean": mean,
        "pixel_std": std,
        "backbone": {
            "name": _backbone_name(d2),
            "norm": _get(d2, "MODEL.RESNETS.NORM", "FrozenBN"),
            "freeze_at": _get(d2, "MODEL.BACKBONE.FREEZE_AT", 2),
            "stem_out_channels": _get(d2, "MODEL.RESNETS.STEM_OUT_CHANNELS", 64),
            "res5_dilation": _get(d2, "MODEL.RESNETS.RES5_DILATION", 1),
            "stride_in_1x1": bool(_get(d2, "MODEL.RESNETS.STRIDE_IN_1X1", False)),
        },
        "fpn": {
            "in_features": tuple(
                _get(d2, "MODEL.FPN.IN_FEATURES", ("res2", "res3", "res4", "res5"))
            ),
            "out_channels": _get(d2, "MODEL.FPN.OUT_CHANNELS", 256),
            "norm": _get(d2, "MODEL.FPN.NORM", ""),
            "fuse_type": _get(d2, "MODEL.FPN.FUSE_TYPE", "sum"),
        },
        "anchor_generator": {
            "sizes": tuple(tuple(s) for s in _get(d2, "MODEL.ANCHOR_GENERATOR.SIZES")),
            "aspect_ratios": tuple(
                tuple(a) for a in _get(d2, "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS")
            ),
            "offset": _get(d2, "MODEL.ANCHOR_GENERATOR.OFFSET", 0.0),
        },
        "rpn": {
            "in_features": tuple(_get(d2, "MODEL.RPN.IN_FEATURES")),
            "pre_nms_topk_train": _get(d2, "MODEL.RPN.PRE_NMS_TOPK_TRAIN", 2000),
            "pre_nms_topk_test": _get(d2, "MODEL.RPN.PRE_NMS_TOPK_TEST", 1000),
            "post_nms_topk_train": _get(d2, "MODEL.RPN.POST_NMS_TOPK_TRAIN", 1000),
            "post_nms_topk_test": _get(d2, "MODEL.RPN.POST_NMS_TOPK_TEST", 1000),
            "nms_thresh": _get(d2, "MODEL.RPN.NMS_THRESH", 0.7),
            "iou_thresholds": tuple(_get(d2, "MODEL.RPN.IOU_THRESHOLDS", (0.3, 0.7))),
            "iou_labels": tuple(_get(d2, "MODEL.RPN.IOU_LABELS", (0, -1, 1))),
            "batch_size_per_image": _get(d2, "MODEL.RPN.BATCH_SIZE_PER_IMAGE", 256),
            "positive_fraction": _get(d2, "MODEL.RPN.POSITIVE_FRACTION", 0.5),
            "bbox_reg_weights": tuple(_get(d2, "MODEL.RPN.BBOX_REG_WEIGHTS", (1.0,) * 4)),
            "smooth_l1_beta": _get(d2, "MODEL.RPN.SMOOTH_L1_BETA", 0.0),
            "loss_weight": _get(d2, "MODEL.RPN.LOSS_WEIGHT", 1.0),
            "box_reg_loss_type": _get(d2, "MODEL.RPN.BBOX_REG_LOSS_TYPE", "smooth_l1"),
        },
        "roi_heads": {
            "in_features": tuple(_get(d2, "MODEL.ROI_HEADS.IN_FEATURES")),
            "num_classes": _get(d2, "MODEL.ROI_HEADS.NUM_CLASSES"),
            "batch_size_per_image": _get(d2, "MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE", 512),
            "positive_fraction": _get(d2, "MODEL.ROI_HEADS.POSITIVE_FRACTION", 0.25),
            "iou_thresholds": tuple(_get(d2, "MODEL.ROI_HEADS.IOU_THRESHOLDS", (0.5,))),
            "iou_labels": tuple(_get(d2, "MODEL.ROI_HEADS.IOU_LABELS", (0, 1))),
            "score_thresh_test": _get(d2, "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.05),
            "nms_thresh_test": _get(d2, "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.5),
            "proposal_append_gt": bool(_get(d2, "MODEL.ROI_HEADS.PROPOSAL_APPEND_GT", True)),
        },
        "roi_box_head": {
            "pooler_resolution": _get(d2, "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION", 7),
            "pooler_sampling_ratio": _get(d2, "MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO", 0),
            "num_conv": _get(d2, "MODEL.ROI_BOX_HEAD.NUM_CONV", 0),
            "conv_dim": _get(d2, "MODEL.ROI_BOX_HEAD.CONV_DIM", 256),
            "num_fc": _get(d2, "MODEL.ROI_BOX_HEAD.NUM_FC", 2),
            "fc_dim": _get(d2, "MODEL.ROI_BOX_HEAD.FC_DIM", 1024),
            "bbox_reg_weights": tuple(
                _get(d2, "MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS", (10.0, 10.0, 5.0, 5.0))
            ),
            "smooth_l1_beta": _get(d2, "MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA", 0.0),
            "box_reg_loss_type": _get(d2, "MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE", "smooth_l1"),
            "cls_agnostic_bbox_reg": bool(
                _get(d2, "MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG", False)
            ),
        },
    }
    if mask_on:
        model["roi_mask_head"] = {
            "pooler_resolution": _get(d2, "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION", 14),
            "pooler_sampling_ratio": _get(d2, "MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO", 0),
            "num_conv": _get(d2, "MODEL.ROI_MASK_HEAD.NUM_CONV", 4),
            "conv_dim": _get(d2, "MODEL.ROI_MASK_HEAD.CONV_DIM", 256),
            "cls_agnostic_mask": bool(_get(d2, "MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK", False)),
        }
    if keypoint_on:
        # D2 keeps keypoint_flip_map in MetadataCatalog, not the config, so
        # flip_indices cannot be recovered. Inference never uses it; a later
        # fine-tune with horizontal flip does.
        model["roi_keypoint_head"] = {
            "pooler_resolution": _get(d2, "MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION", 14),
            "pooler_sampling_ratio": _get(d2, "MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO", 0),
            "conv_dims": tuple(_get(d2, "MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS", (512,) * 8)),
            "num_keypoints": _get(d2, "MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS", 17),
            "min_keypoints_per_image": _get(
                d2, "MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE", 1
            ),
            "normalize_loss_by_visible_keypoints": bool(
                _get(d2, "MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS", True)
            ),
            "loss_weight": _get(d2, "MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT", 1.0),
        }

    min_train = _get(d2, "INPUT.MIN_SIZE_TRAIN", (640, 672, 704, 736, 768, 800))
    config = MayakuConfig.model_validate(
        {
            "model": model,
            "input": {
                # Keep D2's variable shortest-edge geometry: letterbox is
                # Mayaku's own deploy convention and would change the resolution
                # this model was trained to see.
                "resize_mode": "shortest_edge",
                "min_size_train": tuple(min_train) if min_train else (800,),
                "max_size_train": _get(d2, "INPUT.MAX_SIZE_TRAIN", 1333),
                "min_size_train_sampling": _get(d2, "INPUT.MIN_SIZE_TRAIN_SAMPLING", "choice"),
                "min_size_test": _get(d2, "INPUT.MIN_SIZE_TEST", 800),
                "max_size_test": _get(d2, "INPUT.MAX_SIZE_TEST", 1333),
                "mask_format": _get(d2, "INPUT.MASK_FORMAT", "polygon"),
                "random_flip": _get(d2, "INPUT.RANDOM_FLIP", "horizontal"),
            },
            "test": {"detections_per_image": _get(d2, "TEST.DETECTIONS_PER_IMAGE", 100)},
            # A converted checkpoint is a finished model; leave Mayaku's
            # dataset-driven auto-tuning out unless the user opts back in.
            "auto_config": {"enabled": False},
        }
    )
    return config, input_format


# ---------------------------------------------------------------------------
# Weights: D2 module names -> Mayaku module names
# ---------------------------------------------------------------------------

_Replacement = str | Callable[[re.Match[str]], str]

_STEM_CONV = "backbone.bottom_up.stem.0.weight"

# Provenance D2 writes beside the weights; carries no learned state.
_METADATA_KEYS = ("__author__", "matching_heuristics")

# Buffers Mayaku recomputes (cell_anchors) or registers persistent=False.
_DROP = re.compile(
    r"^(proposal_generator\.anchor_generator\.cell_anchors\.\d+|pixel_mean|pixel_std)$"
)

# First match wins. D2's ResNet is its own; Mayaku uses torchvision's, and each
# head flattens its conv stack differently. Nothing is reshaped, only re-keyed.
_RENAME: tuple[tuple[re.Pattern[str], _Replacement], ...] = (
    (re.compile(r"^backbone\.bottom_up\.stem\.conv1\.weight$"), _STEM_CONV),
    (
        re.compile(r"^backbone\.bottom_up\.stem\.conv1\.norm\.(\w+)$"),
        r"backbone.bottom_up.stem.1.\1",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.conv([1-3])\.weight$"),
        r"backbone.bottom_up.\1.\2.conv\3.weight",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.conv([1-3])\.norm\.(\w+)$"),
        r"backbone.bottom_up.\1.\2.bn\3.\4",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.shortcut\.weight$"),
        r"backbone.bottom_up.\1.\2.downsample.0.weight",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res[2-5])\.(\d+)\.shortcut\.norm\.(\w+)$"),
        r"backbone.bottom_up.\1.\2.downsample.1.\3",
    ),
    (
        re.compile(r"^backbone\.fpn_lateral([2-5])\.(weight|bias)$"),
        lambda m: f"backbone.lateral_convs.{int(m.group(1)) - 2}.{m.group(2)}",
    ),
    (
        re.compile(r"^backbone\.fpn_output([2-5])\.(weight|bias)$"),
        lambda m: f"backbone.output_convs.{int(m.group(1)) - 2}.{m.group(2)}",
    ),
    (
        re.compile(r"^proposal_generator\.rpn_head\.(\w+)\.(weight|bias)$"),
        r"rpn.head.\1.\2",
    ),
    (
        re.compile(r"^roi_heads\.box_head\.fc(\d+)\.(weight|bias)$"),
        lambda m: f"roi_heads.box_head.fcs.{int(m.group(1)) - 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.box_head\.conv(\d+)\.(weight|bias)$"),
        lambda m: f"roi_heads.box_head.convs.{int(m.group(1)) - 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.box_predictor\.(cls_score|bbox_pred)\.(weight|bias)$"),
        r"roi_heads.box_predictor.\1.\2",
    ),
    (
        re.compile(r"^roi_heads\.mask_head\.mask_fcn(\d+)\.(weight|bias)$"),
        lambda m: f"roi_heads.mask_head.convs.{int(m.group(1)) - 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.mask_head\.(deconv|predictor)\.(weight|bias)$"),
        r"roi_heads.mask_head.\1.\2",
    ),
    (
        re.compile(r"^roi_heads\.keypoint_head\.conv_fcn(\d+)\.(weight|bias)$"),
        lambda m: f"roi_heads.keypoint_head.convs.{int(m.group(1)) - 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.keypoint_head\.score_lowres\.(weight|bias)$"),
        r"roi_heads.keypoint_head.deconv.\1",
    ),
)


def _load_d2_state(path: Path) -> dict[str, Tensor]:
    """Load a D2 ``.pkl`` (caffe2 numpy) or ``.pth`` (torch) checkpoint.

    ``.pkl`` runs arbitrary code on ``pickle.load``, so only the exact shape
    Detectron2 emits is accepted. That is a structural check, not a substitute
    for trusting the source: only load ``.pkl`` files from the official MODEL_ZOO.
    """
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as fh:
            obj = pickle.load(fh, encoding="latin1")
        if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
            raise D2ConversionError(
                f"refusing to load {path}: not a Detectron2 .pkl "
                "(expected a dict with a dict under 'model')"
            )
        state = {}
        for key, value in obj["model"].items():
            if key in _METADATA_KEYS:
                continue
            if not isinstance(value, np.ndarray):
                raise D2ConversionError(
                    f"refusing to load {path}: model[{key!r}] is {type(value).__name__}, "
                    "not a numpy.ndarray"
                )
            state[key] = torch.from_numpy(value)
        return state

    _sidecar, loaded = load_checkpoint(path)
    if not isinstance(loaded, dict):
        raise D2ConversionError(f"{path}: expected a state_dict, got {type(loaded).__name__}")
    return {k: v for k, v in loaded.items() if k not in _METADATA_KEYS}


def _remap(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Re-key a D2 state_dict onto Mayaku module names."""
    out: dict[str, Tensor] = {}
    unknown: list[str] = []
    for key, value in state.items():
        if _DROP.match(key):
            continue
        for pattern, replacement in _RENAME:
            match = pattern.match(key)
            if match is not None:
                new = replacement(match) if callable(replacement) else pattern.sub(replacement, key)
                out[new] = value
                break
        else:
            unknown.append(key)
    if unknown:
        raise D2ConversionError(
            f"{len(unknown)} checkpoint key(s) have no rename rule, so the conversion "
            f"would be incomplete. First few: {unknown[:5]}"
        )
    return out


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert_d2(
    weights: Path | str,
    *,
    d2_config: Path | str,
    output: Path | str,
    class_names: Sequence[str] | None = None,
    verbose: bool = False,
) -> MayakuConfig:
    """Convert a Detectron2 checkpoint; write ``output`` and return its config.

    ``verbose`` prints a human-readable report of what the config carried over;
    the CLI turns it on, library callers get a quiet function.

    ``class_names`` are ordered class names — Detectron2 keeps them in
    ``MetadataCatalog``, not the config, so they cannot be recovered from
    ``cfg.yaml``. ``class_0…`` is used when omitted; the model still runs
    correctly either way.
    """
    from mayaku.cli._factory import build_detector

    weights, d2_config, output = Path(weights), Path(d2_config), Path(output)
    config, input_format = config_from_d2(load_d2_config(d2_config))

    num_classes = config.model.roi_heads.num_classes
    names = list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]
    if len(names) != num_classes:
        raise D2ConversionError(
            f"got {len(names)} class name(s) but MODEL.ROI_HEADS.NUM_CLASSES={num_classes}"
        )

    raw = _load_d2_state(weights)
    state = _remap(raw)
    if input_format == "BGR":
        # Fold the channel swap into the weights so nothing downstream needs a
        # flag — Mayaku is RGB-native (ADR 002).
        weight = state[_STEM_CONV]
        state[_STEM_CONV] = weight[:, [2, 1, 0], :, :]

    # Gate: the weights must fit the detector this config builds, exactly.
    # Built on the meta device — the check is about names and shapes, so
    # allocating (and randomly initialising) a second full copy of the
    # parameters just to overwrite and discard them is pure waste.
    with torch.device("meta"):
        skeleton = build_detector(config)
    skeleton.load_state_dict(state, assign=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "mayaku": build_sidecar(
                config,
                names,
                {
                    "source": "detectron2",
                    "d2_config": str(d2_config),
                    "d2_weights": weights.name,
                    "d2_input_format": input_format,
                },
            ),
        },
        output,
    )
    if verbose:
        defaults = dict(walk_leaves(MayakuConfig().model_dump(mode="json")))
        print(f"Converted {weights.name} -> {output} ({output.stat().st_size / 1e6:.1f} MB)")
        print(f"  architecture : {config.model.meta_architecture} / {config.model.backbone.name}")
        print(f"  classes      : {names}")
        print(f"  tensors      : {len(state)} loaded strict, {len(raw) - len(state)} dropped")
        if input_format == "BGR":
            print("  channels     : BGR source -> stem conv reversed to RGB")
        print("  from cfg.yaml (differs from Mayaku defaults):")
        for key, value in walk_leaves(config.model_dump(mode="json")):
            if defaults.get(key) != value:
                print(f"      {key:46s} {defaults.get(key)!r} -> {value!r}")
    return config
