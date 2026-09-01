"""Tests for the Detectron2 import path (``mayaku convert-d2``).

The end-to-end tests build a Mayaku detector, push its weights *backwards*
through an independently written D2 naming map, and convert them back. A
successful round trip proves the rename table is right in both directions, and
comparing predictions against the original module proves nothing was lost.

Covers all three R-CNN meta-architectures: detection, segmentation, keypoints.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mayaku.cli._factory import build_detector
from mayaku.config.schemas import MayakuConfig
from mayaku.d2 import D2ConversionError, config_from_d2, convert_d2, load_d2_config
from mayaku.inference import from_pretrained
from mayaku.utils.checkpoint import load_checkpoint

# ---------------------------------------------------------------------------
# Fixtures: a minimal but complete Detectron2 config
# ---------------------------------------------------------------------------


def d2_config(
    *,
    mask_on: bool = False,
    keypoint_on: bool = False,
    depth: int = 50,
    input_format: str = "BGR",
    **overrides: Any,
) -> dict[str, Any]:
    """A Detectron2 config dict, shaped like a real ``cfg.yaml``.

    Values match D2's own defaults so the tests exercise the same paths a real
    checkpoint takes; ``overrides`` are dotted keys applied on top.
    """
    cfg: dict[str, Any] = {
        "MODEL": {
            "META_ARCHITECTURE": "GeneralizedRCNN",
            "MASK_ON": mask_on,
            "KEYPOINT_ON": keypoint_on,
            # D2 lists these in INPUT.FORMAT order — BGR here.
            "PIXEL_MEAN": [103.53, 116.28, 123.675],
            "PIXEL_STD": [1.0, 1.0, 1.0],
            "BACKBONE": {"NAME": "build_resnet_fpn_backbone", "FREEZE_AT": 2},
            "RESNETS": {
                "DEPTH": depth,
                "NORM": "FrozenBN",
                "NUM_GROUPS": 1,
                "WIDTH_PER_GROUP": 64,
                "STRIDE_IN_1X1": True,
                "STEM_OUT_CHANNELS": 64,
                "RES2_OUT_CHANNELS": 256,
                "RES5_DILATION": 1,
                "DEFORM_ON_PER_STAGE": [False, False, False, False],
            },
            "FPN": {
                "IN_FEATURES": ["res2", "res3", "res4", "res5"],
                "OUT_CHANNELS": 256,
                "NORM": "",
                "FUSE_TYPE": "sum",
            },
            "ANCHOR_GENERATOR": {
                "NAME": "DefaultAnchorGenerator",
                "SIZES": [[32], [64], [128], [256], [512]],
                "ASPECT_RATIOS": [[0.5, 1.0, 2.0]],
                "OFFSET": 0.0,
            },
            "PROPOSAL_GENERATOR": {"NAME": "RPN"},
            "RPN": {
                "HEAD_NAME": "StandardRPNHead",
                "IN_FEATURES": ["p2", "p3", "p4", "p5", "p6"],
                "CONV_DIMS": [-1],
                "PRE_NMS_TOPK_TEST": 1000,
                "POST_NMS_TOPK_TEST": 1000,
                "NMS_THRESH": 0.7,
                "BBOX_REG_WEIGHTS": [1.0, 1.0, 1.0, 1.0],
            },
            "ROI_HEADS": {
                "NAME": "StandardROIHeads",
                "IN_FEATURES": ["p2", "p3", "p4", "p5"],
                "NUM_CLASSES": 3,
                "SCORE_THRESH_TEST": 0.05,
                "NMS_THRESH_TEST": 0.5,
            },
            "ROI_BOX_HEAD": {
                "POOLER_TYPE": "ROIAlignV2",
                "POOLER_RESOLUTION": 7,
                "POOLER_SAMPLING_RATIO": 0,
                "NUM_CONV": 0,
                "NUM_FC": 2,
                "FC_DIM": 1024,
                "CONV_DIM": 256,
                "NORM": "",
                "BBOX_REG_WEIGHTS": [10.0, 10.0, 5.0, 5.0],
                "USE_SIGMOID_CE": False,
                "USE_FED_LOSS": False,
            },
            "ROI_MASK_HEAD": {
                "POOLER_TYPE": "ROIAlignV2",
                "POOLER_RESOLUTION": 14,
                "POOLER_SAMPLING_RATIO": 0,
                "NUM_CONV": 4,
                "CONV_DIM": 256,
                "NORM": "",
                "CLS_AGNOSTIC_MASK": False,
            },
            "ROI_KEYPOINT_HEAD": {
                "POOLER_TYPE": "ROIAlignV2",
                "POOLER_RESOLUTION": 14,
                "POOLER_SAMPLING_RATIO": 0,
                "CONV_DIMS": [512] * 8,
                "NUM_KEYPOINTS": 5,
            },
        },
        "INPUT": {
            "FORMAT": input_format,
            "MIN_SIZE_TEST": 800,
            "MAX_SIZE_TEST": 1333,
            "MASK_FORMAT": "polygon",
        },
        "TEST": {"DETECTIONS_PER_IMAGE": 100},
    }
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return cfg


def write_pickled_cfg(path: Path, cfg: dict[str, Any]) -> Path:
    """Write ``cfg`` the way Detectron2 does — as a *pickled* ``CfgNode``.

    Real ``cfg.yaml`` files carry ``!!python/object/new:`` tags that
    ``yaml.safe_load`` refuses and ``yaml.unsafe_load`` would execute. Writing
    the fixture in that exact form is what makes the loader test meaningful.
    """
    tag = "!!python/object/new:detectron2.config.config.CfgNode"

    def emit(node: Any, indent: str) -> str:
        if isinstance(node, dict):
            out = f" {tag}\n{indent}dictitems:\n"
            for key, value in node.items():
                if isinstance(value, dict):
                    out += f"{indent}  {key}:{emit(value, indent + '    ')}"
                else:
                    out += f"{indent}  {key}: {value!r}\n"
            # yacs bookkeeping D2 also dumps; the loader must ignore it.
            out += f"{indent}state:\n{indent}  __immutable__: false\n"
            return out
        return f" {node!r}\n"

    path.write_text("--- " + emit(cfg, "").lstrip() + "\n")
    return path


# ---------------------------------------------------------------------------
# The D2 naming map, written backwards and independently of the library table
# ---------------------------------------------------------------------------

_TO_D2: tuple[tuple[re.Pattern[str], Any], ...] = (
    (re.compile(r"^backbone\.bottom_up\.stem\.0\.weight$"), "backbone.bottom_up.stem.conv1.weight"),
    (
        re.compile(r"^backbone\.bottom_up\.stem\.1\.(\w+)$"),
        r"backbone.bottom_up.stem.conv1.norm.\1",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res\d)\.(\d+)\.bn(\d)\.(\w+)$"),
        r"backbone.bottom_up.\1.\2.conv\3.norm.\4",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res\d)\.(\d+)\.downsample\.0\.weight$"),
        r"backbone.bottom_up.\1.\2.shortcut.weight",
    ),
    (
        re.compile(r"^backbone\.bottom_up\.(res\d)\.(\d+)\.downsample\.1\.(\w+)$"),
        r"backbone.bottom_up.\1.\2.shortcut.norm.\3",
    ),
    (
        re.compile(r"^backbone\.lateral_convs\.(\d)\.(\w+)$"),
        lambda m: f"backbone.fpn_lateral{int(m.group(1)) + 2}.{m.group(2)}",
    ),
    (
        re.compile(r"^backbone\.output_convs\.(\d)\.(\w+)$"),
        lambda m: f"backbone.fpn_output{int(m.group(1)) + 2}.{m.group(2)}",
    ),
    (re.compile(r"^rpn\.head\.(\w+)\.(\w+)$"), r"proposal_generator.rpn_head.\1.\2"),
    (
        re.compile(r"^roi_heads\.box_head\.fcs\.(\d+)\.(\w+)$"),
        lambda m: f"roi_heads.box_head.fc{int(m.group(1)) + 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.box_head\.convs\.(\d+)\.(\w+)$"),
        lambda m: f"roi_heads.box_head.conv{int(m.group(1)) + 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.mask_head\.convs\.(\d+)\.(\w+)$"),
        lambda m: f"roi_heads.mask_head.mask_fcn{int(m.group(1)) + 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.keypoint_head\.convs\.(\d+)\.(\w+)$"),
        lambda m: f"roi_heads.keypoint_head.conv_fcn{int(m.group(1)) + 1}.{m.group(2)}",
    ),
    (
        re.compile(r"^roi_heads\.keypoint_head\.deconv\.(\w+)$"),
        r"roi_heads.keypoint_head.score_lowres.\1",
    ),
)


def to_d2_key(key: str) -> str:
    """Mayaku module name -> the Detectron2 name for the same tensor."""
    for pattern, replacement in _TO_D2:
        match = pattern.match(key)
        if match is None:
            continue
        return replacement(match) if callable(replacement) else pattern.sub(replacement, key)
    return key  # conv/bbox_pred/cls_score/mask deconv+predictor names already agree


ARCHS = ("faster_rcnn", "mask_rcnn", "keypoint_rcnn")


def build_pair(arch: str, input_format: str = "BGR") -> tuple[MayakuConfig, dict[str, Any]]:
    """A ``(mayaku_config, d2_config)`` pair describing the same detector."""
    d2 = d2_config(
        mask_on=arch == "mask_rcnn",
        keypoint_on=arch == "keypoint_rcnn",
        input_format=input_format,
    )
    config, _ = config_from_d2(d2)
    return config, d2


def d2_state_from(model: torch.nn.Module, *, is_bgr: bool) -> dict[str, torch.Tensor]:
    """A D2-named ``state_dict`` for ``model``, as a real D2 checkpoint would hold it."""
    state = {to_d2_key(k): v.clone() for k, v in model.state_dict().items()}
    if is_bgr:
        # Undo the RGB->BGR the converter will re-apply, so the round trip lands
        # back on the original weights.
        stem = "backbone.bottom_up.stem.conv1.weight"
        state[stem] = state[stem][:, [2, 1, 0], :, :].clone()
    return state


# ---------------------------------------------------------------------------
# Loading a cfg.yaml
# ---------------------------------------------------------------------------


def test_load_d2_config_reads_pickled_cfgnode_without_executing_it(tmp_path: Path) -> None:
    """The pickled ``CfgNode`` tags parse as plain data — no code runs."""
    path = write_pickled_cfg(tmp_path / "cfg.yaml", d2_config(keypoint_on=True))
    loaded = load_d2_config(path)

    assert loaded["MODEL"]["RESNETS"]["STRIDE_IN_1X1"] is True
    assert loaded["MODEL"]["ROI_KEYPOINT_HEAD"]["NUM_KEYPOINTS"] == 5
    assert loaded["INPUT"]["FORMAT"] == "BGR"
    # `state` is yacs bookkeeping, not config; `dictitems` is what we keep.
    assert "dictitems" not in loaded["MODEL"]


def test_load_d2_config_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(D2ConversionError, match="expected a Detectron2 config mapping"):
        load_d2_config(path)


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "mask_on", "keypoint_on"),
    [("faster_rcnn", False, False), ("mask_rcnn", True, False), ("keypoint_rcnn", False, True)],
)
def test_meta_architecture_and_heads(arch: str, mask_on: bool, keypoint_on: bool) -> None:
    config, _ = config_from_d2(d2_config(mask_on=mask_on, keypoint_on=keypoint_on))

    assert config.model.meta_architecture == arch
    assert config.model.mask_on is mask_on
    assert config.model.keypoint_on is keypoint_on
    assert (config.model.roi_mask_head is not None) is mask_on
    assert (config.model.roi_keypoint_head is not None) is keypoint_on


def test_carries_the_settings_a_strict_load_cannot_catch() -> None:
    """``pixel_std`` and ``stride_in_1x1`` leave tensor shapes untouched.

    They are the two ways a hand-written config produces a model that loads
    cleanly and predicts wrongly, so they get their own test.
    """
    config, _ = config_from_d2(d2_config())

    assert config.model.pixel_std == (1.0, 1.0, 1.0)  # caffe2, not Mayaku's default
    assert config.model.backbone.stride_in_1x1 is True
    assert MayakuConfig().model.pixel_std != config.model.pixel_std
    assert MayakuConfig().model.backbone.stride_in_1x1 != config.model.backbone.stride_in_1x1


@pytest.mark.parametrize(
    ("input_format", "expected_mean"),
    [("BGR", (123.675, 116.28, 103.53)), ("RGB", (103.53, 116.28, 123.675))],
)
def test_pixel_mean_is_reordered_to_rgb(
    input_format: str, expected_mean: tuple[float, ...]
) -> None:
    """D2 lists PIXEL_MEAN in INPUT.FORMAT order; Mayaku always stores RGB."""
    config, reported = config_from_d2(d2_config(input_format=input_format))

    assert reported == input_format
    assert config.model.pixel_mean == pytest.approx(expected_mean)


@pytest.mark.parametrize(
    ("depth", "groups", "width", "expected"),
    [(50, 1, 64, "resnet50"), (101, 1, 64, "resnet101"), (101, 32, 8, "resnext101_32x8d")],
)
def test_backbone_name(depth: int, groups: int, width: int, expected: str) -> None:
    config, _ = config_from_d2(
        d2_config(
            depth=depth,
            **{"MODEL.RESNETS.NUM_GROUPS": groups, "MODEL.RESNETS.WIDTH_PER_GROUP": width},
        )
    )
    assert config.model.backbone.name == expected


def test_inference_affecting_values_are_read_not_defaulted() -> None:
    """Box-decode weights and test thresholds come from the file, not defaults."""
    config, _ = config_from_d2(
        d2_config(
            **{
                "MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS": [5.0, 5.0, 2.5, 2.5],
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.3,
                "MODEL.RPN.NMS_THRESH": 0.6,
                "TEST.DETECTIONS_PER_IMAGE": 42,
            }
        )
    )
    assert config.model.roi_box_head.bbox_reg_weights == (5.0, 5.0, 2.5, 2.5)
    assert config.model.roi_heads.score_thresh_test == pytest.approx(0.3)
    assert config.model.rpn.nms_thresh == pytest.approx(0.6)
    assert config.test.detections_per_image == 42


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"MODEL.META_ARCHITECTURE": "RetinaNet"}, "META_ARCHITECTURE"),
        ({"MODEL.RESNETS.DEFORM_ON_PER_STAGE": [False, False, True, True]}, "deformable"),
        ({"MODEL.RESNETS.DEPTH": 152}, "DEPTH"),
        ({"MODEL.ROI_HEADS.NAME": "CascadeROIHeads"}, "ROI_HEADS.NAME"),
        ({"MODEL.ROI_BOX_HEAD.NORM": "GN"}, "ROI_BOX_HEAD.NORM"),
        ({"MODEL.ROI_BOX_HEAD.POOLER_TYPE": "ROIAlign"}, "POOLER_TYPE"),
        ({"MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE": True}, "USE_SIGMOID_CE"),
        ({"MODEL.RPN.CONV_DIMS": [256, 256]}, "CONV_DIMS"),
        ({"MODEL.BACKBONE.NAME": "build_retinanet_resnet_fpn_backbone"}, "BACKBONE.NAME"),
        ({"INPUT.FORMAT": "YUV"}, "INPUT.FORMAT"),
    ],
)
def test_rejects_what_mayaku_cannot_reproduce(overrides: dict[str, Any], match: str) -> None:
    """Refuse rather than convert to a near-equivalent that predicts differently."""
    with pytest.raises(D2ConversionError, match=match):
        config_from_d2(d2_config(**overrides))


def test_rejects_mask_and_keypoint_together() -> None:
    with pytest.raises(D2ConversionError, match="no Mayaku meta_architecture"):
        config_from_d2(d2_config(mask_on=True, keypoint_on=True))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_round_trip_reproduces_the_source_model(arch: str, tmp_path: Path) -> None:
    """D2-named weights -> convert -> a checkpoint that predicts identically.

    The strongest check available without a real D2 install: every tensor must
    land on the module it came from, or the outputs diverge.
    """
    config, d2 = build_pair(arch)
    torch.manual_seed(0)
    source = build_detector(config).eval()

    weights = tmp_path / "d2_model.pth"
    torch.save({"model": d2_state_from(source, is_bgr=True)}, weights)
    out = convert_d2(
        weights,
        d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
        output=tmp_path / "converted.pth",
        class_names=["a", "b", "c"],
    )
    assert out.model.meta_architecture == arch

    converted = build_detector(out)
    converted.load_state_dict(load_checkpoint(tmp_path / "converted.pth")[1])
    converted.eval()

    image = torch.rand(3, 64, 64) * 255
    batch = [{"image": image, "height": 64, "width": 64}]
    with torch.no_grad():
        expected = source(batch)[0]["instances"]
        actual = converted(batch)[0]["instances"]

    assert len(expected) == len(actual)
    torch.testing.assert_close(expected.pred_boxes.tensor, actual.pred_boxes.tensor)
    torch.testing.assert_close(expected.scores, actual.scores)


@pytest.mark.parametrize("arch", ARCHS)
def test_converted_checkpoint_is_deploy_ready(arch: str, tmp_path: Path) -> None:
    """``from_pretrained`` runs the output with no config file and no extra step."""
    config, d2 = build_pair(arch)
    torch.manual_seed(0)
    source = build_detector(config).eval()

    torch.save({"model": d2_state_from(source, is_bgr=True)}, tmp_path / "d2_model.pth")
    convert_d2(
        tmp_path / "d2_model.pth",
        d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
        output=tmp_path / "converted.pth",
        class_names=["a", "b", "c"],
    )

    predictor = from_pretrained(tmp_path / "converted.pth", device="cpu")
    assert predictor.class_names == ["a", "b", "c"]

    instances = predictor(np.zeros((64, 64, 3), dtype=np.uint8))
    assert instances.has("pred_boxes")
    assert instances.has("pred_masks") == (arch == "mask_rcnn")
    assert instances.has("pred_keypoints") == (arch == "keypoint_rcnn")


def test_rgb_source_leaves_the_stem_conv_alone(tmp_path: Path) -> None:
    """Only a BGR-trained checkpoint gets its stem channels reversed."""
    config, d2 = build_pair("faster_rcnn", input_format="RGB")
    torch.manual_seed(0)
    source = build_detector(config).eval()

    torch.save({"model": d2_state_from(source, is_bgr=False)}, tmp_path / "d2_model.pth")
    convert_d2(
        tmp_path / "d2_model.pth",
        d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
        output=tmp_path / "converted.pth",
        class_names=["a", "b", "c"],
    )

    stem = "backbone.bottom_up.stem.0.weight"
    torch.testing.assert_close(
        load_checkpoint(tmp_path / "converted.pth")[1][stem], source.state_dict()[stem]
    )


def test_sidecar_records_where_the_model_came_from(tmp_path: Path) -> None:
    config, d2 = build_pair("keypoint_rcnn")
    source = build_detector(config).eval()
    torch.save({"model": d2_state_from(source, is_bgr=True)}, tmp_path / "d2_model.pth")
    convert_d2(
        tmp_path / "d2_model.pth",
        d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
        output=tmp_path / "converted.pth",
        class_names=["a", "b", "c"],
    )

    sidecar, _ = load_checkpoint(tmp_path / "converted.pth")
    assert sidecar is not None
    assert sidecar["class_names"] == ["a", "b", "c"]
    assert sidecar["provenance"]["source"] == "detectron2"
    assert sidecar["provenance"]["d2_input_format"] == "BGR"


def test_class_name_count_must_match_num_classes(tmp_path: Path) -> None:
    config, d2 = build_pair("faster_rcnn")
    source = build_detector(config).eval()
    torch.save({"model": d2_state_from(source, is_bgr=True)}, tmp_path / "d2_model.pth")

    with pytest.raises(D2ConversionError, match="NUM_CLASSES=3"):
        convert_d2(
            tmp_path / "d2_model.pth",
            d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
            output=tmp_path / "converted.pth",
            class_names=["only-one"],
        )


def test_default_class_names_when_none_given(tmp_path: Path) -> None:
    """D2 keeps names in MetadataCatalog, so cfg.yaml alone yields placeholders."""
    config, d2 = build_pair("faster_rcnn")
    source = build_detector(config).eval()
    torch.save({"model": d2_state_from(source, is_bgr=True)}, tmp_path / "d2_model.pth")
    convert_d2(
        tmp_path / "d2_model.pth",
        d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
        output=tmp_path / "converted.pth",
    )

    sidecar, _ = load_checkpoint(tmp_path / "converted.pth")
    assert sidecar is not None
    assert sidecar["class_names"] == ["class_0", "class_1", "class_2"]


def test_unknown_checkpoint_keys_are_fatal(tmp_path: Path) -> None:
    """A key with no rename rule means the table is incomplete — never guess."""
    config, d2 = build_pair("faster_rcnn")
    source = build_detector(config).eval()
    state = d2_state_from(source, is_bgr=True)
    state["roi_heads.some_future_head.weight"] = torch.zeros(1)
    torch.save({"model": state}, tmp_path / "d2_model.pth")

    with pytest.raises(D2ConversionError, match="no rename rule"):
        convert_d2(
            tmp_path / "d2_model.pth",
            d2_config=write_pickled_cfg(tmp_path / "cfg.yaml", d2),
            output=tmp_path / "converted.pth",
            class_names=["a", "b", "c"],
        )
