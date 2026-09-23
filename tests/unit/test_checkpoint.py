"""The self-describing sidecar: what it records, how it is read back, and
what it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mayaku.config import InputConfig, KeypointConfig, MayakuConfig, ModelConfig
from mayaku.model import TINY, Detector
from mayaku.model.detector import output_names
from mayaku.utils.checkpoint import (
    SIDECAR_SCHEMA_VERSION,
    build_sidecar,
    check_sidecar,
    load_checkpoint,
    read_deploy_checkpoint,
    save_checkpoint,
    select_final_weights,
)

NAMES = ["cat", "dog", "bird"]


def _cfg(**model) -> MayakuConfig:
    return MayakuConfig(model=ModelConfig(num_classes=len(NAMES), **model),
                        input=InputConfig(canvas_hw=(576, 1024)))


def _sidecar(cfg, names=NAMES, **kw):
    return build_sidecar(cfg, names, cfg.model.build(cfg.input.canvas_hw), **kw)


def test_sidecar_describes_the_deploy_contract() -> None:
    s = _sidecar(_cfg(), provenance={"run": "unit"})
    assert s["schema_version"] == SIDECAR_SCHEMA_VERSION == 2
    assert s["class_names"] == NAMES and s["canvas_hw"] == [576, 1024]
    assert s["outputs"] == output_names() == ["cls0", "box0", "cls1", "box1", "cls2", "box2"]
    assert s["decode"]["strides"] == [8, 16, 32] and s["decode"]["reg_max"] == 16
    assert s["preprocess"]["pad_value"] == 114 and s["preprocess"]["channels"] == "RGB"
    assert s["mask"] is None and s["keypoints"] is None
    assert s["quant"]["qat"] is True                     # tier n builds quantization-aware
    assert s["provenance"]["run"] == "unit" and "mayaku_version" in s["provenance"]
    json.dumps(s)                                        # every value is plain JSON


def test_sidecar_carries_mask_and_keypoint_decode_constants() -> None:
    kp = KeypointConfig(num=3, names=("left_a", "right_a", "c"), flip_pairs=((0, 1),))
    s = _sidecar(_cfg(tier="m", seg=True, keypoints=kp))
    assert s["outputs"] == output_names(seg=True, kpt=3)
    assert s["mask"]["kernel_layout"] == [[10, 8], [8, 8], [8, 1]] and s["mask"]["stride"] == 8
    assert s["keypoints"] == {"num": 3, "sigmas": [0.05] * 3,
                              "names": ["left_a", "right_a", "c"], "flip_pairs": [[0, 1]]}
    assert s["quant"]["qat"] is False                    # tier m builds fp32


def test_sidecar_reports_the_model_not_the_config() -> None:
    cfg = _cfg()                                         # tier n: QAT by default
    s = build_sidecar(cfg, NAMES, Detector(cfg.model.to_tier(), len(NAMES)))
    assert s["quant"]["qat"] is False                    # this model was built without it


def test_sidecar_needs_a_resolved_canvas_and_matching_classes() -> None:
    model = Detector(TINY, len(NAMES))
    with pytest.raises(ValueError, match="canvas_hw"):
        build_sidecar(MayakuConfig(model=ModelConfig(num_classes=3)), NAMES, model)
    with pytest.raises(ValueError, match="class names"):
        build_sidecar(_cfg(), NAMES[:2], model)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = _cfg()
    model = cfg.model.build(cfg.input.canvas_hw)
    path = tmp_path / "best.pt"
    save_checkpoint(model, path, build_sidecar(cfg, NAMES, model))
    sidecar, got_cfg, state = read_deploy_checkpoint(path)
    names = sidecar["class_names"]
    assert got_cfg == cfg and names == NAMES
    # a QAT checkpoint loads into the model its own config builds
    got_cfg.model.build(got_cfg.input.canvas_hw, len(names)).load_state_dict(state, strict=True)


def test_bare_state_dict_has_no_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"
    save_checkpoint(Detector(TINY, 2), path)
    sidecar, state = load_checkpoint(path)
    assert sidecar is None and "head.cls.0.bias" in state
    with pytest.raises(ValueError, match="no mayaku sidecar"):
        read_deploy_checkpoint(path)


def test_v2_and_unknown_sidecars_are_refused() -> None:
    with pytest.raises(ValueError, match="mayaku<3"):
        check_sidecar({"schema_version": 1, "config": {}}, "old.pth")
    with pytest.raises(ValueError, match="schema 7"):
        check_sidecar({"schema_version": 7, "config": {}}, "new.pth")
    ok = {"schema_version": 2, "config": {}}
    assert check_sidecar(ok, "x") is ok


def test_select_final_weights_prefers_best(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no checkpoint"):
        select_final_weights(tmp_path)
    torch.save({}, tmp_path / "last.pt")
    assert select_final_weights(tmp_path).name == "last.pt"
    torch.save({}, tmp_path / "best.pt")
    assert select_final_weights(tmp_path).name == "best.pt"
