"""The Python API and the command line: a real (tiny) run through
`mayaku.train`, resuming it, warm-starting from it, and every CLI command
against its checkpoint."""

from __future__ import annotations

import json

import pytest
import torch
from typer.testing import CliRunner

import mayaku
from mayaku.cli import app
from mayaku.config import parse_assignments
from mayaku.utils.checkpoint import read_deploy_checkpoint

from ._coco_fixture import synthetic_coco

# A run small enough for CPU CI: 96-pixel canvas, two epochs, no workers.
FAST = {"train": {"batch": 4, "lr_ref_batch": 4, "recalibrate_images": 4,
                  "warmup_iters_min": 2},
        "dataloader": {"num_workers": 0}, "model": {"qat": False}}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    tr, va = root / "train", root / "val"
    return {"train_annotations": synthetic_coco(str(tr), n=12, seed=1), "train_images": tr,
            "val_annotations": synthetic_coco(str(va), n=6, seed=2), "val_images": va}


@pytest.fixture(scope="module")
def run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return mayaku.train(**data, output_dir=out, size_budget=96, num_epochs=2,
                        overrides=FAST, device="cpu", log=lambda *_: None)


def test_parse_assignments() -> None:
    got = parse_assignments(["train.lr=0.005", "model.tier=s", "input.canvas_hw=[64, 96]",
                             "train.aug.mosaic=0", "model.qat=null"])
    assert got == {"train": {"lr": 0.005, "aug": {"mosaic": 0}},
                   "model": {"tier": "s", "qat": None}, "input": {"canvas_hw": [64, 96]}}
    with pytest.raises(ValueError):
        parse_assignments(["train.lr"])


@pytest.mark.slow
def test_train_writes_a_complete_run(run) -> None:
    train_dir = run["output_dir"] / "train"
    for f in ("config.yaml", "best.pt", "last.pt", "state.pt", "log.jsonl", "metadata.json"):
        assert (train_dir / f).exists(), f
    assert run["final_weights"] == train_dir / "best.pt"
    assert set(run["metrics"]) >= {"AP", "AP50", "AP-S"} and run["best"]["epoch"] in (0, 1)
    meta = json.loads((train_dir / "metadata.json").read_text())
    assert meta["canvas_hw"][0] <= 96 and meta["num_classes"] == 4 and meta["epochs"] == 2
    sidecar, cfg, _ = read_deploy_checkpoint(run["final_weights"])
    assert sidecar["class_names"] == ["c0", "c1", "c2", "c3"] and not cfg.model.qat_enabled
    # user values survive auto-config
    assert cfg.input.size_budget == 96 and cfg.train.epochs == 2 and cfg.train.batch == 4


@pytest.mark.slow
def test_resume_continues_the_run(run, data, tmp_path) -> None:
    import shutil

    train_dir = tmp_path / "train"
    shutil.copytree(run["output_dir"] / "train", train_dir)
    state = torch.load(train_dir / "state.pt", weights_only=True)
    state["epoch"] = 0                                 # as if stopped after epoch 0
    torch.save(state, train_dir / "state.pt")
    mayaku.train(**data, resume=train_dir, device="cpu", log=lambda *_: None)
    epochs = [json.loads(line)["epoch"] for line in (train_dir / "log.jsonl").open()]
    assert epochs == [0, 1, 1]                         # the log is appended to, not replaced
    with pytest.raises(ValueError, match="resume"):
        mayaku.train(**data, resume=train_dir, num_epochs=3)


@pytest.mark.slow
def test_warm_start_takes_the_architecture_from_the_checkpoint(run, data, tmp_path) -> None:
    out = mayaku.train(**{**data, "val_annotations": None, "val_images": None},
                       weights=run["final_weights"], output_dir=tmp_path, size_budget=96,
                       num_epochs=1, overrides={k: v for k, v in FAST.items() if k != "model"},
                       device="cpu", log=lambda *_: None)
    assert out["metrics"] is None and out["final_weights"].name == "last.pt"
    _, cfg, _ = read_deploy_checkpoint(out["final_weights"])
    # the checkpoint's qat=False carried over; the fine-tune recipe was applied
    assert cfg.model.qat is False and cfg.train.lr == 2e-3


@pytest.mark.slow
def test_cli(run, data, tmp_path) -> None:
    cli, weights = CliRunner(), str(run["final_weights"])
    image = str(data["val_images"] / "000.jpg")
    res = cli.invoke(app, ["predict", weights, image, "--conf", "0.01", "--device", "cpu"])
    assert res.exit_code == 0, res.output
    dets = json.loads(res.output)["detections"]
    assert dets and set(dets[0]) == {"class_id", "class", "score", "box_xyxy"}

    res = cli.invoke(app, ["eval", weights, "--annotations", data["val_annotations"],
                           "--images", str(data["val_images"]), "--device", "cpu"])
    assert res.exit_code == 0 and json.loads(res.output) == pytest.approx(run["metrics"])

    onnx = tmp_path / "m.onnx"
    res = cli.invoke(app, ["export", "onnx", weights, "--output", str(onnx)])
    assert res.exit_code == 0 and onnx.exists(), res.output
    assert cli.invoke(app, ["export", "tflite", weights]).exit_code != 0

    res = cli.invoke(app, ["train", "--annotations", data["train_annotations"],
                           "--images", str(data["train_images"]), "--set", "train.nope=1"])
    assert res.exit_code != 0
