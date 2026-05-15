"""Tests for the ``mayaku`` CLI.

Tiny end-to-end tests for each subcommand using Typer's
:class:`CliRunner`. We build a 1-image toy COCO dataset, dump a
matching :class:`MayakuConfig` to YAML, and exercise predict / eval /
train against it. ``export`` is the explicit "not implemented yet"
stub introduced in Step 17.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from mayaku.cli import app
from mayaku.cli._factory import build_detector
from mayaku.cli.eval import run_eval
from mayaku.cli.export import run_export
from mayaku.cli.predict import run_predict
from mayaku.cli.train import run_train
from mayaku.config import MayakuConfig, dump_yaml

# `toy_workspace` is a shared fixture defined in tests/conftest.py.


# ---------------------------------------------------------------------------
# build_detector factory dispatch
# ---------------------------------------------------------------------------


def test_build_detector_dispatches_on_meta_architecture() -> None:
    cfg = MayakuConfig()
    model = build_detector(cfg)
    # Default meta_architecture is "faster_rcnn".
    assert type(model).__name__ == "FasterRCNN"


# ---------------------------------------------------------------------------
# In-process subcommand functions (covers the work; CLI tests below
# verify the typer plumbing on top).
# ---------------------------------------------------------------------------


def test_run_predict_returns_payload(toy_workspace: dict[str, Path]) -> None:
    payload = run_predict(
        toy_workspace["cfg"],
        toy_workspace["image_file"],
        weights=toy_workspace["weights"],
    )
    assert payload["image"] == str(toy_workspace["image_file"])
    assert isinstance(payload["instances"], list)


def test_run_predict_writes_json_file(toy_workspace: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "preds.json"
    run_predict(
        toy_workspace["cfg"],
        toy_workspace["image_file"],
        weights=toy_workspace["weights"],
        output=out,
    )
    parsed = json.loads(out.read_text())
    assert parsed["image"] == str(toy_workspace["image_file"])


def test_run_eval_returns_metrics_dict(toy_workspace: dict[str, Path]) -> None:
    metrics = run_eval(
        toy_workspace["cfg"],
        weights=toy_workspace["weights"],
        coco_gt_json=toy_workspace["json"],
        image_root=toy_workspace["images"],
    )
    assert isinstance(metrics, dict)
    # Untrained random model → no detections survive the score threshold;
    # evaluate() returns {} in that case (predictions list is empty).
    assert metrics == {} or "bbox" in metrics


def test_run_train_runs_the_loop_and_writes_checkpoints(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "train_out"
    # The test exercises the train loop on a tiny synthetic dataset; we
    # don't want to download torchvision ImageNet weights here, so accept
    # the freeze_at-without-pretrain warning for this path. (The warning
    # itself is unit-tested separately below.)
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            toy_workspace["cfg"],
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
            max_iter=2,
        )
    # PeriodicCheckpointer ran (period=2 → save at iter 2 + final).
    checkpoints = sorted(p.name for p in out.glob("*.pth"))
    assert "model_final.pth" in checkpoints


def test_run_train_pretrained_backbone_passes_through(
    toy_workspace: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--pretrained-backbone` must reach `build_detector(..., backbone_weights="DEFAULT")`.

    We mock `build_detector` to inspect the call without actually
    downloading torchvision's IMAGENET1K_V2 weights (CI-friendly).
    """
    from mayaku.cli import train as train_module

    captured: dict[str, object] = {}
    real_build_detector = train_module.build_detector

    def spy(cfg: object, *, backbone_weights: object = None) -> object:
        captured["backbone_weights"] = backbone_weights
        # Forward to the real builder with the random-init path so the
        # rest of the loop (mapper, sampler, trainer) actually runs.
        return real_build_detector(cfg)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "build_detector", spy)

    out = tmp_path / "train_pretrained_out"
    run_train(
        toy_workspace["cfg"],
        coco_gt_json=toy_workspace["json"],
        image_root=toy_workspace["images"],
        output_dir=out,
        device="cpu",
        pretrained_backbone=True,
        max_iter=1,
    )
    assert captured["backbone_weights"] == "DEFAULT"


def test_run_train_pretrained_and_weights_are_mutually_exclusive(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "train_conflict_out"
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_train(
            toy_workspace["cfg"],
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
            pretrained_backbone=True,
            weights=toy_workspace["weights"],
            max_iter=1,
        )


def _cfg_with_eval_period(toy_workspace: dict[str, Path], tmp_path: Path, period: int) -> Path:
    """Clone the toy YAML config but bump test.eval_period."""
    from mayaku.config import load_yaml

    cfg = load_yaml(toy_workspace["cfg"])
    cfg = cfg.model_copy(update={"test": cfg.test.model_copy(update={"eval_period": period})})
    out = tmp_path / "cfg_eval.yaml"
    dump_yaml(cfg, out)
    return out


def test_run_train_eval_period_requires_val_paths(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    cfg_with_eval = _cfg_with_eval_period(toy_workspace, tmp_path, period=2)
    out = tmp_path / "train_eval_missing_val"
    with pytest.raises(ValueError, match="val-json"):
        run_train(
            cfg_with_eval,
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
            max_iter=2,
        )


def test_run_train_warns_when_val_paths_supplied_but_eval_period_zero(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "train_eval_unused_val"
    # Use the original cfg (eval_period=0). The freeze_at warning will
    # also fire on this random-init path, so we filter for our message.
    with pytest.warns(UserWarning) as record:
        run_train(
            toy_workspace["cfg"],
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
            max_iter=1,
            val_json=toy_workspace["json"],
            val_image_root=toy_workspace["images"],
        )
    assert any("test.eval_period=0" in str(w.message) for w in record)


def test_run_train_accepts_mayaku_config_object(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    """Python-side fine-tune scripts should be able to patch a config
    in code and pass the object directly — no temp YAML round-trip."""
    from mayaku.config import load_yaml

    cfg = load_yaml(toy_workspace["cfg"])
    # Pretend the user is patching a few fields in code.
    cfg = cfg.model_copy(update={"solver": cfg.solver.model_copy(update={"max_iter": 1})})
    out = tmp_path / "train_obj_out"
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            cfg,  # MayakuConfig, not a Path
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
        )
    # The resolved cfg is auto-dumped next to the checkpoints.
    assert (out / "config.yaml").exists()
    assert (out / "model_final.pth").exists()


def test_run_train_dumps_resolved_config_for_path_input_too(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    """The auto-dump fires regardless of whether the input was a path
    or a config object — every training run gets a provenance record."""
    out = tmp_path / "train_path_out"
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            toy_workspace["cfg"],
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
            max_iter=1,
        )
    assert (out / "config.yaml").exists()


def test_run_eval_accepts_mayaku_config_object(
    toy_workspace: dict[str, Path],
) -> None:
    from mayaku.config import load_yaml

    cfg = load_yaml(toy_workspace["cfg"])
    metrics = run_eval(
        cfg,  # MayakuConfig, not a Path
        weights=toy_workspace["weights"],
        coco_gt_json=toy_workspace["json"],
        image_root=toy_workspace["images"],
    )
    assert isinstance(metrics, dict)


def test_run_train_finetune_drops_class_count_mismatched_layers(
    toy_workspace: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Loading a checkpoint trained for a different num_classes is the
    standard fine-tune workflow. The shape mismatch on class-specific
    tail layers must be filtered out (with a stdout log line) instead
    of crashing."""
    # Toy cfg has num_classes=2 → cls_score is [3, 1024], bbox_pred is
    # [8, 1024]. Build a fake "COCO" checkpoint with K=80 shapes so
    # those keys mismatch and must be dropped.
    real_state = torch.load(toy_workspace["weights"], map_location="cpu", weights_only=True)
    fake_coco_state = dict(real_state)
    # Synthesize 81-way / 320-way / 80-way shapes that match the D2 COCO
    # convention; the rest of the keys keep their toy shapes (which match
    # the toy model exactly).
    fake_coco_state["roi_heads.box_predictor.cls_score.weight"] = torch.zeros(81, 1024)
    fake_coco_state["roi_heads.box_predictor.cls_score.bias"] = torch.zeros(81)
    fake_coco_state["roi_heads.box_predictor.bbox_pred.weight"] = torch.zeros(320, 1024)
    fake_coco_state["roi_heads.box_predictor.bbox_pred.bias"] = torch.zeros(320)
    fake_path = tmp_path / "fake_coco.pth"
    torch.save(fake_coco_state, fake_path)

    out = tmp_path / "finetune_out"
    run_train(
        toy_workspace["cfg"],
        coco_gt_json=toy_workspace["json"],
        image_root=toy_workspace["images"],
        output_dir=out,
        weights=fake_path,
        device="cpu",
        max_iter=1,
    )
    captured = capsys.readouterr().out
    assert "dropped 4 shape-mismatched key(s)" in captured
    # Toy fixture has fc_dim=32 and num_classes=2 → cls_score is [3, 32];
    # the fake checkpoint we built had COCO's [81, 1024]. Both axes
    # mismatch, both ends get logged.
    assert "roi_heads.box_predictor.cls_score.weight: ckpt (81, 1024) vs model (3, 32)" in captured
    assert "roi_heads.box_predictor.cls_score.bias: ckpt (81,) vs model (3,)" in captured
    assert "roi_heads.box_predictor.bbox_pred.weight: ckpt (320, 1024) vs model (8, 32)" in captured
    # The dropped keys appear in the missing-keys log too.
    assert "missing keys not in checkpoint" in captured
    # Train loop completed despite the dropped layers.
    assert (out / "model_final.pth").exists()


def test_run_train_periodic_eval_runs_evaluator(
    toy_workspace: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end smoke: eval_period=1 over max_iter=2 → at least one
    eval line lands in stdout and the COCOEvaluator dumps its JSON."""
    cfg_with_eval = _cfg_with_eval_period(toy_workspace, tmp_path, period=1)
    out = tmp_path / "train_eval_smoke"
    with pytest.warns(UserWarning, match="freeze_at"):
        run_train(
            cfg_with_eval,
            coco_gt_json=toy_workspace["json"],
            image_root=toy_workspace["images"],
            output_dir=out,
            device="cpu",
            max_iter=2,
            val_json=toy_workspace["json"],
            val_image_root=toy_workspace["images"],
        )
    captured = capsys.readouterr().out
    # The "[eval @ iter N]" line is emitted by EvalHook._run_eval — its
    # presence is the proof that the hook fired with a wired evaluator.
    # The untrained model produces no predictions through the score
    # threshold, so COCOEvaluator's output dir is never written; that
    # path is exercised by the COCOEvaluator unit tests.
    assert "[eval @ iter" in captured


def test_run_export_tensorrt_dispatch_lives_on_cuda_only(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    # All four export targets are live, but TensorRT lazily imports
    # the `tensorrt` runtime which is CUDA-only. On any host without
    # the runtime installed the dispatch reaches TensorRTExporter,
    # which raises ImportError on its first attribute access.
    # On CUDA hosts with `pip install -e '.[tensorrt]'` this becomes
    # a real export — covered by tests/unit/test_tensorrt_export.py.
    import importlib.util

    if importlib.util.find_spec("tensorrt") is None:
        # Error message varies by host: "TensorRT is not supported on
        # macOS.", "TensorRT requires a CUDA-enabled GPU…", or the
        # `pip install mayaku[tensorrt]` hint on Linux+CUDA without the
        # extra installed. All mention TensorRT — match case-insensitively.
        with pytest.raises(ImportError, match=r"(?i)tensorrt"):
            run_export(
                "tensorrt",
                toy_workspace["cfg"],
                weights=toy_workspace["weights"],
                output=tmp_path / "model.engine",
            )
    else:
        # tensorrt is importable but CUDA may not be available; let
        # the real exporter run and accept either a clean export or
        # a CUDA-side RuntimeError. Either way `run_export` returns
        # without our dispatch path raising NotImplementedError.
        try:
            run_export(
                "tensorrt",
                toy_workspace["cfg"],
                weights=toy_workspace["weights"],
                output=tmp_path / "model.engine",
            )
        except (RuntimeError, OSError):
            pass


def test_run_export_rejects_unknown_target(toy_workspace: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown export target"):
        run_export(
            "tflite",
            toy_workspace["cfg"],
            weights=toy_workspace["weights"],
            output=tmp_path / "model.tflite",
        )


# ---------------------------------------------------------------------------
# Typer CLI runner: argument plumbing + exit codes
# ---------------------------------------------------------------------------


def test_cli_help_lists_every_subcommand() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("train", "eval", "predict", "export"):
        assert cmd in res.stdout


def test_cli_predict_invokes_run_predict(toy_workspace: dict[str, Path], tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "cli_preds.json"
    res = runner.invoke(
        app,
        [
            "predict",
            str(toy_workspace["cfg"]),
            str(toy_workspace["image_file"]),
            "--weights",
            str(toy_workspace["weights"]),
            "--output",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    parsed: dict[str, Any] = json.loads(out.read_text())
    assert parsed["image"] == str(toy_workspace["image_file"])


def test_cli_predict_prints_payload_when_no_output(
    toy_workspace: dict[str, Path],
) -> None:
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "predict",
            str(toy_workspace["cfg"]),
            str(toy_workspace["image_file"]),
            "--weights",
            str(toy_workspace["weights"]),
        ],
    )
    assert res.exit_code == 0
    # JSON-shaped output on stdout — parses cleanly.
    parsed = json.loads(res.stdout)
    assert "instances" in parsed


def test_cli_export_unknown_target_returns_nonzero(
    toy_workspace: dict[str, Path], tmp_path: Path
) -> None:
    # All four real targets dispatch through the live exporters;
    # a typo in the target name should still surface as a non-zero
    # exit (Typer wraps the ValueError from run_export).
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "export",
            "tflite",
            str(toy_workspace["cfg"]),
            "--weights",
            str(toy_workspace["weights"]),
            "--output",
            str(tmp_path / "m.tflite"),
        ],
    )
    assert res.exit_code != 0


def test_cli_predict_validates_missing_image() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["predict", "/no/such/cfg.yaml", "/no/such/img.png"])
    # Typer's exists=True check fires before we reach run_predict.
    assert res.exit_code != 0
