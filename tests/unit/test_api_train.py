"""Smoke tests for :func:`mayaku.api.train`.

Same toy-COCO pattern as :mod:`tests.unit.test_cli`. Each test runs 2
iters end-to-end (train → checkpoint → optional eval → metadata) and
asserts the orchestration produced the expected artefacts on disk and
the expected keys in the returned dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mayaku.api import train

# `toy_workspace` is the shared fixture defined in tests/conftest.py;
# returns {"images": Path, "json": Path, "cfg": Path, "cfg_obj":
# MayakuConfig, "weights": Path, "image_file": Path}.

# ---------------------------------------------------------------------------
# Happy path — config from path, full train + eval, all artefacts written
# ---------------------------------------------------------------------------


def test_train_returns_result_dict_and_writes_artefacts(
    toy_workspace: dict[str, Any], tmp_path: Path
) -> None:
    out = tmp_path / "run_happy"
    result = train(
        config=toy_workspace["cfg"],
        train_json=toy_workspace["json"],
        train_images=toy_workspace["images"],
        val_json=toy_workspace["json"],
        val_images=toy_workspace["images"],
        output_dir=out,
        device="cpu",
    )

    expected_keys = {
        "final_box_ap",
        "final_box_ap50",
        "final_box_ap75",
        "final_weights",
        "output_dir",
        "train_seconds",
        "eval_seconds",
        "metadata",
    }
    assert set(result) == expected_keys
    assert result["output_dir"] == out
    assert result["final_weights"].exists()
    assert isinstance(result["train_seconds"], float)
    assert isinstance(result["eval_seconds"], float)
    # box AP can be 0 on a 1-image toy run but should be a float, not None.
    assert isinstance(result["final_box_ap"], float)

    # Artefacts on disk.
    assert (out / "train" / "config.yaml").exists()
    assert (out / "train" / "metadata.json").exists()
    meta = json.loads((out / "train" / "metadata.json").read_text())
    assert meta["backbone"] == "resnet50"
    assert meta["max_iter"] == 2
    assert meta["effective_batch_size"] == 1
    assert meta["final_box_ap"] == result["final_box_ap"]


# ---------------------------------------------------------------------------
# `val_json=None` short-circuits all eval
# ---------------------------------------------------------------------------


def test_train_skips_eval_when_no_val_set(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "run_no_val"
    result = train(
        config=toy_workspace["cfg"],
        train_json=toy_workspace["json"],
        train_images=toy_workspace["images"],
        # val_json / val_images intentionally omitted
        output_dir=out,
        device="cpu",
    )
    assert result["final_box_ap"] is None
    assert result["eval_seconds"] is None
    assert result["final_weights"].exists()
    # eval/ subdir should not be created.
    assert not (out / "eval").exists()
    # metadata reflects the skip.
    meta = json.loads((out / "train" / "metadata.json").read_text())
    assert meta["final_box_ap"] is None
    assert meta["eval_seconds"] is None


def test_train_rejects_half_configured_val(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both be provided"):
        train(
            config=toy_workspace["cfg"],
            train_json=toy_workspace["json"],
            train_images=toy_workspace["images"],
            val_json=toy_workspace["json"],
            # val_images intentionally missing
            output_dir=tmp_path / "run_half_val",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# `config` accepts a MayakuConfig object directly
# ---------------------------------------------------------------------------


def test_train_accepts_mayaku_config_object(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "run_obj"
    result = train(
        config=toy_workspace["cfg_obj"],
        train_json=toy_workspace["json"],
        train_images=toy_workspace["images"],
        output_dir=out,
        device="cpu",
    )
    assert result["final_weights"].exists()
    # When config is an object, stem falls back to "mayaku_run" — but since
    # we passed output_dir explicitly, that's what should win.
    assert result["output_dir"] == out


# ---------------------------------------------------------------------------
# `output_dir=None` → auto-derive ./runs/<config_stem>/
# ---------------------------------------------------------------------------


def test_train_auto_derives_output_dir_from_config_stem(
    toy_workspace: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default Path("./runs") is CWD-relative. chdir into tmp_path so
    # the run lands under tmp_path/runs/<stem> and we don't pollute the
    # actual repo root.
    monkeypatch.chdir(tmp_path)
    result = train(
        config=toy_workspace["cfg"],
        train_json=toy_workspace["json"],
        train_images=toy_workspace["images"],
        device="cpu",
    )
    expected = Path("runs") / "cfg"  # cfg.yaml → stem "cfg"
    assert result["output_dir"] == expected
    assert (expected / "train" / "metadata.json").exists()


# ---------------------------------------------------------------------------
# `overrides` is forwarded to merge_overrides
# ---------------------------------------------------------------------------


def test_train_forwards_overrides_to_merge(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "run_override"
    train(
        config=toy_workspace["cfg"],
        train_json=toy_workspace["json"],
        train_images=toy_workspace["images"],
        output_dir=out,
        overrides={"solver": {"base_lr": 5e-5}},
        device="cpu",
    )
    meta = json.loads((out / "train" / "metadata.json").read_text())
    # base_lr should reflect the override, not the YAML's 1e-4.
    assert meta["base_lr"] == 5e-5


# ---------------------------------------------------------------------------
# Invalid overrides raise pydantic's standard error (no auto-drop magic)
# ---------------------------------------------------------------------------


def test_train_invalid_override_raises(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        train(
            config=toy_workspace["cfg"],
            train_json=toy_workspace["json"],
            train_images=toy_workspace["images"],
            output_dir=tmp_path / "run_invalid",
            overrides={"solver": {"foo_bar_no_such_field": 42}},
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Missing train_json / train_images raise clear errors early
# ---------------------------------------------------------------------------


def test_train_rejects_missing_train_json(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train_json"):
        train(
            config=toy_workspace["cfg"],
            train_json=tmp_path / "does_not_exist.json",
            train_images=toy_workspace["images"],
            output_dir=tmp_path / "run_no_json",
            device="cpu",
        )


def test_train_rejects_non_directory_train_images(
    toy_workspace: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(NotADirectoryError, match="train_images"):
        train(
            config=toy_workspace["cfg"],
            train_json=toy_workspace["json"],
            train_images=toy_workspace["json"],  # a file, not a directory
            output_dir=tmp_path / "run_bad_images",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Multi-process gloo (num_gpus > 1) — exercises the launch() integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_num_gpus_2_on_cpu_via_gloo(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    """Spawn 2 gloo workers via :func:`mayaku.engine.launch`.

    Validates the multi-process DDP code paths end-to-end on CPU
    (sampler rank slicing, rank-0 hook gating, post-train metadata
    write in the parent). The 2-rank CPU/gloo path is the cheapest
    way to exercise the DDP integration without requiring multi-GPU
    hardware.

    The shared ``toy_workspace`` fixture has 1 image, but
    ``TrainingSampler`` refuses ``size < num_replicas`` (any rank with
    no indices would starve and deadlock the DDP backward), so we
    extend the workspace to 4 images for this test only.

    ``dataloader.num_workers=0`` because on macOS each rank's
    DataLoader would re-spawn the test interpreter for every worker —
    8 spawned children (2 ranks × 4 dataloader workers) inflate the
    test runtime by an order of magnitude. The 0-worker path still
    exercises every distributed code path we care about here.
    """
    import numpy as np
    from PIL import Image

    images_dir = toy_workspace["images"]
    coco = json.loads(toy_workspace["json"].read_text())
    rng = np.random.default_rng(1)
    for i in range(2, 5):  # add img2/img3/img4 alongside the fixture's img.png
        arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(images_dir / f"img{i}.png")
        coco["images"].append({"id": i, "file_name": f"img{i}.png", "height": 64, "width": 64})
        coco["annotations"].append(
            {
                "id": 100 + i,
                "image_id": i,
                "category_id": 1,
                "bbox": [10.0, 10.0, 30.0, 30.0],
                "area": 900.0,
                "iscrowd": 0,
            }
        )
    toy_workspace["json"].write_text(json.dumps(coco))

    out = tmp_path / "run_ddp_cpu"
    result = train(
        config=toy_workspace["cfg"],
        train_json=toy_workspace["json"],
        train_images=toy_workspace["images"],
        output_dir=out,
        overrides={"dataloader": {"num_workers": 0}},
        device="cpu",
        num_gpus=2,
    )
    # Same artefact shape as the single-GPU run.
    assert result["final_weights"].exists()
    assert (out / "train" / "config.yaml").exists()
    assert (out / "train" / "metadata.json").exists()
    meta = json.loads((out / "train" / "metadata.json").read_text())
    assert meta["num_gpus"] == 2
