"""Unit tests for :func:`mayaku.evaluate`.

Uses the pre-built ``toy_workspace["weights"]`` checkpoint, so these run
without a training loop — they exercise the public wrapper (path validation,
device resolution, delegation to ``run_eval``), not the COCO metric math
(covered by the engine's own tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mayaku import evaluate


def test_evaluate_returns_metrics_dict(toy_workspace: dict[str, Any]) -> None:
    metrics = evaluate(
        toy_workspace["weights"],
        annotations=toy_workspace["json"],
        images=toy_workspace["images"],
        device="cpu",
    )
    assert isinstance(metrics, dict)
    # An untrained random model yields no surviving detections, so the evaluator
    # returns {} (empty predictions); a trained one would yield a "bbox" entry.
    assert metrics == {} or "bbox" in metrics


def test_evaluate_writes_metrics_json(toy_workspace: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "eval_out"
    evaluate(
        toy_workspace["weights"],
        annotations=toy_workspace["json"],
        images=toy_workspace["images"],
        output_dir=out,
        device="cpu",
    )
    assert (out / "metrics.json").exists()


def test_evaluate_rejects_missing_annotations(
    toy_workspace: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="annotations not found"):
        evaluate(
            toy_workspace["weights"],
            annotations=tmp_path / "nope.json",
            images=toy_workspace["images"],
            device="cpu",
        )


def test_evaluate_rejects_non_directory_images(toy_workspace: dict[str, Any]) -> None:
    with pytest.raises(NotADirectoryError, match="images is not a directory"):
        evaluate(
            toy_workspace["weights"],
            annotations=toy_workspace["json"],
            images=toy_workspace["json"],  # a file, not a directory
            device="cpu",
        )
