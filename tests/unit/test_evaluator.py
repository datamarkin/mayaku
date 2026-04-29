"""Tests for :mod:`mayaku.engine.evaluator`.

A toy 2-image, 1-category COCO dataset stands in for COCO val. We
synthesise predictions that *exactly* match the GT (perfect detector)
and verify ``COCOEvaluator`` reports AP ≈ 1.0 across bbox / segm /
keypoints.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from mayaku.engine.evaluator import (
    COCOEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    instances_to_coco_json,
)
from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances

# ---------------------------------------------------------------------------
# Toy COCO GT fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_coco_gt(tmp_path: Path) -> Path:
    """Two images, one annotation each, a 64x64 box + a square mask + 17 KP."""
    # All-true 64x64 mask; encode to RLE for the GT.
    from pycocotools import mask as coco_mask

    bitmap = np.zeros((64, 64), dtype=np.uint8)
    bitmap[10:50, 10:50] = 1
    rle = coco_mask.encode(np.asfortranarray(bitmap))
    rle["counts"] = rle["counts"].decode("utf-8")

    # 17 visible keypoints inside the box, COCO Person convention.
    keypoints: list[float] = []
    for x in range(15, 32):
        keypoints.extend([float(x), 25.0, 2])

    coco = {
        "info": {"description": "toy"},
        "images": [
            {"id": 1, "file_name": "img1.png", "height": 64, "width": 64},
            {"id": 2, "file_name": "img2.png", "height": 64, "width": 64},
        ],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
        "annotations": [
            {
                # Each annotation needs a unique id; pycocotools uses
                # ann["id"] as a primary key and silently dedupes
                # duplicates (which collapses AP to ~0.25).
                "id": 100 + img_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [10.0, 10.0, 40.0, 40.0],  # XYWH
                "area": 40.0 * 40.0,
                "iscrowd": 0,
                "segmentation": rle,
                "keypoints": keypoints,
                "num_keypoints": 17,
            }
            for img_id in (1, 2)
        ],
    }
    path = tmp_path / "toy_gt.json"
    path.write_text(json.dumps(coco))
    return path


def _make_pred_instances(
    *, with_mask: bool, with_kp: bool, image_size: tuple[int, int]
) -> Instances:
    """Single perfect prediction matching the toy GT for one image."""
    inst = Instances(image_size=image_size)
    # Box already in XYXY at the original image size.
    inst.pred_boxes = Boxes(torch.tensor([[10.0, 10.0, 50.0, 50.0]]))
    inst.scores = torch.tensor([0.99])
    # The toy fixture has a single category with COCO id=1. Mayaku's
    # dataloader maps that to contiguous index 0, so a perfect
    # prediction's pred_classes is 0; COCOEvaluator's reverse map
    # writes it back as category_id=1.
    inst.pred_classes = torch.tensor([0])
    if with_mask:
        # (R, H, W) bool — already pasted to the image canvas.
        bitmap = torch.zeros(1, 64, 64, dtype=torch.bool)
        bitmap[0, 10:50, 10:50] = True
        inst.pred_masks = bitmap
    if with_kp:
        # 17 keypoints matching the GT positions; visibility column is
        # rewritten to 2 by instances_to_coco_json. We pre-add +0.5
        # because the encoder undoes the data-layer +0.5 ingest shift.
        kp = torch.zeros(1, 17, 3)
        for k in range(17):
            kp[0, k, 0] = 15.0 + float(k) + 0.5
            kp[0, k, 1] = 25.0 + 0.5
            kp[0, k, 2] = 2.0
        inst.pred_keypoints = kp
    return inst


# ---------------------------------------------------------------------------
# instances_to_coco_json
# ---------------------------------------------------------------------------


def test_instances_to_coco_json_emits_xywh_bbox() -> None:
    inst = Instances(image_size=(64, 64))
    inst.pred_boxes = Boxes(torch.tensor([[5.0, 10.0, 25.0, 50.0]]))
    inst.scores = torch.tensor([0.7])
    inst.pred_classes = torch.tensor([3])
    record = next(iter(instances_to_coco_json(inst, image_id=42)))
    assert record["image_id"] == 42
    assert record["category_id"] == 3
    assert record["bbox"] == [5.0, 10.0, 20.0, 40.0]  # XYWH
    assert record["score"] == pytest.approx(0.7)
    assert "segmentation" not in record
    assert "keypoints" not in record


def test_instances_to_coco_json_includes_mask_when_present() -> None:
    inst = _make_pred_instances(with_mask=True, with_kp=False, image_size=(64, 64))
    record = next(iter(instances_to_coco_json(inst, 1)))
    assert "segmentation" in record
    assert record["segmentation"]["counts"]  # non-empty RLE


def test_instances_to_coco_json_undoes_keypoint_half_pixel_shift() -> None:
    inst = _make_pred_instances(with_mask=False, with_kp=True, image_size=(64, 64))
    record = next(iter(instances_to_coco_json(inst, 1)))
    kp = record["keypoints"]
    assert len(kp) == 17 * 3
    # First keypoint x: ingest-side stored 15.5; encoder subtracts 0.5 → 15.
    assert kp[0] == pytest.approx(15.0)
    assert kp[1] == pytest.approx(25.0)
    assert kp[2] == 2.0  # visibility forced to visible


def test_instances_to_coco_json_empty_yields_nothing() -> None:
    inst = Instances(image_size=(8, 8))
    inst.pred_boxes = Boxes(torch.zeros(0, 4))
    inst.scores = torch.zeros(0)
    inst.pred_classes = torch.zeros(0, dtype=torch.long)
    assert list(instances_to_coco_json(inst, 0)) == []


# ---------------------------------------------------------------------------
# inference_on_dataset
# ---------------------------------------------------------------------------


class _CountingEvaluator(DatasetEvaluator):
    def __init__(self) -> None:
        self.batches = 0
        self.last_outputs: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.batches = 0
        self.last_outputs = []

    def process(
        self,
        inputs: Sequence[dict[str, Any]],
        outputs: Sequence[dict[str, Any]],
    ) -> None:
        self.batches += 1
        self.last_outputs = list(outputs)

    def evaluate(self) -> dict[str, Any]:
        return {"batches": self.batches}


class _StubModel(nn.Module):
    """Model that returns a deterministic per-image dict in eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self._was_train = False

    def forward(self, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        # Record the eval-mode contract.
        self._was_train = self.training
        return [{"instances": Instances(image_size=(8, 8))} for _ in batch]


def test_inference_on_dataset_runs_loop_and_returns_evaluate_result() -> None:
    model = _StubModel()
    model.train()
    loader: Iterable[Sequence[dict[str, Any]]] = [
        [{"image_id": 1}, {"image_id": 2}],
        [{"image_id": 3}],
    ]
    evaluator = _CountingEvaluator()
    out = inference_on_dataset(model, loader, evaluator)
    assert out == {"batches": 2}
    # Model is in eval mode during the loop.
    assert model._was_train is False
    # Train/eval mode is restored on exit.
    assert model.training is True


def test_inference_on_dataset_rejects_non_list_output() -> None:
    class _BadModel(nn.Module):
        def forward(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
            return {"this": "is not a list"}

    with pytest.raises(TypeError, match="list"):
        inference_on_dataset(_BadModel(), [[{"image_id": 1}]], _CountingEvaluator())


def test_inference_on_dataset_progress_lines(capsys: pytest.CaptureFixture[str]) -> None:
    """progress_period=2 over a 5-batch loader fires at i=2, 4, and 5 (final flush)."""
    model = _StubModel()
    loader: list[Sequence[dict[str, Any]]] = [[{"image_id": i}] for i in range(5)]
    inference_on_dataset(model, loader, _CountingEvaluator(), progress_period=2)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[eval] ")]
    assert [ln.split(" — ")[0] for ln in lines] == [
        "[eval] 2/5 (40.0%)",
        "[eval] 4/5 (80.0%)",
        "[eval] 5/5 (100.0%)",
    ]


def test_inference_on_dataset_progress_silenced_with_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inference_on_dataset(
        _StubModel(),
        [[{"image_id": 1}], [{"image_id": 2}]],
        _CountingEvaluator(),
        progress_period=0,
    )
    out = capsys.readouterr().out
    assert "[eval]" not in out


# ---------------------------------------------------------------------------
# COCOEvaluator end-to-end with perfect predictions
# ---------------------------------------------------------------------------


def _perfect_outputs(*, with_mask: bool, with_kp: bool):  # type: ignore[no-untyped-def]
    """Build the (inputs, outputs) pair for the toy 2-image dataset."""
    inputs = [
        {"image_id": 1, "height": 64, "width": 64},
        {"image_id": 2, "height": 64, "width": 64},
    ]
    outputs = [
        {
            "instances": _make_pred_instances(
                with_mask=with_mask, with_kp=with_kp, image_size=(64, 64)
            )
        }
        for _ in inputs
    ]
    return inputs, outputs


def test_coco_evaluator_bbox_perfect_predictions_yield_ap_one(
    toy_coco_gt: Path,
) -> None:
    evaluator = COCOEvaluator(toy_coco_gt)
    evaluator.reset()
    inputs, outputs = _perfect_outputs(with_mask=False, with_kp=False)
    evaluator.process(inputs, outputs)
    metrics = evaluator.evaluate()
    assert "bbox" in metrics
    assert metrics["bbox"]["AP"] == pytest.approx(1.0, abs=1e-3)
    # Auto-detection didn't spuriously turn on segm or keypoints.
    assert "segm" not in metrics
    assert "keypoints" not in metrics


def test_coco_evaluator_segm_perfect_predictions_yield_ap_one(
    toy_coco_gt: Path,
) -> None:
    evaluator = COCOEvaluator(toy_coco_gt)
    evaluator.reset()
    inputs, outputs = _perfect_outputs(with_mask=True, with_kp=False)
    evaluator.process(inputs, outputs)
    metrics = evaluator.evaluate()
    assert metrics["bbox"]["AP"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["segm"]["AP"] == pytest.approx(1.0, abs=1e-3)


def test_coco_evaluator_keypoints_perfect_predictions_yield_ap_one(
    toy_coco_gt: Path,
) -> None:
    evaluator = COCOEvaluator(toy_coco_gt)
    evaluator.reset()
    inputs, outputs = _perfect_outputs(with_mask=False, with_kp=True)
    evaluator.process(inputs, outputs)
    metrics = evaluator.evaluate()
    assert metrics["bbox"]["AP"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["keypoints"]["AP"] == pytest.approx(1.0, abs=1e-3)


def test_coco_evaluator_remaps_gappy_coco_category_ids(tmp_path: Path) -> None:
    """COCO category_ids with gaps (1, 5, 12, 80) must round-trip through
    the contiguous-index reverse map, not be written as raw ``int(classes[k])``.
    Without the reverse map, perfect predictions get matched against the
    wrong GT category and AP collapses to ~0.
    """
    cat_ids = (1, 5, 12, 80)  # deliberately gappy, like real COCO
    coco = {
        "info": {"description": "gappy"},
        "images": [
            {"id": img_id, "file_name": f"img{img_id}.png", "height": 64, "width": 64}
            for img_id in range(1, len(cat_ids) + 1)
        ],
        "categories": [
            {"id": cid, "name": f"cls_{cid}", "supercategory": "stuff"} for cid in cat_ids
        ],
        "annotations": [
            {
                "id": 1000 + img_id,
                "image_id": img_id,
                "category_id": cat_ids[img_id - 1],
                "bbox": [10.0, 10.0, 40.0, 40.0],
                "area": 40.0 * 40.0,
                "iscrowd": 0,
            }
            for img_id in range(1, len(cat_ids) + 1)
        ],
    }
    gt_path = tmp_path / "gappy_gt.json"
    gt_path.write_text(json.dumps(coco))

    inputs = [
        {"image_id": img_id, "height": 64, "width": 64} for img_id in range(1, len(cat_ids) + 1)
    ]
    outputs = []
    for i in range(len(cat_ids)):
        inst = Instances(image_size=(64, 64))
        inst.pred_boxes = Boxes(torch.tensor([[10.0, 10.0, 50.0, 50.0]]))
        inst.scores = torch.tensor([0.99])
        # Contiguous index i corresponds to COCO category_id cat_ids[i].
        inst.pred_classes = torch.tensor([i])
        outputs.append({"instances": inst})

    evaluator = COCOEvaluator(gt_path)
    evaluator.reset()
    evaluator.process(inputs, outputs)
    metrics = evaluator.evaluate()
    assert metrics["bbox"]["AP"] == pytest.approx(1.0, abs=1e-3)


def test_coco_evaluator_no_predictions_yields_empty_dict(toy_coco_gt: Path) -> None:
    evaluator = COCOEvaluator(toy_coco_gt)
    evaluator.reset()
    # Empty Instances per image → process appends nothing.
    inputs = [{"image_id": 1, "height": 64, "width": 64}]
    inst = Instances(image_size=(64, 64))
    inst.pred_boxes = Boxes(torch.zeros(0, 4))
    inst.scores = torch.zeros(0)
    inst.pred_classes = torch.zeros(0, dtype=torch.long)
    evaluator.process(inputs, [{"instances": inst}])
    assert evaluator.evaluate() == {}


def test_coco_evaluator_writes_results_json_when_output_dir_set(
    toy_coco_gt: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "eval"
    evaluator = COCOEvaluator(toy_coco_gt, output_dir=out_dir)
    evaluator.reset()
    inputs, outputs = _perfect_outputs(with_mask=False, with_kp=False)
    evaluator.process(inputs, outputs)
    evaluator.evaluate()
    written = out_dir / "coco_instances_results.json"
    assert written.exists()
    parsed = json.loads(written.read_text())
    assert isinstance(parsed, list) and len(parsed) == 2  # one record per image
