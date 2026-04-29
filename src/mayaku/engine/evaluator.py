"""Evaluation loop + COCO mAP evaluator.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §4.4 + §7.7
(`evaluation/evaluator.py`, `evaluation/coco_evaluation.py`):

* :class:`DatasetEvaluator` — three-method protocol
  (``reset`` / ``process`` / ``evaluate``) so different metrics share
  the same inference loop.
* :func:`inference_on_dataset` — the canonical eval loop:
  ``reset → for batch in loader: process(inputs, model(inputs))
   → evaluate``.
* :class:`COCOEvaluator` — auto-detects bbox / segm / keypoints from
  the first prediction and runs ``pycocotools.cocoeval.COCOeval``
  per task. DDP-aware: ranks > 0 contribute their predictions via
  :func:`mayaku.engine.distributed.all_gather_object` and rank 0
  produces the final metrics dict.

The evaluator consumes whatever the model returns (the
``[{"instances": Instances}, ...]`` format from
:class:`mayaku.models.detectors.FasterRCNN`) and an optional
"original image size" per input (from the data mapper's
``height`` / ``width`` keys, or from
:attr:`Instances.image_size` if the caller already postprocessed).
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor, nn

from mayaku.engine.distributed import all_gather_object, is_main_process, synchronize
from mayaku.inference.postprocess import detector_postprocess
from mayaku.structures.boxes import BoxMode
from mayaku.structures.instances import Instances

__all__ = [
    "COCOEvaluator",
    "DatasetEvaluator",
    "inference_on_dataset",
    "instances_to_coco_json",
]


# ---------------------------------------------------------------------------
# Protocol + canonical loop
# ---------------------------------------------------------------------------


class DatasetEvaluator:
    """Default no-op base. Subclasses override ``process`` and ``evaluate``."""

    def reset(self) -> None:
        return None

    def process(
        self,
        inputs: Sequence[dict[str, Any]],
        outputs: Sequence[dict[str, Any]],
    ) -> None:
        return None

    def evaluate(self) -> dict[str, Any]:
        return {}


def _fmt_hms(seconds: float) -> str:
    if not (seconds >= 0) or seconds == float("inf"):
        return "?"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def inference_on_dataset(
    model: nn.Module,
    data_loader: Iterable[Sequence[dict[str, Any]]],
    evaluator: DatasetEvaluator,
    *,
    progress_period: int = 50,
) -> dict[str, Any]:
    """Run ``model`` over ``data_loader`` and feed every batch to the evaluator.

    Mirrors `spec §4.4`. The model is forced into ``eval()`` for the
    duration; the original training/eval mode is restored on exit.

    ``progress_period`` controls a periodic ``[eval] i/total ...`` line
    printed to stdout: every ``progress_period`` batches plus the final
    one when the loader has a known length. Set to ``0`` (or any value
    ``<= 0``) to silence — useful for embedding the loop programmatically.
    """
    was_training = model.training
    model.eval()
    evaluator.reset()
    total: int | None
    try:
        total = len(data_loader)  # type: ignore[arg-type]
    except TypeError:
        total = None
    start = time.monotonic()
    try:
        with torch.no_grad():
            for i, inputs in enumerate(data_loader, start=1):
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    raise TypeError(
                        f"model must return list[dict] in eval mode; got {type(outputs).__name__}"
                    )
                evaluator.process(inputs, outputs)
                if progress_period > 0 and (
                    i % progress_period == 0 or (total is not None and i == total)
                ):
                    elapsed = time.monotonic() - start
                    rate = i / elapsed if elapsed > 0 else 0.0
                    if total is not None:
                        pct = 100.0 * i / total
                        eta = (total - i) / rate if rate > 0 else float("inf")
                        print(
                            f"[eval] {i}/{total} ({pct:.1f}%) — "
                            f"{rate:.1f} it/s — ETA {_fmt_hms(eta)}",
                            flush=True,
                        )
                    else:
                        print(f"[eval] {i} batches — {rate:.1f} it/s", flush=True)
        return evaluator.evaluate() or {}
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# COCOEvaluator
# ---------------------------------------------------------------------------


_COCO_TASKS = ("bbox", "segm", "keypoints")


class COCOEvaluator(DatasetEvaluator):
    """COCO mAP for boxes (always), masks (if present), keypoints (if present).

    Args:
        coco_gt_json: Path to a COCO-format ground-truth JSON for the
            dataset being evaluated. The evaluator is dataset-agnostic
            — pass the GT for the loader you ran ``inference_on_dataset``
            against.
        tasks: Optional explicit task list. Defaults to auto-detection
            from the first prediction (bbox always; segm if any
            prediction carries ``pred_masks``; keypoints if any carries
            ``pred_keypoints``).
        output_dir: Optional directory to dump the raw
            ``coco_instances_results.json`` to. Skipped on non-main ranks.
        kpt_oks_sigmas: 17 OKS sigmas for COCO Person keypoints. Defaults
            to the standard COCO Person sigmas; pass a tuple for custom
            datasets.
    """

    # Standard COCO Person OKS sigmas (`pycocotools.cocoeval` defaults).
    DEFAULT_KPT_OKS_SIGMAS: tuple[float, ...] = (
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    )

    def __init__(
        self,
        coco_gt_json: str | Path,
        *,
        tasks: Sequence[str] | None = None,
        output_dir: str | Path | None = None,
        kpt_oks_sigmas: Sequence[float] | None = None,
    ) -> None:
        self.coco_gt_json = str(coco_gt_json)
        self.tasks = tuple(tasks) if tasks is not None else None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.kpt_oks_sigmas = tuple(kpt_oks_sigmas or self.DEFAULT_KPT_OKS_SIGMAS)
        self._predictions: list[dict[str, Any]] = []
        # Defer COCO-GT load until evaluate() to keep reset/process
        # cheap and to avoid IO on workers that won't run COCOeval.
        self._coco_gt: COCO | None = None
        # Reverse of the dataloader's contiguous remap (sorted cat_ids
        # → 0..K-1). Lazily built from the GT JSON on first call.
        self._class_id_map: dict[int, int] | None = None

    # ------------------------------------------------------------------
    # Loop hooks
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._predictions = []

    def process(
        self,
        inputs: Sequence[dict[str, Any]],
        outputs: Sequence[dict[str, Any]],
    ) -> None:
        for inp, out in zip(inputs, outputs, strict=True):
            instances = out.get("instances")
            if instances is None or len(instances) == 0:
                continue
            assert isinstance(instances, Instances)
            # Rescale to the original image size when we have it.
            output_h = int(inp.get("height", instances.image_size[0]))
            output_w = int(inp.get("width", instances.image_size[1]))
            if (output_h, output_w) != instances.image_size:
                instances = detector_postprocess(instances, output_h, output_w)
            json_records = list(
                instances_to_coco_json(
                    instances,
                    int(inp["image_id"]),
                    class_id_map=self._get_class_id_map(),
                )
            )
            self._predictions.extend(json_records)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        # DDP gather: each rank contributed predictions to its own list;
        # rank 0 stitches them together and runs COCOeval.
        synchronize()
        gathered = all_gather_object(self._predictions)
        merged: list[dict[str, Any]] = [r for chunk in gathered for r in chunk]

        if not is_main_process():
            return {}

        if not merged:
            return {}

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "coco_instances_results.json").write_text(json.dumps(merged))

        tasks = self.tasks if self.tasks is not None else _detect_tasks(merged)
        results: dict[str, Any] = {}
        coco_gt = self._load_gt()
        for task in tasks:
            metrics = _evaluate_one_task(
                coco_gt,
                merged,
                task,
                kpt_oks_sigmas=self.kpt_oks_sigmas,
            )
            results[task] = metrics
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_gt(self) -> COCO:
        if self._coco_gt is None:
            # Silence pycocotools' chatty constructor.
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_gt = COCO(self.coco_gt_json)
        return self._coco_gt

    def _get_class_id_map(self) -> dict[int, int]:
        """Contiguous-class-index → original COCO ``category_id``.

        Mirrors the dataloader's ``sorted(coco.cats.keys())`` remap
        (`src/mayaku/data/datasets/coco.py:_build_metadata_from_coco`),
        so the model's contiguous class predictions get translated back
        to the integers ``COCOeval`` expects.
        """
        if self._class_id_map is None:
            cats = self._load_gt().cats
            self._class_id_map = {i: cid for i, cid in enumerate(sorted(cats.keys()))}
        return self._class_id_map


# ---------------------------------------------------------------------------
# Instances → COCO JSON
# ---------------------------------------------------------------------------


def instances_to_coco_json(
    instances: Instances,
    image_id: int,
    *,
    class_id_map: dict[int, int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield one COCO-format detection per instance.

    Mirrors `spec §7.7`. ``pred_boxes`` are converted XYXY → XYWH;
    ``pred_masks`` (bool ``(N, H, W)``) become RLE-encoded
    ``segmentation`` dicts via ``pycocotools.mask.encode``;
    ``pred_keypoints`` are flattened to ``[x, y, v=2, ...]`` per
    keypoint with the ``+0.5`` ingest shift undone.

    ``class_id_map`` reverses the contiguous index the model predicts
    back to the original COCO ``category_id`` (e.g. ``0 → 1`` for
    ``person`` in COCO2017). When ``None``, predicted classes are
    written through unchanged — appropriate when GT category ids are
    already contiguous from 0.
    """
    if len(instances) == 0:
        return
    boxes = instances.pred_boxes.tensor.detach().cpu()
    boxes_xywh = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()
    classes = instances.pred_classes.detach().cpu().tolist()
    scores = instances.scores.detach().cpu().tolist()

    has_mask = instances.has("pred_masks")
    has_kp = instances.has("pred_keypoints")
    masks: Tensor | None = instances.pred_masks.detach().cpu() if has_mask else None
    keypoints: Tensor | None = instances.pred_keypoints.detach().cpu() if has_kp else None

    if has_mask:
        # Lazy import — pycocotools.mask is only needed when masks fire,
        # and we already pay for pycocotools at the COCO-GT load.
        from pycocotools import mask as coco_mask
    for k in range(len(instances)):
        cls = int(classes[k])
        if class_id_map is not None:
            # Predictions whose class isn't present in the GT category
            # map can't possibly match any GT, so drop them rather than
            # emit a record with an out-of-vocabulary category_id (which
            # would either KeyError or pollute COCOeval). This typically
            # happens when the model's num_classes is larger than the
            # dataset's category count (e.g. random-init smoke runs).
            if cls not in class_id_map:
                continue
            category_id = class_id_map[cls]
        else:
            category_id = cls
        record: dict[str, Any] = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(v) for v in boxes_xywh[k]],
            "score": float(scores[k]),
        }
        if has_mask:
            assert masks is not None
            mask_np = masks[k].numpy().astype(np.uint8)
            rle = coco_mask.encode(np.asfortranarray(mask_np))
            # encode() returns bytes for `counts`; JSON-friendly str.
            rle["counts"] = rle["counts"].decode("utf-8")
            record["segmentation"] = rle
        if has_kp:
            assert keypoints is not None
            kp = keypoints[k].numpy().copy()  # (K, 3)
            # Undo the +0.5 the data layer added at ingest (spec §5.4 / §7.7).
            kp[:, 0] -= 0.5
            kp[:, 1] -= 0.5
            # COCO eval ignores per-prediction visibility but expects 17*3
            # floats. Set v=2 (visible) uniformly; the evaluator only
            # uses the (x, y) and the per-keypoint OKS sigmas.
            kp[:, 2] = 2.0
            record["keypoints"] = kp.flatten().tolist()
        yield record


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------


def _detect_tasks(predictions: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    tasks: list[str] = ["bbox"]
    if any("segmentation" in p for p in predictions):
        tasks.append("segm")
    if any("keypoints" in p for p in predictions):
        tasks.append("keypoints")
    return tuple(tasks)


def _evaluate_one_task(
    coco_gt: COCO,
    predictions: Sequence[dict[str, Any]],
    task: str,
    *,
    kpt_oks_sigmas: tuple[float, ...],
) -> dict[str, float]:
    """Run ``COCOeval`` for one task; return AP, AP50, AP75, APs/m/l."""
    if task not in _COCO_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {_COCO_TASKS}")

    # Pycocotools' ``loadRes`` takes either a JSON path or a list-of-dicts.
    # Filter to the keys this task actually consumes; otherwise ``loadRes``
    # complains that bbox-only predictions lack ``segmentation`` etc.
    filtered = _filter_task_predictions(predictions, task)
    if not filtered:
        return {}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=task)
        if task == "keypoints":
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # length-12 (or 10 for keypoints) array of AP / AR
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }


def _filter_task_predictions(
    predictions: Sequence[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in predictions:
        if task == "bbox":
            if "bbox" not in p:
                continue
            out.append(
                {
                    "image_id": p["image_id"],
                    "category_id": p["category_id"],
                    "bbox": p["bbox"],
                    "score": p["score"],
                }
            )
        elif task == "segm":
            if "segmentation" not in p:
                continue
            out.append(
                {
                    "image_id": p["image_id"],
                    "category_id": p["category_id"],
                    "segmentation": p["segmentation"],
                    "score": p["score"],
                }
            )
        else:  # keypoints
            if "keypoints" not in p:
                continue
            out.append(
                {
                    "image_id": p["image_id"],
                    "category_id": p["category_id"],
                    "keypoints": p["keypoints"],
                    "score": p["score"],
                }
            )
    return out
