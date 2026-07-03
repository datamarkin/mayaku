"""Score a loaded predictor against a COCO split — the shared eval loop.

:func:`run_eval` drives images through a ``from_pretrained`` predictor's
``__call__(image) -> Instances`` contract and feeds a
:class:`~mayaku.engine.COCOEvaluator`. That contract is shared by
:class:`~mayaku.inference.Predictor` (a ``.pth`` checkpoint) and
:class:`~mayaku.inference.ArtifactPredictor` (an exported artifact), so a
checkpoint and its ONNX/CoreML/OpenVINO/TensorRT exports all score through one
path — each with its own as-deployed preprocessing and precision.

The public entry is :func:`mayaku.evaluate`; :func:`mayaku.train`'s final eval
routes through it too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mayaku.data import build_coco_metadata, load_coco_json
from mayaku.engine import COCOEvaluator

if TYPE_CHECKING:
    from mayaku.inference import ArtifactPredictor, Predictor

__all__ = ["run_eval"]


def run_eval(
    predictor: Predictor | ArtifactPredictor,
    *,
    coco_gt_json: Path,
    image_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a loaded predictor against a COCO ground-truth split.

    ``predictor`` is any :func:`mayaku.from_pretrained` result (checkpoint or
    artifact); it returns instances in original-image coordinates, so — with no
    letterbox transform on the records — :class:`COCOEvaluator` serialises them
    straight against the ground truth.

    Returns the per-task metrics dict; also written to
    ``<output_dir>/metrics.json`` when ``output_dir`` is set.
    """
    metadata = build_coco_metadata(name="cli_eval", json_path=coco_gt_json)
    dataset_dicts = load_coco_json(
        coco_gt_json, image_root, metadata, keep_segmentation=False, keep_keypoints=False
    )
    evaluator = COCOEvaluator(coco_gt_json, output_dir=output_dir, class_names=predictor.class_names)
    evaluator.reset()
    total = len(dataset_dicts)
    for i, record in enumerate(dataset_dicts, start=1):
        instances = predictor(record["file_name"])
        evaluator.process([record], [{"instances": instances}])
        if i % 100 == 0 or i == total:
            print(f"[eval] {i}/{total}", flush=True)
    metrics = evaluator.evaluate()
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
