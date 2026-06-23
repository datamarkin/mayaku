"""``mayaku predict`` — run a trained detector on a single image.

Reads the architecture from the checkpoint's embedded sidecar, builds the
detector, loads its weights, runs :class:`Predictor` over ``image_path`` and
prints the per-instance results (or writes them to ``--output`` as JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mayaku.inference import from_pretrained
from mayaku.structures.boxes import BoxMode
from mayaku.structures.instances import Instances

__all__ = ["run_predict"]


def run_predict(
    weights: Path | str,
    image_path: Path,
    *,
    output: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run inference and return a JSON-friendly dict of detections.

    ``weights`` is a trained ``.pth`` (or a bundled model name); its embedded
    sidecar defines the architecture. The returned dict matches what gets
    written to ``output`` so the same payload can be inspected
    programmatically. Useful in tests.
    """
    predictor = from_pretrained(weights, device=device if device is not None else "auto")
    instances = predictor(image_path)

    payload = {
        "image": str(image_path),
        "instances": _instances_to_payload(instances),
    }
    if output is not None:
        output.write_text(json.dumps(payload, indent=2))
    return payload


def _instances_to_payload(inst: Instances) -> list[dict[str, Any]]:
    """Compact dict-per-detection rendering. Skips heavy mask/heatmap fields."""
    if len(inst) == 0:
        return []
    boxes = inst.pred_boxes.tensor.detach().cpu()
    boxes_xywh = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).tolist()
    scores = inst.scores.detach().cpu().tolist()
    classes = inst.pred_classes.detach().cpu().tolist()
    out: list[dict[str, Any]] = []
    for i in range(len(inst)):
        out.append(
            {
                "category_id": int(classes[i]),
                "score": float(scores[i]),
                "bbox_xywh": [float(v) for v in boxes_xywh[i]],
            }
        )
    return out
