"""``mayaku predict`` — run a trained detector on a single image.

Loads ``config.yaml``, builds the right detector, optionally loads a
trained checkpoint via ``--weights``, runs :class:`Predictor` over
``image_path`` and prints the per-instance results (or writes them to
``--output`` as JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mayaku.cli._factory import build_detector
from mayaku.cli._weights import resolve_weights
from mayaku.config import MayakuConfig, load_yaml
from mayaku.inference import Predictor
from mayaku.structures.boxes import BoxMode
from mayaku.structures.instances import Instances

__all__ = ["run_predict"]


def run_predict(
    config: Path | MayakuConfig,
    image_path: Path,
    *,
    weights: Path | str | None = None,
    output: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run inference and return a JSON-friendly dict of detections.

    ``config`` accepts a YAML path or a constructed
    :class:`MayakuConfig`. The returned dict matches what gets written
    to ``output`` so the same payload can be inspected
    programmatically. Useful in tests.
    """
    cfg = config if isinstance(config, MayakuConfig) else load_yaml(config)
    model = build_detector(cfg)
    weights_path = resolve_weights(weights)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    if device is not None:
        model = model.to(torch.device(device))
    predictor = Predictor.from_config(cfg, model)
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
