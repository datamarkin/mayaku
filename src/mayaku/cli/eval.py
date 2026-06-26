"""``mayaku eval`` — run COCO mAP evaluation against a trained checkpoint.

Loads a config + checkpoint, builds a Predictor-style data loader from
the COCO ground-truth JSON + image directory, runs
:func:`mayaku.engine.inference_on_dataset` with a
:class:`mayaku.engine.COCOEvaluator`, and prints / returns the per-task
metrics dict.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from mayaku.backends.mps import apply_mps_environment
from mayaku.cli._factory import build_resize_augmentation, load_detector
from mayaku.data import (
    DatasetMapper,
    InferenceSampler,
    SerializedList,
    build_coco_metadata,
    load_coco_json,
    trivial_batch_collator,
)
from mayaku.engine import COCOEvaluator, inference_on_dataset

__all__ = ["run_eval"]


class _LazyMappedDataset(Dataset[dict[str, Any]]):
    """Map-style dataset that runs the mapper on demand.

    Avoids the ~50 GB RAM blow-up from eagerly materialising every
    val image as a float32 tensor before iteration starts. The
    DataLoader requests one index at a time and the mapper only ever
    has one decoded image in flight.
    """

    def __init__(self, dicts: Sequence[dict[str, Any]], mapper: DatasetMapper) -> None:
        self._dicts = dicts
        self._mapper = mapper

    def __len__(self) -> int:
        return len(self._dicts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._mapper(self._dicts[idx])


def run_eval(
    weights: Path | str,
    *,
    coco_gt_json: Path,
    image_root: Path,
    output_dir: Path | None = None,
    device: str | None = None,
    backbone_mlpackage: Path | None = None,
    coreml_compute_units: str = "CPU_AND_GPU",
    backbone_onnx: Path | None = None,
    onnx_providers: str | None = None,
) -> dict[str, Any]:
    """Run COCO evaluation.

    ``weights`` is a trained ``.pth`` (or a bundled model name); its embedded
    sidecar defines the architecture — the same resolved config that produced
    the checkpoint, so eval matches training without a separate config file.
    """
    cfg, model = load_detector(weights)
    # The model's authoritative class identity (contiguous index → name) lives in
    # the checkpoint sidecar; the evaluator decodes predictions to the GT's
    # category_id by name, so a GT split that numbers its categories differently
    # can't silently misalign AP (C7).
    from mayaku.cli._weights import resolve_weights
    from mayaku.utils.checkpoint import class_names_from_checkpoint

    weights_path = resolve_weights(weights)
    class_names = class_names_from_checkpoint(weights_path) if weights_path is not None else None
    if device is not None:
        if device == "mps":
            apply_mps_environment()
        model = model.to(torch.device(device))

    if backbone_mlpackage is not None:
        # Hybrid eval: CoreML for backbone+FPN, PyTorch for RPN/ROI/postprocess.
        # The exported `.mlpackage` is a fixed-shape graph; CoreMLBackbone
        # pads each input up to the export shape and crops the FPN
        # outputs back so the rest of the model sees its expected dict.
        # Square (1344, 1344) covers both landscape and portrait
        # orientations — ResizeShortestEdge can produce either, with
        # the long edge ≤ max_size_test (1333), padded up to 1344.
        from mayaku.inference.export import CoreMLBackbone

        prev_size_div = getattr(model.backbone, "size_divisibility", 32)
        coreml_backbone = CoreMLBackbone(
            backbone_mlpackage,
            input_height=1344,
            input_width=1344,
            size_divisibility=prev_size_div,
            compute_units=coreml_compute_units,
        )
        model.backbone = coreml_backbone
        print(
            f"[eval] using CoreML backbone from {backbone_mlpackage} "
            f"(compute_units={coreml_compute_units})",
            flush=True,
        )

    if backbone_onnx is not None:
        # Same hybrid pattern as the CoreML branch above. ONNX Runtime
        # is cross-platform; provider selection drives where the
        # backbone+FPN actually executes.
        from mayaku.inference.export import ONNXBackbone

        prev_size_div = getattr(model.backbone, "size_divisibility", 32)
        providers = (
            tuple(p.strip() for p in onnx_providers.split(",") if p.strip())
            if onnx_providers
            else None
        )
        onnx_backbone = ONNXBackbone(
            backbone_onnx,
            input_height=1344,
            input_width=1344,
            size_divisibility=prev_size_div,
            providers=providers,
        )
        model.backbone = onnx_backbone
        print(
            f"[eval] using ONNX backbone from {backbone_onnx} "
            f"(active_providers={onnx_backbone.active_providers})",
            flush=True,
        )

    metadata = build_coco_metadata(name="cli_eval", json_path=coco_gt_json)
    # Inference mapper drops annotations (is_train=False). The COCO
    # ground-truth itself is loaded separately by COCOEvaluator, so
    # there's no consumer of polygons/keypoints in the eval path.
    dataset_dicts = load_coco_json(
        coco_gt_json,
        image_root,
        metadata,
        keep_segmentation=False,
        keep_keypoints=False,
    )

    # Letterbox eval measures at the fixed deploy size (size_budget); the mapper
    # records the transform so the evaluator un-letterboxes.
    mapper = DatasetMapper(
        [build_resize_augmentation(cfg, for_train=False)],
        is_train=False,
        keypoint_on=cfg.model.meta_architecture == "keypoint_rcnn",
        metadata=metadata if cfg.model.meta_architecture == "keypoint_rcnn" else None,
        deepcopy_input=False,
    )
    # Lazy mapping — eagerly applying the mapper materialises every
    # decoded image as a float32 tensor (~10 MB each on COCO val2017).
    # For 5k images that's ~50 GB pinned in RAM. The DataLoader pulls
    # one at a time on demand instead.
    mapped = _LazyMappedDataset(SerializedList(dataset_dicts), mapper)
    sampler = InferenceSampler(len(mapped))
    loader: DataLoader[Any] = DataLoader(
        mapped,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        collate_fn=trivial_batch_collator,
    )

    evaluator = COCOEvaluator(coco_gt_json, output_dir=output_dir, class_names=class_names)
    metrics = inference_on_dataset(model, loader, evaluator)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
