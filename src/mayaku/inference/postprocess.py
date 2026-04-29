"""Per-image postprocessing: rescale predictions to original image coords.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.9 (`detector_postprocess`):

* Scale ``pred_boxes`` from network-input pixel coords back to the
  original image size.
* Clip the rescaled boxes to the image canvas; drop any that collapsed
  to zero area.
* If ``pred_masks`` is present, paste the per-RoI ``(R, 1, M, M)`` soft
  masks back to ``(R, output_h, output_w)`` bool bitmaps via
  :class:`mayaku.structures.masks.ROIMasks`.
* If ``pred_keypoints`` is present, scale their ``(x, y)`` into the
  output coordinate system.

Lives in :mod:`mayaku.inference` because it's pure post-graph plumbing
— no learned parameters, no autocast — and it's what the predictor
(Step 16) and the evaluator (Step 15) both consume. Kept out of the
exported graph entirely so ONNX / CoreML / OpenVINO targets stay
clean (`BACKEND_PORTABILITY_REPORT.md` §6).
"""

from __future__ import annotations

import torch

from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances
from mayaku.structures.masks import ROIMasks

__all__ = ["detector_postprocess"]


def detector_postprocess(
    instances: Instances,
    output_height: int,
    output_width: int,
    *,
    mask_threshold: float = 0.5,
) -> Instances:
    """Return a new :class:`Instances` rescaled to ``(output_height,
    output_width)``.

    The input ``instances.image_size`` is the *network input* size
    (post-resize, pre-padding); ``output_height`` / ``output_width`` are
    the *original image* dimensions the caller wants predictions in.
    """
    if not instances.has("pred_boxes"):
        # Nothing to rescale; just stamp the new image_size.
        out = Instances(image_size=(output_height, output_width))
        for k, v in instances.get_fields().items():
            out.set(k, v)
        return out

    in_h, in_w = instances.image_size
    scale_x = float(output_width) / float(in_w)
    scale_y = float(output_height) / float(in_h)

    pred_boxes = instances.pred_boxes
    assert isinstance(pred_boxes, Boxes)
    rescaled = pred_boxes.clone()
    rescaled.scale(scale_x, scale_y)
    rescaled.clip((output_height, output_width))

    # Drop instances whose boxes collapsed to zero area after clipping.
    # Matches Detectron2's `detector_postprocess` (`postprocessing.py`),
    # which calls `results = results[output_boxes.nonempty()]` right
    # after the clip. Zero-area boxes otherwise consume slots in the
    # top-`detections_per_image` cap and count as FPs at every IoU.
    keep = rescaled.nonempty()

    out = Instances(image_size=(output_height, output_width))
    out.pred_boxes = Boxes(rescaled.tensor[keep])

    # Carry simple per-instance fields, filtered to the kept set.
    for name in ("scores", "pred_classes"):
        if instances.has(name):
            out.set(name, instances.get(name)[keep])

    if instances.has("pred_masks"):
        masks = instances.pred_masks
        assert isinstance(masks, torch.Tensor)
        # The mask head emits (R, 1, M, M) soft masks. Drop the K=1 dim
        # so ROIMasks sees its expected (N, M, M). Filter to the kept
        # set before pasting so we don't waste work on collapsed boxes.
        kept_masks = masks[keep][:, 0]
        roi_masks = ROIMasks(kept_masks)
        bitmasks = roi_masks.to_bitmasks(
            out.pred_boxes.tensor,
            output_height,
            output_width,
            threshold=mask_threshold,
        )
        out.pred_masks = bitmasks.tensor

    if instances.has("pred_keypoints"):
        kp = instances.pred_keypoints
        assert isinstance(kp, torch.Tensor)
        # (R, K, 3) — columns 0,1 are x,y; column 2 is the score.
        rescaled_kp = kp[keep].clone()
        rescaled_kp[..., 0] *= scale_x
        rescaled_kp[..., 1] *= scale_y
        out.pred_keypoints = rescaled_kp
        # Pass through the raw heatmaps if the keypoint head emitted them.
        if instances.has("pred_keypoint_heatmaps"):
            out.set("pred_keypoint_heatmaps", instances.pred_keypoint_heatmaps[keep])

    return out
