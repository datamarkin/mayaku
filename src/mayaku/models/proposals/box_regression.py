"""Faster R-CNN box-delta encoding (`Box2BoxTransform`).

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.3
(``modeling/box_regression.py:20-116``). Encodes
``(src_box, target_box)`` pairs as ``(dx, dy, dw, dh)`` with per-axis
weights, and decodes the inverse operation. The same class is used by
the RPN (``weights=(1, 1, 1, 1)``) and by the box head
(``weights=(10, 10, 5, 5)``).

Two safety details from the reference, preserved here:

* ``apply_deltas`` casts deltas (and box centers / sizes) to float32
  regardless of input dtype, because under autocast the prediction
  tensor is fp16 / bf16 and the ``exp(dw)`` / ``exp(dh)`` math becomes
  unstable for large deltas (`box_regression.py:88`).
* ``dw`` and ``dh`` are clamped to ``ln(1000 / 16) ≈ 4.135`` before
  applying ``exp`` to bound the post-decode size at ~1000 pixels even
  on rare unstable predictions (`box_regression.py:14, 102-104`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["Box2BoxTransform"]

# Detectron2 default — exp(this) ≈ 62.5, so a 16-pixel anchor decodes to
# at most ~1000 pixels even with a wildly out-of-distribution delta.
_DEFAULT_SCALE_CLAMP: float = math.log(1000.0 / 16.0)


class Box2BoxTransform:
    """Encode/decode XYXY box pairs as weighted deltas.

    Args:
        weights: ``(wx, wy, ww, wh)`` per-axis multipliers. RPN uses
            ``(1, 1, 1, 1)``; the box head uses ``(10, 10, 5, 5)``
            (`spec §6.1`).
        scale_clamp: Maximum value of ``ww * dw`` and ``wh * dh`` after
            decoding the per-axis multiplier. Defaults to
            ``ln(1000/16)`` per the reference.
    """

    def __init__(
        self,
        weights: tuple[float, float, float, float],
        scale_clamp: float = _DEFAULT_SCALE_CLAMP,
    ) -> None:
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        """Encode ``target_boxes`` relative to ``src_boxes`` as deltas.

        Args:
            src_boxes: ``(N, 4)`` XYXY (e.g. anchors or proposals).
            target_boxes: ``(N, 4)`` XYXY ground-truth boxes paired with
                each source.

        Returns:
            ``(N, 4)`` ``(dx, dy, dw, dh)`` tensor on the same device
            and dtype as ``src_boxes``.
        """
        if src_boxes.shape != target_boxes.shape:
            raise ValueError(
                f"src/target box shape mismatch: {src_boxes.shape} vs {target_boxes.shape}"
            )
        src_w = src_boxes[:, 2] - src_boxes[:, 0]
        src_h = src_boxes[:, 3] - src_boxes[:, 1]
        src_cx = src_boxes[:, 0] + 0.5 * src_w
        src_cy = src_boxes[:, 1] + 0.5 * src_h

        tgt_w = target_boxes[:, 2] - target_boxes[:, 0]
        tgt_h = target_boxes[:, 3] - target_boxes[:, 1]
        tgt_cx = target_boxes[:, 0] + 0.5 * tgt_w
        tgt_cy = target_boxes[:, 1] + 0.5 * tgt_h

        # D2 guard: if a degenerate box slips past the DatasetMapper's
        # filter_empty_instances, log(tgt_w/0) returns -inf and silently
        # NaN-s training. Fail loud instead.
        assert (src_w > 0).all().item() and (src_h > 0).all().item(), (
            "Box2BoxTransform.get_deltas received zero-area src boxes; check "
            "that filter_empty_instances was applied in the DatasetMapper."
        )

        wx, wy, ww, wh = self.weights
        dx = wx * (tgt_cx - src_cx) / src_w
        dy = wy * (tgt_cy - src_cy) / src_h
        dw = ww * torch.log(tgt_w / src_w)
        dh = wh * torch.log(tgt_h / src_h)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def apply_deltas(self, deltas: Tensor, boxes: Tensor) -> Tensor:
        """Decode ``deltas`` against ``boxes`` to produce ``(N, 4)`` XYXY.

        ``deltas`` is reshaped under the hood so callers can pass either
        ``(N, 4)`` (single class) or ``(N, K*4)`` (per-class). The
        per-class form returns ``(N, K*4)``.
        """
        # FP32 cast for safety under autocast (`spec §2.3`).
        deltas = deltas.float()
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        # Stride-4 view: column 0 is dx for every prediction, column 1 dy, etc.
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        # Stack in stride-4 column layout so output column 0 is x1 of
        # class 0, column 1 is y1 of class 0, etc.
        out = torch.stack([x1, y1, x2, y2], dim=2).reshape(deltas.shape[0], deltas.shape[1])
        return out
