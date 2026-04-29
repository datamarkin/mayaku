"""Axis-aligned bounding boxes.

The :class:`Boxes` container stores ``(N, 4)`` tensors in ``XYXY_ABS``
format (absolute pixel coordinates, ``(x0, y0, x1, y1)``). This is the
canonical internal representation used by every detection head; data
loaders convert from COCO's ``XYWH_ABS`` at the ingest boundary via
:func:`BoxMode.convert`.

Rotated boxes (``XYWHA_ABS``) are intentionally absent — they are out of
scope (`BACKEND_PORTABILITY_REPORT.md` §3, master plan non-goals) and
were the only sites in Detectron2 that required ``float64`` arithmetic
on the hot path. Keeping all conversions in fp32 means
:class:`Boxes` is safe under autocast on every backend without special
guards.

The IoU helpers follow the standard area-of-intersection / area-of-union
formulation. They return a dense ``(N, M)`` matrix; pairwise rather than
batched-paired form because every Detectron2-style matcher consumes the
full matrix and indexes into it (see ``proposal_utils.py`` and
``matcher.py`` in the upstream reference).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import IntEnum

import torch
from torch import Tensor

__all__ = [
    "BoxMode",
    "Boxes",
    "pairwise_intersection",
    "pairwise_ioa",
    "pairwise_iou",
]


class BoxMode(IntEnum):
    """Supported box coordinate conventions.

    The rotated box mode (``XYWHA_ABS`` in Detectron2) is not included;
    rotated detection is out of scope (see module docstring).

    * ``XYXY_ABS`` — ``(x0, y0, x1, y1)`` in absolute pixel coordinates.
      Internal canonical form.
    * ``XYWH_ABS`` — ``(x0, y0, w, h)`` in absolute pixel coordinates.
      COCO's on-disk format.
    * ``XYXY_REL`` / ``XYWH_REL`` — same axis layout but coordinates
      normalized to ``[0, 1]`` of image width/height. Provided for
      callers that ingest from formats that store boxes that way; needs
      ``image_size=(h, w)`` to round-trip.
    """

    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3

    @staticmethod
    def convert(
        box: Tensor,
        from_mode: BoxMode,
        to_mode: BoxMode,
        image_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Convert a ``(..., 4)`` tensor between two non-rotated modes.

        ``image_size`` is ``(h, w)`` and only consulted when one side of
        the conversion is a ``_REL`` mode. The output dtype matches the
        input; we do not upcast to fp64 (rotated paths required it; the
        non-rotated math is exact in fp32 for any realistic box).
        """
        if from_mode == to_mode:
            return box.clone()

        # Stage 1: lift to XYXY_ABS, the internal pivot.
        b = _to_xyxy_abs(box, from_mode, image_size)
        # Stage 2: lower from XYXY_ABS to the requested mode.
        return _from_xyxy_abs(b, to_mode, image_size)


def _require_image_size(image_size: tuple[int, int] | None, mode: BoxMode) -> tuple[int, int]:
    if image_size is None:
        raise ValueError(f"image_size=(h, w) is required to convert {mode.name}")
    return image_size


def _to_xyxy_abs(box: Tensor, mode: BoxMode, image_size: tuple[int, int] | None) -> Tensor:
    if mode == BoxMode.XYXY_ABS:
        return box.clone()
    if mode == BoxMode.XYWH_ABS:
        x0, y0, w, h = box.unbind(-1)
        return torch.stack([x0, y0, x0 + w, y0 + h], dim=-1)
    if mode == BoxMode.XYXY_REL:
        h_img, w_img = _require_image_size(image_size, mode)
        scale = box.new_tensor([w_img, h_img, w_img, h_img])
        return box * scale
    # XYWH_REL
    h_img, w_img = _require_image_size(image_size, mode)
    x0, y0, w, h = box.unbind(-1)
    return torch.stack([x0 * w_img, y0 * h_img, (x0 + w) * w_img, (y0 + h) * h_img], dim=-1)


def _from_xyxy_abs(box: Tensor, mode: BoxMode, image_size: tuple[int, int] | None) -> Tensor:
    if mode == BoxMode.XYXY_ABS:
        return box.clone()
    if mode == BoxMode.XYWH_ABS:
        x0, y0, x1, y1 = box.unbind(-1)
        return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)
    if mode == BoxMode.XYXY_REL:
        h_img, w_img = _require_image_size(image_size, mode)
        scale = box.new_tensor([w_img, h_img, w_img, h_img])
        return box / scale
    # XYWH_REL
    h_img, w_img = _require_image_size(image_size, mode)
    x0, y0, x1, y1 = box.unbind(-1)
    return torch.stack([x0 / w_img, y0 / h_img, (x1 - x0) / w_img, (y1 - y0) / h_img], dim=-1)


class Boxes:
    """A wrapper around a ``(N, 4)`` xyxy float tensor.

    The tensor is exposed as the public ``.tensor`` attribute; methods
    return new :class:`Boxes` instances so callers can chain operations
    without mutating shared state. Indexing with an int, slice, bool
    mask, or long index tensor is supported and forwards to the
    underlying tensor.
    """

    def __init__(self, tensor: Tensor) -> None:
        if tensor.numel() == 0:
            tensor = tensor.reshape(-1, 4)
        if tensor.dim() != 2 or tensor.shape[-1] != 4:
            raise ValueError(f"Boxes expects a (N, 4) tensor, got shape {tuple(tensor.shape)}")
        self.tensor: Tensor = tensor

    # --- conversion / movement --------------------------------------------

    def to(self, device: torch.device | str) -> Boxes:
        return Boxes(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def clone(self) -> Boxes:
        return Boxes(self.tensor.clone())

    # --- geometry ---------------------------------------------------------

    def area(self) -> Tensor:
        """Per-box area, shape ``(N,)``. Negative widths/heights clamp to 0."""
        wh = (self.tensor[:, 2:] - self.tensor[:, :2]).clamp(min=0)
        return wh[:, 0] * wh[:, 1]

    def clip(self, image_size: tuple[int, int]) -> None:
        """Clip boxes in-place to fit inside ``(h, w)``."""
        h, w = image_size
        self.tensor[:, 0::2].clamp_(min=0, max=w)
        self.tensor[:, 1::2].clamp_(min=0, max=h)

    def scale(self, scale_x: float, scale_y: float) -> None:
        """Scale x/y coordinates in place."""
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    def nonempty(self, threshold: float = 0.0) -> Tensor:
        """Bool mask of boxes whose width *and* height exceed ``threshold``."""
        widths = self.tensor[:, 2] - self.tensor[:, 0]
        heights = self.tensor[:, 3] - self.tensor[:, 1]
        return (widths > threshold) & (heights > threshold)

    def inside_box(self, image_size: tuple[int, int], threshold: int = 0) -> Tensor:
        """Bool mask of boxes whose every corner lies inside the image,
        with ``threshold`` pixels of slack around each side."""
        h, w = image_size
        t = self.tensor
        return (
            (t[:, 0] >= -threshold)
            & (t[:, 1] >= -threshold)
            & (t[:, 2] <= w + threshold)
            & (t[:, 3] <= h + threshold)
        )

    def get_centers(self) -> Tensor:
        """``(N, 2)`` (cx, cy) centers."""
        return (self.tensor[:, :2] + self.tensor[:, 2:]) * 0.5

    # --- container protocol ----------------------------------------------

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, item: int | slice | Tensor) -> Boxes:
        if isinstance(item, int):
            # Detectron2 returns a 1-row Boxes for int indexing.
            return Boxes(self.tensor[item : item + 1])
        return Boxes(self.tensor[item])

    def __iter__(self) -> Iterator[Tensor]:
        # Yields per-box (4,) tensors — same as iterating .tensor.
        yield from self.tensor

    def __repr__(self) -> str:
        return f"Boxes({self.tensor!r})"

    @classmethod
    def cat(cls, boxes_list: Sequence[Boxes]) -> Boxes:
        """Concatenate a sequence of :class:`Boxes` along ``dim=0``."""
        if len(boxes_list) == 0:
            return cls(torch.zeros(0, 4))
        tensors = [b.tensor for b in boxes_list]
        return cls(torch.cat(tensors, dim=0))


# ---------------------------------------------------------------------------
# Pairwise IoU / IoA
# ---------------------------------------------------------------------------


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> Tensor:
    """Pairwise intersection area, shape ``(N, M)``."""
    a = boxes1.tensor
    b = boxes2.tensor
    # Broadcast to (N, M, 2) for the top-left max and bottom-right min.
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> Tensor:
    """Intersection-over-union matrix, shape ``(N, M)``.

    Empty boxes (zero area) yield an IoU of 0 against every other box,
    matching the Detectron2 convention used by :class:`Matcher`.
    """
    inter = pairwise_intersection(boxes1, boxes2)
    a1 = boxes1.area()
    a2 = boxes2.area()
    union = a1[:, None] + a2[None, :] - inter
    # Avoid 0/0 — emit 0 where union is 0 (i.e. both boxes empty).
    iou = torch.where(
        union > 0, inter / union.clamp(min=torch.finfo(inter.dtype).eps), inter.new_zeros(())
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> Tensor:
    """Intersection-over-area-of-``boxes2``, shape ``(N, M)``.

    Useful when assessing how much of each anchor falls inside the
    ground-truth, regardless of anchor area. Symmetric to ``pairwise_iou``
    where the denominator is fixed to the area of the second argument.
    """
    inter = pairwise_intersection(boxes1, boxes2)
    a2 = boxes2.area()
    return torch.where(
        a2[None, :] > 0,
        inter / a2[None, :].clamp(min=torch.finfo(inter.dtype).eps),
        inter.new_zeros(()),
    )
