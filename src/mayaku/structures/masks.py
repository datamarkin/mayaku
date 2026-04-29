"""Mask containers + paste-back helper.

Three structures, mirroring Detectron2 (`structures/masks.py`):

* :class:`BitMasks` — dense ``(N, H, W)`` boolean tensor; the canonical
  in-memory form once a mask has been rasterised.
* :class:`PolygonMasks` — per-instance list of polygon coordinate
  tensors. The COCO-native annotation format. ``crop_and_resize``
  rasterises into an ``MxM`` boolean grid aligned to a box, ready to
  use as a mask-head training target.
* :class:`ROIMasks` — soft-mask output of the mask head, pre-paste.
  ``to_bitmasks`` pastes each ``MxM`` soft mask back to image
  resolution and binarises it.

We use ``torchvision``'s ``roi_align`` (via :mod:`mayaku.backends.ops`)
for ``BitMasks.crop_and_resize`` and a chunked ``F.grid_sample`` for the
paste-back. The paste-back chunking budget is conservative: the
upstream constant of 1 GB / float was tuned for CUDA in 2019; we keep
it but cap chunk count at the number of masks so an empty input is a
no-op.

Polygon rasterisation goes through ``pycocotools``
(``mask.frPyObjects`` + ``mask.decode``), which is the fastest correct
implementation in the ecosystem and is already a transitive dependency
of every COCO-trained model. It runs CPU-side (host-side rasterisation,
not a hot-path GPU op), which matches `BACKEND_PORTABILITY_REPORT.md`
§5 — no GPU fp64 sites.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from torch import Tensor

from mayaku.backends.ops.roi_align import roi_align

__all__ = [
    "BitMasks",
    "PolygonMasks",
    "ROIMasks",
    "paste_masks_in_image",
]


# ---------------------------------------------------------------------------
# BitMasks
# ---------------------------------------------------------------------------


class BitMasks:
    """Dense bool ``(N, H, W)`` masks tied to a single image size."""

    def __init__(self, tensor: Tensor) -> None:
        if tensor.dim() != 3:
            raise ValueError(f"BitMasks expects (N, H, W); got shape {tuple(tensor.shape)}")
        self.tensor: Tensor = tensor.to(dtype=torch.bool)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def image_size(self) -> tuple[int, int]:
        """``(H, W)`` of the underlying canvas."""
        return int(self.tensor.shape[1]), int(self.tensor.shape[2])

    def to(self, device: torch.device | str) -> BitMasks:
        return BitMasks(self.tensor.to(device))

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, item: int | slice | Tensor) -> BitMasks:
        if isinstance(item, int):
            return BitMasks(self.tensor[item : item + 1])
        return BitMasks(self.tensor[item])

    def __repr__(self) -> str:
        return f"BitMasks(num={len(self)}, image_size={self.image_size})"

    def nonempty(self) -> Tensor:
        """Bool mask of instances with at least one ``True`` pixel."""
        return self.tensor.flatten(1).any(dim=1)

    def crop_and_resize(self, boxes: Tensor, mask_size: int) -> Tensor:
        """Bilinear-sample each mask inside its box at ``MxM``.

        Returns a bool ``(N, M, M)`` tensor. Used to build ``mask_head``
        training targets aligned to the predicted ``28x28`` logits
        (`DETECTRON2_TECHNICAL_SPEC.md` §3.5).
        """
        if boxes.shape[0] != len(self):
            raise ValueError(f"boxes batch {boxes.shape[0]} != BitMasks batch {len(self)}")
        if len(self) == 0:
            return self.tensor.new_zeros((0, mask_size, mask_size), dtype=torch.bool)
        # roi_align expects (N_feat, C, H, W) and (K, 5).
        feat = self.tensor.unsqueeze(1).to(dtype=boxes.dtype)
        idx = torch.arange(len(self), dtype=boxes.dtype, device=boxes.device).unsqueeze(1)
        rois = torch.cat([idx, boxes.to(feat.device)], dim=1)
        sampled = roi_align(
            feat,
            rois,
            (mask_size, mask_size),
            spatial_scale=1.0,
            sampling_ratio=0,
            aligned=True,
        )
        return (sampled[:, 0] >= 0.5).to(dtype=torch.bool)

    @classmethod
    def cat(cls, masks_list: Sequence[BitMasks]) -> BitMasks:
        if len(masks_list) == 0:
            raise ValueError("BitMasks.cat requires at least one element")
        return cls(torch.cat([m.tensor for m in masks_list], dim=0))

    @classmethod
    def from_polygon_masks(cls, polygons: PolygonMasks, height: int, width: int) -> BitMasks:
        """Rasterise each polygon set into a full-resolution bool mask."""
        if len(polygons) == 0:
            return cls(torch.zeros(0, height, width, dtype=torch.bool))
        out = np.zeros((len(polygons), height, width), dtype=np.bool_)
        for i, polys in enumerate(polygons.polygons):
            out[i] = _polygons_to_bitmask(polys, height, width)
        return cls(torch.from_numpy(out))


# ---------------------------------------------------------------------------
# PolygonMasks
# ---------------------------------------------------------------------------


_FloatArray = npt.NDArray[np.float32]
_BoolArray = npt.NDArray[np.bool_]
_PolygonsPerInstance = list[_FloatArray]


class PolygonMasks:
    """Per-instance list of polygon coordinate arrays.

    Each instance is a list of one or more polygons; each polygon is a
    flat ``np.ndarray`` of ``[x0, y0, x1, y1, ...]`` floats. This is the
    COCO on-disk format. Conversion to a bitmap happens lazily —
    typically on demand from :meth:`crop_and_resize` (training) or
    :meth:`BitMasks.from_polygon_masks` (visualisation / evaluation).
    """

    def __init__(self, polygons: Sequence[Sequence[Sequence[float] | _FloatArray]]) -> None:
        self.polygons: list[_PolygonsPerInstance] = [
            [_as_float_array(p) for p in per_instance] for per_instance in polygons
        ]

    def __len__(self) -> int:
        return len(self.polygons)

    def __getitem__(self, item: int | slice | Tensor) -> PolygonMasks:
        if isinstance(item, int):
            return PolygonMasks([self.polygons[item]])
        if isinstance(item, slice):
            return PolygonMasks(self.polygons[item])
        # Assume Tensor of bool or long indices
        if item.dtype == torch.bool:
            indices = torch.nonzero(item, as_tuple=False).flatten().tolist()
        else:
            indices = item.tolist()
        return PolygonMasks([self.polygons[i] for i in indices])

    def __repr__(self) -> str:
        return f"PolygonMasks(num={len(self)})"

    def to(self, device: torch.device | str) -> PolygonMasks:
        # Polygons live as host-side numpy arrays; the rasterised bitmap
        # is what reaches the device. .to is a no-op kept for the
        # Instances.to() loop.
        del device
        return self

    @property
    def device(self) -> torch.device:
        # Polygons are CPU-bound numpy arrays. Report cpu for symmetry
        # with BitMasks; consumers that need GPU bitmaps must call
        # crop_and_resize and use the result.
        return torch.device("cpu")

    @classmethod
    def cat(cls, masks_list: Sequence[PolygonMasks]) -> PolygonMasks:
        if len(masks_list) == 0:
            raise ValueError("PolygonMasks.cat requires at least one element")
        merged: list[_PolygonsPerInstance] = []
        for m in masks_list:
            merged.extend(m.polygons)
        return cls(merged)

    def area(self) -> Tensor:
        """Per-instance polygon area via the shoelace formula.

        Multi-polygon instances sum across components. Returned as a
        ``(N,)`` float tensor.
        """
        out = torch.zeros(len(self))
        for i, polys in enumerate(self.polygons):
            out[i] = sum(_polygon_area(p) for p in polys)
        return out

    def crop_and_resize(self, boxes: Tensor, mask_size: int) -> Tensor:
        """Rasterise each polygon set inside its box at ``MxM``.

        Polygon coordinates are translated into the box-local frame and
        scaled by ``M / box_side``; the rasterisation itself runs on
        CPU via ``pycocotools``. Returns a bool ``(N, M, M)`` tensor on
        the same device as ``boxes``.
        """
        if boxes.shape[0] != len(self):
            raise ValueError(f"boxes batch {boxes.shape[0]} != PolygonMasks batch {len(self)}")
        if len(self) == 0:
            return torch.zeros((0, mask_size, mask_size), dtype=torch.bool, device=boxes.device)

        boxes_cpu = boxes.detach().cpu().numpy()
        out = np.zeros((len(self), mask_size, mask_size), dtype=np.bool_)
        for i, polys in enumerate(self.polygons):
            x0, y0, x1, y1 = boxes_cpu[i]
            w = max(x1 - x0, 1e-6)
            h = max(y1 - y0, 1e-6)
            ratio_w = mask_size / w
            ratio_h = mask_size / h
            shifted = []
            for p in polys:
                pp = p.copy()
                pp[0::2] = (pp[0::2] - x0) * ratio_w
                pp[1::2] = (pp[1::2] - y0) * ratio_h
                shifted.append(pp)
            out[i] = _polygons_to_bitmask(shifted, mask_size, mask_size)
        return torch.from_numpy(out).to(boxes.device)


# ---------------------------------------------------------------------------
# ROIMasks
# ---------------------------------------------------------------------------


class ROIMasks:
    """Soft-mask output of the mask head, before paste-back.

    Stores ``(N, M, M)`` floats in ``[0, 1]`` (or logits — paste-back
    only needs them to be probabilities for the threshold step). The
    typical caller is the inference postprocess: convert
    ``pred_mask_logits`` to probabilities, wrap in :class:`ROIMasks`,
    then call :meth:`to_bitmasks` with the predicted boxes.
    """

    def __init__(self, tensor: Tensor) -> None:
        if tensor.dim() != 3:
            raise ValueError(f"ROIMasks expects (N, M, M); got {tuple(tensor.shape)}")
        self.tensor: Tensor = tensor

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def to(self, device: torch.device | str) -> ROIMasks:
        return ROIMasks(self.tensor.to(device))

    def to_bitmasks(
        self, boxes: Tensor, height: int, width: int, threshold: float = 0.5
    ) -> BitMasks:
        """Paste each soft mask onto the image canvas and binarise."""
        bit = paste_masks_in_image(self.tensor, boxes, (height, width), threshold)
        return BitMasks(bit)


# ---------------------------------------------------------------------------
# Paste-back
# ---------------------------------------------------------------------------

# Soft masks are stored as float32 (4 bytes). The chunking heuristic
# below limits each chunk to roughly 512 MB of intermediate sampling
# tensors. Detectron2 used 1 GB; we use half that to be friendlier on
# MPS, where the unified-memory budget is shared with the rest of the
# system.
_BYTES_PER_FLOAT = 4
_PASTE_CHUNK_BYTES = 1 << 29  # 512 MB


def paste_masks_in_image(
    masks: Tensor,
    boxes: Tensor,
    image_size: tuple[int, int],
    threshold: float = 0.5,
) -> Tensor:
    """Paste ``(N, M, M)`` soft masks back onto a full image canvas.

    Args:
        masks: ``(N, M, M)`` soft masks (probabilities or logits — the
            threshold is applied directly so logits work iff your
            threshold is in logit space).
        boxes: ``(N, 4)`` xyxy boxes in image-pixel coords.
        image_size: ``(H, W)`` of the destination canvas.
        threshold: Binarisation cut-off (default 0.5 matches
            Detectron2's mask-RCNN postprocess).

    Returns:
        ``(N, H, W)`` bool tensor on the same device as ``masks``.
    """
    if masks.dim() != 3:
        raise ValueError(f"masks must be (N, M, M); got {tuple(masks.shape)}")
    if boxes.shape != (masks.shape[0], 4):
        raise ValueError(
            f"boxes shape mismatch: expected ({masks.shape[0]}, 4), got {tuple(boxes.shape)}"
        )
    n = masks.shape[0]
    h, w = image_size
    out = masks.new_zeros((n, h, w), dtype=torch.bool)
    if n == 0:
        return out

    # Each chunk allocates ~ chunk * H * W * 4 bytes for the sampled
    # output. Cap at the number of masks so empty / tiny inputs are a
    # single chunk.
    bytes_per_mask = max(h * w * _BYTES_PER_FLOAT, 1)
    chunks = max(1, min(n, (n * bytes_per_mask + _PASTE_CHUNK_BYTES - 1) // _PASTE_CHUNK_BYTES))
    for indices in torch.chunk(torch.arange(n, device=masks.device), chunks):
        idx = indices.tolist()
        if not idx:
            continue
        chunk_masks = masks[idx]
        chunk_boxes = boxes[idx].to(masks.device)
        pasted = _paste_mask_chunk(chunk_masks, chunk_boxes, h, w)
        out[idx] = pasted >= threshold
    return out


def _paste_mask_chunk(masks: Tensor, boxes: Tensor, img_h: int, img_w: int) -> Tensor:
    """Bilinear-resample a chunk of masks to image resolution.

    Builds a sampling grid that maps each output pixel back to the
    corresponding ``(MxM)`` mask coordinate, then does a single
    ``F.grid_sample`` call. Output: ``(K, H, W)`` float in ``[0, 1]``.
    """
    k, m, _ = masks.shape
    x0, y0, x1, y1 = boxes.unbind(dim=1)  # each (K,)
    # Per-mask scale from image-pixel index to mask-cell index.
    scale_x = m / (x1 - x0).clamp(min=1.0)
    scale_y = m / (y1 - y0).clamp(min=1.0)

    iy = torch.arange(img_h, device=masks.device, dtype=masks.dtype) + 0.5
    ix = torch.arange(img_w, device=masks.device, dtype=masks.dtype) + 0.5
    # Per-mask grid in mask coords (K, H or W).
    gy = (iy[None, :] - y0[:, None]) * scale_y[:, None]
    gx = (ix[None, :] - x0[:, None]) * scale_x[:, None]
    # Normalise to grid_sample's [-1, 1] over the (M, M) source tensor.
    ny = gy / m * 2 - 1  # (K, H)
    nx = gx / m * 2 - 1  # (K, W)
    grid = torch.stack(
        [
            nx[:, None, :].expand(k, img_h, img_w),
            ny[:, :, None].expand(k, img_h, img_w),
        ],
        dim=-1,
    )
    sampled = F.grid_sample(
        masks.unsqueeze(1).to(dtype=torch.float32),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, 0]


# ---------------------------------------------------------------------------
# Polygon helpers (CPU)
# ---------------------------------------------------------------------------


def _as_float_array(p: Sequence[float] | _FloatArray) -> _FloatArray:
    arr = np.asarray(p, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < 6 or arr.shape[0] % 2 != 0:
        raise ValueError(
            f"Polygon must be flat [x0, y0, x1, y1, ...] with >=3 points; got shape {arr.shape}"
        )
    return arr


def _polygons_to_bitmask(polygons: Sequence[_FloatArray], height: int, width: int) -> _BoolArray:
    """Rasterise ``polygons`` (a single instance) into a ``(H, W)`` bool mask.

    Uses pycocotools' RLE backend; multi-polygon instances are merged via
    the COCO ``merge`` op so disjoint components rasterise to a single
    mask.
    """
    if len(polygons) == 0:
        return np.zeros((height, width), dtype=np.bool_)
    rles = coco_mask.frPyObjects([p.tolist() for p in polygons], height, width)
    rle = coco_mask.merge(rles)
    decoded: _BoolArray = coco_mask.decode(rle).astype(np.bool_)
    return decoded


def _polygon_area(polygon: _FloatArray) -> float:
    """Shoelace area of a single closed polygon."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return float(0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
