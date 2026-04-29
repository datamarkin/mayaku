"""FPN-aware multi-level RoI Align (`ROIPooler`).

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §7.1 + §2.4 (`modeling/poolers.py`).

The pooler decides, per RoI, which FPN level to sample from based on the
box's scale: tiny boxes pull from the highest-resolution level (``p2``,
stride 4), large boxes pull from the coarsest (``p5``, stride 32). The
mapping is the FPN paper's eq. (1):

    level = floor(canonical_level + log2(sqrt(area) / canonical_box_size))

with ``canonical_box_size = 224`` and ``canonical_level = 4`` (so a
``224x224`` box maps to ``p4``).

The forward dispatches each level's RoIs through
:func:`mayaku.backends.ops.roi_align.roi_align`, which is the
torchvision wrapper with our pure-PyTorch fallback (`Step 3`). Output
order matches the input order.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mayaku.backends.ops.roi_align import roi_align
from mayaku.structures.boxes import Boxes

__all__ = ["ROIPooler", "assign_boxes_to_levels"]


class ROIPooler(nn.Module):
    """Multi-level RoI Align ("ROIAlignV2", `aligned=True`).

    Args:
        output_size: ``(P, P)`` pooled feature side length (e.g. ``7``
            for the box head, ``14`` for the mask head). A single int
            is interpreted as a square.
        scales: One spatial scale per FPN input level
            ``(1 / stride, ...)`` in the same order the level features
            are passed at forward time. e.g. ``(1/4, 1/8, 1/16, 1/32)``
            for ``p2..p5``.
        sampling_ratio: Samples-per-output-bin per axis; ``0`` means
            "use ``ceil(roi_size / output_size)``" (`spec §7.1`, the
            in-scope default).
        canonical_box_size: Anchor box size (in pixels) that maps to
            ``canonical_level``. Default ``224``.
        canonical_level: FPN level (0-based from ``p2``) that
            ``canonical_box_size`` lands on. Default ``4`` so a
            ``224x224`` box samples from ``p4``.
    """

    def __init__(
        self,
        output_size: int | tuple[int, int],
        scales: Sequence[float],
        sampling_ratio: int = 0,
        *,
        canonical_box_size: int = 224,
        canonical_level: int = 4,
    ) -> None:
        super().__init__()
        if isinstance(output_size, int):
            self.output_size: tuple[int, int] = (output_size, output_size)
        else:
            self.output_size = output_size
        if len(scales) == 0:
            raise ValueError("ROIPooler requires at least one input level")
        self.scales: tuple[float, ...] = tuple(float(s) for s in scales)
        self.sampling_ratio = sampling_ratio
        # Levels are 0-indexed within the input feature list. The
        # canonical level / box size determines the *base*; the
        # actual range is derived from the spread of scales.
        self.canonical_box_size = canonical_box_size
        self.canonical_level = canonical_level
        self._min_level, self._max_level = _level_range(self.scales, canonical_level)

    def forward(self, features: Sequence[Tensor], box_lists: Sequence[Boxes]) -> Tensor:
        """Pool features for every RoI across every image.

        Args:
            features: One ``(N, C, H, W)`` feature map per FPN level,
                in the same order as ``scales``.
            box_lists: One :class:`Boxes` per image (length ``N``).
                The total number of pooled outputs is
                ``sum(len(b) for b in box_lists)``.

        Returns:
            ``(R, C, P, P)`` pooled features, where ``R`` is the total
            box count across all images. Order: image 0 boxes, then
            image 1, etc.
        """
        if len(features) != len(self.scales):
            raise ValueError(
                f"got {len(features)} feature maps but pooler was configured "
                f"for {len(self.scales)} scales"
            )
        device = features[0].device
        dtype = features[0].dtype
        channels = features[0].shape[1]
        ph, pw = self.output_size

        # (R, 5) RoIs in (batch_idx, x0, y0, x1, y1) form.
        pooler_fmt_boxes = _convert_boxes_to_pooler_format(box_lists, device, dtype)
        if pooler_fmt_boxes.shape[0] == 0:
            return features[0].new_zeros((0, channels, ph, pw))

        levels = assign_boxes_to_levels(
            box_lists,
            min_level=self._min_level,
            max_level=self._max_level,
            canonical_box_size=self.canonical_box_size,
            canonical_level=self.canonical_level,
        )

        out = features[0].new_zeros((pooler_fmt_boxes.shape[0], channels, ph, pw))
        for level, (scale, feat) in enumerate(zip(self.scales, features, strict=True)):
            mask = levels == level
            if not bool(mask.any()):
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            level_boxes = pooler_fmt_boxes[inds]
            pooled = roi_align(
                feat,
                level_boxes,
                output_size=self.output_size,
                spatial_scale=scale,
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
            out[inds] = pooled
        return out


# ---------------------------------------------------------------------------
# Level assignment
# ---------------------------------------------------------------------------


def assign_boxes_to_levels(
    box_lists: Sequence[Boxes],
    *,
    min_level: int,
    max_level: int,
    canonical_box_size: int,
    canonical_level: int,
) -> Tensor:
    """Map each RoI to an FPN level by box scale (FPN paper eq. (1)).

    The 0-indexed level index is relative to the *input list* passed to
    :class:`ROIPooler`. ``min_level`` and ``max_level`` clamp the
    assignment so an extreme box stays inside the configured range.
    """
    flat = torch.cat([b.tensor for b in box_lists], dim=0) if box_lists else torch.zeros(0, 4)
    if flat.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=flat.device)
    sizes = (flat[:, 2:] - flat[:, :2]).clamp(min=0)
    box_areas = sizes[:, 0] * sizes[:, 1]
    # `+1e-8` matches Detectron2's epsilon for log-of-zero stability when
    # a degenerate zero-area box sneaks through (`spec §7.1` reference).
    scales = torch.sqrt(box_areas)
    level_assignments = torch.floor(
        canonical_level + torch.log2(scales / canonical_box_size + 1e-8)
    )
    level_assignments = level_assignments.clamp(min=min_level, max=max_level).to(torch.long)
    return level_assignments - min_level


def _level_range(scales: Sequence[float], canonical_level: int) -> tuple[int, int]:
    """Derive (min_level, max_level) from the spread of input scales.

    Detectron2 derives the FPN level indices from the strides themselves
    so a pooler built for ``(1/4, 1/8, 1/16, 1/32)`` knows the input
    levels are 2..5. We keep a slightly simpler convention: levels are
    integers around ``canonical_level``, indexed by the input list's
    position. ``min_level = canonical_level - canonical_offset``,
    ``max_level = min_level + len(scales) - 1``.
    """
    # canonical_offset is the position (in the input list) of the level
    # whose stride matches `canonical_box_size`.
    canonical_offset = 2  # the standard FPN convention places p4 at index 2 of (p2,p3,p4,p5)
    min_level = canonical_level - canonical_offset
    max_level = min_level + len(scales) - 1
    return min_level, max_level


def _convert_boxes_to_pooler_format(
    box_lists: Sequence[Boxes], device: torch.device, dtype: torch.dtype
) -> Tensor:
    """``[(N, 4)]`` per-image boxes → ``(R, 5)`` ``(batch_idx, x0, y0, x1, y1)``."""
    parts: list[Tensor] = []
    for img_idx, boxes in enumerate(box_lists):
        b = boxes.tensor.to(device=device, dtype=dtype)
        idx_col = torch.full((b.shape[0], 1), float(img_idx), dtype=dtype, device=device)
        parts.append(torch.cat([idx_col, b], dim=1))
    if not parts:
        return torch.zeros(0, 5, dtype=dtype, device=device)
    return torch.cat(parts, dim=0)
