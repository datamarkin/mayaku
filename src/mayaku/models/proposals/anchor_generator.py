"""Per-FPN-level anchor generation.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.3:

* ``generate_cell_anchors(sizes, aspect_ratios)`` produces a
  ``(len(sizes) * len(aspect_ratios), 4)`` tensor of XYXY anchors
  centred at ``(0, 0)``. For each ``s`` and ``a``:
  ``w = sqrt(s² / a)``, ``h = a * w``.
* ``forward(features)`` shifts each cell anchor by integer multiples
  of the level stride (with offset ``offset * stride``, default 0)
  and returns one :class:`Boxes` per FPN level.

For the standard FPN-RCNN config: ``sizes = [(32,), (64,), (128,),
(256,), (512,)]``, ``aspect_ratios = [(0.5, 1.0, 2.0)]`` (shared
across levels). That gives ``A = 1 * 3 = 3`` anchors per cell per
level.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mayaku.config.schemas import AnchorGeneratorConfig
from mayaku.structures.boxes import Boxes

__all__ = ["DefaultAnchorGenerator", "build_anchor_generator"]


class DefaultAnchorGenerator(nn.Module):
    """FPN-friendly anchor generator.

    Args:
        sizes: One tuple per FPN level. Each tuple lists the anchor
            scales (in image-pixel units) for that level.
        aspect_ratios: One tuple per FPN level, *or* a single tuple
            shared across levels. Each ratio is ``h / w``.
        strides: One stride per FPN level (e.g. ``(4, 8, 16, 32, 64)``
            for ``p2..p6``). Anchors are centred on integer multiples
            of stride, shifted by ``offset * stride``.
        offset: Sub-pixel offset of the anchor grid in units of stride.
            Default ``0`` puts anchor centres at ``(0, 0), (stride, 0),
            (2*stride, 0), ...`` The Detectron2 default is also ``0``.
    """

    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        aspect_ratios: Sequence[Sequence[float]],
        strides: Sequence[int],
        offset: float = 0.0,
    ) -> None:
        super().__init__()
        if len(sizes) != len(strides):
            raise ValueError(f"sizes has {len(sizes)} levels but strides has {len(strides)}")
        if len(aspect_ratios) == 1:
            aspect_ratios = list(aspect_ratios) * len(sizes)
        if len(aspect_ratios) != len(sizes):
            raise ValueError(
                "aspect_ratios must be a single shared tuple or one tuple per "
                f"level (got {len(aspect_ratios)} for {len(sizes)} levels)"
            )
        if not 0.0 <= offset < 1.0:
            raise ValueError(f"offset must be in [0, 1); got {offset}")

        self.strides: tuple[int, ...] = tuple(int(s) for s in strides)
        self.offset = float(offset)
        # Cell anchors are dataset-time constants → register as buffers
        # so .to(device) carries them along with the rest of the model.
        cell_anchors = [
            _generate_cell_anchors(level_sizes, level_ratios)
            for level_sizes, level_ratios in zip(sizes, aspect_ratios, strict=True)
        ]
        for i, ca in enumerate(cell_anchors):
            self.register_buffer(f"_cell_anchors_{i}", ca, persistent=False)
        self._num_levels = len(cell_anchors)
        self._anchors_per_cell = tuple(int(ca.shape[0]) for ca in cell_anchors)

    @property
    def num_anchors_per_cell(self) -> tuple[int, ...]:
        """Number of anchors per spatial location, per FPN level."""
        return self._anchors_per_cell

    def cell_anchors(self, level: int) -> Tensor:
        """Per-level cell anchors as ``(A, 4)`` XYXY centred at origin."""
        return self.get_buffer(f"_cell_anchors_{level}")

    def forward(self, features: Sequence[Tensor]) -> list[Boxes]:
        """Generate one :class:`Boxes` per FPN level.

        Args:
            features: List of feature maps ``(N, C, H_i, W_i)``, one per
                level, in the same order as ``strides``.

        Returns:
            List of :class:`Boxes`, one per level. Each contains
            ``H_i * W_i * A_i`` anchors.
        """
        if len(features) != self._num_levels:
            raise ValueError(
                f"got {len(features)} feature maps but anchor generator was "
                f"configured for {self._num_levels} levels"
            )
        out: list[Boxes] = []
        for level, feat in enumerate(features):
            grid_h, grid_w = int(feat.shape[-2]), int(feat.shape[-1])
            stride = self.strides[level]
            cell = self.cell_anchors(level)
            shifts_x, shifts_y = _grid_shifts(
                grid_h, grid_w, stride, self.offset, cell.device, cell.dtype
            )
            shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)
            # (Hi*Wi, 1, 4) + (1, A, 4) → (Hi*Wi, A, 4)
            anchors = shifts[:, None, :] + cell[None, :, :]
            out.append(Boxes(anchors.reshape(-1, 4)))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_cell_anchors(sizes: Sequence[int], aspect_ratios: Sequence[float]) -> Tensor:
    """Anchors centred at ``(0, 0)`` for one FPN level.

    For each ``s`` (anchor area's square root in pixels) and ``a``
    (height/width), produces the XYXY box ``(-w/2, -h/2, w/2, h/2)``
    where ``area = s² = w * h`` and ``a = h / w``.
    """
    anchors: list[list[float]] = []
    for size in sizes:
        area = float(size) ** 2
        for ratio in aspect_ratios:
            w = (area / ratio) ** 0.5
            h = ratio * w
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return torch.tensor(anchors, dtype=torch.float32)


def _grid_shifts(
    h: int, w: int, stride: int, offset: float, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Per-cell ``(x, y)`` shifts in image-pixel coordinates."""
    shifts_x = (torch.arange(w, device=device, dtype=dtype) + offset) * stride
    shifts_y = (torch.arange(h, device=device, dtype=dtype) + offset) * stride
    yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def build_anchor_generator(
    cfg: AnchorGeneratorConfig, strides: Sequence[int]
) -> DefaultAnchorGenerator:
    """Construct a :class:`DefaultAnchorGenerator` from a typed config."""
    return DefaultAnchorGenerator(
        sizes=cfg.sizes,
        aspect_ratios=cfg.aspect_ratios,
        strides=strides,
        offset=cfg.offset,
    )
