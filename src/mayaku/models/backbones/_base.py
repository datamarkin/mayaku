"""Backbone abstract base + ``ShapeSpec`` dataclass.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §2.1's backbone ABC contract:
every backbone produces a name → tensor dict and exposes per-feature
shape specs so downstream FPN, RPN, and ROI heads can size their
convolutions and poolers without instantiating the backbone first.

The contract is deliberately tiny — three properties + one forward —
because every concrete backbone we ship is just torchvision under the
hood (`Step 7 implementation note`). Making the protocol any heavier
would force test code to mock more than is necessary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = ["Backbone", "ShapeSpec"]


@dataclass(frozen=True)
class ShapeSpec:
    """Per-feature output description used by downstream necks/heads.

    ``channels`` is the feature map's ``C`` and ``stride`` is the
    cumulative downsampling from input image pixels (e.g. ``stride=4``
    for ``res2``, ``32`` for ``res5``).
    """

    channels: int
    stride: int


class Backbone(nn.Module):
    """Backbone protocol.

    Concrete subclasses populate ``_out_features``,
    ``_out_feature_channels``, ``_out_feature_strides`` in
    ``__init__`` and implement ``forward(x) -> Dict[str, Tensor]``.
    The ``size_divisibility`` property defaults to the largest output
    stride; backbones that need a different alignment (e.g. ViT
    patch-aligned padding) override it.
    """

    _out_features: tuple[str, ...]
    _out_feature_channels: dict[str, int]
    _out_feature_strides: dict[str, int]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self) -> int:
        return max(self._out_feature_strides[name] for name in self._out_features)

    @property
    def out_features(self) -> tuple[str, ...]:
        return self._out_features

    def freeze_parameters(self) -> None:
        """Set ``requires_grad=False`` on every parameter. Used by
        :meth:`freeze_at`-style helpers in concrete backbones."""
        for p in self.parameters():
            p.requires_grad_(False)

    def dummy_input(self, batch: int = 1, h: int = 64, w: int = 64) -> Tensor:
        """Convenience for shape tests: an RGB tensor on the right device."""
        # Use the device of the first parameter so callers don't have to
        # query the backbone's device explicitly.
        device = next(iter(self.parameters())).device
        return torch.zeros(batch, 3, h, w, device=device)
