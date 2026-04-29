"""Padded batch of images with per-image original sizes.

:class:`ImageList` is the standard way to feed a variable-resolution
mini-batch through the backbone: each image is padded up to the largest
``(H, W)`` in the batch (optionally rounded up to a stride multiple), and
the original ``(h_i, w_i)`` are kept on the side so postprocessing can
crop predictions back later.

Channel order is **RGB** per ADR 002 (`docs/decisions/002-rgb-native-image-ingestion.md`).
This file is channel-agnostic — it pads whatever channel layout the
caller provides — but every consumer in the codebase assumes ``(C, H, W)``
with channels in ``[R, G, B]`` order. Do not pass BGR tensors through
ImageList; convert at the data-loader boundary instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ImageList"]


class ImageList:
    """Padded ``(N, C, Hmax, Wmax)`` tensor + per-image ``(h, w)`` sizes.

    Constructed via :meth:`from_tensors` in essentially every code path;
    the bare ``__init__`` is exposed only for tests and for the rare
    consumer that already has a padded tensor (e.g. ONNX export, where
    the padding is baked into the input).
    """

    def __init__(self, tensor: Tensor, image_sizes: list[tuple[int, int]]) -> None:
        if tensor.dim() != 4:
            raise ValueError(
                f"ImageList tensor must be (N, C, H, W); got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != len(image_sizes):
            raise ValueError(
                f"image_sizes length {len(image_sizes)} != batch size {tensor.shape[0]}"
            )
        self.tensor: Tensor = tensor
        self.image_sizes: list[tuple[int, int]] = list(image_sizes)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """Return the ``idx``-th image cropped to its original ``(h, w)``."""
        h, w = self.image_sizes[idx]
        return self.tensor[idx, ..., :h, :w]

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, device: torch.device | str) -> ImageList:
        return ImageList(self.tensor.to(device), self.image_sizes)

    def __repr__(self) -> str:
        return f"ImageList(tensor={tuple(self.tensor.shape)}, image_sizes={self.image_sizes})"

    @staticmethod
    def from_tensors(
        tensors: Sequence[Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        square: bool = False,
    ) -> ImageList:
        """Pad a list of ``(C, H, W)`` images into a single batch tensor.

        Args:
            tensors: Non-empty sequence of same-rank, same-channel
                tensors. Each may have a different ``H`` and ``W``.
            size_divisibility: If positive, round padded ``H`` and ``W``
                up to a multiple of this. FPN models set this to the
                bottom-up backbone stride (commonly 32) so feature maps
                divide cleanly through every level.
            pad_value: Constant fill for the padded region. The default
                of ``0`` is correct for *normalized* inputs (mean
                already subtracted); see the data-mapper docs for why
                we don't pad with a per-channel mean.
            square: If true, pad to a single side length ``max(Hmax, Wmax)``
                (still subject to ``size_divisibility``). This was added
                in Detectron2 for ViTDet; FPN models leave it off. We
                expose it as a bool flag rather than the upstream
                ``padding_constraints`` dict because we have no other
                constraints worth supporting.

        Returns:
            An :class:`ImageList` with ``tensor`` of shape
            ``(N, C, Hmax, Wmax)`` on the same device/dtype as the inputs.
        """
        if len(tensors) == 0:
            raise ValueError("ImageList.from_tensors requires at least one tensor")
        first = tensors[0]
        if first.dim() != 3:
            raise ValueError(f"Each tensor must be (C, H, W); got shape {tuple(first.shape)}")
        c = first.shape[0]
        device = first.device
        dtype = first.dtype
        for t in tensors:
            if t.dim() != 3 or t.shape[0] != c:
                raise ValueError("All tensors must be (C, H, W) with the same channel count")
            if t.device != device or t.dtype != dtype:
                raise ValueError("All tensors must share device and dtype before batching")

        image_sizes = [(int(t.shape[1]), int(t.shape[2])) for t in tensors]
        max_h = max(h for h, _ in image_sizes)
        max_w = max(w for _, w in image_sizes)
        if square:
            max_h = max_w = max(max_h, max_w)
        if size_divisibility > 1:
            stride = size_divisibility
            max_h = (max_h + stride - 1) // stride * stride
            max_w = (max_w + stride - 1) // stride * stride

        # Single-image fast path: pad once with F.pad.
        if len(tensors) == 1:
            t = tensors[0]
            pad = (0, max_w - t.shape[2], 0, max_h - t.shape[1])
            batched = F.pad(t, pad, value=pad_value).unsqueeze(0)
            return ImageList(batched.contiguous(), image_sizes)

        batched = first.new_full((len(tensors), c, max_h, max_w), pad_value)
        for i, t in enumerate(tensors):
            batched[i, :, : t.shape[1], : t.shape[2]] = t
        return ImageList(batched, image_sizes)
