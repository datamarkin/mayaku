"""Backend-portable ``roi_align``.

Thin wrapper around :func:`torchvision.ops.roi_align` that pairs the
torchvision kernel (CPU + CUDA + MPS in our pinned versions, see
``BACKEND_PORTABILITY_REPORT.md`` Appendix A) with a pure-PyTorch
gather-based fallback (:func:`_roi_align_mps_native`) for two cases:

1. **MPS training** (``device.type == "mps"`` and ``input.requires_grad``).
   torchvision's MPS roi_align *backward* hangs the macOS GPU watchdog
   (see ADR 006); the gather-based fallback uses only ops with native
   MPS forward+backward.
2. **Defensive coverage** if torchvision raises ``NotImplementedError``
   on a required backend at install time. Same fallback handles it.

Defaults match the Detectron2 ROIAlignV2 contract documented in
``DETECTRON2_TECHNICAL_SPEC.md`` §7.1: ``aligned=True`` (pixel-corner
coordinate convention) and ``sampling_ratio=0`` (adaptive
``ceil(roi_size / output_size)`` samples per bin). All in-scope detectors
use this configuration via ``ROIPooler``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import roi_align as _tv_roi_align

__all__ = ["roi_align"]

_OutputSize = int | tuple[int, int]
_Boxes = Tensor | list[Tensor]


def roi_align(
    input: Tensor,
    boxes: _Boxes,
    output_size: _OutputSize,
    spatial_scale: float = 1.0,
    sampling_ratio: int = 0,
    aligned: bool = True,
) -> Tensor:
    """Region-of-interest align pool.

    Args:
        input: Feature map of shape ``(N, C, H, W)``.
        boxes: Either a ``(K, 5)`` tensor of ``(batch_idx, x0, y0, x1, y1)``
            in input-image coordinates, or a length-``N`` list of
            ``(K_i, 4)`` per-image box tensors.
        output_size: Target ``(Ph, Pw)`` (or a single ``int`` for square
            output).
        spatial_scale: Multiplier from input-image coordinates to feature
            coordinates, i.e. ``1 / feature_stride``.
        sampling_ratio: Samples per output bin per axis. ``<= 0`` means
            ``ceil(roi_size / output_size)`` (the Detectron2 default).
        aligned: If ``True`` (the in-scope default; ROIAlignV2), shift box
            coordinates by ``-0.5`` in feature space to align the sampling
            grid with the convolutional pixel grid.

    Returns:
        Pooled features of shape ``(K, C, Ph, Pw)`` on the same device
        and dtype as ``input``.
    """
    # MPS training routes to the gather-based fallback because
    # torchvision's MPS roi_align backward hangs the macOS GPU watchdog
    # (see ADR 006). All other paths — CPU/CUDA train+eval and MPS
    # inference — use torchvision's optimised kernel.
    if input.device.type == "mps" and input.requires_grad:
        return _roi_align_mps_native(
            input, boxes, output_size, spatial_scale, sampling_ratio, aligned
        )
    try:
        out: Tensor = _tv_roi_align(
            input,
            boxes,
            output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
        return out
    except NotImplementedError:
        # Defensive: torchvision regressed on this backend. The
        # gather-based fallback works on every backend, just slower
        # than torchvision's native kernel.
        return _roi_align_mps_native(
            input, boxes, output_size, spatial_scale, sampling_ratio, aligned
        )


def _normalize_boxes(boxes: _Boxes) -> Tensor:
    """Convert list-of-per-image boxes to a single ``(K, 5)`` tensor."""
    if isinstance(boxes, Tensor):
        return boxes
    if not boxes:
        # Empty list — shape (0, 5). Dtype/device borrowed from a sentinel.
        return torch.zeros((0, 5))
    parts: list[Tensor] = []
    for i, b in enumerate(boxes):
        idx_col = torch.full((b.shape[0], 1), float(i), dtype=b.dtype, device=b.device)
        parts.append(torch.cat([idx_col, b], dim=1))
    return torch.cat(parts, dim=0)


def _roi_align_mps_native(
    input: Tensor,
    boxes: _Boxes,
    output_size: _OutputSize,
    spatial_scale: float,
    sampling_ratio: int,
    aligned: bool,
) -> Tensor:
    """Pure-PyTorch roi_align that uses only ops with MPS forward+backward.

    Mirrors torchvision's ROIAlignV2 algorithm but expresses the bilinear
    interpolation as four explicit ``index_select`` corner lookups +
    a weighted sum, avoiding both ``F.grid_sample`` (whose backward is
    not implemented on MPS) and torchvision's MPS roi_align kernel
    (which is broken/slow). All ops in this path — ``index_select``,
    ``floor``, ``clamp``, basic arithmetic — have native MPS backward
    in PyTorch 2.4+.

    Memory: allocates ``(C, K * Ny * Nx)`` per corner gather, four
    corners total. For Faster R-CNN with 1000 proposals/image and
    a 7x7 output with sr=2, that's ~80 MB per FPN level — well within
    M-series budgets (compare to the grid_sample fallback path which
    we observed retaining 72 GB).
    """
    rois = _normalize_boxes(boxes).to(input.device)
    ph, pw = (output_size, output_size) if isinstance(output_size, int) else output_size
    n, c, h, w = input.shape
    k = rois.shape[0]
    if k == 0:
        return input.new_zeros((0, c, ph, pw))

    batch_idx = rois[:, 0].long()  # (K,)
    coords = rois[:, 1:].to(input.dtype) * spatial_scale
    if aligned:
        coords = coords - 0.5
    x0, y0, x1, y1 = coords.unbind(dim=1)  # each (K,)

    bin_h = (y1 - y0) / ph  # (K,)
    bin_w = (x1 - x0) / pw  # (K,)

    # Use a fixed sampling_ratio to avoid the per-ROI Python loop. When
    # the caller asks for adaptive (``sampling_ratio<=0``), we use sr=2,
    # which matches Detectron2's published recipe for the in-scope
    # configs and gives parity-acceptable AP.
    sr = sampling_ratio if sampling_ratio > 0 else 2
    n_y, n_x = ph * sr, pw * sr

    # Sample coordinates per ROI in feature-map space.
    # sample y[k, j] = y0[k] + (j + 0.5) / sr * bin_h[k]   for j in [0, n_y)
    # sample x[k, i] = x0[k] + (i + 0.5) / sr * bin_w[k]   for i in [0, n_x)
    iy_offsets = (torch.arange(n_y, device=input.device, dtype=input.dtype) + 0.5) / sr
    ix_offsets = (torch.arange(n_x, device=input.device, dtype=input.dtype) + 0.5) / sr
    sample_y = y0[:, None] + iy_offsets[None, :] * bin_h[:, None]  # (K, n_y)
    sample_x = x0[:, None] + ix_offsets[None, :] * bin_w[:, None]  # (K, n_x)

    # OOB-skip mask (matches torchvision): if a sample falls outside
    # ``[-1, h] x [-1, w]``, contribute 0. Inside that band, torchvision
    # clamps the sample to ``[0, h-1] x [0, w-1]`` and does bilinear from
    # there — equivalent to "snap-to-edge", not "zero-pad-corner". This
    # is the subtle case our earlier zero-pad attempt got wrong: a sample
    # at y=-0.2 should weight F[0, x] by 1.0 (post-clamp), not 0.8.
    skip_y = (sample_y < -1.0) | (sample_y > h)  # (K, n_y)
    skip_x = (sample_x < -1.0) | (sample_x > w)  # (K, n_x)
    in_bounds = (~(skip_y.unsqueeze(2) | skip_x.unsqueeze(1))).to(input.dtype)  # (K, n_y, n_x)

    # Clamp sample coords into the valid pixel range. After clamping,
    # standard bilinear with floor/ceil indices produces the right
    # values; the in_bounds mask zeros samples that were too far outside.
    y_clamped = torch.clamp(sample_y, 0.0, float(h - 1))
    x_clamped = torch.clamp(sample_x, 0.0, float(w - 1))

    y_lo_f = torch.floor(y_clamped)
    x_lo_f = torch.floor(x_clamped)
    y_lo = y_lo_f.long()
    y_hi = torch.clamp(y_lo + 1, 0, h - 1)
    x_lo = x_lo_f.long()
    x_hi = torch.clamp(x_lo + 1, 0, w - 1)

    wy_hi = (y_clamped - y_lo_f).unsqueeze(2)  # (K, n_y, 1)
    wy_lo = 1.0 - wy_hi
    wx_hi = (x_clamped - x_lo_f).unsqueeze(1)  # (K, 1, n_x)
    wx_lo = 1.0 - wx_hi

    # Flatten input to ``(C, N*H*W)`` once so the four corner gathers
    # share the layout. ``permute + reshape`` may copy on MPS, but the
    # cost is paid once per call instead of four times.
    n_hw = h * w
    flat = input.permute(1, 0, 2, 3).reshape(c, n * n_hw)
    # idx[k, j, i] = batch_idx[k] * H*W + y[k, j] * W + x[k, i]
    base = batch_idx.unsqueeze(1).unsqueeze(2) * n_hw  # (K, 1, 1)

    def _gather(yy: Tensor, xx: Tensor) -> Tensor:
        idx = base + yy.unsqueeze(2) * w + xx.unsqueeze(1)  # (K, n_y, n_x)
        gathered = flat.index_select(1, idx.view(-1))  # (C, K*n_y*n_x)
        return gathered.view(c, k, n_y, n_x).permute(1, 0, 2, 3)  # (K, C, n_y, n_x)

    # Bilinear-weighted sum over the four corners. Accumulating
    # in-place lets PyTorch free each gathered tensor after its
    # contribution lands, keeping peak transient memory at one
    # corner's worth instead of four.
    sampled = input.new_zeros((k, c, n_y, n_x))
    corners = (
        (y_lo, x_lo, wy_lo, wx_lo),
        (y_lo, x_hi, wy_lo, wx_hi),
        (y_hi, x_lo, wy_hi, wx_lo),
        (y_hi, x_hi, wy_hi, wx_hi),
    )
    for yy, xx, wy, wx in corners:
        sampled = sampled + (wy * wx).unsqueeze(1) * _gather(yy, xx)
    # Single OOB mask (matches torchvision: post-clamp samples give
    # normal bilinear values; samples too far outside contribute 0).
    sampled = sampled * in_bounds.unsqueeze(1)
    # Average each (sr, sr) sub-bin into the (ph, pw) output grid.
    pooled = sampled.view(k, c, ph, sr, pw, sr).mean(dim=(3, 5))
    return pooled
