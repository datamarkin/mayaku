"""Keypoint container + heatmap encoder/decoder.

The :class:`Keypoints` structure wraps an ``(N, K, 3)`` ``(x, y, v)``
tensor (``v`` ∈ {0 unlabeled, 1 occluded, 2 visible}) and carries the
**flip-pair metadata** required for horizontal-flip augmentation (the
``flip_indices`` attribute), as called out in `PROJECT_STATUS.md` notes
for Step 4. Without this metadata, flipping a keypoint annotation
silently swaps semantically distinct landmarks (left elbow ↔ right
elbow), which corrupts training in a way that's invisible in loss
curves but obvious in qualitative results — exactly the failure mode
this rewrite is intended to avoid.

The encoder/decoder match Detectron2's ``_keypoints_to_heatmap`` and
``heatmaps_to_keypoints`` (`DETECTRON2_TECHNICAL_SPEC.md` §2.6) with
two adjustments:

1. We never upcast to ``float64``. The reference does this in
   ``data/detection_utils.py`` for numpy ingest only; we keep tensors in
   their input dtype (typically fp32) on every backend. See
   `BACKEND_PORTABILITY_REPORT.md` §5.

2. The decoder's bicubic-resize-to-integer-ROI-pixel-size loop is
   identified by the portability report as an ONNX-export liability for
   keypoint models. The implementation here lives in plain Python and
   is meant to run as **postprocessing**, outside the exported graph
   (per Step 16 of the spec). The exported graph stops at the
   ``(R, K, 56, 56)`` heatmap.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "Keypoints",
    "heatmaps_to_keypoints",
    "keypoints_to_heatmap",
]


class Keypoints:
    """Per-instance keypoint annotations.

    Args:
        tensor: ``(N, K, 3)`` of ``(x, y, v)``. ``x``, ``y`` are absolute
            image-pixel coordinates after augmentation; ``v`` is 0
            (unlabeled), 1 (occluded), 2 (visible).
        flip_indices: Optional length-``K`` long tensor used by horizontal
            flip augmentation: after flipping x-coordinates, the keypoint
            order is permuted by ``flip_indices`` so that, e.g., a
            right-shoulder index maps onto a left-shoulder index. Must
            be a permutation of ``range(K)``.
    """

    def __init__(self, tensor: Tensor, flip_indices: Tensor | None = None) -> None:
        if tensor.dim() != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"Keypoints expects (N, K, 3), got shape {tuple(tensor.shape)}")
        if flip_indices is not None:
            if flip_indices.dim() != 1 or flip_indices.shape[0] != tensor.shape[1]:
                raise ValueError(
                    f"flip_indices must be a 1D tensor of length K={tensor.shape[1]}, "
                    f"got shape {tuple(flip_indices.shape)}"
                )
            if not torch.equal(
                torch.sort(flip_indices.cpu()).values,
                torch.arange(tensor.shape[1], dtype=flip_indices.dtype),
            ):
                raise ValueError("flip_indices must be a permutation of range(K)")
            flip_indices = flip_indices.long()
        self.tensor: Tensor = tensor
        self.flip_indices: Tensor | None = flip_indices

    # --- shape / device ----------------------------------------------------

    @property
    def num_keypoints(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to(self, device: torch.device | str) -> Keypoints:
        flip = None if self.flip_indices is None else self.flip_indices.to(device)
        return Keypoints(self.tensor.to(device), flip)

    # --- container protocol ------------------------------------------------

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, item: int | slice | Tensor) -> Keypoints:
        if isinstance(item, int):
            return Keypoints(self.tensor[item : item + 1], self.flip_indices)
        return Keypoints(self.tensor[item], self.flip_indices)

    def __repr__(self) -> str:
        flip = "yes" if self.flip_indices is not None else "no"
        return (
            f"Keypoints(num_instances={len(self)}, num_keypoints={self.num_keypoints}, "
            f"flip_indices={flip})"
        )

    # --- helpers ----------------------------------------------------------

    @classmethod
    def cat(cls, keypoints_list: Sequence[Keypoints]) -> Keypoints:
        """Concatenate along the instance axis. Flip indices must agree
        across all entries (they describe a per-dataset convention)."""
        if len(keypoints_list) == 0:
            raise ValueError("Keypoints.cat requires at least one element")
        flip = keypoints_list[0].flip_indices
        for kp in keypoints_list[1:]:
            if (flip is None) != (kp.flip_indices is None):
                raise ValueError("Cannot concatenate Keypoints with mixed flip metadata")
            if flip is not None and not torch.equal(flip, kp.flip_indices):  # type: ignore[arg-type]
                raise ValueError("Cannot concatenate Keypoints with differing flip_indices")
        return cls(torch.cat([k.tensor for k in keypoints_list], dim=0), flip)

    def to_heatmap(self, boxes: Tensor, heatmap_size: int) -> tuple[Tensor, Tensor]:
        """Encode keypoints into discrete heatmap target indices.

        Convenience wrapper around :func:`keypoints_to_heatmap`. ``boxes``
        is the ``(N, 4)`` xyxy proposal box for each instance.
        """
        return keypoints_to_heatmap(self.tensor, boxes, heatmap_size)


# ---------------------------------------------------------------------------
# Heatmap codec
# ---------------------------------------------------------------------------


def keypoints_to_heatmap(
    keypoints: Tensor, rois: Tensor, heatmap_size: int
) -> tuple[Tensor, Tensor]:
    """Encode keypoints into discrete heatmap target indices.

    Implements the math from `DETECTRON2_TECHNICAL_SPEC.md` §2.6 verbatim.

    Args:
        keypoints: ``(N, K, 3)`` of ``(x, y, v)`` in image-pixel coords.
        rois: ``(N, 4)`` xyxy proposal boxes (one per instance).
        heatmap_size: Side length ``S`` (56 in all in-scope configs).

    Returns:
        ``(targets, valid)`` where ``targets`` is a ``(N, K)`` long tensor
        of linearised heatmap indices ``y * S + x`` (zero where invalid)
        and ``valid`` is a ``(N, K)`` bool tensor — true for keypoints
        that are visible (``v > 0``) and land inside the heatmap. The
        loss must mask by ``valid``.
    """
    if keypoints.shape[0] != rois.shape[0]:
        raise ValueError(f"keypoints batch {keypoints.shape[0]} != rois batch {rois.shape[0]}")
    s = heatmap_size

    offset_x = rois[:, 0:1]
    offset_y = rois[:, 1:2]
    # If RoI is degenerate (zero width/height), produce all-invalid below.
    scale_x = s / (rois[:, 2:3] - rois[:, 0:1]).clamp(min=torch.finfo(rois.dtype).eps)
    scale_y = s / (rois[:, 3:4] - rois[:, 1:2]).clamp(min=torch.finfo(rois.dtype).eps)

    kx = keypoints[..., 0]
    ky = keypoints[..., 1]
    vis = keypoints[..., 2] > 0

    x = ((kx - offset_x) * scale_x).floor().long()
    y = ((ky - offset_y) * scale_y).floor().long()

    # Edge fixup (Heckbert 1990): a keypoint exactly on the right/bottom
    # RoI edge maps to S after the floor. Snap it back to S - 1.
    on_right = kx == rois[:, 2:3]
    on_bottom = ky == rois[:, 3:4]
    x = torch.where(on_right, torch.full_like(x, s - 1), x)
    y = torch.where(on_bottom, torch.full_like(y, s - 1), y)

    valid_loc = (x >= 0) & (x < s) & (y >= 0) & (y < s)
    valid = valid_loc & vis

    lin_ind = y * s + x
    # Zero out where invalid so consumers of the index can index without
    # going out of bounds — the loss masks these positions out anyway.
    lin_ind = lin_ind * valid.long()
    return lin_ind, valid


def heatmaps_to_keypoints(maps: Tensor, rois: Tensor) -> Tensor:
    """Decode heatmap logits back to image-pixel keypoint predictions.

    Bicubic-resizes each per-ROI heatmap up to the integer pixel size of
    the corresponding ROI, takes the per-channel argmax with a half-pixel
    correction, and returns ``(R, K, 4)`` of ``(x, y, logit, score)``.

    Args:
        maps: ``(R, K, S, S)`` raw heatmap logits (S=56 in all in-scope
            configs).
        rois: ``(R, 4)`` xyxy proposal boxes in image-pixel coords.

    Returns:
        ``(R, K, 4)`` float tensor. Inference plumbs columns
        ``[0, 1, 3]`` into ``pred_keypoints`` as ``(x, y, score)``.

    Notes:
        The bicubic + variable-target-size loop is intentionally
        Python-side (postprocessing) rather than part of any exported
        graph; see `BACKEND_PORTABILITY_REPORT.md` §6 for the export
        rationale.
    """
    if maps.dim() != 4:
        raise ValueError(f"heatmaps_to_keypoints expects (R, K, S, S), got {tuple(maps.shape)}")
    r, k, _, _ = maps.shape
    if rois.shape != (r, 4):
        raise ValueError(f"rois shape mismatch: expected ({r}, 4), got {tuple(rois.shape)}")
    out = maps.new_zeros((r, k, 4))
    if r == 0:
        return out

    # Per-ROI loop. Unavoidable: the bicubic target shape is per-ROI.
    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1.0).ceil().long()
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1.0).ceil().long()

    for i in range(r):
        roi_w = int(widths[i].item())
        roi_h = int(heights[i].item())
        # Bicubic up to integer ROI pixel size.
        roi_map = F.interpolate(
            maps[i : i + 1].float(),  # bicubic requires float
            size=(roi_h, roi_w),
            mode="bicubic",
            align_corners=False,
        )[0]  # (K, h, w)
        # Score normalization uses the *low-res* sum as denominator so
        # scores are comparable across object sizes (spec §2.6).
        flat_high = roi_map.view(k, -1)
        max_score = flat_high.max(dim=1).values  # (K,)
        full_res = torch.exp(roi_map - max_score[:, None, None])
        pool_res = torch.exp(maps[i] - max_score[:, None, None])
        denom = pool_res.view(k, -1).sum(dim=1).clamp(min=torch.finfo(roi_map.dtype).eps)
        scores_full = full_res / denom[:, None, None]

        pos = flat_high.argmax(dim=1)  # (K,)
        x_int = (pos % roi_w).to(roi_map.dtype)
        y_int = (pos // roi_w).to(roi_map.dtype)

        width_corr = (rois[i, 2] - rois[i, 0]) / max(roi_w, 1)
        height_corr = (rois[i, 3] - rois[i, 1]) / max(roi_h, 1)

        x = (x_int + 0.5) * width_corr + rois[i, 0]
        y = (y_int + 0.5) * height_corr + rois[i, 1]

        # Pull per-channel logit and score at the argmax position.
        flat_score = scores_full.view(k, -1)
        kp_idx = torch.arange(k, device=maps.device)
        logit_at = flat_high[kp_idx, pos]
        score_at = flat_score[kp_idx, pos]

        out[i, :, 0] = x
        out[i, :, 1] = y
        out[i, :, 2] = logit_at
        out[i, :, 3] = score_at
    return out
