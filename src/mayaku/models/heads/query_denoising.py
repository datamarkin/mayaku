"""DN-DETR-style query denoising for QueryRCNN (box-only).

Builds auxiliary queries from noised GT boxes. The head feeds them through
the same refinement stages — isolated from the matching queries by an
attention mask — and regresses them back to the clean GT. This gives the
box head a dense, matching-free "denoise toward the truth" signal that
stabilizes the early bipartite matching. Training-only: not generated at
inference, so zero deployment cost.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["build_dn_groups", "dn_attention_mask"]


@torch.no_grad()
def build_dn_groups(
    targets: list[dict[str, Tensor]],
    *,
    dn_groups: int,
    box_noise_scale: float,
    device: torch.device,
) -> dict[str, Tensor] | None:
    """Noised-GT auxiliary boxes, padded to a fixed width across the batch.

    Returns ``None`` when the batch has no GT boxes (nothing to denoise).
    Otherwise a dict with, for ``M = max_b(G_b) * dn_groups``:
        boxes:     (B, M, 4) absolute xyxy noised boxes (pad rows = 0)
        tgt_boxes: (B, M, 4) clean GT box per slot (pad rows = 0)
        valid:     (B, M) bool, True for real (non-pad) DN queries
    """
    counts = [int(t["boxes_xyxy"].shape[0]) for t in targets]
    max_gt = max(counts) if counts else 0
    if max_gt == 0:
        return None

    batch_size = len(targets)
    width = max_gt * dn_groups
    boxes = torch.zeros(batch_size, width, 4, device=device)
    tgt = torch.zeros(batch_size, width, 4, device=device)
    valid = torch.zeros(batch_size, width, dtype=torch.bool, device=device)

    for b, t in enumerate(targets):
        gt = t["boxes_xyxy"].to(device)
        g = gt.shape[0]
        if g == 0:
            continue
        gt = gt.repeat(dn_groups, 1)  # (g*K, 4)
        img_w = float(t["image_size_xyxy"][0])
        img_h = float(t["image_size_xyxy"][1])

        x1, y1, x2, y2 = gt.unbind(1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)

        # center shift uniform in (-s, s) * half-size; size scale by (1 +- s)
        cx = cx + (torch.rand_like(cx) * 2 - 1) * box_noise_scale * bw * 0.5
        cy = cy + (torch.rand_like(cy) * 2 - 1) * box_noise_scale * bh * 0.5
        bw = (bw * (1.0 + (torch.rand_like(bw) * 2 - 1) * box_noise_scale)).clamp(min=1.0)
        bh = (bh * (1.0 + (torch.rand_like(bh) * 2 - 1) * box_noise_scale)).clamp(min=1.0)

        noised = torch.stack(
            [
                (cx - bw * 0.5).clamp(0, img_w),
                (cy - bh * 0.5).clamp(0, img_h),
                (cx + bw * 0.5).clamp(0, img_w),
                (cy + bh * 0.5).clamp(0, img_h),
            ],
            dim=1,
        )

        # Drop boxes that collapsed to <1px after clipping (off-edge noise):
        # they stay as harmless attention-isolated inputs but are excluded
        # from the loss via valid=False (matches the FRCNN reference's guard).
        non_degenerate = (noised[:, 2] - noised[:, 0] > 1.0) & (noised[:, 3] - noised[:, 1] > 1.0)

        n = g * dn_groups
        boxes[b, :n] = noised
        tgt[b, :n] = gt
        valid[b, :n] = non_degenerate

    return {"boxes": boxes, "tgt_boxes": tgt, "valid": valid}


def dn_attention_mask(num_match: int, num_dn: int, device: torch.device) -> Tensor:
    """Block matching <-> DN cross-attention (both directions).

    Returns a (T, T) bool mask, T = num_match + num_dn, where True positions
    are *not* allowed to attend (nn.MultiheadAttention convention). Matching
    queries must not see DN queries (prevents GT leakage); DN queries refine
    independently of the matching set. Within-block attention is allowed.
    """
    total = num_match + num_dn
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:num_match, num_match:] = True
    mask[num_match:, :num_match] = True
    return mask
