"""From samples to a device batch: tensors, collate, the target table, and
multi-scale rescaling.

The target table is [image, class, x1, y1, x2, y2] then a tail in a fixed
order: has_mask (one column) when the dataset carries masks, then the 3K
keypoint values when it carries keypoints. `split_extras` owns that order.
"""

import numpy as np
import torch
import torch.nn.functional as F

from mayaku.data.geometry import as_canvas
from mayaku.model.aux import MASK_STRIDE


def split_extras(extras, seg, k):
    """(..., E) tail columns -> (has_mask (...) or None, keypoints (..., 3k)
    or None). A keypoint-only table has no has_mask column, so its keypoints
    start at column 0. Slices are views, so writing through them edits
    `extras`."""
    ko = 1 if seg else 0
    has_mask = extras[..., 0] if seg and extras.shape[-1] >= 1 else None
    kpts = extras[..., ko:ko + 3 * k] if k and extras.shape[-1] >= ko + 3 * k else None
    return has_mask, kpts


def seed_worker(worker_id):
    """Give each dataloader worker its own augmentation stream.

    A worker forks the dataset, random state included, so without this all of
    them draw the same sequence: every image in a batch of `workers` would
    share its mosaic partners, its scale, its translation and its flip.
    `info.seed` is already per worker and per epoch.
    """
    info = torch.utils.data.get_worker_info()
    info.dataset.rng.seed(info.seed)


def to_tensor(img):
    """HWC BGR uint8 -> CHW RGB uint8, contiguous.

    The one copy in the sample path: a horizontal flip upstream leaves a
    reversed view, which composes with the channel reversal here into one
    copy. Deliberately still uint8 -- a quarter of the bytes through the
    worker, pinned memory and PCIe; `batch_to` scales on the device.
    """
    return torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)


def batch_to(imgs, device):
    """The other half of `to_tensor`: uint8 batch -> float 0..1 on `device`."""
    return imgs.to(device, non_blocking=True).float().div_(255)


def collate(batch):
    """-> images (B, 3, H, W) uint8, targets (n, 6 + E), metas, masks.

    The target rows are exactly what the loss takes, so nothing between here
    and the loss reshapes labels. `masks` is the (B, H/8, W/8) uint8 instance
    raster when the dataset carries masks, else None.
    """
    rows = [torch.cat((torch.full((len(b["labels"]), 1), float(i)), b["labels"]), 1)
            for i, b in enumerate(batch) if len(b["labels"])]
    width = batch[0]["labels"].shape[1] + 1
    return (torch.stack([b["img"] for b in batch]),
            torch.cat(rows) if rows else torch.zeros(0, width),
            [b["meta"] for b in batch],
            torch.stack([b["masks"] for b in batch]) if "masks" in batch[0] else None)


def multiscale_sizes(lo, hi, step=32):
    """Square input sizes for multi-scale training: multiples of `step` from
    `lo` up to and including `hi`, the operating point."""
    lo = max(step, (lo // step) * step)
    return list(range(lo, (hi // step) * step + 1, step))


def rescale_batch(imgs, targets, masks, size, seg=False, kpt=0):
    """Resize one rendered batch to `size` (a side or (h, w)) for multi-scale
    training.

    The dataset renders at the operating point, the maximum; this downscales
    the batch, so there is only ever one resize and it never invents detail.
    Box and keypoint x / y scale by the same per-axis ratios as the image, and
    the stride-8 instance raster resizes nearest so its integer indices
    survive. Coordinates stay in pixels of the resized batch.
    """
    h, w = as_canvas(size)
    H, W = imgs.shape[-2:]
    if (h, w) == (H, W):
        return imgs, targets, masks
    rx, ry = w / W, h / H
    imgs = F.interpolate(imgs, size=(h, w), mode="bilinear", align_corners=False)
    if len(targets):
        targets = targets.clone()
        targets[:, 2:6:2] *= rx
        targets[:, 3:6:2] *= ry
        _, kp = split_extras(targets[:, 6:], seg, kpt)
        if kp is not None:
            xyv = kp.view(len(targets), kpt, 3)
            xyv[..., 0] *= rx
            xyv[..., 1] *= ry
    if masks is not None:
        masks = F.interpolate(masks[:, None].float(),
                              size=(h // MASK_STRIDE, w // MASK_STRIDE),
                              mode="nearest")[:, 0].to(masks.dtype)
    return imgs, targets, masks
