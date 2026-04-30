"""Multi-image augmentations (Mosaic / MixUp / CopyPaste).

These augmentations cross sample boundaries: each combines two or more
``dataset_dict`` entries into one synthetic training example before the
mapper runs. The :class:`MultiSampleMappedDataset` wrapper holds the
underlying dataset_dicts list so concrete augmentations can sample
extras on demand.

Architecture notes:

* The mapper (`DatasetMapper`) takes ONE dataset_dict and returns one
  output. Multi-sample augs would normally have to bypass it, but we
  added a tiny escape hatch — if a dict carries ``__image`` (a numpy
  uint8 RGB array), the mapper uses that instead of reading from
  ``file_name``. This lets multi-sample augs do the image combination,
  hand the synthetic dict to the *unmodified* mapper, and inherit all
  of its annotation-transform machinery (resize / flip / color jitter)
  for free.
* The wrapper preserves ``len(wrapper) == len(dataset_dicts)`` so
  :class:`mayaku.data.samplers.TrainingSampler` keeps working unchanged.
* Annotation handling: each augmentation converts bboxes to
  ``BoxMode.XYXY_ABS`` and scales / translates per-image, then hands
  the merged annotation list to the mapper. The mapper itself runs
  ``BoxMode.convert(..., XYXY_ABS)`` so re-converting from XYXY_ABS to
  XYXY_ABS is a no-op — this is the correctness path.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from mayaku.data.mapper import DatasetMapper
from mayaku.structures.boxes import BoxMode
from mayaku.utils.image import read_image

__all__ = [
    "CopyPaste",
    "MixUp",
    "Mosaic",
    "MultiSampleAugmentation",
    "MultiSampleMappedDataset",
]


class MultiSampleAugmentation(ABC):
    """Base class for multi-image augmentations.

    Each concrete subclass implements:

    * :meth:`fires` — sample a Bernoulli; True means "apply this augmentation
      to the current ``__getitem__`` call". Probability is per-instance so
      different augs can have independent rates.
    * :meth:`num_extras` — how many EXTRA dataset_dicts beyond the primary
      this augmentation needs. ``Mosaic`` returns 3 (4 images total),
      ``MixUp`` returns 1, ``CopyPaste`` returns 1.
    * :meth:`apply_and_map` — given primary + extras + the mapper, return
      the final mapped output dict. Implementations typically do their
      multi-image combination, build a synthetic ``dataset_dict`` carrying
      the combined image via ``__image``, then delegate to ``mapper(...)``.
    """

    @abstractmethod
    def fires(self, rng: np.random.Generator) -> bool: ...

    @abstractmethod
    def num_extras(self) -> int: ...

    @abstractmethod
    def apply_and_map(
        self,
        dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        rng: np.random.Generator,
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Wrapper dataset
# ---------------------------------------------------------------------------


class MultiSampleMappedDataset:
    """List-style accessor that interposes multi-sample augmentations.

    Quacks like :class:`mayaku.cli.train._MappedList` (``__len__`` +
    ``__getitem__``) so it drops into the existing ``_SamplerView``
    pipeline without changes.

    On every ``__getitem__``, each registered augmentation samples its
    own Bernoulli. The first one that fires gets to combine the primary
    sample with extras and produce the final dict. If none fire, the
    regular per-sample mapper path runs.

    Args:
        dataset_dicts: The full list of training dicts. The wrapper
            references this directly (no copy) and assumes it's stable
            across the training run, like a normal Detectron2 dataset.
        mapper: The :class:`DatasetMapper` that turns dicts into batched
            tensors. Augmentations delegate to it for the per-sample
            transforms (resize / flip / color jitter / annotation build).
        multi_sample_augs: Active multi-sample augmentations. Tried in
            order; first to fire wins. Pass an empty list to make this
            a no-op wrapper (useful as a regression check).
        rng: Optional NumPy ``Generator`` for sampling. Default seeds
            from the OS so behaviour matches the existing single-sample
            pipeline (where ``np.random.default_rng()`` also OS-seeds).
    """

    def __init__(
        self,
        dataset_dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        multi_sample_augs: Sequence[MultiSampleAugmentation],
        rng: np.random.Generator | None = None,
    ) -> None:
        # Accepts either a plain ``list[dict]`` or a
        # :class:`mayaku.data.SerializedList` — both quack the same way
        # for ``__len__`` / ``__getitem__``. Using a Sequence type keeps
        # the wrapper independent of which storage the caller picks.
        self._dataset_dicts = dataset_dicts
        self._mapper = mapper
        self._augs: list[MultiSampleAugmentation] = list(multi_sample_augs)
        self._rng = rng if rng is not None else np.random.default_rng()

    def __len__(self) -> int:
        return len(self._dataset_dicts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        primary = self._dataset_dicts[idx]
        for aug in self._augs:
            if aug.fires(self._rng):
                extras = self._sample_extras(idx, aug.num_extras())
                return aug.apply_and_map([primary, *extras], self._mapper, self._rng)
        return self._mapper(primary)

    def _sample_extras(self, exclude_idx: int, n: int) -> list[dict[str, Any]]:
        """Draw ``n`` distinct dataset_dicts that aren't the primary."""
        if n <= 0:
            return []
        size = len(self._dataset_dicts)
        if size <= 1:
            # Pathological — only one image in the dataset. Augmentation
            # has nothing to combine with; fall through to "no-op" by
            # returning the same dict twice. The aug should handle this
            # gracefully (MixUp blending an image with itself produces
            # the same image; not useful but not wrong).
            return [self._dataset_dicts[exclude_idx]] * n
        chosen: list[int] = []
        while len(chosen) < n:
            j = int(self._rng.integers(0, size))
            if j == exclude_idx or j in chosen:
                continue
            chosen.append(j)
        return [self._dataset_dicts[j] for j in chosen]


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------


class MixUp(MultiSampleAugmentation):
    """YOLOv8-style MixUp: λ-blend two images, concat their annotations.

    The detection-tuned recipe differs from classification's MixUp in two
    ways: ``α`` is much larger (8 vs 0.2) so λ peaks near 0.5 instead of
    near 0/1, and annotations are kept from BOTH images — both sets of
    bboxes / masks / keypoints contribute to the loss, which the per-anchor
    R-CNN matcher handles fine.

    Args:
        prob: Probability of firing per training sample.
        alpha: Beta distribution shape parameter for sampling λ. Default 8.0
            (peaked near 0.5 — equal contribution from both images).
    """

    def __init__(self, *, prob: float = 0.5, alpha: float = 8.0) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0; got {alpha}")
        self.prob = float(prob)
        self.alpha = float(alpha)

    def fires(self, rng: np.random.Generator) -> bool:
        return float(rng.random()) < self.prob

    def num_extras(self) -> int:
        return 1

    def apply_and_map(
        self,
        dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if len(dicts) != 2:
            raise ValueError(f"MixUp expects exactly 2 dicts (primary + 1 extra); got {len(dicts)}")
        primary, secondary = dicts

        # Read both images; the primary's native size is the canvas.
        img_primary = read_image(primary["file_name"])
        img_secondary = read_image(secondary["file_name"])
        h, w = img_primary.shape[:2]

        # Resize secondary to match primary's native (pre-mapper) shape so
        # the two arrays are blendable. Bboxes from secondary need the
        # same scale factor applied so they end up in the canvas frame.
        sec_h, sec_w = img_secondary.shape[:2]
        if (sec_h, sec_w) != (h, w):
            img_secondary = _resize_uint8(img_secondary, h, w)
        sx = w / sec_w
        sy = h / sec_h

        # λ ~ Beta(α, α) — α=8 peaks near 0.5; α<1 would peak at the
        # extremes (which is what classification MixUp wants). For
        # detection, equal contribution is what helps regularise.
        lam = float(rng.beta(self.alpha, self.alpha))
        blended = lam * img_primary.astype(np.float32) + (1.0 - lam) * img_secondary.astype(
            np.float32
        )
        blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        # Concatenate annotations. Primary's stay in their native frame
        # (no scaling); secondary's get scaled to match the canvas.
        annos = list(copy.deepcopy(primary.get("annotations", [])))
        for ann in copy.deepcopy(secondary.get("annotations", [])):
            annos.append(_scale_annotation(ann, sx=sx, sy=sy))

        synthetic: dict[str, Any] = {
            "file_name": primary["file_name"],
            "image_id": primary.get("image_id", -1),
            "height": h,
            "width": w,
            "annotations": annos,
            "__image": blended,
        }
        return mapper(synthetic)


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------


class Mosaic(MultiSampleAugmentation):
    """YOLOv4-style 2×2 mosaic — stitch 4 images into one canvas.

    Each of 4 source images is scaled aspect-preserving to fit inside its
    quadrant, then placed so all four touch the *pivot* point (a random
    interior point of the canvas). Pixels outside the placed images are
    zero-padded; bboxes / polygons / keypoints are translated by the
    quadrant offset and clipped to the canvas. Bboxes whose post-clip
    area falls below ``min_box_area`` are dropped.

    Args:
        prob: Probability of firing per training sample.
        canvas_size: Output canvas (h, w). Default ``(1024, 1024)`` —
            comfortably larger than the typical 640-short-edge resize so
            no information is lost when the mapper's ResizeShortestEdge
            runs after.
        min_box_area: Minimum bbox area (pixels²) post-clip; bboxes
            below this are dropped. Default 16 (matches YOLOv8).
        pivot_range: ``(lo, hi)`` fraction of the canvas in which the
            pivot is sampled. Default ``(0.25, 0.75)`` — keeps every
            quadrant non-degenerate. ``(0.5, 0.5)`` would always centre
            the pivot (deterministic).
    """

    def __init__(
        self,
        *,
        prob: float = 0.5,
        canvas_size: tuple[int, int] = (1024, 1024),
        min_box_area: float = 16.0,
        pivot_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob}")
        if any(s <= 0 for s in canvas_size):
            raise ValueError(f"canvas_size must be positive; got {canvas_size}")
        if min_box_area < 0.0:
            raise ValueError(f"min_box_area must be >= 0; got {min_box_area}")
        lo, hi = pivot_range
        if not 0.0 < lo <= hi < 1.0:
            raise ValueError(f"pivot_range must satisfy 0 < lo <= hi < 1; got {pivot_range}")
        self.prob = float(prob)
        self.canvas_size = (int(canvas_size[0]), int(canvas_size[1]))
        self.min_box_area = float(min_box_area)
        self.pivot_range = (float(lo), float(hi))

    def fires(self, rng: np.random.Generator) -> bool:
        return float(rng.random()) < self.prob

    def num_extras(self) -> int:
        return 3

    def apply_and_map(
        self,
        dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if len(dicts) != 4:
            raise ValueError(
                f"Mosaic expects exactly 4 dicts (primary + 3 extras); got {len(dicts)}"
            )

        canvas_h, canvas_w = self.canvas_size
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Pivot point — random in the configured inner band of the canvas.
        lo, hi = self.pivot_range
        px = int(rng.integers(int(canvas_w * lo), int(canvas_w * hi) + 1))
        py = int(rng.integers(int(canvas_h * lo), int(canvas_h * hi) + 1))

        # Quadrant sizes follow the pivot.
        # positions: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        quadrant_sizes = [
            (py, px),  # TL: rows 0..py, cols 0..px
            (py, canvas_w - px),  # TR: rows 0..py, cols px..W
            (canvas_h - py, px),  # BL: rows py..H, cols 0..px
            (canvas_h - py, canvas_w - px),  # BR: rows py..H, cols px..W
        ]

        merged_annos: list[dict[str, Any]] = []
        for i, dd in enumerate(dicts):
            quad_h, quad_w = quadrant_sizes[i]
            if quad_h <= 0 or quad_w <= 0:
                # Pivot at the canvas edge would degenerate this quadrant.
                # The pivot_range guard normally prevents it; this is belt-
                # and-suspenders for users who pass a custom range.
                continue

            img = read_image(dd["file_name"])
            h0, w0 = img.shape[:2]
            scale = min(quad_w / w0, quad_h / h0)
            new_h = max(1, round(h0 * scale))
            new_w = max(1, round(w0 * scale))
            resized = _resize_uint8(img, new_h, new_w) if (new_h, new_w) != (h0, w0) else img

            # Place so this image's pivot-facing corner aligns with the
            # pivot point. Offsets are the canvas coordinates of image[0,0].
            if i == 0:  # top-left → bottom-right of image at pivot
                ox, oy = px - new_w, py - new_h
            elif i == 1:  # top-right → bottom-left of image at pivot
                ox, oy = px, py - new_h
            elif i == 2:  # bottom-left → top-right of image at pivot
                ox, oy = px - new_w, py
            else:  # bottom-right → top-left of image at pivot
                ox, oy = px, py

            # Paste — in this branch the image's footprint is bounded
            # within its quadrant by construction (scale chose min of
            # axis ratios, so new_h ≤ quad_h and new_w ≤ quad_w).
            x0 = max(0, ox)
            y0 = max(0, oy)
            x1 = min(canvas_w, ox + new_w)
            y1 = min(canvas_h, oy + new_h)
            if x0 >= x1 or y0 >= y1:
                continue
            src_x0 = x0 - ox
            src_y0 = y0 - oy
            canvas[y0:y1, x0:x1] = resized[src_y0 : src_y0 + (y1 - y0), src_x0 : src_x0 + (x1 - x0)]

            # Annotations: scale to the resized frame, then translate by
            # the placement offset. Then clip to canvas bounds.
            for ann in dd.get("annotations", []):
                scaled = _scale_annotation(ann, sx=scale, sy=scale)
                placed = _translate_annotation(scaled, dx=ox, dy=oy)
                clipped = _clip_annotation_to_canvas(
                    placed, canvas_h=canvas_h, canvas_w=canvas_w, min_box_area=self.min_box_area
                )
                if clipped is not None:
                    merged_annos.append(clipped)

        synthetic: dict[str, Any] = {
            "file_name": dicts[0]["file_name"],
            "image_id": dicts[0].get("image_id", -1),
            "height": canvas_h,
            "width": canvas_w,
            "annotations": merged_annos,
            "__image": canvas,
        }
        return mapper(synthetic)


# ---------------------------------------------------------------------------
# CopyPaste
# ---------------------------------------------------------------------------


class CopyPaste(MultiSampleAugmentation):
    """Ghiasi 2021 Simple Copy-Paste — paste source instances onto a target.

    For each fired sample, a SOURCE image's segmentation instances are
    pasted (pixel-for-pixel where the mask is True) onto the TARGET
    image. The pasted annotations are appended to the target's; target
    instances heavily occluded by the union of pasted masks are dropped
    (Ghiasi 2021, §3.2).

    Requires the dataset config to set ``input.mask_format='bitmask'``.
    Polygon masks would need a lossy raster→polygon round-trip after the
    paste, so :class:`mayaku.config.schemas.InputConfig` rejects
    ``copy_paste_prob > 0`` with ``mask_format='polygon'`` upstream.

    Args:
        prob: Probability of firing per training sample.
        paste_fraction: Probability of including each individual source
            instance in the paste. Default 1.0 — paste every source
            instance. Lowering this gives a sparser composite.
        occlusion_threshold: Drop target instances whose original mask
            area is more than this fraction covered by the union of
            pasted masks. Default 0.7. Use 1.0 to disable the filter.
        min_box_area: Drop pasted instances whose tight bbox is smaller
            than this many pixels². Default 16 (matches ``Mosaic``).
    """

    def __init__(
        self,
        *,
        prob: float = 0.5,
        paste_fraction: float = 1.0,
        occlusion_threshold: float = 0.7,
        min_box_area: float = 16.0,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1]; got {prob}")
        if not 0.0 < paste_fraction <= 1.0:
            raise ValueError(f"paste_fraction must be in (0, 1]; got {paste_fraction}")
        if not 0.0 < occlusion_threshold <= 1.0:
            raise ValueError(f"occlusion_threshold must be in (0, 1]; got {occlusion_threshold}")
        if min_box_area < 0.0:
            raise ValueError(f"min_box_area must be >= 0; got {min_box_area}")
        self.prob = float(prob)
        self.paste_fraction = float(paste_fraction)
        self.occlusion_threshold = float(occlusion_threshold)
        self.min_box_area = float(min_box_area)

    def fires(self, rng: np.random.Generator) -> bool:
        return float(rng.random()) < self.prob

    def num_extras(self) -> int:
        return 1

    def apply_and_map(
        self,
        dicts: Sequence[dict[str, Any]],
        mapper: DatasetMapper,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if len(dicts) != 2:
            raise ValueError(
                f"CopyPaste expects exactly 2 dicts (target + 1 source); got {len(dicts)}"
            )
        target, source = dicts

        # Imported lazily — pycocotools is a hard dep of the data layer
        # but this module is imported during config-loading on machines
        # that may not have a usable pycocotools install (e.g. ARM CI).
        # Fail at the augmentation site, not at import time.
        from pycocotools import mask as coco_mask

        from mayaku.data.mapper import _segmentation_to_bitmask

        target_img = read_image(target["file_name"])
        source_img = read_image(source["file_name"])
        h, w = target_img.shape[:2]
        sh, sw = source_img.shape[:2]
        if (sh, sw) != (h, w):
            source_img = _resize_uint8(source_img, h, w)

        composite = target_img.copy()
        union_mask = np.zeros((h, w), dtype=bool)
        pasted_anns: list[dict[str, Any]] = []

        # Random order so depth-stacking is randomised across pastes —
        # the last one painted ends up on top.
        source_anns = source.get("annotations", [])
        order = rng.permutation(len(source_anns))
        for i in order:
            ann = source_anns[int(i)]
            if ann.get("iscrowd", 0):
                continue
            seg = ann.get("segmentation")
            if not seg:
                continue
            if float(rng.random()) > self.paste_fraction:
                continue
            try:
                mask_native = _segmentation_to_bitmask(seg, sh, sw, coco_mask)
            except (ValueError, KeyError):
                # Malformed seg — skip rather than fail the whole batch.
                continue
            mask_target = _resize_bool_nearest(mask_native, h, w)
            if not mask_target.any():
                continue

            # Blit. Boolean fancy-indexing copies row-by-row; both arrays
            # are (H, W, 3) so broadcasting handles the channel dim.
            composite[mask_target] = source_img[mask_target]
            union_mask |= mask_target

            ys, xs = np.where(mask_target)
            x0, y0 = float(xs.min()), float(ys.min())
            x1 = float(xs.max()) + 1.0
            y1 = float(ys.max()) + 1.0
            if (x1 - x0) * (y1 - y0) < self.min_box_area:
                continue

            # Encode the placed mask as RLE so the mapper's bitmask path
            # can decode it through the same `_segmentation_to_bitmask`
            # helper. ``counts`` comes back as bytes from pycocotools;
            # decode to ASCII for JSON-friendly storage and to match the
            # COCO-format string-RLE convention.
            rle = coco_mask.encode(np.asfortranarray(mask_target.astype(np.uint8)))
            if isinstance(rle.get("counts"), bytes):
                rle["counts"] = rle["counts"].decode("ascii")
            pasted_anns.append(
                {
                    "bbox": [x0, y0, x1, y1],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(ann["category_id"]),
                    "segmentation": rle,
                }
            )

        # Occlusion filter on TARGET annotations: rasterise each target
        # instance against the original target dims, compare with the
        # paste union. Skip when there's no paste (union empty) or the
        # filter is disabled (threshold == 1.0).
        keep_target: list[dict[str, Any]] = []
        run_filter = self.occlusion_threshold < 1.0 and bool(union_mask.any())
        for ann in copy.deepcopy(target.get("annotations", [])):
            if not run_filter or ann.get("iscrowd", 0):
                keep_target.append(ann)
                continue
            seg = ann.get("segmentation")
            if not seg:
                keep_target.append(ann)
                continue
            try:
                tmask = _segmentation_to_bitmask(seg, h, w, coco_mask)
            except (ValueError, KeyError):
                keep_target.append(ann)
                continue
            tmask_area = float(tmask.sum())
            if tmask_area <= 0.0:
                keep_target.append(ann)
                continue
            overlap = float(np.logical_and(tmask, union_mask).sum())
            if overlap / tmask_area >= self.occlusion_threshold:
                continue
            keep_target.append(ann)

        synthetic: dict[str, Any] = {
            "file_name": target["file_name"],
            "image_id": target.get("image_id", -1),
            "height": h,
            "width": w,
            "annotations": keep_target + pasted_anns,
            "__image": composite,
        }
        return mapper(synthetic)


# ---------------------------------------------------------------------------
# Helpers (private, shared across multi-sample augs)
# ---------------------------------------------------------------------------


def _resize_uint8(image: npt.NDArray[np.uint8], new_h: int, new_w: int) -> npt.NDArray[np.uint8]:
    """Pillow-bilinear resize for an ``(H, W, 3)`` uint8 RGB image."""
    pil = Image.fromarray(image)
    out = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.uint8)


def _resize_bool_nearest(
    mask: npt.NDArray[np.bool_], new_h: int, new_w: int
) -> npt.NDArray[np.bool_]:
    """Nearest-neighbor resize for a 2-D boolean mask.

    Bilinear resampling on a binary mask would blur the boundary into
    grey values; we want crisp 0/1 transitions so :class:`CopyPaste`
    can paste pixel-for-pixel without a half-coverage halo.
    """
    if mask.shape == (new_h, new_w):
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    out = pil.resize((new_w, new_h), Image.Resampling.NEAREST)
    return np.asarray(out, dtype=np.uint8) > 0


def _scale_annotation(ann: dict[str, Any], *, sx: float, sy: float) -> dict[str, Any]:
    """Scale a COCO annotation's bbox to a new coordinate frame.

    Produces a normalised annotation in ``BoxMode.XYXY_ABS`` so the
    downstream :class:`DatasetMapper` doesn't have to second-guess
    formats. Mask polygons are scaled by the same ``(sx, sy)`` so they
    stay aligned with the new bbox; bitmask segmentations are not
    scaled here (they're rasterised against the original image dims;
    multi-sample augs that mix bitmask masks should resize the bitmask
    explicitly before reaching the mapper).
    """
    out = copy.deepcopy(ann)
    bbox = np.asarray(out["bbox"], dtype=np.float32)
    bbox_mode = out.get("bbox_mode", BoxMode.XYWH_ABS)
    # Convert to XYXY_ABS so we can scale uniformly.
    import torch

    xyxy = (
        BoxMode.convert(torch.tensor(bbox, dtype=torch.float32), bbox_mode, BoxMode.XYXY_ABS)
        .numpy()
        .astype(np.float32)
    )
    xyxy[0] *= sx
    xyxy[1] *= sy
    xyxy[2] *= sx
    xyxy[3] *= sy
    out["bbox"] = xyxy.tolist()
    out["bbox_mode"] = BoxMode.XYXY_ABS

    # Polygon segmentation: scale x's and y's of every (x, y) pair.
    seg = out.get("segmentation")
    if isinstance(seg, list):
        scaled_seg: list[list[float]] = []
        for poly in seg:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            arr[:, 0] *= sx
            arr[:, 1] *= sy
            scaled_seg.append(arr.reshape(-1).tolist())
        out["segmentation"] = scaled_seg

    # Keypoints: scale (x, y) per keypoint, leave visibility flag alone.
    if "keypoints" in out:
        kp = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 3)
        kp[:, 0] *= sx
        kp[:, 1] *= sy
        out["keypoints"] = kp.reshape(-1).tolist()

    return out


def _translate_annotation(ann: dict[str, Any], *, dx: float, dy: float) -> dict[str, Any]:
    """Translate an annotation in ``BoxMode.XYXY_ABS`` by ``(dx, dy)``.

    Assumes the input has already been normalised to XYXY_ABS by
    :func:`_scale_annotation`. Polygons and keypoints translate by the
    same offset. ``segmentation`` in dict (RLE) format is not
    translatable in-place; left untouched (Mosaic with bitmask masks
    needs a separate path — currently not supported on the polygon
    path).
    """
    out = copy.deepcopy(ann)
    bbox = np.asarray(out["bbox"], dtype=np.float32)
    bbox[0] += dx
    bbox[2] += dx
    bbox[1] += dy
    bbox[3] += dy
    out["bbox"] = bbox.tolist()

    seg = out.get("segmentation")
    if isinstance(seg, list):
        translated_seg: list[list[float]] = []
        for poly in seg:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            arr[:, 0] += dx
            arr[:, 1] += dy
            translated_seg.append(arr.reshape(-1).tolist())
        out["segmentation"] = translated_seg

    if "keypoints" in out:
        kp = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 3)
        kp[:, 0] += dx
        kp[:, 1] += dy
        out["keypoints"] = kp.reshape(-1).tolist()

    return out


def _clip_annotation_to_canvas(
    ann: dict[str, Any],
    *,
    canvas_h: int,
    canvas_w: int,
    min_box_area: float,
) -> dict[str, Any] | None:
    """Clip bbox / polygon / keypoints to ``[0, canvas_w] × [0, canvas_h]``.

    Returns the modified annotation, or ``None`` when the clipped bbox
    has area < ``min_box_area`` (the whole annotation is dropped — too
    small to contribute useful supervision).

    Polygons that fall entirely outside the canvas are dropped from the
    annotation's segmentation list; if all polygons are dropped, the
    list is left empty (the Instances builder then tracks zero polygons
    for that instance — annotators downstream should handle that).
    """
    out = copy.deepcopy(ann)
    bbox = np.asarray(out["bbox"], dtype=np.float32)
    bbox[0] = float(np.clip(bbox[0], 0, canvas_w))
    bbox[1] = float(np.clip(bbox[1], 0, canvas_h))
    bbox[2] = float(np.clip(bbox[2], 0, canvas_w))
    bbox[3] = float(np.clip(bbox[3], 0, canvas_h))
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    if area < min_box_area:
        return None
    out["bbox"] = bbox.tolist()

    seg = out.get("segmentation")
    if isinstance(seg, list):
        clipped_seg: list[list[float]] = []
        for poly in seg:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            clipped = _clip_polygon_to_rect(arr, 0.0, 0.0, float(canvas_w), float(canvas_h))
            if clipped.shape[0] >= 3:
                clipped_seg.append(clipped.reshape(-1).tolist())
        out["segmentation"] = clipped_seg

    if "keypoints" in out:
        kp = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 3)
        # Mark keypoints outside the canvas as invisible (v=0). The
        # mapper does the same OOB check after its own augs; doing it
        # here keeps the contract tight pre-mapper.
        oob = (kp[:, 0] < 0) | (kp[:, 0] > canvas_w) | (kp[:, 1] < 0) | (kp[:, 1] > canvas_h)
        kp[oob, 0] = 0.0
        kp[oob, 1] = 0.0
        kp[oob, 2] = 0.0
        out["keypoints"] = kp.reshape(-1).tolist()

    return out


def _clip_polygon_to_rect(
    poly_xy: npt.NDArray[np.float32],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> npt.NDArray[np.float32]:
    """Sutherland-Hodgman clip an ``(N, 2)`` polygon to an axis-aligned rect.

    Returns an ``(M, 2)`` array; ``M`` may be 0 (entirely outside) or
    less than the input N (some vertices clipped out, intersections added).
    The output preserves the input's winding order.
    """
    if poly_xy.size == 0:
        return poly_xy.astype(np.float32)
    out = poly_xy.astype(np.float32, copy=True)
    # Clip against each edge: x>=x_min, x<=x_max, y>=y_min, y<=y_max.
    out = _clip_against_half_plane(out, axis=0, val=x_min, keep_geq=True)
    if out.shape[0] == 0:
        return out
    out = _clip_against_half_plane(out, axis=0, val=x_max, keep_geq=False)
    if out.shape[0] == 0:
        return out
    out = _clip_against_half_plane(out, axis=1, val=y_min, keep_geq=True)
    if out.shape[0] == 0:
        return out
    out = _clip_against_half_plane(out, axis=1, val=y_max, keep_geq=False)
    return out


def _clip_against_half_plane(
    poly: npt.NDArray[np.float32],
    *,
    axis: int,
    val: float,
    keep_geq: bool,
) -> npt.NDArray[np.float32]:
    """Clip ``poly`` to the half-plane ``poly[:, axis] {>=, <=} val``.

    Sutherland-Hodgman walks the polygon edges (cur -> next) and emits
    output vertices when crossing the boundary. ``keep_geq=True`` keeps
    points where ``poly[:, axis] >= val`` (i.e. the half-plane to the
    right of / below ``val`` for axis 0 / 1 respectively).
    """
    n = poly.shape[0]
    if n == 0:
        return poly
    out: list[npt.NDArray[np.float32]] = []
    for i in range(n):
        cur = poly[i]
        prev = poly[(i - 1) % n]
        cur_inside = cur[axis] >= val if keep_geq else cur[axis] <= val
        prev_inside = prev[axis] >= val if keep_geq else prev[axis] <= val
        if cur_inside:
            if not prev_inside:
                out.append(_segment_intersect_axis(prev, cur, axis=axis, val=val))
            out.append(cur)
        elif prev_inside:
            out.append(_segment_intersect_axis(prev, cur, axis=axis, val=val))
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def _segment_intersect_axis(
    p1: npt.NDArray[np.float32],
    p2: npt.NDArray[np.float32],
    *,
    axis: int,
    val: float,
) -> npt.NDArray[np.float32]:
    """Return the (x, y) where segment p1→p2 crosses ``axis = val``."""
    other = 1 - axis
    denom = p2[axis] - p1[axis]
    # Caller guarantees the segment crosses the boundary, so ``denom``
    # is non-zero. Tiny denominators (near-parallel) are still safe:
    # the resulting intersection is just close to one endpoint.
    t = (val - p1[axis]) / denom if denom != 0 else 0.0
    out = np.empty(2, dtype=np.float32)
    out[axis] = float(val)
    out[other] = float(p1[other] + t * (p2[other] - p1[other]))
    return out
