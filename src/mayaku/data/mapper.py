"""DatasetMapper: standard dataset dict → trainable batch element.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §5.3 (``DatasetMapper.__call__``)
with the simplifications that come out of our scope:

* RGB only (ADR 002) — no ``INPUT.FORMAT`` branch.
* No semantic segmentation (out of scope).
* Polygon vs bitmask mask format is a constructor argument, not a
  config-string switch.
* The keypoint flip-pair permutation is read from the dataset's
  :class:`mayaku.data.catalog.Metadata` (or supplied directly), not
  reconstructed from a ``MetadataCatalog`` global.

Output dict shape (matches the rest of the pipeline's expectations):

```
{
    "image":      Tensor[3, H, W] float32 RGB (post-augmentation),
    "instances":  Instances(image_size=(H, W), gt_boxes, gt_classes,
                            gt_masks?, gt_keypoints?),
    "image_id":   int,
    "height":     int,        # original input height
    "width":      int,        # original input width
}
```

The mapper does *not* normalise pixels — that's done by the model's
forward pass against the configured ``pixel_mean`` / ``pixel_std`` (Step
7 onwards). The mapper does *not* pad — that's :class:`ImageList.from_tensors`
inside the model.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from mayaku.data.catalog import Metadata
from mayaku.data.transforms import (
    AugInput,
    AugmentationList,
    TransformList,
)
from mayaku.structures.boxes import BoxMode
from mayaku.structures.instances import Instances
from mayaku.structures.keypoints import Keypoints
from mayaku.structures.masks import BitMasks, PolygonMasks
from mayaku.utils.image import read_image

__all__ = ["DatasetMapper"]

MaskFormat = Literal["polygon", "bitmask"]


class DatasetMapper:
    """Picklable mapper for a PyTorch ``Dataset`` / ``IterableDataset``.

    Args:
        augmentations: Sequence of :class:`Augmentation` to apply per
            sample. The first one typically resizes; the rest may
            flip / jitter. For inference, pass only a deterministic
            resize.
        is_train: If false, annotations are dropped from the output and
            the bare ``"image"`` is returned (matches Detectron2's test
            mapper behaviour).
        mask_format: ``"polygon"`` keeps :class:`PolygonMasks` lazily —
            cheaper at construction, rasterised once per ROI by the
            mask head. ``"bitmask"`` rasterises eagerly to
            :class:`BitMasks` (correct for datasets that ship RLE-only
            masks, since RLE → polygon is lossy).
        keypoint_on: Whether to materialise ``gt_keypoints``. When
            ``True``, ``metadata.keypoint_flip_indices`` must be set so
            horizontal flip swaps left/right correctly.
        metadata: The dataset's :class:`Metadata`. Read for
            ``keypoint_flip_indices`` only.
    """

    def __init__(
        self,
        augmentations: Sequence[Any],
        *,
        is_train: bool = True,
        mask_format: MaskFormat = "polygon",
        keypoint_on: bool = False,
        metadata: Metadata | None = None,
        deepcopy_input: bool = True,
    ) -> None:
        self.is_train = is_train
        self.mask_format: MaskFormat = mask_format
        self.keypoint_on = keypoint_on
        self.metadata = metadata
        self.deepcopy_input = deepcopy_input
        flip_indices: tuple[int, ...] | None = None
        if keypoint_on:
            if metadata is None or metadata.keypoint_flip_indices is None:
                raise ValueError(
                    "keypoint_on=True requires metadata.keypoint_flip_indices "
                    "(see Metadata / build_coco_metadata)."
                )
            flip_indices = metadata.keypoint_flip_indices
        self._aug_list = AugmentationList(list(augmentations), flip_indices=flip_indices)

    def __call__(self, dataset_dict: dict[str, Any]) -> dict[str, Any]:
        # Detectron2 deepcopies the dict so downstream mutation doesn't
        # bleed back into the dataset cache. When the input came from a
        # :class:`mayaku.data.SerializedList` (the train CLI's default)
        # every dict is already a fresh ``pickle.loads`` allocation, so
        # the deepcopy is wasted — and on a 90k-iter run it's the single
        # biggest source of small-object churn that drives glibc malloc
        # fragmentation. ``deepcopy_input=False`` skips it.
        dd = copy.deepcopy(dataset_dict) if self.deepcopy_input else dataset_dict
        # Multi-sample augmentations (Mosaic / MixUp / CopyPaste) supply a
        # pre-composed image directly via ``__image`` — they've already
        # combined N source files in their own coordinate space and there
        # is no on-disk file to read for the synthetic combination. The
        # dunder name avoids collisions with any user-supplied COCO keys.
        # The explicit if/else form (over a ternary) keeps the contract
        # readable for the multi-sample integration point.
        image = dd.pop("__image") if "__image" in dd else read_image(dd["file_name"])
        _check_image_size(dd, image)
        # Capture pre-augmentation dims: this is the coordinate frame the
        # source polygon / RLE segmentations live in, and the input frame
        # the transform list is calibrated for. Bitmask rasterisation has
        # to happen at this size, not at post-aug size — see _build_masks.
        pre_h, pre_w = image.shape[:2]

        aug_input = AugInput(image=image)
        transforms = self._aug_list(aug_input)
        image = aug_input.image
        h, w = image.shape[:2]
        dd["image"] = _image_to_tensor(image)

        if not self.is_train:
            dd.pop("annotations", None)
            return dd

        annos = dd.pop("annotations", [])
        # Detectron2 also drops iscrowd annotations from training
        # (`dataset_mapper.py:165`); they participate only in evaluation.
        annos = [a for a in annos if not a.get("iscrowd", 0)]
        instances = self._annotations_to_instances(
            annos, transforms, image_size=(h, w), pre_image_size=(pre_h, pre_w)
        )
        dd["instances"] = instances
        return dd

    # ------------------------------------------------------------------

    def _annotations_to_instances(
        self,
        annos: list[dict[str, Any]],
        transforms: TransformList,
        image_size: tuple[int, int],
        pre_image_size: tuple[int, int],
    ) -> Instances:
        h, w = image_size

        # --- boxes (always present)
        if not annos:
            inst = Instances(image_size=(h, w))
            inst.gt_boxes = torch.zeros(0, 4, dtype=torch.float32)
            inst.gt_classes = torch.zeros(0, dtype=torch.int64)
            return inst

        boxes_in = np.stack(
            [
                BoxMode.convert(
                    torch.tensor(a["bbox"], dtype=torch.float32),
                    a["bbox_mode"],
                    BoxMode.XYXY_ABS,
                ).numpy()
                for a in annos
            ],
            axis=0,
        ).astype(np.float32)
        boxes_out = transforms.apply_box(boxes_in)
        # Clip to image bounds (matches detection_utils.py)
        boxes_out[:, 0::2] = np.clip(boxes_out[:, 0::2], 0, w)
        boxes_out[:, 1::2] = np.clip(boxes_out[:, 1::2], 0, h)

        # Drop GT that became degenerate (width<=1e-5 or height<=1e-5) after
        # augmentation+clipping. Mirrors detectron2.data.detection_utils.
        # filter_empty_instances (threshold=1e-5). A zero-width src_box
        # in Box2BoxTransform.get_deltas produces log(0) = -inf during
        # anchor matching, which silently NaN-s the entire training run.
        widths = boxes_out[:, 2] - boxes_out[:, 0]
        heights = boxes_out[:, 3] - boxes_out[:, 1]
        keep_mask = (widths > 1e-5) & (heights > 1e-5)
        if not keep_mask.all():
            boxes_out = boxes_out[keep_mask]
            annos = [a for a, k in zip(annos, keep_mask, strict=True) if k]

        gt_boxes = torch.from_numpy(boxes_out).to(torch.float32)
        gt_classes = torch.tensor([a["category_id"] for a in annos], dtype=torch.int64)

        # --- masks (optional)
        gt_masks: PolygonMasks | BitMasks | None = None
        if any("segmentation" in a for a in annos):
            gt_masks = self._build_masks(
                annos, transforms, image_size=(h, w), pre_image_size=pre_image_size
            )

        # --- keypoints (optional)
        gt_keypoints: Keypoints | None = None
        if self.keypoint_on and any("keypoints" in a for a in annos):
            gt_keypoints = self._build_keypoints(annos, transforms, image_size=(h, w))

        inst = Instances(image_size=(h, w))
        inst.gt_boxes = gt_boxes
        inst.gt_classes = gt_classes
        if gt_masks is not None:
            inst.gt_masks = gt_masks
        if gt_keypoints is not None:
            inst.gt_keypoints = gt_keypoints
        return inst

    def _build_masks(
        self,
        annos: list[dict[str, Any]],
        transforms: TransformList,
        image_size: tuple[int, int],
        pre_image_size: tuple[int, int],
    ) -> PolygonMasks | BitMasks:
        h, w = image_size
        pre_h, pre_w = pre_image_size
        if self.mask_format == "polygon":
            polygons_per_instance: list[list[npt.NDArray[np.float32]]] = []
            for a in annos:
                seg = a.get("segmentation")
                if not seg:
                    polygons_per_instance.append([])
                    continue
                if not isinstance(seg, list):
                    raise ValueError(
                        f"mask_format='polygon' but annotation for image_id="
                        f"{a.get('image_id')!r} carries a non-polygon segmentation "
                        f"of type {type(seg).__name__}; switch to 'bitmask' or "
                        "convert the dataset's RLE entries to polygons offline."
                    )
                polys = [np.asarray(p, dtype=np.float32) for p in seg]
                polygons_per_instance.append(transforms.apply_polygons(polys))
            return PolygonMasks(polygons_per_instance)
        # bitmask: rasterise here (slow but correct for RLE-only datasets).
        # Polygons / RLEs are in the *pre-augmentation* coordinate frame
        # (source-image dims for plain samples, canvas dims for Mosaic-style
        # synthetic samples — both equal `pre_image_size`). Rasterising at
        # post-aug `(h, w)` would clip / scale polygon coords incorrectly
        # AND hand the transform list a wrong-shape mask, since the
        # ResizeTransform was calibrated to consume `(pre_h, pre_w)` input.
        # Output `bit[i]` is post-aug because `apply_segmentation` performs
        # the pre→post resize that the rest of the transform pipeline expects.
        from pycocotools import mask as coco_mask

        bit = np.zeros((len(annos), h, w), dtype=np.bool_)
        for i, a in enumerate(annos):
            seg = a.get("segmentation")
            if not seg:
                continue
            rasterised = _segmentation_to_bitmask(seg, pre_h, pre_w, coco_mask)
            bit[i] = transforms.apply_segmentation(rasterised.astype(np.uint8)).astype(bool)
        return BitMasks(torch.from_numpy(bit))

    def _build_keypoints(
        self,
        annos: list[dict[str, Any]],
        transforms: TransformList,
        image_size: tuple[int, int],
    ) -> Keypoints:
        # K is determined by the first annotation that carries keypoints;
        # all others must agree.
        k = next(len(a["keypoints"]) // 3 for a in annos if "keypoints" in a)
        kp: npt.NDArray[np.float32] = np.zeros((len(annos), k, 3), dtype=np.float32)
        for i, a in enumerate(annos):
            raw = a.get("keypoints")
            if raw is None:
                # Treat missing keypoints as fully invisible.
                continue
            arr = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
            if arr.shape[0] != k:
                raise ValueError(
                    f"All keypoint annotations must have K={k} points; got "
                    f"{arr.shape[0]} for one of the instances"
                )
            kp[i] = arr
        kp = transforms.apply_keypoints(kp)
        # Mark out-of-bounds keypoints as invisible (v=0) per spec §5.3.
        h, w = image_size
        oob = (kp[..., 0] < 0) | (kp[..., 0] > w) | (kp[..., 1] < 0) | (kp[..., 1] > h)
        kp[oob, 0] = 0.0
        kp[oob, 1] = 0.0
        kp[oob, 2] = 0.0
        flip_indices_t: Tensor | None = None
        if self.metadata is not None and self.metadata.keypoint_flip_indices is not None:
            flip_indices_t = torch.tensor(self.metadata.keypoint_flip_indices, dtype=torch.long)
        return Keypoints(torch.from_numpy(kp), flip_indices=flip_indices_t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_image_size(dd: dict[str, Any], image: npt.NDArray[np.uint8]) -> None:
    h, w = image.shape[:2]
    if "height" in dd and dd["height"] != h:
        raise ValueError(
            f"image_id={dd.get('image_id')!r}: dataset dict says height={dd['height']} "
            f"but decoded image is {h} tall"
        )
    if "width" in dd and dd["width"] != w:
        raise ValueError(
            f"image_id={dd.get('image_id')!r}: dataset dict says width={dd['width']} "
            f"but decoded image is {w} wide"
        )


def _image_to_tensor(image: npt.NDArray[np.uint8]) -> Tensor:
    """``(H, W, 3) uint8`` → ``(3, H, W) float32`` (RGB, no normalisation)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"DatasetMapper expects (H, W, 3) RGB; got shape {image.shape}. "
            "Use mayaku.utils.bgr_to_rgb at the boundary if your decoder "
            "produced BGR."
        )
    arr = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(arr).to(dtype=torch.float32)


def _segmentation_to_bitmask(seg: Any, h: int, w: int, coco_mask: Any) -> npt.NDArray[np.bool_]:
    """Decode polygon-list / RLE / encoded-RLE into a bool ``(H, W)`` mask."""
    if isinstance(seg, list):
        rles = coco_mask.frPyObjects(seg, h, w)
        rle = coco_mask.merge(rles)
    elif isinstance(seg, dict):
        # RLE — may need fr-encoding for uncompressed
        rle = coco_mask.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
    else:
        raise ValueError(f"Unknown segmentation format: {type(seg).__name__}")
    decoded: npt.NDArray[np.bool_] = coco_mask.decode(rle).astype(np.bool_)
    return decoded
