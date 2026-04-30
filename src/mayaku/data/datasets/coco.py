"""COCO JSON loader → standard dataset dicts.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §5.4 (``load_coco_json``) for the
three in-scope annotation types: bounding boxes, polygon segmentations,
and 17-keypoint COCO Person.

Two design adjustments relative to the reference:

1. **Channel order is not stored on dataset dicts.** Per ADR 002 the
   channel format is a contract, not metadata; ``"file_name"`` is just
   a path, the mapper decodes it RGB.

2. **Categories are remapped to a contiguous ``[0, num_classes)`` range
   eagerly** when ``thing_dataset_id_to_contiguous_id`` is available on
   the supplied :class:`Metadata`. Otherwise the ids pass through
   unchanged and the caller is responsible for remapping later
   (Detectron2 defers this to ``DatasetMapper`` via the metadata
   catalog; we prefer to fail-fast at load time so a missing remap
   shows up in dataset construction, not in the middle of a training
   epoch).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO

from mayaku.data.catalog import Metadata
from mayaku.structures.boxes import BoxMode

__all__ = ["build_coco_metadata", "load_coco_json"]


def load_coco_json(
    json_path: str | Path,
    image_root: str | Path,
    metadata: Metadata,
    *,
    extra_annotation_keys: Iterable[str] | None = None,
    keep_segmentation: bool = True,
    keep_keypoints: bool = True,
) -> list[dict[str, Any]]:
    """Parse a COCO-format JSON into standard dataset dicts.

    Args:
        json_path: Path to a COCO-format JSON (``annotations/*.json``).
        image_root: Directory that ``image_info["file_name"]`` is
            resolved against.
        metadata: Per-dataset facts. ``thing_dataset_id_to_contiguous_id``
            is consulted to remap category ids; if it is empty the
            on-disk ids are kept verbatim and a sanity-check ensures
            they are already contiguous from 0.
        extra_annotation_keys: Optional iterable of extra per-annotation
            keys to copy through into the dataset dict (e.g. dataset-
            specific attributes). The standard set
            ``{bbox, bbox_mode, category_id, iscrowd, segmentation, keypoints}``
            is always carried (subject to the ``keep_*`` knobs below).
        keep_segmentation: When ``False``, ``ann["segmentation"]`` is
            dropped from every record. Set this for detection-only
            training (no mask head): COCO 2017's polygon segmentations
            are ~3-4 GB of small Python objects that the loader would
            otherwise hold for the lifetime of the run, even though the
            mapper never reads them.
        keep_keypoints: When ``False``, ``ann["keypoints"]`` is dropped.
            Set this when ``meta_architecture != "keypoint_rcnn"``.

    Returns:
        A list of dicts (one per image) in the format documented in
        `DETECTRON2_TECHNICAL_SPEC.md` §5.1.
    """
    json_path = str(json_path)
    image_root = Path(image_root)
    coco = COCO(json_path)

    img_ids = sorted(coco.imgs.keys())
    remap = dict(metadata.thing_dataset_id_to_contiguous_id)
    if not remap:
        # No remap supplied — accept it iff category ids are already
        # contiguous from 0 to num_classes - 1.
        sorted_cat_ids = sorted(coco.cats.keys())
        if sorted_cat_ids != list(range(len(sorted_cat_ids))):
            raise ValueError(
                f"Metadata {metadata.name!r} has no thing_dataset_id_to_contiguous_id "
                "and the COCO JSON's category ids are not already contiguous from 0; "
                "supply an explicit remap via build_coco_metadata()."
            )
        remap = {cid: cid for cid in sorted_cat_ids}

    extra_keys = tuple(extra_annotation_keys or ())

    dataset_dicts: list[dict[str, Any]] = []
    for img_id in img_ids:
        img_info = coco.imgs[img_id]
        record: dict[str, Any] = {
            "file_name": str(image_root / img_info["file_name"]),
            "image_id": img_id,
            "height": int(img_info["height"]),
            "width": int(img_info["width"]),
            "annotations": [],
        }
        for ann in coco.imgToAnns.get(img_id, []):
            if ann.get("ignore", 0):
                continue
            cat_id = ann["category_id"]
            if cat_id not in remap:
                raise ValueError(
                    f"COCO category_id={cat_id} is not in metadata "
                    f"{metadata.name!r} thing_dataset_id_to_contiguous_id"
                )
            obj: dict[str, Any] = {
                "iscrowd": int(ann.get("iscrowd", 0)),
                "bbox": list(ann["bbox"]),
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": remap[cat_id],
            }
            if keep_segmentation and ann.get("segmentation"):
                obj["segmentation"] = ann["segmentation"]
            if keep_keypoints and "keypoints" in ann:
                kp = list(ann["keypoints"])
                # Spec §5.4: COCO uses pixel-corner indexing for keypoints;
                # add 0.5 to convert to the pixel-center convention used
                # by the heatmap encoder (Step 4).
                for i in range(0, len(kp), 3):
                    kp[i] = float(kp[i]) + 0.5
                    kp[i + 1] = float(kp[i + 1]) + 0.5
                obj["keypoints"] = kp
            for k in extra_keys:
                if k in ann:
                    obj[k] = ann[k]
            record["annotations"].append(obj)
        dataset_dicts.append(record)
    return dataset_dicts


def build_coco_metadata(
    name: str,
    json_path: str | Path,
    *,
    keypoint_names: tuple[str, ...] | None = None,
    keypoint_flip_indices: tuple[int, ...] | None = None,
) -> Metadata:
    """Build :class:`Metadata` from a COCO JSON's category list.

    Reads the JSON's ``categories`` array, sorts by id, and produces a
    contiguous-id remap plus the matching ``thing_classes`` tuple.
    Useful for quickly registering a dataset without manually listing
    class names.

    When the JSON carries keypoint annotations (the ``person_keypoints_*``
    files) and the caller didn't pass ``keypoint_names`` /
    ``keypoint_flip_indices`` explicitly, both are derived from the
    JSON: ``keypoint_names`` from the (single) keypoint-bearing
    category's ``keypoints`` array, and ``keypoint_flip_indices`` from
    the ``left_X`` / ``right_X`` naming convention. This mirrors
    Detectron2's `data/datasets/coco.py:load_coco_json` behaviour.
    """
    coco = COCO(str(json_path))
    cat_ids = sorted(coco.cats.keys())
    remap = {cid: i for i, cid in enumerate(cat_ids)}
    classes = tuple(coco.cats[cid]["name"] for cid in cat_ids)

    if keypoint_names is None and keypoint_flip_indices is None:
        cat_kps = [tuple(c["keypoints"]) for c in coco.cats.values() if c.get("keypoints")]
        if len(cat_kps) == 1:
            keypoint_names = cat_kps[0]
            keypoint_flip_indices = _derive_flip_indices(keypoint_names)

    return Metadata(
        name=name,
        thing_classes=classes,
        thing_dataset_id_to_contiguous_id=remap,
        keypoint_names=keypoint_names,
        keypoint_flip_indices=keypoint_flip_indices,
    )


def _derive_flip_indices(names: tuple[str, ...]) -> tuple[int, ...]:
    """Map each keypoint to its horizontal-flip partner index.

    A keypoint named ``left_X`` swaps with ``right_X`` (and vice versa);
    everything else (e.g. ``nose``) maps to itself. Returns an index
    permutation of ``range(len(names))``.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    out: list[int] = []
    for i, n in enumerate(names):
        if n.startswith("left_"):
            partner = "right_" + n.removeprefix("left_")
        elif n.startswith("right_"):
            partner = "left_" + n.removeprefix("right_")
        else:
            partner = n
        out.append(name_to_idx.get(partner, i))
    return tuple(out)
