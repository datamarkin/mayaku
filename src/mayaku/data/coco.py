"""COCO-format annotations -> per-image label arrays.

Reads the annotation JSON directly rather than a converted label layout, so
the image ids `pycocotools` needs for evaluation come from the same file the
boxes came from, and there is no conversion step to get wrong.
"""

import dataclasses
import json
import os

import numpy as np

from mayaku.data.polygons import Polys
from mayaku.data.serialize import SerializedList

# Left/right pairs of the COCO person skeleton, the fallback for 17 unnamed
# keypoints; named keypoints derive their pairs from the names.
COCO_FLIP_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
                   (13, 14), (15, 16))


def keep_ann(a):
    """Whether an annotation is trained on: a real (non-crowd) object bigger
    than a pixel each side."""
    if a.get("iscrowd", 0):
        return False
    w, h = a["bbox"][2], a["bbox"][3]
    return w > 1 and h > 1


def flip_pairs(names, k):
    """Keypoint index pairs swapped by a horizontal flip.

    `left_X` pairs with `right_X` when the dataset names its keypoints; 17
    unnamed keypoints are taken to be COCO's person skeleton; anything else
    has no pairs.
    """
    if names and len(names) == k:
        idx = {n: i for i, n in enumerate(names)}
        return tuple((i, idx["right_" + n[5:]]) for i, n in enumerate(names)
                     if n.startswith("left_") and "right_" + n[5:] in idx)
    return COCO_FLIP_PAIRS if k == 17 else ()


@dataclasses.dataclass
class CocoLabels:
    """Everything a dataset needs from one annotation file, per image in file
    order. `labels[i]` is (n, 5) [class, x1, y1, x2, y2] in original pixels
    with dense class indices; `polys[i]` and `kpts[i]` ((n, 3K)) are empty
    unless requested.

    The per-image fields are `SerializedList`s (one bytes buffer each rather
    than a Python object per image, so DataLoader workers do not copy the
    whole label set page by page as they touch refcounts, and every access
    returns a fresh object the caller may modify), and `shapes` one array.
    """

    ann_path: str
    kpt_ann_path: str      # where keypoint ground truth is scored from
    num_degenerate: int    # non-crowd annotations dropped for a side of a pixel or less
    cat_ids: list          # dense class index -> the file's category id
    class_names: list      # dense class index -> name
    kpt_names: list        # keypoint names from the categories, when the file has them
    kpt_flip_pairs: tuple
    ids: SerializedList
    files: SerializedList
    shapes: np.ndarray     # (n, 2) int32 (height, width)
    labels: SerializedList
    polys: SerializedList
    kpts: SerializedList


def load_coco(images, annotations, masks=False, kpt=0, kpt_annotations=None):
    """Parse a COCO instances JSON.

    `masks` keeps every instance's segmentation (polygons or RLE) as `Polys`.
    `kpt` = K keypoints per instance, read from this file or from a second
    COCO file keyed by annotation id (COCO ships person keypoints separately
    from its instances file). Crowd regions and degenerate boxes are dropped;
    every image is kept, labelled or not.
    """
    with open(annotations) as f:
        data = json.load(f)
    kp_by_id, kpt_names = {}, []
    if kpt:
        src = data
        if kpt_annotations and kpt_annotations != annotations:
            with open(kpt_annotations) as f:
                src = json.load(f)
        kp_by_id = {a["id"]: a["keypoints"] for a in src["annotations"]
                    if a.get("keypoints") and a.get("num_keypoints", 1) > 0}
        kpt_names = next((list(c["keypoints"]) for c in src.get("categories", [])
                          if len(c.get("keypoints", [])) == kpt), [])

    # Category ids can be sparse (COCO's 80 classes run to 90); the network
    # emits a dense 0..nc-1, so keep the map both ways for the evaluator.
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_ids = [c["id"] for c in cats]
    dense = {c: i for i, c in enumerate(cat_ids)}

    per_image, segs, kps = {}, {}, {}
    degenerate = 0
    for a in data["annotations"]:
        if not keep_ann(a):
            degenerate += not a.get("iscrowd", 0)
            continue
        x, y, w, h = a["bbox"]
        per_image.setdefault(a["image_id"], []).append(
            [dense[a["category_id"]], x, y, x + w, y + h])
        if masks:
            seg = a.get("segmentation")
            segs.setdefault(a["image_id"], []).append(seg if isinstance(seg, (list, dict)) else None)
        if kpt:
            kps.setdefault(a["image_id"], []).append(kp_by_id.get(a["id"]) or [0.0] * (3 * kpt))

    ims = data["images"]
    del data
    return CocoLabels(
        ann_path=annotations,
        kpt_ann_path=kpt_annotations or annotations,
        num_degenerate=degenerate,
        cat_ids=cat_ids,
        class_names=[c.get("name", str(c["id"])) for c in cats],
        kpt_names=kpt_names,
        kpt_flip_pairs=flip_pairs(kpt_names, kpt),
        ids=SerializedList([im["id"] for im in ims]),
        files=SerializedList([os.path.join(images, im["file_name"]) for im in ims]),
        shapes=np.array([(im["height"], im["width"]) for im in ims], np.int32).reshape(-1, 2),
        labels=SerializedList([np.array(per_image[im["id"]], np.float32)
                               if im["id"] in per_image else np.zeros((0, 5), np.float32)
                               for im in ims]),
        polys=SerializedList([Polys.from_coco(segs.get(im["id"], [])) for im in ims]
                             if masks else []),
        kpts=SerializedList([np.array(kps.get(im["id"], []), np.float32).reshape(-1, 3 * kpt)
                             for im in ims] if kpt else []),
    )
