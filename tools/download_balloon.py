"""Fetch the matterport balloon dataset and convert to COCO-format JSON.

A tiny public dataset (61 train + 13 val images, single class) used by
``benchmarks/training_validation/tier1_mps.py`` for the MPS sanity /
diagnostic run. Originally bundled with the matterport Mask R-CNN
implementation; reused here because it's small enough to round-trip
on a Mac in minutes and large enough to produce a meaningful loss
curve.

The original archive ships a ``via_region_data.json`` per split with
VIA-format polygon annotations. We extract bounding boxes from those
polygons and emit a minimal COCO-format JSON suitable for Faster
R-CNN box-only training (no segmentation field — the box head ignores
it and the mask head isn't built).

Usage:
    python tools/download_balloon.py --output /path/to/balloon

After running, the layout is::

    /path/to/balloon/
    ├── train/                      # 61 images
    ├── train_coco.json             # COCO-format annotations
    ├── val/                        # 13 images
    └── val_coco.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"


def _download(dst: Path) -> None:
    if dst.exists():
        print(f"[balloon] archive already present: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[balloon] downloading {ARCHIVE_URL} -> {dst}")
    # urlretrieve is fine here — public GitHub release URL, ~38 MB.
    urllib.request.urlretrieve(ARCHIVE_URL, dst)


def _bbox_from_polygon(xs: list[float], ys: list[float]) -> list[float]:
    """COCO bbox = [x_min, y_min, width, height]."""
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _via_to_coco(via_json: Path, image_dir: Path, split_name: str) -> dict[str, object]:
    via = json.loads(via_json.read_text())
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    image_id = 0
    ann_id = 0
    for entry in via.values():
        filename = entry["filename"]
        image_path = image_dir / filename
        if not image_path.exists():
            continue
        # Pillow is already a project dep; use it for the size lookup.
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        # VIA stores regions as either a dict (keyed by region index) or a
        # list, depending on the export version. Normalize to an iterable
        # of region dicts.
        regions = entry.get("regions", [])
        if isinstance(regions, dict):
            regions = list(regions.values())
        for region in regions:
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if not xs or not ys:
                continue
            x, y, w, h = _bbox_from_polygon(xs, ys)
            # COCO-format polygon: list-of-list, alternating x,y. Single
            # polygon per region (balloons aren't multi-piece).
            polygon = [coord for pair in zip(xs, ys, strict=True) for coord in pair]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [polygon],
            })
            ann_id += 1
        image_id += 1
    print(
        f"[balloon] {split_name}: {len(images)} images, "
        f"{len(annotations)} annotations"
    )
    return {
        "info": {"description": f"matterport balloon ({split_name})"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "balloon"}],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--output", type=Path, required=True,
        help="Directory to extract into and write *_coco.json under.",
    )
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "balloon_dataset.zip"
    _download(archive)

    if not (args.output / "balloon").exists():
        print(f"[balloon] extracting {archive}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(args.output)

    extracted = args.output / "balloon"
    for split in ("train", "val"):
        image_dir = extracted / split
        via_path = image_dir / "via_region_data.json"
        if not via_path.exists():
            print(f"[balloon] missing {via_path}", file=sys.stderr)
            return 2
        coco = _via_to_coco(via_path, image_dir, split)
        out = args.output / f"{split}_coco.json"
        out.write_text(json.dumps(coco, indent=2))
        print(f"[balloon] wrote {out}")

    print(f"[balloon] done. images under {extracted}/{{train,val}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
