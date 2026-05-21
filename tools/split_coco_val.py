"""Split a COCO annotations file into train/eval subsets sharing one image directory.

Used to prepare the COCO val 5k research benchmark for R-CNN head improvement
experiments (see ``head_plan.md``). The standard COCO release ships
``annotations/instances_val2017.json`` plus a ``val2017/`` image directory.
For research we want a 4k train / 1k eval split, but the loader hard-fails
if the JSON references missing image files (``src/mayaku/data/mapper.py``
calls ``PIL.Image.open`` lazily and PIL raises ``FileNotFoundError``).

This script splits the **annotations** only — both output JSONs point at the
same unmodified image directory. Image IDs are shuffled deterministically by
seed so the split is reproducible across machines; commit the resulting JSONs
to lock the partition.

Usage:
    python tools/split_coco_val.py --input /path/to/instances_val2017.json
    python tools/split_coco_val.py \\
        --input /path/to/instances_val2017.json \\
        --output-dir /path/to/annotations \\
        --train-size 4000 \\
        --seed 42

Output filenames are derived from ``--train-size`` / total — e.g. 4000 train
out of 5000 → ``instances_val2017_train4k.json`` and
``instances_val2017_eval1k.json``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _split(
    coco: dict,
    train_size: int,
    seed: int,
) -> tuple[dict, dict]:
    ids = sorted(im["id"] for im in coco["images"])
    if train_size >= len(ids):
        raise ValueError(
            f"--train-size ({train_size}) must be < total images ({len(ids)})"
        )

    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    train_ids = set(shuffled[:train_size])
    eval_ids = set(shuffled[train_size:])

    def _subset(selected: set[int]) -> dict:
        return {
            **{k: v for k, v in coco.items() if k not in {"images", "annotations"}},
            "images": [im for im in coco["images"] if im["id"] in selected],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in selected],
        }

    return _subset(train_ids), _subset(eval_ids)


def _size_suffix(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source COCO JSON (e.g. instances_val2017.json).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the two split JSONs (default: same directory as --input).",
    )
    p.add_argument(
        "--train-size",
        type=int,
        default=4000,
        help="Number of images in the train subset; rest go to eval (default: 4000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling (default: 42).",
    )
    args = p.parse_args()

    if not args.input.is_file():
        print(f"error: --input not found: {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(args.input.read_text())
    total = len(coco["images"])
    train_doc, eval_doc = _split(coco, args.train_size, args.seed)

    stem = args.input.stem  # e.g. "instances_val2017"
    train_suffix = _size_suffix(len(train_doc["images"]))
    eval_suffix = _size_suffix(len(eval_doc["images"]))
    train_path = out_dir / f"{stem}_train{train_suffix}.json"
    eval_path = out_dir / f"{stem}_eval{eval_suffix}.json"

    train_path.write_text(json.dumps(train_doc))
    eval_path.write_text(json.dumps(eval_doc))

    print(
        f"wrote {len(train_doc['images'])} images / "
        f"{len(train_doc['annotations'])} annotations → {train_path}"
    )
    print(
        f"wrote {len(eval_doc['images'])} images / "
        f"{len(eval_doc['annotations'])} annotations → {eval_path}"
    )
    print(f"total: {total} images split with seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
