"""Resolve a ``data=`` argument to its train / val / test splits.

One entry point, :func:`resolve_dataset`, accepts either of two shapes
and converges them onto a single internal representation
(``{split: DataSplit}``):

* **A directory** — the Roboflow / convention layout. ``train/`` is
  required; ``val/`` (or ``valid/``) and ``test/`` are optional. Each
  split directory holds its images and a COCO annotation JSON, found by
  :func:`_find_annotations`. No descriptor file needed.

* **A ``.yaml`` descriptor** — the YOLO-style layout, for when images and
  annotations don't sit in one convention tree. ``path`` is an optional
  root the splits resolve against; each split value is either a directory
  string (json auto-found, same as the directory form) or an explicit
  ``{images, annotations}`` mapping.

Class names are **not** an input — they live in the COCO ``categories``
and are read downstream by :func:`mayaku.data.load_coco_json`. The
descriptor answers only "where are the splits", never "what are the
classes", so there is one source of truth and nothing to reconcile.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import yaml

__all__ = ["DataSplit", "resolve_dataset"]

# Probed in order when a split is given as a bare directory. Roboflow's
# COCO export writes ``_annotations.coco.json``; the other two cover the
# common hand-rolled names. A lone ``*.json`` is accepted as a fallback.
_ANNOTATION_FILENAMES = ("_annotations.coco.json", "annotations.json", "instances.json")

# Canonical split -> directory names probed under a dataset root. "val"
# and the Roboflow "valid" both map to the canonical "val".
_SPLIT_DIR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}


class DataSplit(NamedTuple):
    """Resolved location of one split's images and COCO annotations."""

    images: Path
    annotations: Path


def resolve_dataset(data: str | Path) -> dict[str, DataSplit]:
    """Resolve ``data`` to ``{split: DataSplit}`` with ``train`` guaranteed.

    ``data`` is a dataset directory or a ``.yaml`` descriptor (see the
    module docstring). Optional ``val`` / ``test`` keys appear only when
    those splits exist. Raises on a missing train split, a missing or
    ambiguous annotation file, or an unsupported ``data`` value.
    """
    path = Path(data)

    if path.is_dir():
        return _resolve_from_dir(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset descriptor not found: {path}")
        return _resolve_from_yaml(path)

    raise ValueError(
        f"data must be a dataset directory or a .yaml descriptor; got {path}. "
        "A single COCO .json is not accepted — point at the split directory "
        "that contains it, or write a .yaml descriptor."
    )


def _resolve_from_dir(root: Path) -> dict[str, DataSplit]:
    """Discover splits under a dataset root by directory convention."""
    splits: dict[str, DataSplit] = {}
    for canonical, candidates in _SPLIT_DIR_CANDIDATES.items():
        for name in candidates:
            directory = root / name
            if directory.is_dir():
                splits[canonical] = DataSplit(directory, _find_annotations(directory))
                break
    if "train" not in splits:
        raise FileNotFoundError(
            f"No 'train' split under {root}. Expected a 'train/' subdirectory "
            "with images and a COCO annotation JSON (optionally 'val'/'valid' "
            "and 'test' alongside it)."
        )
    return splits


def _resolve_from_yaml(descriptor: Path) -> dict[str, DataSplit]:
    """Resolve splits from a YOLO-style ``.yaml`` descriptor."""
    raw = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{descriptor}: top-level YAML must be a mapping.")

    root = _root_dir(raw.get("path"), descriptor.parent)

    splits: dict[str, DataSplit] = {}
    for key, value in raw.items():
        if key not in ("train", "val", "valid", "test") or not value:
            continue
        canonical = "val" if key in ("val", "valid") else key
        if canonical in splits:
            raise ValueError(f"{descriptor}: specify only one of 'val' / 'valid'.")
        splits[canonical] = _resolve_split(value, root, descriptor)

    if "train" not in splits:
        raise ValueError(f"{descriptor}: descriptor must define a 'train' split.")
    return splits


def _resolve_split(value: Any, root: Path, descriptor: Path) -> DataSplit:
    """Resolve one descriptor split value to a :class:`DataSplit`.

    ``value`` is either a directory string (json auto-found) or an
    ``{images, annotations}`` mapping with explicit paths.
    """
    if isinstance(value, Mapping):
        try:
            images = _abs(value["images"], root)
            annotations = _abs(value["annotations"], root)
        except KeyError as missing:
            raise ValueError(
                f"{descriptor}: split mapping needs both 'images' and "
                f"'annotations' keys (missing {missing})."
            ) from None
        if not images.is_dir():
            raise NotADirectoryError(f"{descriptor}: images directory not found: {images}")
        if not annotations.is_file():
            raise FileNotFoundError(f"{descriptor}: annotations file not found: {annotations}")
        return DataSplit(images, annotations)

    if isinstance(value, str):
        directory = _abs(value, root)
        if not directory.is_dir():
            raise NotADirectoryError(f"{descriptor}: split directory not found: {directory}")
        return DataSplit(directory, _find_annotations(directory))

    raise ValueError(
        f"{descriptor}: split value must be a directory string or an "
        f"{{images, annotations}} mapping; got {type(value).__name__}."
    )


def _find_annotations(directory: Path) -> Path:
    """Find the single COCO annotation JSON inside ``directory``.

    Tries the known filenames first, then falls back to a lone ``*.json``.
    Raises when none — or more than one — candidate exists.
    """
    for name in _ANNOTATION_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate

    loose = sorted(directory.glob("*.json"))
    if len(loose) == 1:
        return loose[0]
    if not loose:
        raise FileNotFoundError(
            f"No COCO annotation JSON in {directory}. Expected one of "
            f"{', '.join(_ANNOTATION_FILENAMES)}, or a single *.json file."
        )
    raise FileNotFoundError(
        f"Multiple JSON files in {directory}; cannot pick the annotation file "
        f"automatically. Rename the one you want to '{_ANNOTATION_FILENAMES[0]}'. "
        f"Found: {[p.name for p in loose]}."
    )


def _root_dir(path_value: Any, descriptor_parent: Path) -> Path:
    """Resolve the descriptor's ``path`` root (relative to the .yaml)."""
    return descriptor_parent if path_value is None else _abs(path_value, descriptor_parent)


def _abs(value: Any, root: Path) -> Path:
    """Resolve ``value`` against ``root`` unless it is already absolute."""
    path = Path(value)
    return path if path.is_absolute() else root / path
