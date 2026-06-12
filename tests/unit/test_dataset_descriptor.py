"""Unit tests for :func:`mayaku.data.resolve_dataset`.

The resolver is path-only — it never parses annotation content — so
these tests just build empty directories and touch JSON files, no real
COCO data or images required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mayaku.data import DataSplit, resolve_dataset


def _split(root: Path, name: str, ann: str = "_annotations.coco.json") -> Path:
    """Create ``root/name`` with an annotation file; return the directory."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / ann).write_text("{}")
    return directory


# ---------------------------------------------------------------------------
# Directory convention
# ---------------------------------------------------------------------------


def test_dir_resolves_train_val_test(tmp_path: Path) -> None:
    _split(tmp_path, "train")
    _split(tmp_path, "valid")  # Roboflow's spelling
    _split(tmp_path, "test")

    splits = resolve_dataset(tmp_path)

    assert set(splits) == {"train", "val", "test"}
    assert splits["train"] == DataSplit(
        tmp_path / "train", tmp_path / "train" / "_annotations.coco.json"
    )
    assert splits["val"].images == tmp_path / "valid"  # 'valid' maps to canonical 'val'


def test_dir_train_only(tmp_path: Path) -> None:
    _split(tmp_path, "train")
    splits = resolve_dataset(tmp_path)
    assert set(splits) == {"train"}


def test_dir_val_spelling(tmp_path: Path) -> None:
    _split(tmp_path, "train")
    _split(tmp_path, "val")  # the 'val' spelling, not 'valid'
    assert resolve_dataset(tmp_path)["val"].images == tmp_path / "val"


def test_dir_missing_train_raises(tmp_path: Path) -> None:
    _split(tmp_path, "valid")  # val but no train
    with pytest.raises(FileNotFoundError, match="No 'train' split"):
        resolve_dataset(tmp_path)


# ---------------------------------------------------------------------------
# Annotation discovery
# ---------------------------------------------------------------------------


def test_lone_json_fallback(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "whatever.json").write_text("{}")  # non-standard name, but the only json
    assert resolve_dataset(tmp_path)["train"].annotations == train / "whatever.json"


def test_no_json_raises(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="No COCO annotation JSON"):
        resolve_dataset(tmp_path)


def test_ambiguous_json_raises(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "a.json").write_text("{}")
    (train / "b.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Multiple JSON files"):
        resolve_dataset(tmp_path)


def test_known_name_wins_over_extra_json(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "_annotations.coco.json").write_text("{}")
    (train / "extra.json").write_text("{}")  # would be ambiguous without the known name
    assert resolve_dataset(tmp_path)["train"].annotations == train / "_annotations.coco.json"


# ---------------------------------------------------------------------------
# YAML descriptor
# ---------------------------------------------------------------------------


def test_yaml_dir_splits_relative_to_path(tmp_path: Path) -> None:
    _split(tmp_path, "train")
    _split(tmp_path, "valid")
    descriptor = tmp_path / "dataset.yaml"
    descriptor.write_text("path: .\ntrain: train\nval: valid\n")

    splits = resolve_dataset(descriptor)
    assert splits["train"].images == tmp_path / "train"
    assert splits["val"].images == tmp_path / "valid"


def test_yaml_explicit_mapping(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    ann = tmp_path / "ann.json"
    ann.write_text("{}")
    descriptor = tmp_path / "d.yaml"
    descriptor.write_text(f"train:\n  images: {images}\n  annotations: {ann}\n")

    assert resolve_dataset(descriptor)["train"] == DataSplit(images, ann)


def test_yaml_path_defaults_to_descriptor_dir(tmp_path: Path) -> None:
    _split(tmp_path, "train")
    descriptor = tmp_path / "dataset.yaml"
    descriptor.write_text("train: train\n")  # no 'path:' → resolves against the .yaml's dir
    assert resolve_dataset(descriptor)["train"].images == tmp_path / "train"


def test_yaml_missing_train_raises(tmp_path: Path) -> None:
    _split(tmp_path, "valid")
    descriptor = tmp_path / "dataset.yaml"
    descriptor.write_text("path: .\nval: valid\n")
    with pytest.raises(ValueError, match="must define a 'train' split"):
        resolve_dataset(descriptor)


def test_yaml_val_and_valid_conflict_raises(tmp_path: Path) -> None:
    _split(tmp_path, "train")
    _split(tmp_path, "valid")
    _split(tmp_path, "val")
    descriptor = tmp_path / "dataset.yaml"
    descriptor.write_text("path: .\ntrain: train\nval: val\nvalid: valid\n")
    with pytest.raises(ValueError, match="only one of 'val' / 'valid'"):
        resolve_dataset(descriptor)


def test_yaml_mapping_missing_key_raises(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    images.mkdir()
    descriptor = tmp_path / "d.yaml"
    descriptor.write_text(f"train:\n  images: {images}\n")  # no 'annotations'
    with pytest.raises(ValueError, match="both 'images' and 'annotations'"):
        resolve_dataset(descriptor)


def test_missing_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="descriptor not found"):
        resolve_dataset(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Rejected inputs
# ---------------------------------------------------------------------------


def test_single_json_rejected(tmp_path: Path) -> None:
    loose = tmp_path / "instances.json"
    loose.write_text("{}")
    with pytest.raises(ValueError, match=r"single COCO \.json is not accepted"):
        resolve_dataset(loose)
