"""Tests for :mod:`mayaku.data.catalog`."""

from __future__ import annotations

import pytest

from mayaku.data.catalog import DatasetCatalog, Metadata


def _toy_loader() -> list[dict[str, object]]:
    return [{"file_name": "a.jpg", "image_id": 1, "annotations": []}]


def test_register_get_metadata_round_trip() -> None:
    cat = DatasetCatalog()
    md = Metadata(name="toy", thing_classes=("a", "b"))
    cat.register("toy", _toy_loader, md)
    assert "toy" in cat
    assert cat.metadata("toy") is md
    assert cat.get("toy")[0]["file_name"] == "a.jpg"
    assert cat.names() == ["toy"]


def test_register_rejects_duplicate_names() -> None:
    cat = DatasetCatalog()
    md = Metadata(name="toy", thing_classes=("a",))
    cat.register("toy", _toy_loader, md)
    with pytest.raises(ValueError, match="already registered"):
        cat.register("toy", _toy_loader, md)


def test_register_rejects_metadata_name_mismatch() -> None:
    cat = DatasetCatalog()
    md = Metadata(name="something_else", thing_classes=("a",))
    with pytest.raises(ValueError, match="does not match"):
        cat.register("toy", _toy_loader, md)


def test_get_unknown_lists_available() -> None:
    cat = DatasetCatalog()
    cat.register("toy", _toy_loader, Metadata(name="toy", thing_classes=("a",)))
    with pytest.raises(KeyError, match="toy"):
        cat.get("missing")


def test_remove_and_clear() -> None:
    cat = DatasetCatalog()
    cat.register("a", _toy_loader, Metadata(name="a", thing_classes=()))
    cat.register("b", _toy_loader, Metadata(name="b", thing_classes=()))
    cat.remove("a")
    assert cat.names() == ["b"]
    cat.clear()
    assert cat.names() == []


def test_metadata_is_frozen_dataclass() -> None:
    md = Metadata(name="toy", thing_classes=("a",))
    with pytest.raises(Exception):
        md.name = "other"  # type: ignore[misc]
