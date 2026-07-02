"""Unit tests for :func:`mayaku.utils.download.download_model`'s local resolution.

Only the file-first (cache-hit) path is exercised here — it must resolve a
present checkpoint with no network and no hashing, and it owns the cosmetic
trailing-``.pth`` strip so ``name`` and ``name.pth`` map to the same file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mayaku.utils.download import download_model


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any network use fail loudly, proving the hit path never reaches it."""

    def boom(*_: object, **__: object) -> None:
        raise AssertionError("download_model hit the network on a present file")

    monkeypatch.setattr("mayaku.utils.download._fetch_manifest", boom)
    monkeypatch.setattr("mayaku.utils.download._download", boom)


def test_present_file_returned_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _no_network(monkeypatch)
    (tmp_path / "mayaku-s-det.pth").write_bytes(b"x")
    assert download_model("mayaku-s-det") == tmp_path / "mayaku-s-det.pth"


def test_cosmetic_pth_strips_to_same_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `name` and `name.pth` resolve to the identical ./name.pth — so every entry
    # point (`--weights`, `mayaku download`) accepts either spelling.
    monkeypatch.chdir(tmp_path)
    _no_network(monkeypatch)
    (tmp_path / "mayaku-s-det.pth").write_bytes(b"x")
    assert download_model("mayaku-s-det.pth") == download_model("mayaku-s-det")


def test_cache_dir_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_network(monkeypatch)
    (tmp_path / "mayaku-s-det.pth").write_bytes(b"x")
    assert download_model("mayaku-s-det", cache_dir=tmp_path) == tmp_path / "mayaku-s-det.pth"
