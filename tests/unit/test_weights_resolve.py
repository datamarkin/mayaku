"""Unit tests for :func:`mayaku.cli._weights.resolve_weights`.

The resolver disambiguates a ``--weights`` argument that may be a local
path or a bare manifest model name. The hyphenated `mayaku-s` family must
resolve (a regression: the old regex only allowed underscores), a cosmetic
``.pth`` suffix must be stripped before the extension-less manifest lookup,
and a typo'd path must NOT trigger a network fetch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mayaku.cli import _weights
from mayaku.cli._weights import resolve_weights
from mayaku.utils.download import DownloadError


@pytest.fixture
def capture_fetch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``download_model`` with a spy that records the name it was asked for."""
    calls: list[str] = []

    def fake_download(name: str, target: str = "pth", **_: object) -> Path:
        calls.append(name)
        assert target == "pth"
        return Path(f"/cache/{name}.pth")

    monkeypatch.setattr(_weights, "download_model", fake_download)
    return calls


def test_none_returns_none() -> None:
    assert resolve_weights(None) is None


def test_existing_path_returned_verbatim(tmp_path: Path, capture_fetch: list[str]) -> None:
    pth = tmp_path / "model.pth"
    pth.write_bytes(b"x")
    assert resolve_weights(pth) == pth
    assert capture_fetch == []  # local hit never touches the hub


def test_hyphenated_name_fetches(capture_fetch: list[str]) -> None:
    assert resolve_weights("mayaku-s") == Path("/cache/mayaku-s.pth")
    assert capture_fetch == ["mayaku-s"]


def test_underscore_zoo_name_still_fetches(capture_fetch: list[str]) -> None:
    assert resolve_weights("faster_rcnn_R_50_FPN_3x") == Path("/cache/faster_rcnn_R_50_FPN_3x.pth")
    assert capture_fetch == ["faster_rcnn_R_50_FPN_3x"]


def test_cosmetic_pth_suffix_is_stripped(capture_fetch: list[str]) -> None:
    # The manifest key is extension-less; `mayaku-s.pth` must look up `mayaku-s`.
    assert resolve_weights("mayaku-s.pth") == Path("/cache/mayaku-s.pth")
    assert capture_fetch == ["mayaku-s"]


def test_missing_path_does_not_fetch(capture_fetch: list[str]) -> None:
    # A directory separator makes it path-like; a missing path must raise,
    # not silently hit the network.
    with pytest.raises(FileNotFoundError):
        resolve_weights("some/dir/typo.pth")
    assert capture_fetch == []


def test_unknown_name_surfaces_manifest_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_download(name: str, target: str = "pth", **_: object) -> Path:
        raise DownloadError(f"model {name!r} not in manifest")

    monkeypatch.setattr(_weights, "download_model", raising_download)
    with pytest.raises(FileNotFoundError, match="not in the manifest"):
        resolve_weights("mayaku-does-not-exist")
