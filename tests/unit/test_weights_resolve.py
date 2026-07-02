"""Unit tests for :func:`mayaku.cli._weights.resolve_weights`.

Local-first, then model name: an existing file wins (cwd-relative); otherwise the
argument is a model name (a trailing ``.pth`` is cosmetic) resolved via the hub.
So ``mayaku-s-det`` and ``mayaku-s-det.pth`` both resolve to the same download,
while a real local file — or a path — loads directly without touching the hub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mayaku.cli import _weights
from mayaku.cli._weights import resolve_weights
from mayaku.utils.download import DownloadError


@pytest.fixture
def capture_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Spy on ``download_model`` + chdir to a clean dir so no stray cwd file interferes."""
    calls: list[str] = []

    def fake_download(name: str, **_: object) -> Path:
        calls.append(name)
        return Path(f"/dl/{name}.pth")

    monkeypatch.setattr(_weights, "download_model", fake_download)
    monkeypatch.chdir(tmp_path)
    return calls


def test_none_returns_none() -> None:
    assert resolve_weights(None) is None


# --- local file wins --------------------------------------------------------


def test_existing_file_returned_verbatim(tmp_path: Path, capture_fetch: list[str]) -> None:
    pth = tmp_path / "model.pth"
    pth.write_bytes(b"x")
    assert resolve_weights(pth) == pth
    assert capture_fetch == []  # a local file never touches the hub


def test_local_file_wins_over_hub(tmp_path: Path, capture_fetch: list[str]) -> None:
    # capture_fetch chdir'd to tmp_path; a local `mayaku-s-det.pth` here shadows
    # the hub model of that name (drop-in override).
    (tmp_path / "mayaku-s-det.pth").write_bytes(b"x")
    assert resolve_weights("mayaku-s-det.pth") == Path("mayaku-s-det.pth")  # verbatim
    assert capture_fetch == []  # local file shadows the hub — no download


# --- model name = hub download ---------------------------------------------


def test_bare_name_fetches(capture_fetch: list[str]) -> None:
    assert resolve_weights("mayaku-s-det") == Path("/dl/mayaku-s-det.pth")
    assert capture_fetch == ["mayaku-s-det"]


def test_pth_name_forwarded_to_download(capture_fetch: list[str]) -> None:
    # No local file (chdir'd to a clean dir), so `mayaku-s-det.pth` is a name — it's
    # forwarded verbatim; download_model owns the cosmetic-.pth strip (tested there).
    resolve_weights("mayaku-s-det.pth")
    assert capture_fetch == ["mayaku-s-det.pth"]


def test_underscore_name_fetches(capture_fetch: list[str]) -> None:
    assert resolve_weights("faster_rcnn_R_50_FPN_3x") == Path("/dl/faster_rcnn_R_50_FPN_3x.pth")
    assert capture_fetch == ["faster_rcnn_R_50_FPN_3x"]


# --- rejections -------------------------------------------------------------


def test_missing_path_with_separator_does_not_fetch(capture_fetch: list[str]) -> None:
    # A separator makes it a path; a missing path errors, it doesn't hit the hub.
    with pytest.raises(FileNotFoundError, match="weights file not found"):
        resolve_weights("some/dir/typo.pth")
    assert capture_fetch == []


def test_unknown_name_surfaces_manifest_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_download(name: str, **_: object) -> Path:
        raise DownloadError(f"model {name!r} not in manifest")

    monkeypatch.setattr(_weights, "download_model", raising_download)
    with pytest.raises(FileNotFoundError, match="not in the manifest"):
        resolve_weights("mayaku-does-not-exist")
