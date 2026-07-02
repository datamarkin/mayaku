"""Fetch hosted Mayaku model checkpoints on demand.

The user uploads a regenerated `models/` tree (and ``manifest.json``) to
``https://dtmfiles.com/mayaku/v1/models/``. This module is the client
side: given a model name, return a local path to a verified, ready-to-use
``.pth`` checkpoint.

Cache layout mirrors the URL structure so a wiped cache rehydrates
identically (under ``<project>/cache/`` by default — see
:func:`_cache_root`)::

    <project>/cache/mayaku/v1/models/<task>/<name>/<revision>/<name>.pth

Only the ``.pth`` checkpoint is hosted; deployment artifacts (onnx/coreml/…) are
produced locally via ``model.export(...)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, cast

__all__ = [
    "DEFAULT_MANIFEST_URL",
    "DownloadError",
    "ManifestEntry",
    "download_model",
    "list_models",
    "list_revisions",
]

DEFAULT_MANIFEST_URL = "https://dtmfiles.com/mayaku/v1/models/manifest.json"

_MANIFEST_CACHE_TTL_S = 3600  # re-fetch manifest after 1 hour


class DownloadError(RuntimeError):
    """Raised when a download or hash verification fails."""


# Type aliases for the two manifest dict shapes (documentation only; both are
# ``dict[str, Any]`` at runtime). ``ManifestEntry`` is a model's entry
# (``{task, latest, revisions}``); ``CheckpointEntry`` is a revision's
# ``{path, size, sha256}``.
ManifestEntry = dict[str, Any]
CheckpointEntry = dict[str, Any]


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Nearest enclosing project root, walked up from the CWD.

    A "root" is the closest ancestor (including the CWD) that holds a
    ``pyproject.toml`` or a ``.git``. Falls back to the CWD when no marker
    is found, so the cache always lands somewhere visible next to where
    you're working — never in a hidden home directory.
    """
    cwd = Path.cwd()
    for directory in (cwd, *cwd.parents):
        if (directory / "pyproject.toml").exists() or (directory / ".git").exists():
            return directory
    return cwd


def _cache_root(version: str = "v1") -> Path:
    """Visible cache root: ``<project>/cache/mayaku/<version>/models``.

    Lands under the project root (see :func:`_project_root`) rather than a
    hidden ``~/.cache`` so downloaded artifacts are easy to find and clear.
    Set ``MAYAKU_CACHE_DIR`` to override the base directory (e.g. a shared
    cache on CI); the ``mayaku/<version>/models`` suffix is still appended.
    """
    override = os.environ.get("MAYAKU_CACHE_DIR")
    base = Path(override).expanduser() if override else _project_root() / "cache"
    return base / "mayaku" / version / "models"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _fetch_manifest(manifest_url: str, *, force: bool = False) -> dict[str, Any]:
    """Get the manifest, with a 1-hour disk cache to avoid repeat HTTPS round-trips."""
    cache_path = _cache_root() / "manifest.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < _MANIFEST_CACHE_TTL_S:
            return cast(dict[str, Any], json.loads(cache_path.read_text()))

    try:
        with urllib.request.urlopen(manifest_url, timeout=30) as resp:
            data = resp.read()
    except OSError as e:
        if cache_path.exists():
            # Stale cache is better than failing offline.
            return cast(dict[str, Any], json.loads(cache_path.read_text()))
        raise DownloadError(f"failed to fetch manifest from {manifest_url}: {e}") from e

    cache_path.write_bytes(data)
    return cast(dict[str, Any], json.loads(data))


def list_models(*, manifest_url: str = DEFAULT_MANIFEST_URL) -> dict[str, list[str]]:
    """Return ``{task: [model_name, ...]}`` for every model in the manifest."""
    manifest = _fetch_manifest(manifest_url)
    out: dict[str, list[str]] = {}
    for name, info in manifest["models"].items():
        out.setdefault(info["task"], []).append(name)
    for task in out:
        out[task].sort()
    return out


def _model_entry(manifest: dict[str, Any], name: str) -> ManifestEntry:
    """Return the manifest entry for ``name`` or raise :class:`DownloadError`."""
    info = manifest["models"].get(name)
    if info is None:
        available = ", ".join(sorted(manifest["models"])) or "(empty manifest)"
        raise DownloadError(f"model {name!r} not in manifest. Available: {available}")
    return cast(ManifestEntry, info)


def list_revisions(name: str, *, manifest_url: str = DEFAULT_MANIFEST_URL) -> list[str]:
    """Return the available revision ids for ``name`` (sorted ascending).

    Raises :class:`DownloadError` if ``name`` isn't in the manifest.
    """
    info = _model_entry(_fetch_manifest(manifest_url), name)
    return sorted(info["revisions"])


def _select_revision(info: ManifestEntry, revision: str | None, name: str) -> CheckpointEntry:
    """Resolve the ``{path, size, sha256}`` checkpoint entry for a model.

    ``revision=None`` selects the ``info["latest"]`` pointer.
    """
    revisions = info["revisions"]
    rev = revision or info["latest"]
    if rev not in revisions:
        available = ", ".join(sorted(revisions)) or "(none)"
        raise DownloadError(f"revision {rev!r} not available for {name!r}. Available: {available}")
    return cast(CheckpointEntry, revisions[rev])


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected_sha256: str, *, expected_size: int | None = None) -> bool:
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return _sha256(path) == expected_sha256


def _download(
    url: str, out: Path, *, expected_size: int | None = None, progress_label: str | None = None
) -> None:
    """Stream `url` to `out` with a coarse progress line if size is known."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    label = progress_label or out.name
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
            total = expected_size or int(resp.headers.get("Content-Length", "0") or 0)
            done = 0
            last_print = time.perf_counter()
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                f.write(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if now - last_print > 1.0 and total:
                    pct = 100 * done / total
                    print(
                        f"  [download] {label}  {pct:5.1f}%  ({done / 1e6:.1f} / {total / 1e6:.1f} MB)",
                        flush=True,
                    )
                    last_print = now
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"failed to download {url}: {e}") from e
    tmp.replace(out)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_url(base_url: str, rel_path: str) -> str:
    """Join base + rel without losing the base's path component (urljoin would)."""
    return f"{base_url.rstrip('/')}/{rel_path.lstrip('/')}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def download_model(
    name: str,
    *,
    revision: str | None = None,
    cache_dir: Path | None = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    verify_sha256: bool = True,
    quiet: bool = False,
) -> Path:
    """Download (if missing) and return the local path to a model's ``.pth``.

    ``name`` is a model name like ``"mayaku-s"`` (or a config name like
    ``"faster_rcnn_R_50_FPN_3x"``). ``revision`` pins a specific published
    snapshot; ``None`` resolves the manifest's ``latest`` pointer for the model.

    Returns the local path to the fetched file. Idempotent — a cache hit
    (size + sha256 match) skips the network.
    """
    cache_root = cache_dir or _cache_root()
    manifest = _fetch_manifest(manifest_url)
    base_url = manifest["base_url"]
    info = _model_entry(manifest, name)
    entry = _select_revision(info, revision, name)

    # The entry comes from JSON (typed `Any`); the manifest schema guarantees
    # path/sha256 are str and size is int.
    rel_path = cast(str, entry["path"])
    size = cast(int, entry["size"])
    sha = cast(str, entry["sha256"])
    local = cache_root / rel_path
    if verify_sha256 and _verify(local, sha, expected_size=size):
        if not quiet:
            print(f"[mayaku.download] cache hit  {rel_path}", flush=True)
    else:
        if not quiet:
            print(f"[mayaku.download] fetching   {rel_path}  ({size / 1e6:.1f} MB)", flush=True)
        _download(_resolve_url(base_url, rel_path), local, expected_size=size,
                  progress_label=Path(rel_path).name)
        if verify_sha256 and not _verify(local, sha, expected_size=size):
            local.unlink(missing_ok=True)
            raise DownloadError(
                f"sha256 mismatch on {rel_path} after download — server may be serving "
                "a stale or corrupted file."
            )
    return local
