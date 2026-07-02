"""Fetch hosted Mayaku model checkpoints on demand.

The user uploads a regenerated `models/` tree (and ``manifest.json``) to
``https://dtmfiles.com/mayaku/v1/models/``. This module is the client side:
given a model name, return a local path to a verified ``.pth`` checkpoint.

**Download-to-project, file-first.**
``download_model("mayaku-s")`` returns ``./mayaku-s.pth`` if it's already there —
no network, no manifest, no re-hash — so a model lands visibly next to your
script and re-runs load instantly and offline. Only on a miss does it fetch the
manifest (in memory), resolve the name, download atomically, verify the sha256,
and write ``./mayaku-s.pth``. The download directory defaults to the current
directory; override with ``MAYAKU_CACHE_DIR`` (e.g. a shared dir on CI/Docker).

Consequences (deliberate): a downloaded model never auto-updates to a newer
revision — delete the file to refresh. Only the ``.pth`` is hosted; deployment
artifacts (onnx/coreml/…) are produced locally via ``model.export(...)``.
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


class DownloadError(RuntimeError):
    """Raised when a download or hash verification fails."""


# A model's manifest entry: ``{task, latest, revisions}`` (documentation alias;
# ``dict[str, Any]`` at runtime).
ManifestEntry = dict[str, Any]


# ---------------------------------------------------------------------------
# Download directory
# ---------------------------------------------------------------------------


def _download_dir() -> Path:
    """Download directory: ``$MAYAKU_CACHE_DIR`` if set, else the cwd (see module docstring)."""
    override = os.environ.get("MAYAKU_CACHE_DIR")
    return Path(override).expanduser() if override else Path.cwd()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _fetch_manifest(manifest_url: str) -> dict[str, Any]:
    """GET + parse the manifest, in memory.

    Fetched only on a cache miss (and by ``list_models`` / ``list_revisions``),
    so there's nothing to cache on disk — a present checkpoint never reaches here.
    """
    try:
        with urllib.request.urlopen(manifest_url, timeout=30) as resp:
            data = resp.read()
    except OSError as e:
        raise DownloadError(f"failed to fetch manifest from {manifest_url}: {e}") from e
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


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------


def _download(
    url: str,
    out: Path,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Stream ``url`` to a temp file (hashing as it goes), verify size + sha256,
    then atomically rename to ``out`` — so ``out`` exists only once fully
    downloaded and verified."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
            total = expected_size or int(resp.headers.get("Content-Length", "0") or 0)
            done = 0
            last_print = time.perf_counter()
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                f.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if now - last_print > 1.0 and total:
                    print(
                        f"  [download] {out.name}  {100 * done / total:5.1f}%  "
                        f"({done / 1e6:.1f} / {total / 1e6:.1f} MB)",
                        flush=True,
                    )
                    last_print = now
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"failed to download {url}: {e}") from e

    if (expected_size is not None and tmp.stat().st_size != expected_size) or (
        expected_sha256 is not None and digest.hexdigest() != expected_sha256
    ):
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"integrity check failed for {out.name} after download — the server may "
            "be serving a stale or corrupted file."
        )
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
) -> Path:
    """Return the path to ``<download-dir>/<name>.pth``, downloading on a miss.

    ``name`` is a bare model name; a trailing ``.pth`` is accepted and stripped
    (it's the stored filename, not part of the name), so callers needn't. File-
    first (see the module docstring): a present file is returned as-is and
    silently, so ``verify_sha256`` only guards the fresh download. ``revision``
    pins a snapshot on the miss path (``None`` → the manifest's ``latest``); the
    name-keyed file can't honor a pin on a hit.
    """
    if name.lower().endswith(".pth"):  # cosmetic — the stored file is always <name>.pth
        name = name[:-4]
    local = (cache_dir or _download_dir()) / f"{name}.pth"
    if local.exists():
        return local

    manifest = _fetch_manifest(manifest_url)
    info = _model_entry(manifest, name)
    revisions = info["revisions"]
    rev = revision or info["latest"]
    if rev not in revisions:
        available = ", ".join(sorted(revisions)) or "(none)"
        raise DownloadError(f"revision {rev!r} not available for {name!r}. Available: {available}")

    # entry is JSON-typed `Any`; the manifest schema guarantees these types.
    entry = revisions[rev]
    size = cast(int, entry["size"])
    print(f"[mayaku] downloading {name}  ({size / 1e6:.1f} MB)", flush=True)
    _download(
        _resolve_url(manifest["base_url"], cast(str, entry["path"])),
        local,
        expected_size=size,
        expected_sha256=cast(str, entry["sha256"]) if verify_sha256 else None,
    )
    return local
