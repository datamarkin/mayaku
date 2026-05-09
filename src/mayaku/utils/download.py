"""Fetch hosted Mayaku model artifacts on demand.

The user uploads a regenerated `models/` tree (and ``manifest.json``) to
``https://dtmfiles.com/mayaku/v1/models/``. This module is the client
side: given a model name and a target variant, return a local path to a
verified, ready-to-use artifact.

Cache layout mirrors the URL structure so a wiped cache rehydrates
identically::

    ~/.cache/mayaku/v1/models/<task>/<config_name>.<ext>

For ``coreml-fp16``: the ``.mlpackage.zip`` is downloaded then extracted
in place and the resulting ``.mlpackage`` directory is what's returned.
``.zip`` is kept around for cache-validation.

For ``openvino``: both ``.xml`` and ``.bin`` are fetched together; the
``.xml`` path is returned (loaders look up the ``.bin`` sibling).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

__all__ = [
    "DEFAULT_MANIFEST_URL",
    "DEFAULT_VARIANT",
    "VARIANTS",
    "DownloadError",
    "ManifestEntry",
    "download_model",
    "engine_cache_path",
    "list_models",
]

DEFAULT_MANIFEST_URL = "https://dtmfiles.com/mayaku/v1/models/manifest.json"
DEFAULT_VARIANT = "pth"
VARIANTS = ("pth", "onnx", "onnx-fixed", "coreml-fp16", "openvino")

_MANIFEST_CACHE_TTL_S = 3600  # re-fetch manifest after 1 hour


class DownloadError(RuntimeError):
    """Raised when a download or hash verification fails."""


# Public alias for type checkers / documentation.
ManifestEntry = dict[str, Any]


# ---------------------------------------------------------------------------
# Cache layout
# ---------------------------------------------------------------------------


def _cache_root(version: str = "v1") -> Path:
    """``$XDG_CACHE_HOME/mayaku/<version>/models`` (or ~/.cache/...)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "mayaku" / version / "models"


def engine_cache_path(
    name: str,
    *,
    pinned_h: int,
    pinned_w: int,
    fp16: bool,
    cache_dir: Path | None = None,
) -> Path:
    """Local cache path for a TensorRT engine built for ``name``.

    Engines are GPU-architecture-specific (a build for ``sm89`` won't run
    on ``sm86``), so the SM string is part of the path. Engines are not
    served from the manifest — they're built locally on first use by
    :class:`mayaku.inference.export.TensorRTExporter`. This helper is the
    single source of truth for *where* to write / look for them.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("engine_cache_path requires a CUDA-capable GPU")
    cap = torch.cuda.get_device_capability()
    sm = f"sm{cap[0]}{cap[1]}"
    precision = "fp16" if fp16 else "fp32"
    base = cache_dir or _cache_root()
    return base / "engines" / sm / f"{name}_{int(pinned_h)}x{int(pinned_w)}_{precision}.engine"


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


def _extract_zip_in_place(zip_path: Path) -> Path:
    """Extract a `.mlpackage.zip` into the same directory; return the .mlpackage dir."""
    # Conservative: re-extract every call but skip if the directory's already there
    # with the right shape — the manifest hash on the .zip is the source of truth.
    target_dir = zip_path.with_suffix("")  # drop .zip → .mlpackage
    if target_dir.exists() and any(target_dir.rglob("Manifest.json")):
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    if not target_dir.exists():
        raise DownloadError(
            f"zip {zip_path.name} did not produce expected directory {target_dir.name}"
        )
    return target_dir


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _resolve_url(base_url: str, rel_path: str) -> str:
    """Join base + rel without losing the base's path component (urljoin would)."""
    return f"{base_url.rstrip('/')}/{rel_path.lstrip('/')}"


def _file_entries(variant: ManifestEntry) -> Iterable[tuple[str, int, str]]:
    """Yield ``(rel_path, size, sha256)`` for each file in a variant entry."""
    if "files" in variant:
        for f in variant["files"]:
            yield f["path"], f["size"], f["sha256"]
    else:
        yield variant["path"], variant["size"], variant["sha256"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def download_model(
    name: str,
    target: str = DEFAULT_VARIANT,
    *,
    cache_dir: Path | None = None,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    verify_sha256: bool = True,
    quiet: bool = False,
) -> Path:
    """Download (if missing) and return the local path to a Mayaku artifact.

    ``name`` is a config name like ``"faster_rcnn_R_50_FPN_3x"``.
    ``target`` is one of :data:`VARIANTS`.

    Returns the path the user actually loads:

    * ``pth`` / ``onnx`` / ``onnx-fixed`` → the file itself.
    * ``coreml-fp16`` → the extracted ``.mlpackage`` directory.
    * ``openvino`` → the ``.xml`` file (loaders look up the ``.bin`` sibling).

    Idempotent — a cache hit (size + sha256 match) skips the network.
    """
    if target not in VARIANTS:
        raise ValueError(f"unknown target {target!r}; expected one of {VARIANTS}")

    cache_root = cache_dir or _cache_root()
    manifest = _fetch_manifest(manifest_url)
    base_url = manifest["base_url"]
    models = manifest["models"]
    if name not in models:
        available = ", ".join(sorted(models)) or "(empty manifest)"
        raise DownloadError(f"model {name!r} not in manifest. Available: {available}")
    info = models[name]
    if target not in info["variants"]:
        available = ", ".join(sorted(info["variants"]))
        raise DownloadError(
            f"variant {target!r} not available for {name!r}. Available: {available}"
        )

    variant = info["variants"][target]
    primary_rel: str | None = None
    for rel_path, size, sha in _file_entries(variant):
        local = cache_root / rel_path
        url = _resolve_url(base_url, rel_path)
        if verify_sha256 and _verify(local, sha, expected_size=size):
            if not quiet:
                print(f"[mayaku.download] cache hit  {rel_path}", flush=True)
        else:
            if not quiet:
                size_mb = size / 1e6
                print(f"[mayaku.download] fetching   {rel_path}  ({size_mb:.1f} MB)", flush=True)
            _download(url, local, expected_size=size, progress_label=Path(rel_path).name)
            if verify_sha256 and not _verify(local, sha, expected_size=size):
                local.unlink(missing_ok=True)
                raise DownloadError(
                    f"sha256 mismatch on {rel_path} after download — server may be serving "
                    "a stale or corrupted file."
                )
        if "primary" in variant and rel_path == variant["primary"]:
            primary_rel = rel_path

    # Resolve the path to return. The variant dict comes from JSON, so
    # entries are typed `Any`; the manifest schema guarantees these are
    # str values.
    if target == "coreml-fp16":
        zip_path = cache_root / cast(str, variant["path"])
        return _extract_zip_in_place(zip_path)
    if "primary" in variant:
        return cache_root / cast(str, primary_rel or variant["primary"])
    return cache_root / cast(str, variant["path"])
