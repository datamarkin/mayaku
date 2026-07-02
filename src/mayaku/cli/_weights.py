"""Resolve a ``weights`` argument that may be a local file *or* a model name.

Local-first: an existing file wins (cwd-relative), so a path — or a checkpoint
already downloaded next to your script — loads directly. Otherwise it's a model
name (a trailing ``.pth`` is cosmetic) resolved via
:func:`mayaku.utils.download.download_model`, which downloads ``./<name>.pth`` on
first use and returns it thereafter. So ``mayaku-n-det`` and ``mayaku-n-det.pth``
both resolve to the same ``./mayaku-n-det.pth``.
"""

from __future__ import annotations

from pathlib import Path

from mayaku.utils.download import DownloadError, download_model

__all__ = ["resolve_weights"]


def resolve_weights(weights: str | Path | None) -> Path | None:
    """Return a real ``Path`` for ``weights``.

    * ``None`` → ``None`` (caller decides whether weights are required).
    * An existing local file → that path (cwd-relative).
    * Otherwise a model name → downloaded to ``./<name>.pth`` via
      :func:`download_model` (which treats a trailing ``.pth`` as cosmetic).
    """
    if weights is None:
        return None

    p = Path(weights)
    if p.exists():  # a local file wins — a path, or an already-downloaded model
        return p

    s = str(weights)
    if "/" in s or "\\" in s:  # a path was given but doesn't exist
        raise FileNotFoundError(f"weights file not found: {s}")

    try:
        return download_model(s)  # a bare model name (a trailing .pth is cosmetic)
    except DownloadError as e:
        raise FileNotFoundError(
            f"weights {s!r}: not a local file and not in the manifest. Run "
            f"`mayaku download --list`, or pass a .pth path. ({e})"
        ) from e
