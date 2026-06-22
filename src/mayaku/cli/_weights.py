"""Resolve a ``--weights`` argument that may be a path *or* a manifest model name.

The CLI accepts ``--weights faster_rcnn_R_50_FPN_3x`` (a name from the
hosted manifest) as well as ``--weights /path/to/model.pth``. This helper
disambiguates: if it looks like a path and exists, use it as-is; if it
looks like a bare name and isn't on disk, fetch it via
:func:`mayaku.utils.download.download_model`.

Path-like inputs that don't exist (contain ``/`` or ``.``) raise rather
than silently triggering a network fetch — typo'd paths shouldn't go
hunting on dtmfiles.
"""

from __future__ import annotations

import re
from pathlib import Path

from mayaku.utils.download import DownloadError, download_model

__all__ = ["resolve_weights"]

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def resolve_weights(weights: str | Path | None) -> Path | None:
    """Return a real ``Path`` for ``weights``, fetching from the hub if needed.

    * ``None`` → ``None`` (caller decides whether weights are required).
    * Existing path → that path.
    * Bare model name (no ``/``, no ``.``) → fetched and cached via
      :func:`download_model` (target = ``"pth"``).
    * Anything else → ``FileNotFoundError`` with a helpful message.
    """
    if weights is None:
        return None

    p = Path(weights)
    if p.exists():
        return p

    s = str(weights)
    if _NAME_RE.fullmatch(s):
        # Looks like a model name. Try the hub.
        try:
            return download_model(s, target="pth")
        except DownloadError as e:
            raise FileNotFoundError(
                f"--weights {s!r}: not a local path and not in the manifest. "
                f"Run `mayaku download --list` to see what's available. ({e})"
            ) from e

    raise FileNotFoundError(
        f"--weights {weights!r}: file not found. Pass an existing .pth path, "
        "or a bare model name like `faster_rcnn_R_50_FPN_3x` to fetch from "
        "the hosted manifest."
    )
