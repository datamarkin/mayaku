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
from typing import TYPE_CHECKING

from mayaku.utils.download import DownloadError, download_model

if TYPE_CHECKING:
    from mayaku.config import MayakuConfig

__all__ = ["config_from_weights", "resolve_weights"]

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


def config_from_weights(weights: str | Path) -> tuple[MayakuConfig, Path, str]:
    """Resolve ``(config, weights_path, stem)`` from a self-describing checkpoint.

    The architecture is read from the model itself, never a separate config file:

    * a bundled model name → its shipped config, plus the fetched ``.pth``;
    * a trained ``.pth`` → the resolved config embedded in its ``"mayaku"``
      sidecar (written by training).

    A ``.pth`` with no sidecar (an older or externally-produced checkpoint)
    raises ``ValueError`` — convert it first. Inference never silently falls
    back to a hand-passed config; the checkpoint is the single source of truth.
    """
    from mayaku import configs
    from mayaku.config import MayakuConfig
    from mayaku.utils.checkpoint import load_embedded_config

    name = str(weights)
    if not Path(weights).exists():
        # A bare name: prefer the bundled config + its hosted weights.
        try:
            cfg = configs.load(name)
        except FileNotFoundError:
            cfg = None
        if cfg is not None:
            fetched = resolve_weights(weights)
            assert fetched is not None  # weights is not None on this path
            return cfg, fetched, name

    weights_path = resolve_weights(weights)
    assert weights_path is not None  # weights is not None on this path
    config_dict = load_embedded_config(weights_path)
    if config_dict is None:
        raise ValueError(
            f"{weights_path} has no embedded config (an older or externally "
            "produced checkpoint). Convert it first — predict/eval/export read "
            "the architecture from the checkpoint's embedded sidecar."
        )
    return MayakuConfig.model_validate(config_dict), weights_path, weights_path.stem
