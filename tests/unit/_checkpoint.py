"""Write a self-describing checkpoint for tests.

Mirrors what training embeds (``cli/train.py``): a pure ``state_dict`` under
``"model"`` plus a ``"mayaku"`` sidecar carrying the resolved config, so
``predict``/``eval``/``export`` can reconstruct the architecture from the
checkpoint alone — no separate config file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from mayaku.config import MayakuConfig
from mayaku.utils import build_sidecar


def save_self_describing(
    path: Path,
    model: nn.Module,
    cfg: MayakuConfig,
    class_names: Sequence[str] = ("thing",),
) -> Path:
    """Save ``model``'s weights with an embedded ``cfg`` sidecar; return ``path``."""
    torch.save({"model": model.state_dict(), "mayaku": build_sidecar(cfg, class_names)}, path)
    return path
