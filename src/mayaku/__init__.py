from __future__ import annotations

import os

# PYTORCH_ENABLE_MPS_FALLBACK must be set BEFORE PyTorch's MPS backend
# initialises, otherwise ops without an MPS implementation raise
# NotImplementedError instead of falling back to CPU. Setting it later
# (inside ``mayaku.cli.train.run_train``) is too late — by then
# ``import torch`` has already snapshotted the env. We use ``setdefault``
# so a user-set value (including the explicit ``"0"`` opt-out) wins.
# The variable is harmless on non-MPS hosts, so unconditional set is fine.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.0.0"

# Lazy top-level export. ``from mayaku import train`` works without
# eagerly pulling ``torch`` + the entire CLI stack on plain ``import
# mayaku``. PEP 562 ``__getattr__`` resolves the name on first access.
__all__ = ["train"]


def __getattr__(name: str) -> object:
    if name == "train":
        from mayaku.api import train

        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
