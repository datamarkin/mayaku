from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

# PYTORCH_ENABLE_MPS_FALLBACK must be set BEFORE PyTorch's MPS backend
# initialises, otherwise ops without an MPS implementation raise
# NotImplementedError instead of falling back to CPU. Setting it later
# (inside ``mayaku.cli.train.run_train``) is too late — by then
# ``import torch`` has already snapshotted the env. We use ``setdefault``
# so a user-set value (including the explicit ``"0"`` opt-out) wins.
# The variable is harmless on non-MPS hosts, so unconditional set is fine.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Single-sourced from the installed distribution metadata (pyproject's
# ``version``) so this can never drift from what pip reports. The fallback
# covers running straight from a source tree with no install.
try:
    __version__ = _metadata_version("mayaku")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0"

# Eager top-level exports. Every entry point pulls in torch anyway (this is a
# PyTorch CV library — there is no torch-free code path to protect), so there
# is nothing to defer, and eager imports are what let IDEs and type checkers
# resolve ``from mayaku import train``. They sit after the env-var set above
# because torch snapshots the env at import time.
from mayaku.api import evaluate, train
from mayaku.health import health_check
from mayaku.inference import from_pretrained

__all__ = ["evaluate", "from_pretrained", "health_check", "train"]
