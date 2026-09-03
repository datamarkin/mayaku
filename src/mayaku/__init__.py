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

# The single source of truth for the version. `pyproject.toml` declares
# `dynamic = ["version"]` and hatchling reads this literal out of the file at
# build time (by regex, without importing it), so the distribution metadata and
# `mayaku.__version__` cannot drift apart. Keeping the literal here rather than
# reading it back from `importlib.metadata` also means a source tree with no
# install reports its real version instead of a `0.0.0` placeholder — that
# string is stamped into every checkpoint's provenance sidecar.
__version__ = "2.0.1"

# Eager top-level exports. Every entry point pulls in torch anyway (this is a
# PyTorch CV library — there is no torch-free code path to protect), so there
# is nothing to defer, and eager imports are what let IDEs and type checkers
# resolve ``from mayaku import train``. They sit after the env-var set above
# because torch snapshots the env at import time.
from mayaku.api import evaluate, train
from mayaku.health import health_check
from mayaku.inference import from_pretrained

__all__ = ["evaluate", "from_pretrained", "health_check", "train"]
