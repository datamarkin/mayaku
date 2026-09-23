from __future__ import annotations

import os

# PYTORCH_ENABLE_MPS_FALLBACK must be set before PyTorch's MPS backend
# initialises, or an op without an MPS kernel raises instead of falling back
# to the CPU; torch snapshots the environment at import, so it is set here,
# first. ``setdefault`` lets a user-set value (including an explicit "0")
# win, and the variable is harmless on hosts without MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# The single source of truth for the version. `pyproject.toml` declares
# `dynamic = ["version"]` and hatchling reads this literal out of the file at
# build time (by regex, without importing it), so the distribution metadata and
# `mayaku.__version__` cannot drift apart. Keeping the literal here rather than
# reading it back from `importlib.metadata` also means a source tree with no
# install reports its real version instead of a `0.0.0` placeholder — that
# string is stamped into every checkpoint's provenance sidecar.
__version__ = "3.0.0.dev0"

# Eager top-level exports. Every entry point pulls in torch anyway (this is a
# PyTorch CV library — there is no torch-free code path to protect), so there
# is nothing to defer, and eager imports are what let IDEs and type checkers
# resolve ``from mayaku import train``. They sit after the env-var set above
# because torch snapshots the env at import time.
from mayaku.api import evaluate, train
from mayaku.health import health_check
from mayaku.inference import from_pretrained

__all__ = ["evaluate", "from_pretrained", "health_check", "train"]
