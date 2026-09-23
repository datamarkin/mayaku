"""Which accelerator: CUDA, MPS or the CPU, as one small value.

MPS is a single device (PyTorch has no multi-device MPS), so ``index`` is
meaningful only for CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeviceKind = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class Device:
    """A backend and, for CUDA, which GPU."""

    kind: DeviceKind
    index: int = 0

    @property
    def torch(self) -> torch.device:
        """As torch spells it: ``cuda:0``, ``mps`` or ``cpu``."""
        if self.kind == "cuda":
            return torch.device(f"cuda:{self.index}")
        return torch.device(self.kind)

    @property
    def dist_backend(self) -> str:
        """The ``torch.distributed`` backend: ``nccl`` on CUDA, else ``gloo``."""
        return "nccl" if self.kind == "cuda" else "gloo"

    @classmethod
    def resolve(cls, setting: str) -> str:
        """A device setting as torch spells it: "auto" becomes the best
        available backend (`auto`), anything else passes through."""
        return cls.auto().kind if setting == "auto" else setting

    @classmethod
    def auto(cls) -> Device:
        """The best available accelerator: CUDA, then MPS, then the CPU."""
        if torch.cuda.is_available():
            return cls("cuda", 0)
        if torch.backends.mps.is_available():
            return cls("mps", 0)
        return cls("cpu", 0)
