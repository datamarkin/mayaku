"""Device facade for the three required backends (CUDA, MPS, CPU).

Every model module, optimizer plumbing, and distributed call site in
Mayaku routes through :class:`Device` instead of touching ``torch.cuda``
or ``torch.backends.mps`` directly. This collapses the ~25 scattered
backend-aware call sites in the Detectron2 reference (see
``BACKEND_PORTABILITY_REPORT.md`` §1) into a single, trivially testable
module.

Notes:
    * MPS is single-device by construction (``torch.backends.mps`` does
      not expose a multi-device API), so ``index`` is meaningful only for
      CUDA.
    * ``amp_dtype`` is ``float16`` on CUDA and ``float32`` on MPS.
      MPS defaults to fp32 because R-CNN's box-reg + mask losses are
      fp16-sensitive and PyTorch's MPS fp16 autocast has not been
      validated end-to-end on this codebase. Users who want fp16 on
      MPS can pass ``dtype=torch.float16`` to :func:`autocast` directly
      via the solver config (``solver.amp_dtype: float16``).
      ``bfloat16`` on CUDA Ampere+ is a future toggle and not exposed yet
      to keep the v1 surface small (see open question recorded in
      ``PROJECT_STATUS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import dtype as TorchDtype  # aliased to avoid the .torch property shadow

DeviceKind = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class Device:
    """Backend-aware device descriptor.

    Args:
        kind: One of ``"cuda"``, ``"mps"``, ``"cpu"``.
        index: Device ordinal. Only meaningful for CUDA; ignored for MPS
            and CPU. Defaults to ``0``.
    """

    kind: DeviceKind
    index: int = 0

    @property
    def torch(self) -> torch.device:
        """Equivalent ``torch.device`` (e.g. ``cuda:0``, ``mps``, ``cpu``)."""
        if self.kind == "cuda":
            return torch.device(f"cuda:{self.index}")
        return torch.device(self.kind)

    @property
    def supports_amp(self) -> bool:
        """True iff this backend has a working ``torch.amp.autocast`` path."""
        return self.kind in ("cuda", "mps")

    @property
    def amp_dtype(self) -> TorchDtype | None:
        """Autocast dtype for this backend; ``None`` when AMP is unsupported.

        CUDA defaults to fp16 (the standard mixed-precision recipe). MPS
        defaults to fp32: the autocast block is still entered for
        consistency, but the dtype change is a no-op until the user
        opts into fp16 explicitly. See the module docstring for why.
        """
        if self.kind == "cuda":
            return torch.float16
        if self.kind == "mps":
            return torch.float32
        return None

    @property
    def dist_backend(self) -> str:
        """``torch.distributed`` backend name: ``nccl`` for CUDA, else ``gloo``."""
        return "nccl" if self.kind == "cuda" else "gloo"

    @property
    def supports_pin_memory(self) -> bool:
        """``DataLoader(pin_memory=...)`` is only meaningful for CUDA hosts."""
        return self.kind == "cuda"

    def synchronize(self) -> None:
        """Block until pending work on this device completes.

        Dispatches to ``torch.cuda.synchronize`` / ``torch.mps.synchronize``
        and is a no-op on CPU.
        """
        if self.kind == "cuda":
            torch.cuda.synchronize(self.index)
        elif self.kind == "mps":
            torch.mps.synchronize()

    @classmethod
    def auto(cls) -> Device:
        """Pick the best available accelerator: CUDA → MPS → CPU."""
        if torch.cuda.is_available():
            return cls("cuda", 0)
        if torch.backends.mps.is_available():
            return cls("mps", 0)
        return cls("cpu", 0)
