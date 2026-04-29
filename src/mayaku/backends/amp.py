"""Device-aware AMP plumbing.

Wraps ``torch.amp.autocast`` and ``torch.amp.GradScaler`` so the trainer
can be written backend-agnostically. On CPU (and any other backend
without an AMP path), :func:`autocast` becomes a no-op context manager
and :func:`make_grad_scaler` returns a :class:`NullGradScaler` that
satisfies the same call surface as the real one.

This is the single place that knows about ``device_type=`` and
``GradScaler`` selection — call sites elsewhere in Mayaku should not
import ``torch.cuda.amp`` directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from .device import Device


@contextmanager
def autocast(device: Device, *, dtype: torch.dtype | None = None) -> Iterator[None]:
    """Backend-aware autocast context.

    On CUDA / MPS this enters ``torch.amp.autocast(device_type=device.kind,
    dtype=...)``. On CPU it is a no-op so trainer code can use the
    same ``with autocast(device):`` block on every backend.

    Args:
        device: The active :class:`Device`.
        dtype: Override the autocast dtype. Defaults to
            :attr:`Device.amp_dtype` (fp16). Pass ``torch.bfloat16`` to
            opt into bf16 on CUDA Ampere+ (`Step 13` decision).

    Yields:
        ``None``. Use as a context manager.
    """
    if not device.supports_amp or device.amp_dtype is None:
        yield
        return
    use_dtype = dtype if dtype is not None else device.amp_dtype
    # Autocast at fp32 is a definitional no-op (no down-cast happens).
    # PyTorch's MPS autocast specifically prints a UserWarning when called
    # with fp32 ("only supports torch.bfloat16, torch.float16"); skip the
    # context entirely on MPS+fp32 to avoid the spurious warning. CPU
    # autocast doesn't do fp32 either, but we already short-circuited
    # CPU above (supports_amp=False).
    if use_dtype == torch.float32:
        yield
        return
    with torch.amp.autocast(device_type=device.kind, dtype=use_dtype):
        yield


class NullGradScaler:
    """No-op replacement for :class:`torch.amp.GradScaler` on non-CUDA backends.

    Matches the public surface used by the trainer (``scale``, ``step``,
    ``update``, ``unscale_``) so the caller does not need a backend
    branch around the AMP-bookkeeping calls. MPS does not yet have a
    working gradient scaler (``GradScaler`` exists generically in PT 2.4+
    but is a no-op on MPS), and CPU AMP doesn't need one — so on those
    backends we substitute this stub.
    """

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return ``outputs`` unchanged (no scaling)."""
        return outputs

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Forward to ``optimizer.step()``; ignore the AMP-specific args."""
        optimizer.step()
        return None

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:
        """No-op; the scale is fixed at 1.0 conceptually."""
        return None

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """No-op; gradients were never scaled in the first place."""
        return None


def make_grad_scaler(device: Device) -> torch.amp.GradScaler | NullGradScaler:
    """Return a real :class:`torch.amp.GradScaler` on CUDA, else a stub.

    MPS autocast supports fp16 but PyTorch does not provide a working
    gradient scaler for it (see ``BACKEND_PORTABILITY_REPORT.md`` §4),
    so we return :class:`NullGradScaler` there too.
    """
    if device.kind == "cuda":
        return torch.amp.GradScaler("cuda")
    return NullGradScaler()
