"""MPS-specific environment + op-fallback diagnostics.

PyTorch's MPS backend doesn't yet implement every ATen op an R-CNN
training loop touches. Without ``PYTORCH_ENABLE_MPS_FALLBACK=1``,
unsupported ops *raise*; with it, they fall back to CPU silently
(modulo a ``UserWarning`` per call).

The user-facing problem with that arrangement:

1. **Without the env var**: training fails immediately on a
   single missing op. Useless.
2. **With the env var, no filtering**: PyTorch emits the same
   warning thousands of times per epoch. The log becomes unreadable.
3. **With blanket warning suppression**: the user can't see *which*
   ops are falling back, so a 100x slowdown caused by a single
   ``aten::nonzero`` round-trip per iter is invisible.

This module strikes the middle path: enable the fallback env var
automatically on MPS, capture each unique fallback op once, then
print a single summary block at end-of-run that surfaces the
bottleneck. Verbose mode (``MAYAKU_VERBOSE_MPS=1``) restores the
raw per-call warnings for users who want to debug at the call site.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

# PyTorch emits the fallback warning with a fairly stable message
# shape; the op name is the only useful payload. Example:
#   "The operator 'aten::nonzero' is not currently supported on the
#    MPS backend and will fall back to run on the CPU. ..."
_FALLBACK_RE = re.compile(r"operator '([^']+)' is not currently supported on the MPS backend")


def _verbose_mode_enabled() -> bool:
    """True iff ``MAYAKU_VERBOSE_MPS=1`` is set in the environment."""
    return os.environ.get("MAYAKU_VERBOSE_MPS", "") == "1"


def apply_mps_environment() -> None:
    """Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` if not already set, and
    print a one-line orientation note on first call.

    Idempotent. ``mayaku/__init__.py`` also sets the env var at
    package-import time so it lands before PyTorch reads it; this
    function is the redundant safety net for callers that import
    torch before mayaku.
    """
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    global _APPLIED_NOTE_PRINTED
    if not _APPLIED_NOTE_PRINTED:
        suffix = (
            "" if _verbose_mode_enabled() else " (set MAYAKU_VERBOSE_MPS=1 for per-call warnings)"
        )
        print(
            f"[mayaku/mps] CPU fallback enabled. Op-fallback summary at end of run.{suffix}",
            flush=True,
        )
        _APPLIED_NOTE_PRINTED = True


_APPLIED_NOTE_PRINTED: bool = False


class OpFallbackTracker:
    """Capture and dedupe MPS op-fallback ``UserWarning``s.

    Use as a context manager around the section of code (training
    loop, eval loop) where you want the dedup behaviour. On exit,
    call :meth:`print_summary` to emit the aggregated table — or
    let ``__exit__`` do it automatically.

    In verbose mode (``MAYAKU_VERBOSE_MPS=1``), the tracker becomes
    a no-op: PyTorch's native per-call warnings flow through to
    stderr and no summary is printed.
    """

    def __init__(self, *, label: str = "train") -> None:
        self.label = label
        self.counts: Counter[str] = Counter()
        self._original_showwarning = warnings.showwarning
        self._verbose = _verbose_mode_enabled()

    def __enter__(self) -> OpFallbackTracker:
        if self._verbose:
            return self
        # Replace the global warning hook so we can intercept the
        # MPS fallback warnings, count them, and silently drop them.
        # Other warnings are passed through to the original handler.
        warnings.showwarning = self._showwarning
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self._verbose:
            return
        # Always restore the original handler, even if printing fails.
        try:
            self.print_summary()
        finally:
            warnings.showwarning = self._original_showwarning

    def _showwarning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        match = _FALLBACK_RE.search(text)
        if match is not None:
            self.counts[match.group(1)] += 1
            return
        # Not an MPS-fallback warning — passthrough to the original.
        self._original_showwarning(message, category, filename, lineno, file, line)

    def print_summary(self) -> None:
        """Emit the aggregated op-fallback summary to stdout.

        No-op when no fallback warnings were recorded (clean run on a
        modern PyTorch, hopefully).
        """
        if not self.counts:
            print(
                f"[mayaku/mps] op-fallback summary ({self.label}): no MPS->CPU fallbacks recorded.",
                flush=True,
            )
            return
        total = sum(self.counts.values())
        print(
            f"[mayaku/mps] op-fallback summary ({self.label}): "
            f"{total} CPU round-trips across {len(self.counts)} unique ops",
            flush=True,
        )
        for op, n in self.counts.most_common():
            print(f"  {op:<32s} {n:>8d} calls", flush=True)


@contextmanager
def track_mps_fallbacks(label: str = "train") -> Iterator[OpFallbackTracker]:
    """Convenience context manager equivalent to ``OpFallbackTracker(label=...)``.

    Lets call sites write::

        with track_mps_fallbacks("train"):
            run_training_loop()

    without manually instantiating the tracker.
    """
    tracker = OpFallbackTracker(label=label)
    with tracker:
        yield tracker
