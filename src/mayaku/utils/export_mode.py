"""Whether the graph is currently being captured for export.

Some ops have to be written one way to satisfy a backend compiler and a
different way to run fast eagerly (see ``ROIPooler._forward_onepass``). The two
formulations must agree numerically; what differs is which one is *recorded
into the artifact*.

Deciding that from the capture mechanism alone — ``torch.jit.is_tracing()`` /
``torch.onnx.is_in_onnx_export()`` — is not reliable, because it depends on how
each exporter happens to capture rather than on our intent. OpenVINO's
``convert_model`` sets neither, so a trace-sniffing check silently reports
"eager" mid-export and bakes the eager formulation into the artifact.

So export mode is stated explicitly: :func:`export_detector` enters
:func:`exporting` around every capture, and the tracing predicates remain only
as a fallback for callers that reach an exporter some other way.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

import torch

__all__ = ["exporting", "is_exporting"]

_EXPORTING: ContextVar[bool] = ContextVar("mayaku_exporting", default=False)


@contextlib.contextmanager
def exporting() -> Iterator[None]:
    """Mark the enclosed block as graph capture for export."""
    token = _EXPORTING.set(True)
    try:
        yield
    finally:
        _EXPORTING.reset(token)


def is_exporting() -> bool:
    """True while a graph is being captured for export.

    Prefer this over the ``torch`` tracing predicates it wraps: those answer
    "how am I being captured", which varies per exporter, rather than "is this
    graph destined for an artifact".
    """
    return (
        _EXPORTING.get()
        or bool(torch.jit.is_tracing())  # type: ignore[no-untyped-call]
        or torch.onnx.is_in_onnx_export()
    )
