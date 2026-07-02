"""Shared full-detector export plumbing.

Some detectors (UniQuery) expose ``export_forward`` — a traceable
``image -> (boxes, scores, labels)`` graph that covers the *whole* detector
(backbone + FPN + head + NMS-free top-k decode). Those export as a single
runnable artifact. Everything else (R-CNN heads don't trace cleanly) exports the
backbone+FPN body only.

Every exporter dispatches on :func:`is_full_detector`; when true it traces
:class:`FullDetectorAdapter` (which surfaces ``export_forward`` as ``forward`` so
``torch.jit.trace`` / ``torch.onnx.export`` pick it up) and names the outputs
:data:`FULL_DETECTOR_OUTPUTS`.
"""

from __future__ import annotations

from typing import Protocol, cast

from torch import Tensor, nn

__all__ = [
    "FULL_DETECTOR_OUTPUTS",
    "FullDetectorAdapter",
    "is_full_detector",
]

# Output tensor names of a full-detector graph, in order.
FULL_DETECTOR_OUTPUTS: tuple[str, ...] = ("boxes", "scores", "labels")


class _FullDetector(Protocol):
    """A detector exposing the traceable image -> (boxes, scores, labels) graph.

    Membership is checked at runtime by :func:`is_full_detector`
    (``hasattr(model, "export_forward")``).
    """

    def export_forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...


def is_full_detector(model: nn.Module) -> bool:
    """Whether ``model`` exports as one full graph (image -> boxes/scores/labels)."""
    return hasattr(model, "export_forward")


class FullDetectorAdapter(nn.Module):
    """Wrap a full-detector model so tracing picks up ``export_forward``.

    ``torch.jit.trace`` / ``torch.onnx.export`` trace ``forward``, not an
    arbitrary method name, so this exposes the whole
    image -> ``(boxes, scores, labels)`` graph as ``forward``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Cast once (validated by is_full_detector upstream) so forward() reads
        # clean; the object is still an nn.Module, registered as a submodule.
        self.model: _FullDetector = cast(_FullDetector, model)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.export_forward(image)
