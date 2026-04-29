"""Backend facade.

The submodules are still the source of truth — this file just re-exports
the few names that *every* model module is expected to use, so call
sites read

    from mayaku.backends import Device, autocast

instead of the fully-qualified two-line import. ``ops`` is a submodule
(``mayaku.backends.ops.{nms, roi_align, ...}``) — we don't re-export
the ops here because callers already write
``from mayaku.backends.ops import roi_align`` and pulling them up would
shadow the ``ops`` namespace.

Resolved at Step 7 (backbones) per the open question recorded in
``PROJECT_STATUS.md`` from Step 3.
"""

from __future__ import annotations

from mayaku.backends.amp import NullGradScaler, autocast, make_grad_scaler
from mayaku.backends.device import Device, DeviceKind

__all__ = [
    "Device",
    "DeviceKind",
    "NullGradScaler",
    "autocast",
    "make_grad_scaler",
]
