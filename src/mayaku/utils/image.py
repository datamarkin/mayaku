"""RGB-native image ingest helpers.

Per ADR 002 (`docs/decisions/002-rgb-native-image-ingestion.md`), Mayaku
is RGB end-to-end. Pillow is the primary decode path because
``PIL.Image.open(...).convert("RGB")`` is the canonical RGB entry point
in the Python ML ecosystem and produces the same byte layout that
torchvision's pretrained ResNet weights expect.

``bgr_to_rgb`` is the conversion helper for users who already have
OpenCV-decoded arrays — it's the only place in the codebase that
*acknowledges* BGR exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

# Disable Pillow's decompression-bomb guard. That limit (~89 Mpix) defends a
# *web service* against malicious uploads; here we read trusted training/eval
# images, and large-image datasets (e.g. Objects365 has frames up to ~96 Mpix,
# satellite/medical sets go higher) legitimately exceed it. Left at the default
# it floods the logs with DecompressionBombWarning and — above 2x the limit
# (~179 Mpix) — raises DecompressionBombError, which would crash a dataloader
# worker and kill a run mid-training. ``None`` removes both; the mapper resizes
# every image to <=max_size right after decode, so the decoded frame is
# transient. Setting it here (the canonical read entry point) covers every
# PIL read in the process, since MAX_IMAGE_PIXELS is a module-global.
Image.MAX_IMAGE_PIXELS = None

__all__ = ["bgr_to_rgb", "read_image"]

_Uint8RGB = npt.NDArray[np.uint8]


def read_image(path: str | Path) -> _Uint8RGB:
    """Read an image from disk and return ``(H, W, 3)`` ``uint8`` RGB.

    EXIF orientation is honoured (`ImageOps.exif_transpose`) so a portrait
    photo doesn't end up rotated 90 degrees once the metadata is dropped
    by ``np.asarray``. Greyscale and RGBA inputs are converted to RGB.
    """
    with Image.open(path) as im:
        rotated = ImageOps.exif_transpose(im)
        rgb: Image.Image = (rotated if rotated is not None else im).convert("RGB")
        arr: _Uint8RGB = np.asarray(rgb, dtype=np.uint8).copy()
    return arr


def bgr_to_rgb(image: npt.NDArray[np.uint8]) -> _Uint8RGB:
    """Convert an ``(H, W, 3)`` BGR uint8 array to RGB.

    For users coming from OpenCV (``cv2.imread`` returns BGR). This is
    the only sanctioned entry point for BGR data — the rest of the
    pipeline assumes RGB. See ADR 002 for the design rationale.
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"bgr_to_rgb expects (H, W, 3); got shape {image.shape}")
    return np.ascontiguousarray(image[..., ::-1])
