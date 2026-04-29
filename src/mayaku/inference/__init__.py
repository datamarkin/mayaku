"""Inference: predictor, postprocess, export targets."""

from __future__ import annotations

from mayaku.inference.postprocess import detector_postprocess
from mayaku.inference.predictor import Predictor

__all__ = ["Predictor", "detector_postprocess"]
