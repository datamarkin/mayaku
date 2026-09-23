"""The v3 detector: a five-op convolutional graph with host-side decode."""

from mayaku.model.blocks import STRIDES
from mayaku.model.contract import assert_contract, check_parity, export_onnx
from mayaku.model.detector import Detector, build, load_pretrained, load_weights, split_outputs
from mayaku.model.quant import enable_qat
from mayaku.model.tiers import TIERS, TINY, Tier

__all__ = [
    "STRIDES",
    "TIERS",
    "TINY",
    "Detector",
    "Tier",
    "assert_contract",
    "build",
    "check_parity",
    "enable_qat",
    "export_onnx",
    "load_pretrained",
    "load_weights",
    "split_outputs",
]
