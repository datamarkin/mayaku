"""The five-op contract, asserted on the exported artifact, plus the fuse-parity check.

The deploy graph is five compute op types and nothing else. The check runs on
the exported ONNX rather than on the Python, because the artifact is what a
runtime consumes and the exporter is what inserts ops nobody wrote.
"""

import collections
import tempfile

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from mayaku.model.blocks import as_canvas
from mayaku.model.quant import deploy_mode, is_qat

DEPLOY_OPS = {"Conv", "Relu", "Add", "Resize", "MaxPool"}

# The int8 graph adds the quantization boundary around every conv: its input
# activation and its weights quantized, then dequantized for the fp32 op a
# runtime fuses them into.
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}

# Compile-time scaffolding, not compute: Constant feeds every Resize's scales
# tensor, and Identity appears when the exporter aliases two byte-identical
# initialisers -- on a freshly initialised model that is most of the biases,
# since they all fold to zero. No backend runs either.
FOLDABLE_OPS = {"Constant", "Identity", "Cast", "Shape", "ConstantOfShape"}


def assert_contract(path, int8=False):
    """Every conv 1x1 or 3x3 at stride 1 or 2 with a bias and no grouping,
    every maxpool 3x3 stride 1, every resize nearest, and no other compute
    op (`int8` also allows the Q/DQ pairs). Returns the op inventory."""
    import onnx

    model = onnx.load(path)
    inv = collections.Counter(n.op_type for n in model.graph.node)
    extra = set(inv) - DEPLOY_OPS - FOLDABLE_OPS - (QDQ_OPS if int8 else set())
    assert not extra, "ops outside the contract: %s" % sorted(extra)
    for n in model.graph.node:
        a = {x.name: x for x in n.attribute}
        if n.op_type == "Conv":
            assert list(a["kernel_shape"].ints) in ([1, 1], [3, 3]), n.name
            assert list(a["strides"].ints) in ([1, 1], [2, 2]), n.name
            assert len(n.input) == 3, "conv without bias: %s" % n.name
            assert "group" not in a or a["group"].i == 1, n.name
        elif n.op_type == "MaxPool":
            assert list(a["kernel_shape"].ints) == [3, 3], n.name
            assert list(a["strides"].ints) == [1, 1], n.name
        elif n.op_type == "Resize":
            assert a["mode"].s == b"nearest", n.name
    return collections.Counter({k: v for k, v in inv.items() if k not in FOLDABLE_OPS})


def export_onnx(model, path, canvas, batch=1, int8=False):
    """Trace the (fused) detector at a fixed (H, W) canvas to ONNX and assert
    the contract on the file.

    The default is the structural fp32 graph: fake-quant is a training
    simulation and is left out. `int8` (a quantization-aware model only)
    writes the explicit int8 graph instead, the trained ranges as
    QuantizeLinear / DequantizeLinear around every conv.
    """
    h, w = as_canvas(canvas)
    if int8 and not is_qat(model):
        raise ValueError("an int8 graph needs a quantization-aware model (model.qat)")
    with deploy_mode(model, int8):
        torch.onnx.export(model, torch.zeros(batch, 3, h, w), path,
                          input_names=["images"], output_names=model.out_names,
                          opset_version=17, dynamo=False)
    return assert_contract(path, int8)


def count(model, canvas=640):
    """Parameters and FLOPs of the model as it stands on an (H, W) canvas
    (fuse it first for the deploy graph)."""
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, *as_canvas(canvas)))
    return sum(p.numel() for p in model.parameters()), counter.get_total_flops()


def randomize_bn(model):
    """Give BatchNorm non-trivial statistics, so a parity check exercises the
    fold arithmetic instead of folding an identity."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.normal_(0, 0.1)


def check_parity(model, canvas):
    """Assert the deploy graph computes the train graph on an (H, W) canvas,
    then that it is in contract. Fuses `model` in place; returns (max error,
    logit scale, compute-op inventory)."""
    canvas = as_canvas(canvas)
    randomize_bn(model)
    x = torch.randn(1, 3, *canvas)
    model.eval()
    with torch.no_grad():
        ref = model(x)
    model.fuse()
    with torch.no_grad():
        got = model(x)
    err = max((r - g).abs().max().item() for r, g in zip(ref, got, strict=True))
    scale = max(r.abs().max().item() for r in ref)
    assert err < 1e-4 * max(scale, 1.0), "fusion is not exact: %.3e" % err
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        inv = export_onnx(model, f.name, canvas)
    return err, scale, inv
