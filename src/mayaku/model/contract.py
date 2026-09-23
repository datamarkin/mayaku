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

from mayaku.model.quant import fake_quant_disabled

DEPLOY_OPS = {"Conv", "Relu", "Add", "Resize", "MaxPool"}

# Compile-time scaffolding, not compute: Constant feeds every Resize's scales
# tensor, and Identity appears when the exporter aliases two byte-identical
# initialisers -- on a freshly initialised model that is most of the biases,
# since they all fold to zero. No backend runs either.
FOLDABLE_OPS = {"Constant", "Identity", "Cast", "Shape", "ConstantOfShape"}


def assert_contract(path):
    """Every conv 1x1 or 3x3 at stride 1 or 2 with a bias and no grouping,
    every maxpool 3x3 stride 1, every resize nearest, and no other compute
    op. Returns the compute-op inventory."""
    import onnx

    model = onnx.load(path)
    inv = collections.Counter(n.op_type for n in model.graph.node)
    extra = set(inv) - DEPLOY_OPS - FOLDABLE_OPS
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


def export_onnx(model, path, imgsz, batch=1):
    """Trace the (fused) detector to ONNX and assert the contract on the file.

    Fake-quant is a train/eval simulation; the deploy graph is structural fp32
    (int8 is applied at runtime from the observed ranges).
    """
    with fake_quant_disabled(model):
        torch.onnx.export(model, torch.zeros(batch, 3, imgsz, imgsz), path,
                          input_names=["images"], output_names=model.out_names,
                          opset_version=17, dynamo=False)
    return assert_contract(path)


def count(model, imgsz=640):
    """Parameters and FLOPs of the model as it stands (fuse it first for the
    deploy graph)."""
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
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


def check_parity(model, imgsz):
    """Assert the deploy graph computes the train graph, then that it is in
    contract. Fuses `model` in place; returns (max error, logit scale,
    compute-op inventory)."""
    randomize_bn(model)
    x = torch.randn(1, 3, imgsz, imgsz)
    model.eval()
    with torch.no_grad():
        ref = model(x)
    model.fuse()
    with torch.no_grad():
        got = model(x)
    err = max((r - g).abs().max().item() for r, g in zip(ref, got))
    scale = max(r.abs().max().item() for r in ref)
    assert err < 1e-4 * max(scale, 1.0), "fusion is not exact: %.3e" % err
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        inv = export_onnx(model, f.name, imgsz)
    return err, scale, inv
