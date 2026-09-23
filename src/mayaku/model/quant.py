"""Quantization-aware training (fake-quant), matching the int8 deploy scheme.

Per-channel symmetric weights, per-tensor affine activations. A model that
never calls `enable_qat` is plain `nn.Conv2d` and unchanged. Once `enable_qat`
swaps its convolutions, they fake-quantize in train and eval, so the scored AP
is the int8 number (ranges freeze in eval). Quantization is a property of the
swapped module (`fake_quant`), not a process flag: two models can hold
different settings and nothing leaks between them. An export traces either
the structural five-op fp32 graph (`fake_quant_disabled`) or the explicit
int8 one (`qdq_export`: the trained ranges as quantize / dequantize ops);
`deploy_mode` picks between them.

The observed ranges are statistics of the weights they were observed with,
like BatchNorm's running statistics, so whenever those weights change without
the observers seeing it (an EMA copy) they are recomputed with
`recalibrate_ranges`, after BatchNorm is recalibrated.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weight_scales(w, n=127):
    """Per-output-channel symmetric int8 scales of a conv weight, (out,)."""
    return w.detach().abs().amax(dim=(1, 2, 3)).clamp_min(1e-8) / n


def _fq_weight_perchannel(w, n=127):
    """Per-output-channel symmetric int8 fake-quant, straight-through."""
    s = _weight_scales(w, n)[:, None, None, None]
    wq = torch.clamp(torch.round(w / s), -n - 1, n) * s
    return w + (wq - w).detach()


class ActFakeQuant(nn.Module):
    """Per-tensor affine (int8, zero-point) activation fake-quant with a
    min/max observer. Training updates the range; eval, or `observe = False`,
    freezes it.

    `momentum` weights the running range as an EMA of per-batch min / max;
    None makes it the cumulative mean instead (a population estimate, as
    BatchNorm's `momentum=None`), which is what `recalibrate_ranges` uses.
    """

    def __init__(self, momentum=0.99):
        super().__init__()
        self.qmin, self.qmax, self.momentum = -128, 127, momentum
        self.observe = True
        self.register_buffer("mn", torch.zeros(()))
        self.register_buffer("mx", torch.zeros(()))
        # Plain Python state, not buffers: reading a CUDA buffer per forward
        # forces a device->host sync, and this runs on every conv every step.
        # Loading a state dict sets `_inited` (see `_load_from_state_dict`),
        # so loaded ranges are continued, not overwritten.
        self.reset()

    def reset(self):
        """Forget the range: the next observed batch sets it outright."""
        self._inited = False
        self._n = 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if prefix + "mn" in state_dict:
            self._inited, self._n = True, 1

    def forward(self, x):
        if self.training and self.observe:
            cmn, cmx = torch.aminmax(x.detach())
            if not self._inited:
                self.mn.copy_(cmn)
                self.mx.copy_(cmx)
                self._inited, self._n = True, 1
            elif self.momentum is None:
                self._n += 1
                self.mn.add_((cmn - self.mn) / self._n)
                self.mx.add_((cmx - self.mx) / self._n)
            else:
                self.mn.mul_(self.momentum).add_((1 - self.momentum) * cmn)
                self.mx.mul_(self.momentum).add_((1 - self.momentum) * cmx)
        scale, zp = self.qparams()
        xq = torch.clamp(torch.round(x / scale) + zp, self.qmin, self.qmax)
        xdq = (xq - zp) * scale
        return x + (xdq - x).detach()

    def qparams(self):
        """(scale, zero point) of the int8 grid over the observed range, as
        0-d tensors; the range is widened to include zero."""
        mn = self.mn.clamp(max=0.0)
        mx = self.mx.clamp(min=0.0)
        scale = ((mx - mn) / (self.qmax - self.qmin)).clamp_min(1e-8)
        zp = torch.round(self.qmin - mn / scale).clamp(self.qmin, self.qmax)
        return scale, zp


class QuantConv2d(nn.Conv2d):
    """Conv2d that fake-quantizes its input activation and its weight.
    Created only by `enable_qat`, which attaches the `act_fq` observer.

    `fake_quant` is on by default; `fake_quant_disabled` turns it off for
    export, so the deploy graph is structural (the runtime does int8 from the
    observed ranges)."""

    fake_quant = True
    qdq = None    # (activation scale, zero point, weight scales), set by `qdq_export`

    def forward(self, x):
        if self.qdq is not None:
            return self._qdq_forward(x)
        if not self.fake_quant:
            return super().forward(x)
        xq = self.act_fq(x)
        wq = _fq_weight_perchannel(self.weight)
        return F.conv2d(xq, wq, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def _qdq_forward(self, x):
        """The same int8 simulation through PyTorch's quantize ops, which
        trace to ONNX QuantizeLinear / DequantizeLinear pairs: the explicit
        int8 graph an int8 runtime executes. The quantization parameters are
        constants (see `qdq_export`), so they trace as values, not as ops."""
        scale, zp, ws = self.qdq
        x = torch.fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)
        w = torch.fake_quantize_per_channel_affine(
            self.weight, ws, torch.zeros_like(ws, dtype=torch.int32), 0, -128, 127)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _quantized(conv, act_fq):
    """`conv` as a QuantConv2d sharing its parameters, observing with `act_fq`."""
    q = QuantConv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, conv.dilation, conv.groups,
                    conv.bias is not None)
    q.weight = conv.weight
    if conv.bias is not None:
        q.bias = conv.bias
    q.add_module("act_fq", act_fq)
    return q.to(conv.weight.device)


def enable_qat(model):
    """Swap every plain Conv2d for a QuantConv2d in place, each with its own
    activation observer. Idempotent."""
    for m in model.modules():
        for name, c in list(m.named_children()):
            if type(c) is nn.Conv2d:
                setattr(m, name, _quantized(c, ActFakeQuant()))
    return model



def is_qat(model):
    """Whether `model` went through `enable_qat`."""
    return any(isinstance(m, QuantConv2d) for m in model.modules())


def strip_fake_quant(model):
    """Run every QuantConv2d in `model` as a plain convolution from now on:
    the deployed fp32 graph (the int8 one is traced with `qdq_export`, which
    overrides this). For a model that is only served, not trained further."""
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.fake_quant = False
    return model


@contextlib.contextmanager
def fake_quant_disabled(model):
    """Every QuantConv2d in `model` runs as a plain convolution inside the
    block, so a trace sees the structural graph rather than the round / clip
    / div ops of the simulation. Restores each module's setting on exit."""
    quant = [m for m in model.modules() if isinstance(m, QuantConv2d)]
    saved = [m.fake_quant for m in quant]
    for m in quant:
        m.fake_quant = False
    try:
        yield model
    finally:
        for m, s in zip(quant, saved, strict=True):
            m.fake_quant = s


@contextlib.contextmanager
def qdq_export(model):
    """Inside the block every QuantConv2d in `model` runs the explicit int8
    graph (`QuantConv2d._qdq_forward`), so an ONNX export carries the trained
    ranges as QuantizeLinear / DequantizeLinear pairs. Use on a fused model."""
    quant = [m for m in model.modules() if isinstance(m, QuantConv2d)]
    for m in quant:
        scale, zp = m.act_fq.qparams()
        m.qdq = float(scale), int(zp), _weight_scales(m.weight)
    try:
        yield model
    finally:
        for m in quant:
            m.qdq = None


def deploy_mode(model, int8=False):
    """The graph an export traces: the explicit int8 graph (`qdq_export`) or
    the structural fp32 one (`fake_quant_disabled`)."""
    return qdq_export(model) if int8 else fake_quant_disabled(model)


@contextlib.contextmanager
def ranges_frozen(model):
    """No activation observer in `model` updates its range inside the block,
    even in train mode, so a pass that exists for another reason (BatchNorm
    recalibration) does not move the int8 ranges."""
    obs = [m for m in model.modules() if isinstance(m, ActFakeQuant)]
    saved = [m.observe for m in obs]
    for m in obs:
        m.observe = False
    try:
        yield model
    finally:
        for m, s in zip(obs, saved, strict=True):
            m.observe = s


@torch.no_grad()
def recalibrate_ranges(model, batches):
    """Recompute every activation range for the model's own weights.

    The counterpart of BatchNorm recalibration: an EMA of the weights, or any
    model whose weights moved since its ranges were observed, carries ranges
    that describe other weights. Every observer is reset and its range
    recomputed as the cumulative mean of per-batch min / max over `batches`,
    with BatchNorm in eval mode, so the ranges describe exactly the
    activations that evaluation and export will see. Recalibrate BatchNorm
    first. A model without QAT is untouched and `batches` is not read.
    """
    obs = [m for m in model.modules() if isinstance(m, ActFakeQuant)]
    if not obs:
        return
    was_training = model.training
    model.eval()
    momenta = [m.momentum for m in obs]
    for m in obs:
        m.reset()
        m.momentum = None
        m.train()
    for x in batches:
        model(x)
    for m, mom in zip(obs, momenta, strict=True):
        m.momentum = mom
    model.train(was_training)
