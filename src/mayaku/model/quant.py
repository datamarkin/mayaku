"""Quantization-aware training (fake-quant), matching the int8 deploy scheme.

Per-channel symmetric weights, per-tensor affine activations. A model that
never calls `enable_qat` is plain `nn.Conv2d` and unchanged. Once `enable_qat`
swaps its convolutions, they fake-quantize in train and eval, so the scored AP
is the int8 number (ranges freeze in eval). Quantization is a property of the
swapped module (`fake_quant`), not a process flag: two models can hold
different settings and nothing leaks between them. The exported deploy graph
is structural five-op fp32 -- int8 is applied at runtime from the observed
ranges -- so the exporter turns `fake_quant` off before tracing.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fq_weight_perchannel(w, n=127):
    """Per-output-channel symmetric int8 fake-quant, straight-through."""
    s = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8) / n
    wq = torch.clamp(torch.round(w / s), -n - 1, n) * s
    return w + (wq - w).detach()


class ActFakeQuant(nn.Module):
    """Per-tensor affine (int8, zero-point) activation fake-quant with an EMA
    min/max observer. Training updates the range; eval freezes it."""

    def __init__(self, momentum=0.99):
        super().__init__()
        self.qmin, self.qmax, self.momentum = -128, 127, momentum
        self.register_buffer("mn", torch.zeros(()))
        self.register_buffer("mx", torch.zeros(()))
        # A plain Python flag, not a buffer: reading a CUDA bool buffer per
        # forward forces a device->host sync, and this runs on every conv
        # every step. It survives a deep copy; only the ranges (mn, mx) need
        # to persist in the state dict.
        self._inited = False

    def forward(self, x):
        if self.training:
            cmn, cmx = torch.aminmax(x.detach())
            if not self._inited:
                self.mn.copy_(cmn)
                self.mx.copy_(cmx)
                self._inited = True
            else:
                self.mn.mul_(self.momentum).add_((1 - self.momentum) * cmn)
                self.mx.mul_(self.momentum).add_((1 - self.momentum) * cmx)
        mn = self.mn.clamp(max=0.0)  # the range must include zero
        mx = self.mx.clamp(min=0.0)
        scale = ((mx - mn) / (self.qmax - self.qmin)).clamp_min(1e-8)
        zp = torch.round(self.qmin - mn / scale).clamp(self.qmin, self.qmax)
        xq = torch.clamp(torch.round(x / scale) + zp, self.qmin, self.qmax)
        xdq = (xq - zp) * scale
        return x + (xdq - x).detach()


class QuantConv2d(nn.Conv2d):
    """Conv2d that fake-quantizes its input activation and its weight.
    Created only by `enable_qat`, which attaches the `act_fq` observer.

    `fake_quant` is on by default; `fake_quant_disabled` turns it off for
    export, so the deploy graph is structural (the runtime does int8 from the
    observed ranges)."""

    fake_quant = True

    def forward(self, x):
        if not self.fake_quant:
            return super().forward(x)
        xq = self.act_fq(x)
        wq = _fq_weight_perchannel(self.weight)
        return F.conv2d(xq, wq, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def enable_qat(model):
    """Swap every plain Conv2d for a QuantConv2d in place, each with its own
    activation observer. Idempotent."""
    for m in model.modules():
        for name, c in list(m.named_children()):
            if type(c) is nn.Conv2d:
                q = QuantConv2d(c.in_channels, c.out_channels, c.kernel_size,
                                c.stride, c.padding, c.dilation, c.groups,
                                c.bias is not None)
                q.weight = c.weight
                if c.bias is not None:
                    q.bias = c.bias
                q.add_module("act_fq", ActFakeQuant())
                q.to(c.weight.device)
                setattr(m, name, q)
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
        for m, s in zip(quant, saved):
            m.fake_quant = s
