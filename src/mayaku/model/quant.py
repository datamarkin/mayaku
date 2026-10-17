"""Quantization-aware training (fake-quant), matching the int8 deploy scheme.

Per-channel symmetric weights, per-tensor activations: symmetric by default
(zero point 0, the one int8 scheme every runtime takes, TensorRT included), or
affine (a zero point, the whole int8 grid over [min, max]; finer for one-sided
ReLU outputs, but TensorRT cannot run it). A model that
never calls `enable_qat` is plain `nn.Conv2d` and unchanged. Once `enable_qat`
swaps its convolutions (and arms its `ResidualQuant` add operands, with one
observer per tensor, see `tie_observers`), they fake-quantize in train and
eval, so the scored AP
is the int8 number (ranges freeze in eval). Quantization is a property of the
swapped module (`fake_quant`), not a process flag: two models can hold
different settings and nothing leaks between them. An export traces either
the structural five-op fp32 graph (`fake_quant_disabled`) or the explicit
int8 one (`qdq_export`: the trained ranges as quantize / dequantize ops);
`deploy_mode` picks between them.

The observed ranges are statistics of the weights they were observed with,
like BatchNorm's running statistics. A trained model keeps its trained
ranges: the network learned to work within them, and min / max over clean
images can be set by a single outlier channel (see
`mayaku.engine.trainer.recalibrate`). Only layers that never trained with
ranges (a freshly attached head) get them from data, with
`recalibrate_ranges`, after BatchNorm is recalibrated.
"""

import contextlib
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

# Smallest int8 scale: fp16's smallest normal number. A scale below it becomes
# 0 in an fp16 graph (an all-zero output channel otherwise gives ~1e-10), and
# TensorRT rejects a zero scale.
SCALE_FLOOR = 2.0 ** -14


def _weight_scales(w, n=127):
    """Per-output-channel symmetric int8 scales of a conv weight, (out,)."""
    return (w.detach().abs().amax(dim=(1, 2, 3)) / n).clamp_min(SCALE_FLOOR)


def _fq_weight_perchannel(w, n=127):
    """Per-output-channel symmetric int8 fake-quant, straight-through."""
    s = _weight_scales(w, n)[:, None, None, None]
    wq = torch.clamp(torch.round(w / s), -n - 1, n) * s
    return w + (wq - w).detach()


class ActFakeQuant(nn.Module):
    """Per-tensor int8 activation fake-quant with a min/max observer:
    symmetric (zero point 0, scale max|range| / 127) or affine (a zero point,
    scale (max - min) / 255). Training updates the range; eval, or
    `observe = False`, freezes it.

    `momentum` weights the running range as an EMA of per-batch min / max;
    None makes it the cumulative mean instead (a population estimate, as
    BatchNorm's `momentum=None`), which is what `recalibrate_ranges` uses.
    """

    def __init__(self, momentum=0.99, symmetric=True, optional=False):
        super().__init__()
        self.qmin, self.qmax, self.momentum = -128, 127, momentum
        self.symmetric, self.optional = symmetric, optional
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
        self._seen = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
        if prefix + "mn" in state_dict:
            self._inited, self._n = True, 1
        elif self.optional:
            # an observer newer than the checkpoint: it stays unobserved (and
            # passes values through) until training or calibration sees data
            for k in ("mn", "mx"):
                if prefix + k in missing_keys:
                    missing_keys.remove(prefix + k)

    def forward(self, x):
        # a tied observer (`tie_observers`) is called by every layer reading
        # the tensor; the statistics are the same, so it updates once
        if self.training and self.observe and (self._seen is None or self._seen() is not x):
            self._seen = weakref.ref(x)
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
                self.mn.lerp_(cmn, 1 - self.momentum)
                self.mx.lerp_(cmx, 1 - self.momentum)
        if not self._inited:
            return x                     # no range yet: nothing to quantize to
        scale, zp = self.qparams()
        # One fused kernel whose gradient is straight-through inside the range
        # and zero outside it: the int8 output of a clipped value does not
        # move with it, so neither may its gradient. An identity STE through
        # the clamp keeps pushing a clipped value the loss wants larger while
        # the forward never improves, and the float activation runs away (one
        # COCO P5 channel: 92 -> 4205 over 34 epochs). Clipped STE as in Yin
        # et al. 2019, arXiv 1903.05662.
        return torch.fake_quantize_per_tensor_affine(
            x, scale.reshape(1), zp.reshape(1).int(), self.qmin, self.qmax)

    def qparams(self):
        """(scale, zero point) of the int8 grid over the observed range, as
        0-d tensors; the range is widened to include zero."""
        if self.symmetric:
            scale = (torch.maximum(self.mn.abs(), self.mx.abs()) / self.qmax).clamp_min(SCALE_FLOOR)
            return scale, torch.zeros_like(scale)
        mn = self.mn.clamp(max=0.0)
        mx = self.mx.clamp(min=0.0)
        scale = ((mx - mn) / (self.qmax - self.qmin)).clamp_min(SCALE_FLOOR)
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
        x = _qdq_act(x, scale, zp)
        w = torch.fake_quantize_per_channel_affine(
            self.weight, ws, torch.zeros_like(ws, dtype=torch.int32), 0, -128, 127)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResidualQuant(nn.Module):
    """Quantizes the operand of an add that does not come straight from a
    convolution, so an int8 runtime can fuse the add (and the ReLU after it)
    into the convolution that produces the other operand.

    TensorRT's placement rule: quantize every input of a weighted layer, and
    the residual input of an elementwise add that follows one; the conv's own
    output stays unquantized, and the add and activation run in its int8
    kernel's epilogue (TensorRT developer guide, Q/DQ placement). Without
    this, every add between two convolutions runs as its own fp16 kernel with
    a requantize before the next conv, and int8 ends up slower than fp16.

    The identity on a float model. `enable_qat` attaches an optional observer
    (`act_fq`, as on a QuantConv2d): a checkpoint older than it loads with
    the observer unobserved, which passes values through until training or
    calibration observes it. Training sees exactly the quantization the
    deployed graph applies."""

    fake_quant = True
    qdq = None    # (scale, zero point), set by `qdq_export`

    def __init__(self):
        super().__init__()
        self.act_fq = None

    @property
    def observed(self):
        return self.act_fq is not None and self.act_fq._inited

    def forward(self, x):
        if self.qdq is not None:
            return _qdq_act(x, *self.qdq)
        if self.act_fq is None or not self.fake_quant:
            return x
        return self.act_fq(x)


def _qdq_act(x, scale, zp):
    """Per-tensor int8 quantize -> dequantize with constant parameters: what
    traces to an ONNX QuantizeLinear / DequantizeLinear pair."""
    return torch.fake_quantize_per_tensor_affine(x, scale, zp, -128, 127)


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


def enable_qat(model, symmetric=True, example=None):
    """Swap every plain Conv2d for a QuantConv2d in place, give every
    ResidualQuant an observer (symmetric or affine, `ActFakeQuant`), then tie
    the observers that read one tensor into one (`tie_observers`; `example`
    is the input that traces the model, default a small image batch).
    Idempotent."""
    for m in model.modules():
        for name, c in list(m.named_children()):
            if type(c) is nn.Conv2d:
                setattr(m, name, _quantized(c, ActFakeQuant(symmetric=symmetric)))
        if isinstance(m, ResidualQuant) and m.act_fq is None:
            m.act_fq = ActFakeQuant(symmetric=symmetric, optional=True).to(_device(model))
    tie_observers(model, example)
    return model


@torch.no_grad()
def tie_observers(model, example=None):
    """Give every tensor one int8 scale: the observers of all the layers that
    read the same tensor become one shared observer.

    TensorRT folds a quantize into the layer that produces the tensor, and
    the dequantize into each consumer, only when every consumer quantizes
    that tensor identically; with one observer per consumer the scales differ
    by rounding and each consumer pays a separate quantize kernel (on COCO
    nano at batch 8, 0.6 ms of requantizing, and int8 slower than fp16).
    Observers that read one tensor see the same values, so sharing them
    changes nothing the network learns. Tied observers tolerate a checkpoint
    holding only one of their keys (`optional`). Found by tracing `example`
    once, by default a small image batch for a detector (anything else is
    left untied unless an `example` is given): a forward pre-hook per
    quantized layer groups the layers by input tensor, and every input is
    kept alive until the grouping is done, so a freed tensor's storage cannot
    be reused by an unrelated one and tie it by mistake."""
    if example is None:
        if not hasattr(model, "out_names"):          # not a detector
            return model
        example = torch.zeros(1, 3, 64, 64, device=_device(model))
    readers, alive, hooks = {}, [], []

    def hook(mod, inp):
        alive.append(inp[0])
        readers.setdefault(inp[0].data_ptr(), []).append(mod)

    for m in _quant_modules(model):
        if m.act_fq is not None:
            hooks.append(m.register_forward_pre_hook(hook))
    was = model.training
    model.eval()
    try:
        with fake_quant_disabled(model):
            model(example)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    for mods in readers.values():
        shared = mods[0].act_fq
        if len(mods) > 1:
            shared.optional = True
        for m in mods[1:]:
            m.act_fq = shared
    return model


def _device(model):
    p = next(model.parameters(), None)
    return p.device if p is not None else "cpu"


def _observers(model):
    return [m for m in model.modules() if isinstance(m, ActFakeQuant)]


def activation_scheme(model):
    """"symmetric" or "affine": how a quantization-aware model's activations
    quantize; None for a float model."""
    obs = [m.symmetric for m in _observers(model)]
    if not obs:
        return None
    return "symmetric" if all(obs) else "affine"



def is_qat(model):
    """Whether `model` went through `enable_qat`."""
    return any(isinstance(m, QuantConv2d) for m in model.modules())


def _quant_modules(model):
    return [m for m in model.modules() if isinstance(m, (QuantConv2d, ResidualQuant))]


def strip_fake_quant(model):
    """Run every QuantConv2d in `model` as a plain convolution, and every
    ResidualQuant as the identity, from now on: the deployed fp32 graph (the
    int8 one is traced with `qdq_export`, which overrides this). For a model
    that is only served, not trained further."""
    for m in _quant_modules(model):
        m.fake_quant = False
    return model


@contextlib.contextmanager
def fake_quant_disabled(model):
    """Every QuantConv2d in `model` runs as a plain convolution inside the
    block, so a trace sees the structural graph rather than the round / clip
    / div ops of the simulation. Restores each module's setting on exit."""
    quant = _quant_modules(model)
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
    graph (`QuantConv2d._qdq_forward`), and every observed ResidualQuant
    quantizes its operand the same way, so an ONNX export carries the trained
    ranges as QuantizeLinear / DequantizeLinear pairs. Use on a fused model."""
    # an unobserved residual operand stays a plain add
    quant = [m for m in _quant_modules(model) if isinstance(m, QuantConv2d) or m.observed]
    for m in quant:
        scale, zp = m.act_fq.qparams()
        m.qdq = (float(scale), int(zp)) + ((_weight_scales(m.weight),) if isinstance(m, QuantConv2d) else ())
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
    obs = _observers(model)
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
    """Compute every activation range from data, for layers that never
    trained with ranges (a freshly attached head). A trained model keeps
    its trained ranges instead (module docstring).

    Every observer is reset and its range
    recomputed as the cumulative mean of per-batch min / max over `batches`,
    with BatchNorm in eval mode, so the ranges describe exactly the
    activations that evaluation and export will see. Recalibrate BatchNorm
    first. A model without QAT is untouched and `batches` is not read.
    """
    obs = _observers(model)
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
