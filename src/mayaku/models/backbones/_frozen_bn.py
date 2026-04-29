"""Frozen batch normalisation.

Mirrors `DETECTRON2_TECHNICAL_SPEC.md` §7.5: ``FrozenBatchNorm2d``
stores ``weight, bias, running_mean, running_var`` as buffers (no
grad), and the forward bakes the affine into a single
``y = x * scale + shift``. This is the default backbone normalisation
for the in-scope detectors (`spec §6.1`,
``MODEL.RESNETS.NORM = "FrozenBN"``).

Why frozen rather than ``nn.BatchNorm2d.eval()``? Because the head
optimiser still touches the BN affine parameters when they're trainable
parameters of an ``nn.BatchNorm2d``, even in eval mode. ``FrozenBN``
makes the buffers explicitly non-trainable, so weight decay and the
optimiser leave them alone.

`BACKEND_PORTABILITY_REPORT.md` §3 confirms this is pure PyTorch and
runs on every required backend without special handling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["FrozenBatchNorm2d", "convert_frozen_batchnorm"]


class FrozenBatchNorm2d(nn.Module):
    """Per-channel affine + running stats, all non-trainable.

    Args:
        num_features: ``C`` in the input ``(N, C, H, W)``.
        eps: Numerical floor for ``rsqrt(running_var + eps)``. Matches
            torchvision's BN default of ``1e-5``.
    """

    # Class-level annotations so mypy knows the registered buffers are
    # Tensor (otherwise nn.Module.__getattr__ returns Tensor | Module).
    weight: Tensor
    bias: Tensor
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Buffers, not parameters: optimisers and weight decay must skip them.
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x: Tensor) -> Tensor:
        # Compute fold once per forward; PyTorch can fuse it with the
        # surrounding ops downstream of fp32 / fp16. Folding to a single
        # scale + shift is the same recipe Detectron2 uses (§7.5).
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        shift = self.bias - self.running_mean * scale
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


def convert_frozen_batchnorm(module: nn.Module) -> nn.Module:
    """Recursively replace every ``BatchNorm2d`` / ``SyncBatchNorm`` in
    ``module`` with a :class:`FrozenBatchNorm2d` carrying the same
    running stats and affine.

    Operates in place; returns ``module`` for chaining. Any submodule
    that is *already* a :class:`FrozenBatchNorm2d` is left untouched.
    The replacement preserves dtype/device by reading them off the
    source BN's buffers.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            continue
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            # Copy state (and dtype/device) from the source BN.
            assert child.running_mean is not None
            assert child.running_var is not None
            frozen.running_mean.copy_(child.running_mean)
            frozen.running_var.copy_(child.running_var)
            if child.affine and child.weight is not None and child.bias is not None:
                frozen.weight.copy_(child.weight.data)
                frozen.bias.copy_(child.bias.data)
            frozen = frozen.to(device=child.running_mean.device, dtype=child.running_mean.dtype)
            setattr(module, name, frozen)
        else:
            convert_frozen_batchnorm(child)
    return module
