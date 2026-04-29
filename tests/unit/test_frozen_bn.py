"""Tests for :mod:`mayaku.models.backbones._frozen_bn`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mayaku.models.backbones import FrozenBatchNorm2d, convert_frozen_batchnorm


def test_frozen_bn_matches_eval_bn_numerically(device: torch.device) -> None:
    # A trainable BN in eval mode and a FrozenBN with the same buffers
    # must produce identical outputs.
    torch.manual_seed(0)
    bn = nn.BatchNorm2d(8).to(device)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.uniform_(-0.5, 0.5)
    bn.running_mean.uniform_(-1.0, 1.0)
    bn.running_var.uniform_(0.1, 2.0)
    bn.eval()

    frozen = convert_frozen_batchnorm(nn.BatchNorm2d(8).to(device))
    # convert_frozen_batchnorm needs the source BN as the child; use a
    # wrapper to exercise the recursive replacement path.
    holder = nn.Sequential(bn).to(device)
    convert_frozen_batchnorm(holder)
    frozen = holder[0]
    assert isinstance(frozen, FrozenBatchNorm2d)

    x = torch.randn(2, 8, 4, 5, device=device)
    bn_eval_y = bn(x)  # bn was already in eval before conversion
    frozen_y = frozen(x)
    torch.testing.assert_close(bn_eval_y, frozen_y, atol=1e-6, rtol=1e-6)


def test_frozen_bn_buffers_have_no_grad(device: torch.device) -> None:
    fb = FrozenBatchNorm2d(4).to(device)
    # All four state tensors are buffers, not parameters.
    assert list(fb.parameters()) == []
    for name in ("weight", "bias", "running_mean", "running_var"):
        t = getattr(fb, name)
        assert isinstance(t, torch.Tensor)
        assert not t.requires_grad


def test_frozen_bn_forward_shape_dtype(device: torch.device) -> None:
    fb = FrozenBatchNorm2d(3).to(device)
    x = torch.randn(2, 3, 5, 7, device=device)
    y = fb(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device.type == device.type


def test_convert_replaces_recursively() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 8, 1),
        nn.BatchNorm2d(8),
        nn.Sequential(nn.Conv2d(8, 16, 1), nn.BatchNorm2d(16)),
    )
    convert_frozen_batchnorm(model)
    assert isinstance(model[1], FrozenBatchNorm2d)
    assert isinstance(model[2][1], FrozenBatchNorm2d)


def test_convert_preserves_running_stats() -> None:
    bn = nn.BatchNorm2d(4)
    bn.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    bn.running_var.copy_(torch.tensor([0.5, 1.5, 2.5, 3.5]))
    bn.weight.data.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    bn.bias.data.copy_(torch.tensor([-0.1, -0.2, -0.3, -0.4]))

    holder = nn.Sequential(bn)
    convert_frozen_batchnorm(holder)
    frozen = holder[0]
    assert isinstance(frozen, FrozenBatchNorm2d)
    torch.testing.assert_close(frozen.running_mean, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(frozen.running_var, torch.tensor([0.5, 1.5, 2.5, 3.5]))
    torch.testing.assert_close(frozen.weight, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    torch.testing.assert_close(frozen.bias, torch.tensor([-0.1, -0.2, -0.3, -0.4]))


def test_convert_idempotent() -> None:
    holder = nn.Sequential(nn.BatchNorm2d(4))
    convert_frozen_batchnorm(holder)
    # Second call must be a no-op (does not blow up, does not double-wrap).
    convert_frozen_batchnorm(holder)
    assert isinstance(holder[0], FrozenBatchNorm2d)


def test_convert_handles_modules_without_bn() -> None:
    pure = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
    out = convert_frozen_batchnorm(pure)
    assert out is pure  # in-place + returns the input


def test_repr_is_informative() -> None:
    s = repr(FrozenBatchNorm2d(8, eps=1e-3))
    assert "num_features=8" in s
    # Don't pin the exact eps formatting; just check the substring.
    assert "eps" in s


def test_convert_skips_when_already_frozen() -> None:
    # Direct assertion: a FrozenBN child is left intact, not replaced.
    holder = nn.Sequential(FrozenBatchNorm2d(4))
    target = holder[0]
    convert_frozen_batchnorm(holder)
    assert holder[0] is target


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_eps_used_in_forward(eps: float) -> None:
    fb = FrozenBatchNorm2d(2, eps=eps)
    fb.running_var.copy_(torch.tensor([0.0, 0.0]))  # rsqrt(0+eps) = sqrt(1/eps)
    fb.weight.copy_(torch.tensor([1.0, 1.0]))
    fb.bias.copy_(torch.tensor([0.0, 0.0]))
    fb.running_mean.copy_(torch.tensor([0.0, 0.0]))
    x = torch.ones(1, 2, 1, 1)
    y = fb(x)
    expected = (1.0 / eps) ** 0.5
    torch.testing.assert_close(y[0, 0, 0, 0].item(), expected, atol=1e-3, rtol=1e-3)
