"""Tests for :mod:`mayaku.structures.image_list`."""

from __future__ import annotations

import pytest
import torch

from mayaku.structures.image_list import ImageList


def test_from_tensors_pads_to_max(device: torch.device) -> None:
    a = torch.ones(3, 4, 5, device=device)
    b = torch.ones(3, 6, 4, device=device) * 2
    il = ImageList.from_tensors([a, b])
    assert il.tensor.shape == (2, 3, 6, 5)
    assert il.image_sizes == [(4, 5), (6, 4)]
    # Image 0: original area filled with 1, the rest 0.
    assert torch.all(il.tensor[0, :, :4, :5] == 1)
    assert torch.all(il.tensor[0, :, 4:, :] == 0)
    # Image 1: original area filled with 2, padded col is 0.
    assert torch.all(il.tensor[1, :, :, :4] == 2)
    assert torch.all(il.tensor[1, :, :, 4:] == 0)


def test_size_divisibility_rounds_up(device: torch.device) -> None:
    a = torch.zeros(3, 5, 7, device=device)
    il = ImageList.from_tensors([a], size_divisibility=8)
    assert il.tensor.shape == (1, 3, 8, 8)
    # Original size preserved on the side.
    assert il.image_sizes == [(5, 7)]


def test_square_padding(device: torch.device) -> None:
    a = torch.zeros(3, 5, 9, device=device)
    il = ImageList.from_tensors([a], square=True)
    assert il.tensor.shape == (1, 3, 9, 9)


def test_pad_value_is_used() -> None:
    a = torch.zeros(1, 2, 3)
    il = ImageList.from_tensors([a, torch.zeros(1, 4, 4)], pad_value=-1.0)
    # Image 0 has padding rows/cols filled with -1.
    assert torch.all(il.tensor[0, :, 2:, :] == -1)


def test_getitem_crops_to_original_size(device: torch.device) -> None:
    a = torch.arange(20, dtype=torch.float32, device=device).view(1, 4, 5)
    b = torch.arange(24, dtype=torch.float32, device=device).view(1, 6, 4)
    il = ImageList.from_tensors([a, b])
    torch.testing.assert_close(il[0], a)
    torch.testing.assert_close(il[1], b)


def test_to_moves_tensor(device: torch.device) -> None:
    a = torch.zeros(3, 4, 5)
    il = ImageList.from_tensors([a]).to(device)
    assert il.tensor.device.type == device.type


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ImageList.from_tensors([])


def test_mismatched_dims_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(C, H, W\)"):
        ImageList.from_tensors([torch.zeros(4, 5)])


def test_mismatched_channels_rejected() -> None:
    with pytest.raises(ValueError, match="channel count"):
        ImageList.from_tensors([torch.zeros(3, 4, 5), torch.zeros(1, 4, 5)])


def test_mismatched_devices_rejected(device: torch.device) -> None:
    if device.type == "cpu":
        pytest.skip("requires a non-cpu accelerator to mix devices")
    with pytest.raises(ValueError, match="device and dtype"):
        ImageList.from_tensors([torch.zeros(3, 4, 5), torch.zeros(3, 4, 5, device=device)])


def test_constructor_validates_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, C, H, W\)"):
        ImageList(torch.zeros(3, 4, 5), [(4, 5)])
    with pytest.raises(ValueError, match="batch size"):
        ImageList(torch.zeros(2, 3, 4, 5), [(4, 5)])
