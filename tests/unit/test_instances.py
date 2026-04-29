"""Tests for :mod:`mayaku.structures.instances`."""

from __future__ import annotations

import pytest
import torch

from mayaku.structures.boxes import Boxes
from mayaku.structures.instances import Instances


def _toy_instances(device: torch.device) -> Instances:
    boxes = Boxes(
        torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0], [2.0, 2.0, 6.0, 6.0]],
            device=device,
        )
    )
    classes = torch.tensor([1, 2, 3], device=device)
    return Instances((10, 10), gt_boxes=boxes, gt_classes=classes)


def test_attribute_set_and_get(device: torch.device) -> None:
    inst = _toy_instances(device)
    assert len(inst) == 3
    assert inst.image_size == (10, 10)
    assert isinstance(inst.gt_boxes, Boxes)
    assert torch.equal(inst.gt_classes, torch.tensor([1, 2, 3], device=device))
    assert inst.has("gt_boxes") and not inst.has("scores")


def test_set_via_attribute(device: torch.device) -> None:
    inst = _toy_instances(device)
    inst.scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    assert inst.has("scores")


def test_field_length_mismatch_rejected(device: torch.device) -> None:
    inst = _toy_instances(device)
    with pytest.raises(ValueError, match="mismatching"):
        inst.set("scores", torch.tensor([0.1, 0.2], device=device))


def test_field_without_len_rejected(device: torch.device) -> None:
    inst = Instances((4, 4))
    with pytest.raises(ValueError, match="__len__"):
        inst.set("bad", 42)


def test_indexing_applies_to_every_field(device: torch.device) -> None:
    inst = _toy_instances(device)
    sub = inst[torch.tensor([True, False, True], device=device)]
    assert len(sub) == 2
    torch.testing.assert_close(sub.gt_classes, torch.tensor([1, 3], device=device))
    assert sub.gt_boxes.tensor.shape == (2, 4)
    # int → 1-element view
    one = inst[1]
    assert len(one) == 1
    torch.testing.assert_close(one.gt_classes, torch.tensor([2], device=device))


def test_remove_and_get_fields(device: torch.device) -> None:
    inst = _toy_instances(device)
    fields = inst.get_fields()
    assert set(fields) == {"gt_boxes", "gt_classes"}
    inst.remove("gt_classes")
    assert not inst.has("gt_classes")


def test_to_moves_tensor_fields(device: torch.device) -> None:
    inst = Instances((4, 4), gt_classes=torch.tensor([1, 2, 3]), tags=["a", "b", "c"])
    moved = inst.to(device)
    assert moved.gt_classes.device.type == device.type
    # list field unchanged
    assert moved.tags == ["a", "b", "c"]


def test_cat_concatenates_fields(device: torch.device) -> None:
    a = Instances(
        (10, 10),
        gt_boxes=Boxes(torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)),
        gt_classes=torch.tensor([5], device=device),
    )
    b = Instances(
        (10, 10),
        gt_boxes=Boxes(torch.tensor([[2.0, 2.0, 3.0, 3.0]], device=device)),
        gt_classes=torch.tensor([7], device=device),
    )
    out = Instances.cat([a, b])
    assert len(out) == 2
    torch.testing.assert_close(out.gt_classes, torch.tensor([5, 7], device=device))
    assert out.gt_boxes.tensor.shape == (2, 4)


def test_cat_rejects_image_size_mismatch(device: torch.device) -> None:
    a = Instances((10, 10), gt_classes=torch.tensor([1], device=device))
    b = Instances((20, 20), gt_classes=torch.tensor([2], device=device))
    with pytest.raises(ValueError, match="image sizes"):
        Instances.cat([a, b])


def test_len_with_no_fields_raises() -> None:
    inst = Instances((4, 4))
    with pytest.raises(ValueError, match="no fields"):
        len(inst)


def test_iter_not_supported() -> None:
    inst = Instances((4, 4))
    with pytest.raises(NotImplementedError):
        iter(inst)


def test_missing_field_attribute_raises() -> None:
    inst = Instances((4, 4))
    with pytest.raises(AttributeError, match="no field"):
        _ = inst.does_not_exist


def test_repr_does_not_crash_on_empty() -> None:
    inst = Instances((4, 4))
    s = repr(inst)
    assert "num_instances=0" in s
