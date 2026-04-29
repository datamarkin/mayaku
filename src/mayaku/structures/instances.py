"""Per-image dict-of-fields container.

:class:`Instances` is the universal carrier for everything attached to a
single image during the forward pass: ground-truth annotations
(``gt_boxes``, ``gt_classes``, ``gt_masks``, ``gt_keypoints``), proposal
fields (``proposal_boxes``, ``objectness_logits``), and predictions
(``pred_boxes``, ``pred_classes``, ``scores``, ``pred_masks``,
``pred_keypoints``, ``pred_keypoint_heatmaps``).

The container itself does **not** enforce naming or types — those
conventions live in the heads that produce/consume the fields. The only
invariant is structural: every field must support ``len()`` and have the
same length, so that ``instances[mask]`` slices every field consistently.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch

__all__ = ["Instances"]

# Field names used internally on the instance so we can store / retrieve
# the user-visible fields via __setattr__ / __getattr__ without recursion.
_RESERVED = ("_image_size", "_fields")


class Instances:
    """Per-image bag of fields, all of the same length.

    A field can be a :class:`torch.Tensor`, a :class:`Boxes`, a
    :class:`Keypoints`, a :class:`BitMasks`, a Python ``list``, or
    anything else that supports ``__len__`` and ``__getitem__``.
    """

    def __init__(self, image_size: tuple[int, int], **fields: Any) -> None:
        self._image_size = image_size
        self._fields: dict[str, Any] = {}
        for k, v in fields.items():
            self.set(k, v)

    # --- dataclass-like attribute access -----------------------------------

    @property
    def image_size(self) -> tuple[int, int]:
        """``(h, w)`` of the image these instances belong to."""
        return self._image_size

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RESERVED:
            object.__setattr__(self, name, value)
            return
        self.set(name, value)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is called only when normal lookup fails — so the
        # reserved attributes set in __init__ go through fast and we
        # only land here for user fields.
        if name == "_fields" or name not in self._fields:
            raise AttributeError(f"Instances has no field {name!r}")
        return self._fields[name]

    # --- field operations --------------------------------------------------

    def set(self, name: str, value: Any) -> None:
        """Add or overwrite a field. Length must match existing fields."""
        if not hasattr(value, "__len__"):
            raise ValueError(f"Field {name!r} value of type {type(value).__name__} has no __len__")
        n = len(value)
        if self._fields and n != len(self):
            raise ValueError(
                f"Field {name!r} has length {n}, mismatching existing length {len(self)}"
            )
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get(self, name: str) -> Any:
        return self._fields[name]

    def get_fields(self) -> dict[str, Any]:
        """Shallow copy of the underlying name → value mapping."""
        return dict(self._fields)

    # --- container protocol ------------------------------------------------

    def __len__(self) -> int:
        if not self._fields:
            # Match Detectron2: an Instances with no fields has no length.
            raise ValueError("Cannot take len() of an Instances with no fields")
        return len(next(iter(self._fields.values())))

    def __getitem__(self, item: int | slice | torch.Tensor) -> Instances:
        out = Instances(self._image_size)
        for k, v in self._fields.items():
            if isinstance(item, int):
                # Detectron2 normalizes int indexing to a 1-element slice
                # so every field still has __len__.
                out.set(k, v[item : item + 1])
            else:
                out.set(k, v[item])
        return out

    def __iter__(self) -> Iterator[Any]:
        # Iterating Instances yields per-element views — but only
        # meaningful for tensor-like fields. We deliberately don't
        # implement this (matches Detectron2's behavior of raising
        # NotImplementedError) so that misuse fails loudly.
        raise NotImplementedError("Instances is not iterable; index with [] instead")

    # --- device movement ---------------------------------------------------

    def to(self, device: torch.device | str) -> Instances:
        """Return a new :class:`Instances` with every tensor-like field
        moved to ``device``. Fields without a ``.to`` method are passed
        through unchanged (e.g. plain Python lists)."""
        out = Instances(self._image_size)
        for k, v in self._fields.items():
            mover = getattr(v, "to", None)
            out.set(k, mover(device) if callable(mover) else v)
        return out

    # --- concatenation -----------------------------------------------------

    @staticmethod
    def cat(instances_list: Sequence[Instances]) -> Instances:
        """Concatenate a list of :class:`Instances` along the field axis.

        All entries must share the same image size and the same set of
        keys; we keep the first item's key set, matching Detectron2.
        """
        if len(instances_list) == 0:
            raise ValueError("Instances.cat requires at least one element")
        image_size = instances_list[0]._image_size
        for inst in instances_list[1:]:
            if inst._image_size != image_size:
                raise ValueError(
                    "Cannot Instances.cat across different image sizes "
                    f"({inst._image_size} vs {image_size})"
                )
        keys = list(instances_list[0]._fields.keys())
        out = Instances(image_size)
        for k in keys:
            values = [inst.get(k) for inst in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                out.set(k, torch.cat(values, dim=0))
            elif hasattr(type(v0), "cat"):
                # Boxes, Keypoints, BitMasks etc. all expose a classmethod cat.
                out.set(k, type(v0).cat(values))
            elif isinstance(v0, list):
                merged: list[Any] = []
                for v in values:
                    merged.extend(v)
                out.set(k, merged)
            else:
                raise TypeError(
                    f"Don't know how to concatenate field {k!r} of type "
                    f"{type(v0).__name__}; provide a .cat classmethod"
                )
        return out

    # --- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            n = len(self)
        except ValueError:
            n = 0
        parts = ", ".join(f"{k}: {type(v).__name__}" for k, v in self._fields.items())
        return f"Instances(num_instances={n}, image_size={self._image_size}, fields=[{parts}])"
