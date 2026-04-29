"""Dataset catalog + per-dataset metadata.

The Detectron2 reference (`DETECTRON2_TECHNICAL_SPEC.md` §5.1) keeps
``DatasetCatalog`` and ``MetadataCatalog`` as **module-global**
singletons. Spec §9.1 calls this out as one of the rewrite's
improvement opportunities — global state makes test isolation hard,
makes plugin code surprising, and makes parallel pytest runs flaky if
two tests register the same name.

This module instead exposes :class:`DatasetCatalog` as a regular class.
Construct one in your training script (or in a fixture); pass it to
the loaders / mapper that need it. The module-level
:data:`default_catalog` is provided as a convenience for one-shot
scripts but **is not** consulted implicitly by anything in the library.

The :class:`Metadata` dataclass carries the small per-dataset facts the
mapper actually consumes — class names and the keypoint flip-pair
permutation. It deliberately mirrors :class:`mayaku.config.schemas.ROIKeypointHeadConfig.flip_indices`
so configs and datasets agree on the same K-permutation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DatasetCatalog", "Metadata", "default_catalog"]


@dataclass(frozen=True)
class Metadata:
    """Per-dataset facts the rest of the pipeline reads.

    ``thing_classes`` is the contiguous list of category names (index =
    contiguous category id). ``thing_dataset_id_to_contiguous_id`` maps
    arbitrary on-disk category ids (e.g. COCO's sparse 1..90 range) to
    that contiguous range.

    ``keypoint_names`` and ``keypoint_flip_indices`` are non-None only
    when the dataset carries keypoint annotations. ``keypoint_flip_indices``
    is the permutation passed straight into
    :class:`mayaku.data.transforms.TransformList` so horizontal flip
    augmentation swaps left/right keypoints correctly (Step 4 contract).
    """

    name: str
    thing_classes: tuple[str, ...]
    thing_dataset_id_to_contiguous_id: dict[int, int] = field(default_factory=dict)
    keypoint_names: tuple[str, ...] | None = None
    keypoint_flip_indices: tuple[int, ...] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class DatasetCatalog:
    """Instance-based dataset registry.

    Keys are dataset names; values are ``(loader_fn, metadata)`` pairs.
    ``loader_fn`` is a no-arg callable returning the standard dataset
    dicts (`DETECTRON2_TECHNICAL_SPEC.md` §5.1 format).
    """

    def __init__(self) -> None:
        self._datasets: dict[str, tuple[Callable[[], list[dict[str, Any]]], Metadata]] = {}

    def register(
        self,
        name: str,
        loader: Callable[[], list[dict[str, Any]]],
        metadata: Metadata,
    ) -> None:
        if name in self._datasets:
            raise ValueError(f"Dataset {name!r} is already registered in this catalog")
        if metadata.name != name:
            raise ValueError(
                f"Metadata.name={metadata.name!r} does not match registered name {name!r}"
            )
        self._datasets[name] = (loader, metadata)

    def get(self, name: str) -> list[dict[str, Any]]:
        """Materialise the dataset dicts for ``name``."""
        loader, _meta = self._lookup(name)
        return loader()

    def metadata(self, name: str) -> Metadata:
        _loader, meta = self._lookup(name)
        return meta

    def names(self) -> list[str]:
        return sorted(self._datasets.keys())

    def remove(self, name: str) -> None:
        del self._datasets[name]

    def clear(self) -> None:
        self._datasets.clear()

    def __contains__(self, name: object) -> bool:
        return name in self._datasets

    def _lookup(self, name: str) -> tuple[Callable[[], list[dict[str, Any]]], Metadata]:
        if name not in self._datasets:
            available = ", ".join(self.names()) or "(empty)"
            raise KeyError(
                f"Dataset {name!r} is not registered in this catalog. Registered: {available}"
            )
        return self._datasets[name]


# Module-level convenience instance for ad-hoc scripts. Library code
# should accept a ``DatasetCatalog`` argument instead of reaching for
# this — using it implicitly recreates the global-state foot-gun.
default_catalog = DatasetCatalog()
