"""Tests for :class:`mayaku.data.serialize.SerializedList`."""

from __future__ import annotations

import pickle

import pytest

from mayaku.data.serialize import SerializedList


def test_empty_list() -> None:
    sl = SerializedList([])
    assert len(sl) == 0
    with pytest.raises(IndexError):
        _ = sl[0]


def test_len_matches_input() -> None:
    items = [{"i": i} for i in range(123)]
    sl = SerializedList(items)
    assert len(sl) == 123


def test_roundtrip_equals_input() -> None:
    items = [
        {"file_name": f"img_{i}.png", "image_id": i, "annotations": [{"bbox": [i, i, 4, 4]}]}
        for i in range(10)
    ]
    sl = SerializedList(items)
    for i, item in enumerate(items):
        assert sl[i] == item


def test_returns_fresh_objects_each_call() -> None:
    """Each ``__getitem__`` must return a new object so callers can mutate."""
    sl = SerializedList([{"annotations": [1, 2, 3]}])
    a = sl[0]
    b = sl[0]
    a["annotations"].append(99)
    # b is a fresh decode, untouched.
    assert b["annotations"] == [1, 2, 3]


def test_negative_index() -> None:
    sl = SerializedList([{"i": 0}, {"i": 1}, {"i": 2}])
    assert sl[-1] == {"i": 2}
    assert sl[-3] == {"i": 0}
    with pytest.raises(IndexError):
        _ = sl[-4]


def test_out_of_range_raises() -> None:
    sl = SerializedList([{"i": 0}])
    with pytest.raises(IndexError):
        _ = sl[1]


def test_handles_nested_polygon_lists() -> None:
    """Realistic COCO-shaped dict with nested polygon segmentation."""
    item = {
        "file_name": "x.png",
        "image_id": 7,
        "height": 400,
        "width": 600,
        "annotations": [
            {
                "bbox": [10.5, 20.5, 30.0, 40.0],
                "category_id": 3,
                "segmentation": [[10.0, 20.0, 40.0, 20.0, 40.0, 60.0, 10.0, 60.0]],
            }
        ],
    }
    sl = SerializedList([item])
    assert sl[0] == item


def test_pickle_roundtrip() -> None:
    """``SerializedList`` must be picklable so DataLoader workers under
    ``spawn`` / ``forkserver`` start methods receive it without
    ``TypeError``. Regression for the 'cannot pickle memoryview'
    failure that broke 7 ``test_cli`` cases on macOS (and which would
    bite Linux on Python 3.14+ when the runtime default flips to
    ``forkserver``)."""
    src = [
        {"file_name": f"img_{i}.jpg", "annotations": [{"bbox": [0, 0, 10, 10]}]}
        for i in range(5)
    ]
    sl = SerializedList(src)
    blob = pickle.dumps(sl)
    sl2 = pickle.loads(blob)
    assert len(sl2) == len(sl) == 5
    for i in range(5):
        assert sl2[i] == sl[i] == src[i]


def test_pickle_roundtrip_empty() -> None:
    """Empty list edge case — the ``offsets = np.zeros(0)`` branch in
    ``__init__`` must round-trip cleanly too."""
    sl = SerializedList([])
    sl2 = pickle.loads(pickle.dumps(sl))
    assert len(sl2) == 0
