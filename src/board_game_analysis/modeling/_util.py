"""Shared modeling helpers. Internal; not part of the stable public API."""

from __future__ import annotations

from collections.abc import Sequence

from board_game_analysis.modeling.examples import ObservationExample


def mean_floats(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_vector(vectors: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("cannot average an empty vector set")
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vector in vectors:
        if len(vector) != dim:
            raise ValueError("vectors must share a dim")
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(vectors))
    return tuple(total / count for total in totals)


def as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {value!r}")
    return float(value)


def jaccard_distance[T](left: set[T], right: set[T]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return 1.0 - (len(left & right) / len(union))


def known_item_ids(example: ObservationExample) -> set[str]:
    return {
        str(item["id"])
        for item in example.items
        if item.get("content_known") and item.get("id")
    }
