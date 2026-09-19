"""Logical-key registry invariants."""

from __future__ import annotations

import board_game_analysis.modeling.logical_keys as lk


def test_logical_keys_are_unique() -> None:
    by_value: dict[str, list[str]] = {}
    for name in lk.__all__:
        by_value.setdefault(getattr(lk, name), []).append(name)
    duplicates = {value: names for value, names in by_value.items() if len(names) > 1}
    assert duplicates == {
        "bga:model/sequence-encoder:v0": [
            "SEQUENCE_ENCODER_LOGICAL_KEY",
            "SEQUENCE_MODEL_LOGICAL_KEY",
        ],
    }


def test_logical_keys_use_bga_namespace() -> None:
    for name in lk.__all__:
        value = getattr(lk, name)
        assert value.startswith("bga:"), f"{name} must start with bga:"


def test_evaluation_keys_use_evaluation_namespace() -> None:
    for name in lk.__all__:
        if not name.endswith("_EVAL_LOGICAL_KEY"):
            continue
        value = getattr(lk, name)
        assert value.startswith("bga:evaluation/"), name
