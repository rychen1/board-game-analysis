"""Game-level splits so fixture members do not leak across folds."""

from __future__ import annotations

from collections.abc import Sequence

from ds_platform.modeling.spec import SplitSpec
from ds_platform.modeling.split import SplitAssignment, split_groups


def split_entities_by_game(
    entity_ids: Sequence[str],
    game_ids: Sequence[str],
    spec: SplitSpec,
) -> SplitAssignment:
    """Split entities by ``game_id``. One game never appears in two partitions."""
    if len(entity_ids) != len(game_ids):
        raise ValueError("entity_ids length must match game_ids length")
    group_assignment = split_groups(game_ids, spec)[0]
    train_games = set(group_assignment.train_ids)
    validation_games = set(group_assignment.validation_ids)
    test_games = set(group_assignment.test_ids)
    return SplitAssignment(
        train_ids=tuple(
            entity_id
            for entity_id, game_id in zip(entity_ids, game_ids, strict=True)
            if game_id in train_games
        ),
        validation_ids=tuple(
            entity_id
            for entity_id, game_id in zip(entity_ids, game_ids, strict=True)
            if game_id in validation_games
        ),
        test_ids=tuple(
            entity_id
            for entity_id, game_id in zip(entity_ids, game_ids, strict=True)
            if game_id in test_games
        ),
    )


def split_transitions_by_game(
    entity_ids: Sequence[str],
    game_ids: Sequence[str],
    spec: SplitSpec,
) -> SplitAssignment:
    """Game-safe transition split. Same invariant as :func:`split_entities_by_game`."""
    return split_entities_by_game(entity_ids, game_ids, spec)


def split_sequences_by_game(
    entity_ids: Sequence[str],
    game_ids: Sequence[str],
    spec: SplitSpec,
) -> SplitAssignment:
    """Game-safe sequence split. A game never appears on both sides."""
    return split_entities_by_game(entity_ids, game_ids, spec)
