"""BGA adapters onto platform PairTable. No BGA copy of the table type."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ds_platform.hashing import payload_id
from ds_platform.modeling.pairs import PairTable
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import Action, PlaySituation
from board_game_analysis.modeling.examples import (
    PairExample,
    action_set_entity_id,
    examples_from_situation,
    examples_from_situations,
    situation_id_for,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActionExample(_FrozenModel):
    """Structural action-set record for one authored transition."""

    entity_id: str
    game_id: str
    pair_entity_id: str
    action_ids: tuple[str, ...]
    action_types: tuple[str, ...]
    parameter_keys: tuple[str, ...]
    n_actions: int
    player_ids: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_encoder_record(self) -> dict[str, object]:
        """Action type, parameter keys, and multiplicity. No titles or values."""
        return {
            "action_types": list(self.action_types),
            "parameter_keys": list(self.parameter_keys),
            "n_actions": self.n_actions,
        }


class TransitionPairBundle(_FrozenModel):
    pairs: tuple[PairExample, ...] = ()
    actions: tuple[ActionExample, ...] = ()
    skipped: tuple[str, ...] = ()


def transition_pairs_from_situation(situation: PlaySituation) -> TransitionPairBundle:
    """Keep authored transitions only. Empty action sets are skipped, not filled."""
    return _bundle_from_pairs(situation, examples_from_situation(situation).pairs)


def transition_pairs_from_situations(
    situations: Sequence[PlaySituation],
) -> TransitionPairBundle:
    pairs = examples_from_situations(situations).pairs
    by_situation = {situation_id_for(situation): situation for situation in situations}
    kept: list[PairExample] = []
    actions: list[ActionExample] = []
    skipped: list[str] = []
    for pair in pairs:
        situation = by_situation[pair.situation_id]
        resolved = _actionable_pair(situation, pair)
        if isinstance(resolved, str):
            skipped.append(resolved)
            continue
        kept.append(pair)
        actions.append(resolved)
    return TransitionPairBundle(
        pairs=tuple(kept),
        actions=tuple(actions),
        skipped=tuple(skipped),
    )


def pair_table_from_examples(
    pairs: Sequence[PairExample],
    *,
    source_payload_ids: Sequence[str] | None = None,
) -> PairTable:
    """Platform PairTable: context=from-state, condition=action-set."""
    if source_payload_ids is None:
        sources = tuple(_pair_source_id(pair) for pair in pairs)
    else:
        if len(source_payload_ids) != len(pairs):
            raise ValueError("source_payload_ids length must match pairs length")
        sources = tuple(source_payload_ids)
    return PairTable(
        pair_ids=tuple(pair.entity_id for pair in pairs),
        context_ids=tuple(pair.from_entity_id for pair in pairs),
        condition_ids=tuple(pair.action_entity_id for pair in pairs),
        source_payload_ids=sources,
    )


def _bundle_from_pairs(
    situation: PlaySituation, pairs: Sequence[PairExample]
) -> TransitionPairBundle:
    kept: list[PairExample] = []
    actions: list[ActionExample] = []
    skipped: list[str] = []
    for pair in pairs:
        resolved = _actionable_pair(situation, pair)
        if isinstance(resolved, str):
            skipped.append(resolved)
            continue
        kept.append(pair)
        actions.append(resolved)
    return TransitionPairBundle(
        pairs=tuple(kept),
        actions=tuple(actions),
        skipped=tuple(skipped),
    )


def _actionable_pair(
    situation: PlaySituation, pair: PairExample
) -> ActionExample | str:
    if not pair.action_ids:
        return f"{pair.entity_id}: no action_ids"
    situation.state(pair.from_state_id)
    situation.state(pair.to_state_id)
    resolved = _actions_for(situation, pair.action_ids)
    return ActionExample(
        entity_id=action_set_entity_id(pair.situation_id, pair.action_ids),
        game_id=pair.game_id,
        pair_entity_id=pair.entity_id,
        action_ids=pair.action_ids,
        action_types=tuple(action.action_type for action in resolved),
        parameter_keys=_parameter_keys(resolved),
        n_actions=len(resolved),
        player_ids=tuple(action.player_id for action in resolved),
    )


def _actions_for(situation: PlaySituation, action_ids: Sequence[str]) -> list[Action]:
    by_id = {action.id: action for action in situation.actions}
    resolved: list[Action] = []
    for action_id in action_ids:
        try:
            resolved.append(by_id[action_id])
        except KeyError as exc:
            raise KeyError(f"no action {action_id}") from exc
    return resolved


def _parameter_keys(actions: Sequence[Action]) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for action in actions:
        for key in sorted(str(item) for item in action.parameters):
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


def _pair_source_id(pair: PairExample) -> str:
    return payload_id(
        (
            f"{pair.entity_id}|{pair.from_state_id}|{pair.to_state_id}|"
            f"{','.join(pair.action_ids)}"
        ).encode()
    )
