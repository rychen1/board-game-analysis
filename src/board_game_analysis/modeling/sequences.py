"""BGA trajectories: authored transition chains onto platform SequenceTable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ds_platform.hashing import payload_id
from ds_platform.modeling.sequences import SequenceTable
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import PlaySituation
from board_game_analysis.modeling.examples import (
    PairExample,
    SequenceExample,
    SequenceStep,
    examples_from_situation,
    pair_entity_id,
    situation_id_for,
    state_entity_id,
    trajectory_entity_id,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrefixExample(_FrozenModel):
    """Observed prefix ending at a from-state. The target to-state is excluded."""

    entity_id: str
    trajectory_id: str
    game_id: str
    situation_id: str
    step_index: int
    state_ids: tuple[str, ...]
    action_id_groups: tuple[tuple[str, ...], ...]
    pair_entity_id: str
    target_state_id: str
    observer_ids: tuple[str, ...] = ()
    actor_ids: tuple[str | None, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class SequenceBundle(_FrozenModel):
    sequences: tuple[SequenceExample, ...] = ()
    skipped: tuple[str, ...] = ()


def sequence_from_situation(situation: PlaySituation) -> SequenceBundle:
    """Build transition-linked trajectories. Missing actions stay empty."""
    examples = examples_from_situation(situation)
    if not examples.pairs:
        return SequenceBundle(
            skipped=(f"{situation.game.id}: no authored transitions",)
        )
    chains, skipped = _ordered_chains(examples.pairs)
    sequences: list[SequenceExample] = []
    for chain in chains:
        sequences.append(_trajectory_from_chain(situation, chain))
    return SequenceBundle(sequences=tuple(sequences), skipped=tuple(skipped))


def sequences_from_situations(situations: Sequence[PlaySituation]) -> SequenceBundle:
    """Keep situation boundaries. Situations are never concatenated."""
    sequences: list[SequenceExample] = []
    skipped: list[str] = []
    for situation in situations:
        bundle = sequence_from_situation(situation)
        sequences.extend(bundle.sequences)
        skipped.extend(bundle.skipped)
    return SequenceBundle(sequences=tuple(sequences), skipped=tuple(skipped))


def prefixes_from_situations(
    situations: Sequence[PlaySituation],
) -> tuple[PrefixExample, ...]:
    """Prefixes from each situation's trajectories. Target states stay out."""
    bundle = sequences_from_situations(situations)
    by_id = {situation_id_for(situation): situation for situation in situations}
    return prefixes_from_sequences(bundle.sequences, by_id)


def prefixes_from_sequences(
    sequences: Sequence[SequenceExample],
    situation_by_id: dict[str, PlaySituation] | None = None,
) -> tuple[PrefixExample, ...]:
    """One prefix per authored step. Target to-state is never in ``state_ids``."""
    prefixes: list[PrefixExample] = []
    for sequence in sequences:
        if not sequence.steps:
            continue
        for step in sequence.steps:
            state_ids = _prefix_state_ids(sequence, step.index)
            prefixes.append(
                PrefixExample(
                    entity_id=step.pair_entity_id,
                    trajectory_id=sequence.entity_id,
                    game_id=sequence.game_id,
                    situation_id=sequence.situation_id,
                    step_index=step.index,
                    state_ids=state_ids,
                    action_id_groups=tuple(
                        item.action_ids for item in sequence.steps[: step.index + 1]
                    ),
                    pair_entity_id=step.pair_entity_id,
                    target_state_id=step.to_state_id,
                    observer_ids=_prefix_observer_ids(
                        sequence, state_ids, situation_by_id
                    ),
                    actor_ids=sequence.actor_ids[: len(state_ids)],
                )
            )
    return tuple(prefixes)


def sequence_table_from_examples(
    sequences: Sequence[SequenceExample],
    *,
    source_payload_ids: Sequence[str] | None = None,
) -> SequenceTable:
    """Platform SequenceTable: one row per authored step, grouped by trajectory."""
    group_ids: list[str] = []
    event_ids: list[str] = []
    positions: list[int] = []
    actor_ids: list[str | None] = []
    sources: list[str] = []
    for sequence in sequences:
        for step in sequence.steps:
            group_ids.append(sequence.group_id)
            event_ids.append(step.pair_entity_id)
            positions.append(step.index)
            actor_ids.append(step.actor_id)
            sources.append(_step_source_id(sequence, step))
    if source_payload_ids is not None:
        if len(source_payload_ids) != len(event_ids):
            raise ValueError("source_payload_ids length must match step count")
        sources = list(source_payload_ids)
    return SequenceTable(
        group_ids=tuple(group_ids),
        event_ids=tuple(event_ids),
        positions=tuple(positions),
        actor_ids=tuple(actor_ids),
        source_payload_ids=tuple(sources),
    )


def prefix_sequence_table(prefixes: Sequence[PrefixExample]) -> SequenceTable:
    """One group per prefix. Events are prefix from-states in order."""
    group_ids: list[str] = []
    event_ids: list[str] = []
    positions: list[int] = []
    actor_ids: list[str | None] = []
    sources: list[str] = []
    for prefix in prefixes:
        for index, state_id in enumerate(prefix.state_ids):
            group_ids.append(prefix.entity_id)
            event_ids.append(_prefix_event_id(prefix.entity_id, index))
            positions.append(index)
            actor = prefix.actor_ids[index] if index < len(prefix.actor_ids) else None
            actor_ids.append(actor)
            sources.append(
                payload_id(
                    f"{prefix.entity_id}|{index}|{state_id}|{prefix.target_state_id}".encode()
                )
            )
    return SequenceTable(
        group_ids=tuple(group_ids),
        event_ids=tuple(event_ids),
        positions=tuple(positions),
        actor_ids=tuple(actor_ids),
        source_payload_ids=tuple(sources),
    )


def prefix_event_id(prefix_entity_id: str, index: int) -> str:
    return _prefix_event_id(prefix_entity_id, index)


def reverse_sequence(sequence: SequenceExample) -> SequenceExample:
    """Reverse the observed walk. Actions stay attached; none are invented."""
    if not sequence.steps:
        return sequence.model_copy(
            update={
                "entity_id": f"{sequence.entity_id}/rev",
                "group_id": f"{sequence.group_id}/rev",
                "event_ids": tuple(reversed(sequence.event_ids)),
                "actor_ids": tuple(reversed(sequence.actor_ids)),
            }
        )
    reversed_steps = tuple(
        SequenceStep(
            index=index,
            from_state_id=step.to_state_id,
            to_state_id=step.from_state_id,
            action_ids=step.action_ids,
            transition_id=step.transition_id,
            pair_entity_id=f"{step.pair_entity_id}/rev",
            observer_ids=step.observer_ids,
            actor_id=step.actor_id,
        )
        for index, step in enumerate(reversed(sequence.steps))
    )
    state_ids = _state_ids_from_steps(reversed_steps)
    return SequenceExample(
        entity_id=f"{sequence.entity_id}/rev",
        game_id=sequence.game_id,
        situation_id=sequence.situation_id,
        group_id=f"{sequence.group_id}/rev",
        event_ids=state_ids,
        positions=tuple(range(len(state_ids))),
        actor_ids=tuple(reversed(sequence.actor_ids)),
        steps=reversed_steps,
        observer_ids=sequence.observer_ids,
    )


def _trajectory_from_chain(
    situation: PlaySituation, chain: Sequence[PairExample]
) -> SequenceExample:
    game_id = situation.game.id
    situation_id = situation_id_for(situation)
    steps = tuple(
        SequenceStep(
            index=index,
            from_state_id=pair.from_state_id,
            to_state_id=pair.to_state_id,
            action_ids=pair.action_ids,
            transition_id=pair.transition_id,
            pair_entity_id=pair.entity_id,
            observer_ids=pair.observer_ids,
            actor_id=situation.state(pair.from_state_id).active_player_id,
        )
        for index, pair in enumerate(chain)
    )
    state_ids = _state_ids_from_steps(steps)
    actor_ids = tuple(
        situation.state(state_id).active_player_id for state_id in state_ids
    )
    observers: list[str] = []
    for step in steps:
        for observer_id in step.observer_ids:
            if observer_id not in observers:
                observers.append(observer_id)
    first_transition = chain[0].transition_id
    entity_id = trajectory_entity_id(situation_id, first_transition)
    return SequenceExample(
        entity_id=entity_id,
        game_id=game_id,
        situation_id=situation_id,
        group_id=entity_id,
        event_ids=state_ids,
        positions=tuple(range(len(state_ids))),
        actor_ids=actor_ids,
        steps=steps,
        observer_ids=tuple(observers),
    )


def _ordered_chains(
    pairs: Sequence[PairExample],
) -> tuple[list[list[PairExample]], list[str]]:
    by_from: dict[str, list[PairExample]] = {}
    incoming = {pair.to_state_id for pair in pairs}
    for pair in pairs:
        by_from.setdefault(pair.from_state_id, []).append(pair)
    used: set[str] = set()
    chains: list[list[PairExample]] = []
    skipped: list[str] = []

    def unused_from(state_id: str, chain_ids: set[str]) -> list[PairExample]:
        return [
            pair
            for pair in by_from.get(state_id, [])
            if pair.entity_id not in used and pair.entity_id not in chain_ids
        ]

    def walk(
        start: PairExample,
    ) -> tuple[list[PairExample] | None, str | None, list[str]]:
        if start.from_state_id == start.to_state_id:
            return None, f"{start.entity_id}: cycle", [start.entity_id]
        chain = [start]
        seen = {start.from_state_id, start.to_state_id}
        current = start
        while True:
            nxts = unused_from(current.to_state_id, {item.entity_id for item in chain})
            if len(nxts) != 1:
                break
            nxt = nxts[0]
            if nxt.from_state_id != current.to_state_id:
                consumed = [item.entity_id for item in chain] + [nxt.entity_id]
                return None, f"{nxt.entity_id}: non-contiguous", consumed
            if nxt.to_state_id in seen:
                consumed = [item.entity_id for item in chain] + [nxt.entity_id]
                return None, f"{nxt.entity_id}: cycle", consumed
            chain.append(nxt)
            seen.add(nxt.to_state_id)
            current = nxt
        return chain, None, [item.entity_id for item in chain]

    starts = [pair for pair in pairs if pair.from_state_id not in incoming]
    rest = [pair for pair in pairs if pair.from_state_id in incoming]
    for start in [*starts, *rest]:
        if start.entity_id in used:
            continue
        chain, reason, consumed = walk(start)
        used.update(consumed)
        if reason is not None:
            skipped.append(reason)
            continue
        if chain is not None:
            chains.append(chain)
    return chains, skipped


def _state_ids_from_steps(steps: Sequence[SequenceStep]) -> tuple[str, ...]:
    if not steps:
        return ()
    return (steps[0].from_state_id, *(step.to_state_id for step in steps))


def _prefix_state_ids(sequence: SequenceExample, step_index: int) -> tuple[str, ...]:
    return tuple(
        sequence.steps[0].from_state_id
        if index == 0
        else sequence.steps[index - 1].to_state_id
        for index in range(step_index + 1)
    )


def _prefix_observer_ids(
    sequence: SequenceExample,
    state_ids: Sequence[str],
    situation_by_id: dict[str, PlaySituation] | None,
) -> tuple[str, ...]:
    if situation_by_id is None:
        seen: list[str] = []
        wanted = set(state_ids)
        for step in sequence.steps:
            if step.from_state_id not in wanted or step.to_state_id not in wanted:
                continue
            for observer_id in step.observer_ids:
                if observer_id not in seen:
                    seen.append(observer_id)
        return tuple(seen)
    situation = situation_by_id.get(sequence.situation_id)
    if situation is None:
        return ()
    wanted = set(state_ids)
    seen: list[str] = []
    for observation in situation.observations:
        if observation.game_state_id not in wanted:
            continue
        if observation.observer_id in seen:
            continue
        seen.append(observation.observer_id)
    return tuple(seen)


def _prefix_event_id(prefix_entity_id: str, index: int) -> str:
    return f"{prefix_entity_id}/evt/{index}"


def _step_source_id(sequence: SequenceExample, step: SequenceStep) -> str:
    return payload_id(
        (
            f"{sequence.entity_id}|{step.index}|{step.transition_id}|"
            f"{step.from_state_id}|{step.to_state_id}|{','.join(step.action_ids)}"
        ).encode()
    )


def state_entity_ids(situation_id: str, state_ids: Sequence[str]) -> tuple[str, ...]:
    return tuple(state_entity_id(situation_id, state_id) for state_id in state_ids)


def pair_entity_ids(
    situation_id: str, transition_ids: Sequence[str]
) -> tuple[str, ...]:
    return tuple(pair_entity_id(situation_id, item) for item in transition_ids)
