"""Deterministic sequence summaries and a ridge next-state predictor.

Prefix summaries use only from-states and observed actions. The target
to-state is never read. ``fit`` on the encoder is a no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.representations import RepresentationTable, select_entities
from ds_platform.modeling.sequences import SequenceTable, events_for_group, order_events

from board_game_analysis.modeling.examples import SequenceExample, action_set_entity_id
from board_game_analysis.modeling.probes import RidgeRegressor
from board_game_analysis.modeling.sequences import (
    PrefixExample,
    prefix_event_id,
    prefix_sequence_table,
    state_entity_ids,
)

SEQUENCE_FAMILY = "prefix-summary-v0"
TRAJECTORY_FAMILY = "trajectory-summary-v0"
PREDICTOR_FAMILY = "linear-prefix-v0"


class PrefixSequenceEncoder:
    """Order-sensitive prefix summary of event vectors.

    For event i in a group the output is:

    ``concat(first, last, mean, last-first, [n_steps])``

    over events ``0..i`` in position order. This is not an unordered bag.
    """

    family = SEQUENCE_FAMILY

    def fit(self, sequences: SequenceTable, events: RepresentationTable) -> None:
        del sequences, events

    def encode(
        self,
        sequences: SequenceTable,
        events: RepresentationTable,
    ) -> RepresentationTable:
        ordered = order_events(sequences)
        aligned = select_entities(events, ordered.event_ids)
        out_dim = _summary_dim(aligned.dim)
        vectors: list[tuple[float, ...]] = []
        sources: list[str] = []
        group_ids = _unique_in_order(ordered.group_ids)
        by_event = {
            event_id: index for index, event_id in enumerate(aligned.entity_ids)
        }
        output_ids: list[str] = []
        for group_id in group_ids:
            group = events_for_group(ordered, group_id)
            prefix: list[tuple[float, ...]] = []
            for event_id, source in zip(
                group.event_ids, group.source_payload_ids, strict=True
            ):
                prefix.append(aligned.vectors[by_event[event_id]])
                vectors.append(_prefix_summary(prefix, aligned.dim))
                sources.append(source)
                output_ids.append(event_id)
        if not output_ids:
            return RepresentationTable(
                entity_ids=(),
                vectors=(),
                dim=out_dim,
                source_payload_ids=(),
            )
        return RepresentationTable(
            entity_ids=tuple(output_ids),
            vectors=tuple(vectors),
            dim=out_dim,
            source_payload_ids=tuple(sources),
        )


class LastStatePredictor:
    """Baseline: predicted next state is the last prefix from-state."""

    family = "last-state-v0"

    def encode(
        self,
        prefixes: Sequence[PrefixExample],
        z_states: RepresentationTable,
    ) -> RepresentationTable:
        last_ids = [
            state_entity_ids(prefix.situation_id, prefix.state_ids)[-1]
            for prefix in prefixes
        ]
        selected = select_entities(z_states, last_ids)
        return RepresentationTable(
            entity_ids=tuple(prefix.entity_id for prefix in prefixes),
            vectors=selected.vectors,
            dim=selected.dim,
            source_payload_ids=selected.source_payload_ids,
        )


class LinearSequencePredictor:
    """Ridge map from a prefix summary to the next-state representation."""

    family = PREDICTOR_FAMILY

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.weights: tuple[tuple[float, ...], ...] | None = None
        self.output_dim: int | None = None
        self._targets: RepresentationTable | None = None

    def set_targets(self, targets: RepresentationTable) -> None:
        self._targets = targets

    def fit(self, prefixes: RepresentationTable, entity_ids: Sequence[str]) -> None:
        if self._targets is None:
            raise RuntimeError("set_targets must be called before fit")
        _require_keys(prefixes, entity_ids, "prefixes")
        _require_keys(self._targets, entity_ids, "targets")
        features = _feature_table(prefixes)
        heads: list[tuple[float, ...]] = []
        for dim_index in range(self._targets.dim):
            y = [vector[dim_index] for vector in self._targets.vectors]
            model = RidgeRegressor(ridge=self.ridge)
            model.fit(features, y)
            if model.weights is None:
                raise RuntimeError("ridge fit produced no weights")
            heads.append(model.weights)
        self.weights = tuple(heads)
        self.output_dim = self._targets.dim

    def encode(
        self, prefixes: RepresentationTable, entity_ids: Sequence[str]
    ) -> RepresentationTable:
        if self.weights is None or self.output_dim is None:
            raise RuntimeError("LinearSequencePredictor.fit must be called first")
        _require_keys(prefixes, entity_ids, "prefixes")
        features = _feature_table(prefixes)
        vectors: list[tuple[float, ...]] = []
        for row in features.values:
            intercept_row = [1.0, *_numeric_row(row)]
            vectors.append(tuple(_dot(head, intercept_row) for head in self.weights))
        return RepresentationTable(
            entity_ids=tuple(entity_ids),
            vectors=tuple(vectors),
            dim=self.output_dim,
            source_payload_ids=prefixes.source_payload_ids,
        )


def encode_prefix_summaries(
    prefixes: Sequence[PrefixExample],
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
) -> RepresentationTable:
    """Prefix summaries keyed by prefix/pair id. Targets are not read."""
    if not prefixes:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=0,
            source_payload_ids=(),
        )
    table, events = prefix_event_tables(prefixes, z_states, z_actions)
    contextual = PrefixSequenceEncoder().encode(table, events)
    last_ids = [
        prefix_event_id(prefix.entity_id, len(prefix.state_ids) - 1)
        for prefix in prefixes
    ]
    selected = select_entities(contextual, last_ids)
    extras = [
        (
            float(sum(1 for group in prefix.action_id_groups if group)),
            float(len(prefix.observer_ids)),
        )
        for prefix in prefixes
    ]
    vectors = tuple(
        tuple([*vector, *extra])
        for vector, extra in zip(selected.vectors, extras, strict=True)
    )
    return RepresentationTable(
        entity_ids=tuple(prefix.entity_id for prefix in prefixes),
        vectors=vectors,
        dim=selected.dim + 2,
        source_payload_ids=selected.source_payload_ids,
    )


def encode_trajectory_summaries(
    sequences: Sequence[SequenceExample],
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
) -> RepresentationTable:
    """Full observed trajectories, including the final state.

    Not a prediction input. Reversing the walk changes first/last/delta.
    """
    dim = z_states.dim
    action_dim = z_actions.dim
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    for sequence in sequences:
        if not sequence.event_ids:
            raise ValueError(f"{sequence.entity_id}: empty sequence")
        state_vecs = [
            _lookup(z_states, entity_id)
            for entity_id in state_entity_ids(sequence.situation_id, sequence.event_ids)
        ]
        action_vecs = _observed_action_vectors(sequence, z_actions)
        first = state_vecs[0]
        last = state_vecs[-1]
        mean = _mean(state_vecs, dim)
        delta = tuple(a - b for a, b in zip(last, first, strict=True))
        mean_action = _mean(action_vecs, action_dim)
        n_actions = float(sum(1 for step in sequence.steps if step.action_ids))
        extras = (
            float(len(sequence.steps)),
            n_actions,
            float(len(sequence.observer_ids)),
        )
        vectors.append(tuple([*first, *last, *mean, *delta, *mean_action, *extras]))
        sources.append(sequence.entity_id)
    out_dim = 4 * dim + action_dim + 3
    if not sequences:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=out_dim,
            source_payload_ids=(),
        )
    return RepresentationTable(
        entity_ids=tuple(sequence.entity_id for sequence in sequences),
        vectors=tuple(vectors),
        dim=out_dim,
        source_payload_ids=tuple(sources),
    )


def prefix_event_tables(
    prefixes: Sequence[PrefixExample],
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
) -> tuple[SequenceTable, RepresentationTable]:
    """Event table for prefixes: ``concat(z_from, z_action_or_zero)``."""
    table = prefix_sequence_table(prefixes)
    event_dim = z_states.dim + z_actions.dim
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    zero_action = tuple(0.0 for _ in range(z_actions.dim))
    for prefix in prefixes:
        state_ids = state_entity_ids(prefix.situation_id, prefix.state_ids)
        for index, entity_id in enumerate(state_ids):
            state_vec = _lookup(z_states, entity_id)
            action_ids = (
                prefix.action_id_groups[index]
                if index < len(prefix.action_id_groups)
                else ()
            )
            if action_ids:
                action_vec = _lookup(
                    z_actions, action_set_entity_id(prefix.situation_id, action_ids)
                )
            else:
                action_vec = zero_action
            vectors.append(tuple([*state_vec, *action_vec]))
            sources.append(table.source_payload_ids[len(vectors) - 1])
    return table, RepresentationTable(
        entity_ids=table.event_ids,
        vectors=tuple(vectors),
        dim=event_dim,
        source_payload_ids=tuple(sources),
    )


def target_states_for_prefixes(
    z_states: RepresentationTable,
    prefixes: Sequence[PrefixExample],
) -> RepresentationTable:
    """Encoded actual to-states. Missing targets raise."""
    target_ids = [
        state_entity_ids(prefix.situation_id, (prefix.target_state_id,))[0]
        for prefix in prefixes
    ]
    selected = select_entities(z_states, target_ids)
    return RepresentationTable(
        entity_ids=tuple(prefix.entity_id for prefix in prefixes),
        vectors=selected.vectors,
        dim=selected.dim,
        source_payload_ids=selected.source_payload_ids,
    )


def last_step_pairs(
    prefixes: Sequence[PrefixExample],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """From-state, action-set, and prefix ids for the Phase 4 baseline."""
    from_ids: list[str] = []
    action_ids: list[str] = []
    pair_ids: list[str] = []
    for prefix in prefixes:
        from_ids.append(state_entity_ids(prefix.situation_id, prefix.state_ids)[-1])
        actions = prefix.action_id_groups[-1] if prefix.action_id_groups else ()
        if not actions:
            raise KeyError(f"{prefix.entity_id}: no observed action for transition")
        action_ids.append(action_set_entity_id(prefix.situation_id, actions))
        pair_ids.append(prefix.entity_id)
    return tuple(from_ids), tuple(action_ids), tuple(pair_ids)


def _observed_action_vectors(
    sequence: SequenceExample, z_actions: RepresentationTable
) -> list[tuple[float, ...]]:
    vectors: list[tuple[float, ...]] = []
    for step in sequence.steps:
        if not step.action_ids:
            continue
        vectors.append(
            _lookup(
                z_actions, action_set_entity_id(sequence.situation_id, step.action_ids)
            )
        )
    return vectors


def _prefix_summary(
    vectors: Sequence[tuple[float, ...]], dim: int
) -> tuple[float, ...]:
    first = vectors[0]
    last = vectors[-1]
    mean = _mean(vectors, dim)
    delta = tuple(a - b for a, b in zip(last, first, strict=True))
    return tuple([*first, *last, *mean, *delta, float(len(vectors))])


def _summary_dim(event_dim: int) -> int:
    return 4 * event_dim + 1


def _mean(vectors: Sequence[tuple[float, ...]], dim: int) -> tuple[float, ...]:
    if not vectors:
        return tuple(0.0 for _ in range(dim))
    count = float(len(vectors))
    return tuple(
        sum(vector[index] for vector in vectors) / count for index in range(dim)
    )


def _lookup(table: RepresentationTable, entity_id: str) -> tuple[float, ...]:
    try:
        index = table.entity_ids.index(entity_id)
    except ValueError as exc:
        raise KeyError(f"entity not found: {entity_id!r}") from exc
    return table.vectors[index]


def _feature_table(table: RepresentationTable) -> FeatureTable:
    columns = tuple(f"p{index}" for index in range(table.dim))
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=columns,
        values=table.vectors,
        source_payload_ids=table.source_payload_ids,
    )


def _require_keys(
    table: RepresentationTable, entity_ids: Sequence[str], name: str
) -> None:
    if tuple(table.entity_ids) != tuple(entity_ids):
        raise ValueError(f"{name} must be keyed by entity_ids")


def _numeric_row(row: Sequence[object]) -> list[float]:
    values: list[float] = []
    for cell in row:
        if isinstance(cell, bool) or cell is None:
            raise TypeError("sequence features must be numeric")
        if isinstance(cell, (int, float)):
            values.append(float(cell))
            continue
        raise TypeError(f"expected numeric feature, got {type(cell)!r}")
    return values


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _unique_in_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)
