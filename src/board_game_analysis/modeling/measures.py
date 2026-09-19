"""Naive ontology-v0 operationalizations of catalog measurements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from board_game_analysis.analysis.spec import MeasurementScope, spec_by_id
from board_game_analysis.domain import (
    DerivedMeasurement,
    InformationItem,
    Observation,
    PlaySituation,
    Visibility,
)

MEASURE_METHOD = "ontology-v0-naive"
MEASURE_VERSION = "ontology-v0"

SUPPORTED_METRIC_IDS: tuple[str, ...] = (
    "information_volume",
    "information_visibility",
    "hidden_information",
    "information_asymmetry",
    "information_ownership",
    "self_knowledge_asymmetry",
    "information_gain",
    "information_loss",
    "legal_action_type_count",
    "available_decision_count",
    "decision_branching_factor",
    "decision_constraint",
    "structural_interaction",
)

_TARGETING_TOKENS = (
    "clue",
    "communicate",
    "give",
    "guess",
    "treat",
    "block",
)


class MeasureSpec(BaseModel):
    """Hashable BGA config for the ontology-v0 measurement engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    catalog_version: str = "ontology-v0"
    method: str = MEASURE_METHOD
    skip_incomplete_decision_spaces: bool = True
    metric_ids: tuple[str, ...] = Field(default=SUPPORTED_METRIC_IDS)


class PairAsymmetry(BaseModel):
    """Same-state information difference between two observers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: str
    left_id: str
    right_id: str
    value: float

    @property
    def pair_column(self) -> str:
        first, second = sorted((self.left_id, self.right_id))
        return f"{first}__{second}"


def default_measure_spec() -> MeasureSpec:
    return MeasureSpec()


def measure_situation(
    situation: PlaySituation,
    spec: MeasureSpec | None = None,
) -> list[DerivedMeasurement]:
    """Compute supported v0 metrics. Incomplete legal counts are omitted."""
    chosen = spec or default_measure_spec()
    _validate_metric_ids(chosen.metric_ids)
    measurements: list[DerivedMeasurement] = []
    for metric_id in chosen.metric_ids:
        measurements.extend(_MEASURE_FNS[metric_id](situation, chosen))
    return measurements


def measure_situations(
    situations: Sequence[PlaySituation],
    spec: MeasureSpec | None = None,
) -> list[DerivedMeasurement]:
    chosen = spec or default_measure_spec()
    return [
        measurement
        for situation in situations
        for measurement in measure_situation(situation, chosen)
    ]


def pairwise_information_asymmetry(
    situation: PlaySituation,
) -> list[PairAsymmetry]:
    """Jaccard distance of content_known item keys for each observer pair."""
    scores: list[PairAsymmetry] = []
    for game_state in situation.states:
        observations = _observations_at(situation, game_state.id)
        known_sets = [
            (
                observation.observer_id,
                _known_keys(_items(situation, observation)),
            )
            for observation in observations
        ]
        for left_index, (left_id, left_keys) in enumerate(known_sets):
            for right_id, right_keys in known_sets[left_index + 1 :]:
                scores.append(
                    PairAsymmetry(
                        state_id=game_state.id,
                        left_id=left_id,
                        right_id=right_id,
                        value=_jaccard_distance(left_keys, right_keys),
                    )
                )
    return scores


def _validate_metric_ids(metric_ids: tuple[str, ...]) -> None:
    unknown = [metric_id for metric_id in metric_ids if metric_id not in _MEASURE_FNS]
    if unknown:
        raise ValueError(f"unsupported measure ids: {unknown}")
    for metric_id in metric_ids:
        catalog = spec_by_id(metric_id)
        if not catalog.computable_from_ontology_v0:
            raise ValueError(f"{metric_id} is not computable from ontology v0")


def _information_volume(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    return [
        _measurement(
            situation,
            name="information_volume",
            value=float(len(_items(situation, observation))),
            scope=MeasurementScope.OBSERVATION,
            player_id=observation.observer_id,
            state_id=observation.game_state_id,
            scope_key=observation.id,
            unit="items",
        )
        for observation in situation.observations
    ]


def _information_visibility(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    measurements: list[DerivedMeasurement] = []
    for observation in situation.observations:
        items = _items(situation, observation)
        n_items = len(items)
        known = sum(1 for item in items if item.content_known)
        measurements.append(
            _measurement(
                situation,
                name="information_visibility",
                value=(known / n_items) if n_items else 0.0,
                scope=MeasurementScope.OBSERVATION,
                player_id=observation.observer_id,
                state_id=observation.game_state_id,
                scope_key=observation.id,
            )
        )
    return measurements


def _hidden_information(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    hidden_counts = []
    for observation in situation.observations:
        items = _items(situation, observation)
        hidden_counts.append(float(sum(1 for item in items if not item.content_known)))
    return [
        _measurement(
            situation,
            name="hidden_information",
            value=hidden_counts[index],
            scope=MeasurementScope.OBSERVATION,
            player_id=observation.observer_id,
            state_id=observation.game_state_id,
            scope_key=observation.id,
            unit="items",
        )
        for index, observation in enumerate(situation.observations)
    ]


def _information_asymmetry(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    measurements: list[DerivedMeasurement] = []
    by_state: dict[str, list[float]] = {}
    for score in pairwise_information_asymmetry(situation):
        by_state.setdefault(score.state_id, []).append(score.value)
    for state_id, values in by_state.items():
        measurements.append(
            _measurement(
                situation,
                name="information_asymmetry",
                value=sum(values) / len(values),
                scope=MeasurementScope.STATE,
                state_id=state_id,
                scope_key=state_id,
            )
        )
    return measurements


def _information_ownership(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    measurements: list[DerivedMeasurement] = []
    for game_state in situation.states:
        by_holder: dict[str, int] = {}
        known_to_sizes: dict[str, int] = {}
        for observation in _observations_at(situation, game_state.id):
            for item in _items(situation, observation):
                holder = item.holder_id if item.holder_id is not None else ""
                by_holder[holder] = by_holder.get(holder, 0) + 1
                key = item.about if item.about is not None else item.id
                known_to_sizes[key] = len(item.known_to)
        measurements.append(
            _measurement(
                situation,
                name="information_ownership",
                value={
                    "by_holder": by_holder,
                    "n_holders": len(by_holder),
                    "known_to_sizes": known_to_sizes,
                },
                scope=MeasurementScope.STATE,
                state_id=game_state.id,
                scope_key=game_state.id,
            )
        )
    return measurements


def _self_knowledge_asymmetry(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    measurements: list[DerivedMeasurement] = []
    for observation in situation.observations:
        items = _items(situation, observation)
        observer_id = observation.observer_id
        own_unknown = [
            item
            for item in items
            if item.holder_id == observer_id
            and (
                item.visibility == Visibility.UNKNOWN_TO_SELF or not item.content_known
            )
        ]
        other_known = [
            item
            for item in items
            if item.holder_id is not None
            and item.holder_id != observer_id
            and item.visibility == Visibility.OTHER_PRIVATE
            and item.content_known
        ]
        inverted = bool(own_unknown) and bool(other_known)
        value = float(len(own_unknown) + len(other_known)) if inverted else 0.0
        measurements.append(
            _measurement(
                situation,
                name="self_knowledge_asymmetry",
                value=value,
                scope=MeasurementScope.OBSERVATION,
                player_id=observer_id,
                state_id=observation.game_state_id,
                scope_key=observation.id,
            )
        )
    return measurements


def _information_gain(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    return _gain_loss(situation, name="information_gain", loss=False)


def _information_loss(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    return _gain_loss(situation, name="information_loss", loss=True)


def _gain_loss(
    situation: PlaySituation,
    *,
    name: str,
    loss: bool,
) -> list[DerivedMeasurement]:
    measurements: list[DerivedMeasurement] = []
    by_key = {
        (observation.observer_id, observation.game_state_id): observation
        for observation in situation.observations
    }
    for transition in situation.transitions:
        observers = {
            observation.observer_id
            for observation in situation.observations
            if observation.game_state_id
            in {transition.from_state_id, transition.to_state_id}
        }
        for observer_id in sorted(observers):
            before = by_key.get((observer_id, transition.from_state_id))
            after = by_key.get((observer_id, transition.to_state_id))
            if before is None or after is None:
                continue
            known_before = _known_keys(_items(situation, before))
            known_after = _known_keys(_items(situation, after))
            delta = known_before - known_after if loss else known_after - known_before
            measurements.append(
                _measurement(
                    situation,
                    name=name,
                    value=float(len(delta)),
                    scope=MeasurementScope.TRAJECTORY,
                    player_id=observer_id,
                    state_id=transition.to_state_id,
                    scope_key=f"{transition.id}:{observer_id}",
                    unit="items",
                )
            )
    return measurements


def _legal_action_type_count(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    return [
        _measurement(
            situation,
            name="legal_action_type_count",
            value=float(len(situation.definition.legal_action_types)),
            scope=MeasurementScope.GAME,
            scope_key=situation.game.id,
            unit="action_types",
        )
    ]


def _available_decision_count(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    measurements: list[DerivedMeasurement] = []
    for space in situation.decision_spaces:
        if spec.skip_incomplete_decision_spaces and not space.complete:
            continue
        value = float(
            sum(1 for option in space.options if option.legal and option.available)
        )
        measurements.append(
            _measurement(
                situation,
                name="available_decision_count",
                value=value,
                scope=MeasurementScope.PLAYER,
                player_id=space.player_id,
                state_id=space.game_state_id,
                scope_key=space.id,
                unit="actions",
            )
        )
    return measurements


def _decision_branching_factor(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    measurements: list[DerivedMeasurement] = []
    for game_state in situation.states:
        if game_state.active_player_id is None:
            continue
        space = _decision_space(situation, game_state.active_player_id, game_state.id)
        if space is None:
            continue
        if spec.skip_incomplete_decision_spaces and not space.complete:
            continue
        value = float(
            sum(1 for option in space.options if option.legal and option.available)
        )
        measurements.append(
            _measurement(
                situation,
                name="decision_branching_factor",
                value=value,
                scope=MeasurementScope.STATE,
                player_id=space.player_id,
                state_id=game_state.id,
                scope_key=game_state.id,
                unit="actions",
            )
        )
    return measurements


def _decision_constraint(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    measurements: list[DerivedMeasurement] = []
    for space in situation.decision_spaces:
        value = float(
            sum(1 for option in space.options if option.legal and not option.available)
        )
        measurements.append(
            _measurement(
                situation,
                name="decision_constraint",
                value=value,
                scope=MeasurementScope.PLAYER,
                player_id=space.player_id,
                state_id=space.game_state_id,
                scope_key=space.id,
                unit="actions",
            )
        )
    return measurements


def _structural_interaction(
    situation: PlaySituation,
    spec: MeasureSpec,
) -> list[DerivedMeasurement]:
    del spec
    targeting = [
        action_type
        for action_type in situation.definition.legal_action_types
        if any(token in action_type for token in _TARGETING_TOKENS)
    ]
    labels = sorted({item.label for item in situation.interpretations})
    return [
        _measurement(
            situation,
            name="structural_interaction",
            value={
                "interaction": situation.definition.interaction,
                "decision_timing": situation.definition.decision_timing,
                "targeting_action_types": targeting,
                "labels": labels,
            },
            scope=MeasurementScope.GAME,
            scope_key=situation.game.id,
        )
    ]


def _measurement(
    situation: PlaySituation,
    *,
    name: str,
    value: float | dict[str, Any],
    scope: str,
    scope_key: str,
    player_id: str | None = None,
    state_id: str | None = None,
    unit: str | None = None,
) -> DerivedMeasurement:
    catalog = spec_by_id(name)
    return DerivedMeasurement(
        id=f"{situation.game.id}:{name}:{scope_key}",
        game_id=situation.game.id,
        name=name,
        value=value,
        unit=unit if unit is not None else catalog.unit,
        method=MEASURE_METHOD,
        scope=scope,
        player_id=player_id,
        state_id=state_id,
        version=MEASURE_VERSION,
    )


def _items(situation: PlaySituation, observation: Observation) -> list[InformationItem]:
    return list(
        situation.space_for(observation.observer_id, observation.game_state_id).items
    )


def _observations_at(situation: PlaySituation, state_id: str) -> list[Observation]:
    return [
        observation
        for observation in situation.observations
        if observation.game_state_id == state_id
    ]


def _decision_space(situation: PlaySituation, player_id: str, state_id: str):
    for space in situation.decision_spaces:
        if space.player_id != player_id:
            continue
        if space.game_state_id in {None, state_id}:
            return space
    return None


def _known_keys(
    items: Sequence[InformationItem],
) -> set[tuple[str | None, str | None]]:
    return {(item.about, item.holder_id) for item in items if item.content_known}


def _jaccard_distance(
    left: set[tuple[str | None, str | None]],
    right: set[tuple[str | None, str | None]],
) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left ^ right) / len(union)


_MEASURE_FNS = {
    "information_volume": _information_volume,
    "information_visibility": _information_visibility,
    "hidden_information": _hidden_information,
    "information_asymmetry": _information_asymmetry,
    "information_ownership": _information_ownership,
    "self_knowledge_asymmetry": _self_knowledge_asymmetry,
    "information_gain": _information_gain,
    "information_loss": _information_loss,
    "legal_action_type_count": _legal_action_type_count,
    "available_decision_count": _available_decision_count,
    "decision_branching_factor": _decision_branching_factor,
    "decision_constraint": _decision_constraint,
    "structural_interaction": _structural_interaction,
}
