"""PlaySituation → JSON-serializable example records for later encoders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ds_platform.hashing import payload_id
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import core_schema

from board_game_analysis.domain import PlaySituation

_ACTION_SET_SEP = "\x1f"


class _FrozenDict(dict[str, Any]):
    def _blocked(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("mapping is immutable")

    __setitem__ = _blocked  # type: ignore[assignment]
    __delitem__ = _blocked  # type: ignore[assignment]
    clear = _blocked  # type: ignore[assignment]
    pop = _blocked  # type: ignore[assignment]
    popitem = _blocked  # type: ignore[assignment]
    update = _blocked  # type: ignore[assignment]
    setdefault = _blocked  # type: ignore[assignment]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: object
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.dict_schema(core_schema.str_schema(), core_schema.any_schema()),
        )

    @classmethod
    def _validate(cls, value: object) -> _FrozenDict:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return _freeze_mapping(value)
        raise TypeError("expected mapping")


def _freeze_mapping(value: Mapping[str, Any]) -> _FrozenDict:
    if isinstance(value, _FrozenDict):
        return value
    return _FrozenDict(
        {
            key: _freeze_mapping(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    )


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StateExample(_FrozenModel):
    """One omniscient state. ``data`` stays a mapping, not a FeatureTable."""

    entity_id: str
    game_id: str
    situation_id: str
    state_id: str
    turn_number: int | None = None
    phase: str | None = None
    active_player_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_encoder_record(self) -> dict[str, object]:
        """Fields an Embedder may read. No title, mechanics, or catalog tags."""
        return {
            "state_id": self.state_id,
            "turn_number": self.turn_number,
            "phase": self.phase,
            "active_player_id": self.active_player_id,
            "data": self.data,
        }


class ObservationExample(_FrozenModel):
    """One observer's view of a state, with the items in that view."""

    entity_id: str
    game_id: str
    situation_id: str
    observation_id: str
    state_id: str
    observer_id: str
    information_space_id: str
    items: tuple[_FrozenDict, ...] = ()

    @field_validator("items", mode="before")
    @classmethod
    def _freeze_items(cls, value: object) -> tuple[_FrozenDict, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("items must be a sequence of mappings")
        frozen: list[_FrozenDict] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise TypeError("each observation item must be a mapping")
            frozen.append(_freeze_mapping(item))
        return tuple(frozen)

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def to_encoder_record(self) -> dict[str, object]:
        """Bag fields only: visibility, known flag, about, holder, payload keys."""
        return {
            "state_id": self.state_id,
            "observer_id": self.observer_id,
            "items": [_encoder_item(item) for item in self.items],
        }


class PairExample(_FrozenModel):
    """One authored transition, possibly with several simultaneous actions."""

    entity_id: str
    game_id: str
    situation_id: str
    transition_id: str
    from_state_id: str
    to_state_id: str
    action_ids: tuple[str, ...]
    kind: str
    observer_ids: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    @property
    def from_entity_id(self) -> str:
        return state_entity_id(self.situation_id, self.from_state_id)

    @property
    def to_entity_id(self) -> str:
        return state_entity_id(self.situation_id, self.to_state_id)

    @property
    def action_entity_id(self) -> str:
        return action_set_entity_id(self.situation_id, self.action_ids)


class SequenceStep(_FrozenModel):
    """One authored transition in a trajectory. IDs only; no GameState.data."""

    index: int
    from_state_id: str
    to_state_id: str
    action_ids: tuple[str, ...]
    transition_id: str
    pair_entity_id: str
    observer_ids: tuple[str, ...] = ()
    actor_id: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class SequenceExample(_FrozenModel):
    """Ordered states of one situation or one transition-linked trajectory.

    Phase 1 fills ``event_ids`` from ``situation.states`` and leaves
    ``steps`` empty. Phase 5 fills ``steps`` from authored transition
    chains and sets ``event_ids`` to the chain's state ids.
    """

    entity_id: str
    game_id: str
    situation_id: str
    group_id: str
    event_ids: tuple[str, ...]
    positions: tuple[int, ...]
    actor_ids: tuple[str | None, ...]
    steps: tuple[SequenceStep, ...] = ()
    observer_ids: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class ExampleBundle(_FrozenModel):
    states: tuple[StateExample, ...] = ()
    observations: tuple[ObservationExample, ...] = ()
    pairs: tuple[PairExample, ...] = ()
    sequences: tuple[SequenceExample, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "states": [example.to_mapping() for example in self.states],
            "observations": [example.to_mapping() for example in self.observations],
            "pairs": [example.to_mapping() for example in self.pairs],
            "sequences": [example.to_mapping() for example in self.sequences],
        }


def examples_from_situation(situation: PlaySituation) -> ExampleBundle:
    """Project one play fragment into encoder-ready records."""
    game_id = situation.game.id
    situation_id = _situation_id(situation)
    states = tuple(
        StateExample(
            entity_id=state_entity_id(situation_id, game_state.id),
            game_id=game_id,
            situation_id=situation_id,
            state_id=game_state.id,
            turn_number=game_state.turn_number,
            phase=game_state.phase,
            active_player_id=game_state.active_player_id,
            data=dict(game_state.data),
        )
        for game_state in situation.states
    )
    observations = tuple(
        ObservationExample(
            entity_id=observation_entity_id(situation_id, observation.id),
            game_id=game_id,
            situation_id=situation_id,
            observation_id=observation.id,
            state_id=observation.game_state_id,
            observer_id=observation.observer_id,
            information_space_id=observation.information_space_id,
            items=tuple(
                _freeze_mapping(item.model_dump(mode="python"))
                for item in situation.space_for(
                    observation.observer_id, observation.game_state_id
                ).items
            ),
        )
        for observation in situation.observations
    )
    pairs = tuple(
        PairExample(
            entity_id=pair_entity_id(situation_id, transition.id),
            game_id=game_id,
            situation_id=situation_id,
            transition_id=transition.id,
            from_state_id=transition.from_state_id,
            to_state_id=transition.to_state_id,
            action_ids=tuple(transition.action_ids),
            kind=transition.kind,
            observer_ids=_observer_ids(
                situation, transition.from_state_id, transition.to_state_id
            ),
        )
        for transition in situation.transitions
    )
    sequences = (
        SequenceExample(
            entity_id=sequence_entity_id(situation_id),
            game_id=game_id,
            situation_id=situation_id,
            group_id=sequence_entity_id(situation_id),
            event_ids=tuple(game_state.id for game_state in situation.states),
            positions=tuple(index for index, _state in enumerate(situation.states)),
            actor_ids=tuple(
                game_state.active_player_id for game_state in situation.states
            ),
        ),
    )
    return ExampleBundle(
        states=states,
        observations=observations,
        pairs=pairs,
        sequences=sequences,
    )


def examples_from_situations(situations: Sequence[PlaySituation]) -> ExampleBundle:
    """Concatenate example records from several situations."""
    states: list[StateExample] = []
    observations: list[ObservationExample] = []
    pairs: list[PairExample] = []
    sequences: list[SequenceExample] = []
    for situation in situations:
        bundle = examples_from_situation(situation)
        states.extend(bundle.states)
        observations.extend(bundle.observations)
        pairs.extend(bundle.pairs)
        sequences.extend(bundle.sequences)
    return ExampleBundle(
        states=tuple(states),
        observations=tuple(observations),
        pairs=tuple(pairs),
        sequences=tuple(sequences),
    )


def _encoder_item(item: Mapping[str, Any]) -> dict[str, object]:
    payload = item.get("payload")
    payload_keys: list[str] = []
    if isinstance(payload, Mapping):
        payload_keys = sorted(str(key) for key in payload)
    return {
        "visibility": item.get("visibility"),
        "content_known": item.get("content_known"),
        "about": item.get("about"),
        "holder_id": item.get("holder_id"),
        "payload_keys": payload_keys,
    }


def state_entity_id(situation_id: str, state_id: str) -> str:
    return f"{situation_id}/state/{state_id}"


def observation_entity_id(situation_id: str, observation_id: str) -> str:
    return f"{situation_id}/obs/{observation_id}"


def pair_entity_id(situation_id: str, transition_id: str) -> str:
    return f"{situation_id}/pair/{transition_id}"


def action_set_entity_id(situation_id: str, action_ids: Sequence[str]) -> str:
    ordered = sorted(action_ids)
    return f"{situation_id}/actions/{_ACTION_SET_SEP.join(ordered)}"


def sequence_entity_id(situation_id: str) -> str:
    return f"{situation_id}/seq"


def trajectory_entity_id(situation_id: str, first_transition_id: str) -> str:
    return f"{situation_id}/seq/{first_transition_id}"


def situation_id_for(situation: PlaySituation) -> str:
    return _situation_id(situation)


def _situation_topology_key(situation: PlaySituation) -> str:
    if not situation.states:
        return situation.game.id
    state_ids = ",".join(sorted(game_state.id for game_state in situation.states))
    transition_ids = ",".join(
        sorted(transition.id for transition in situation.transitions)
    )
    if transition_ids:
        return f"{situation.game.id}:{state_ids}|{transition_ids}"
    return f"{situation.game.id}:{state_ids}"


def _situation_content_record(situation: PlaySituation) -> dict[str, Any]:
    """Canonical play content beyond shared local topology ids."""
    return {
        "actions": sorted(
            [action.model_dump(mode="python") for action in situation.actions],
            key=lambda item: item["id"],
        ),
        "decision_spaces": sorted(
            [space.model_dump(mode="python") for space in situation.decision_spaces],
            key=lambda item: item["id"],
        ),
        "information_spaces": sorted(
            [space.model_dump(mode="python") for space in situation.information_spaces],
            key=lambda item: item["id"],
        ),
        "observations": sorted(
            [
                observation.model_dump(mode="python")
                for observation in situation.observations
            ],
            key=lambda item: item["id"],
        ),
        "states": sorted(
            [
                {
                    "id": game_state.id,
                    "turn_number": game_state.turn_number,
                    "phase": game_state.phase,
                    "active_player_id": game_state.active_player_id,
                    "data": game_state.data,
                }
                for game_state in situation.states
            ],
            key=lambda item: str(item["id"]),
        ),
        "transitions": sorted(
            [
                {
                    "id": transition.id,
                    "from_state_id": transition.from_state_id,
                    "to_state_id": transition.to_state_id,
                    "action_ids": sorted(transition.action_ids),
                }
                for transition in situation.transitions
            ],
            key=lambda item: item["id"],
        ),
    }


def _situation_content_digest(situation: PlaySituation) -> str:
    record = _situation_content_record(situation)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload_id(encoded)[:8]


def _situation_id(situation: PlaySituation) -> str:
    topology = _situation_topology_key(situation)
    if not situation.states:
        return topology
    return f"{topology}@{_situation_content_digest(situation)}"


def _observer_ids(
    situation: PlaySituation, from_state_id: str, to_state_id: str
) -> tuple[str, ...]:
    states = {from_state_id, to_state_id}
    seen: list[str] = []
    for observation in situation.observations:
        if observation.game_state_id not in states:
            continue
        if observation.observer_id in seen:
            continue
        seen.append(observation.observer_id)
    return tuple(seen)
