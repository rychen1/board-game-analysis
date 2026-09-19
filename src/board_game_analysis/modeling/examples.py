"""PlaySituation → JSON-serializable example records for later encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from board_game_analysis.domain import PlaySituation


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StateExample(_FrozenModel):
    """One omniscient state. ``data`` stays a mapping, not a FeatureTable."""

    entity_id: str
    game_id: str
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
    observation_id: str
    state_id: str
    observer_id: str
    information_space_id: str
    items: tuple[dict[str, Any], ...] = ()

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
    transition_id: str
    from_state_id: str
    to_state_id: str
    action_ids: tuple[str, ...]
    kind: str

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class SequenceExample(_FrozenModel):
    """Ordered states of one situation. Short fixtures are still one group."""

    entity_id: str
    game_id: str
    group_id: str
    event_ids: tuple[str, ...]
    positions: tuple[int, ...]
    actor_ids: tuple[str | None, ...]

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
    states = tuple(
        StateExample(
            entity_id=_state_entity_id(game_id, game_state.id),
            game_id=game_id,
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
            entity_id=_observation_entity_id(game_id, observation.id),
            game_id=game_id,
            observation_id=observation.id,
            state_id=observation.game_state_id,
            observer_id=observation.observer_id,
            information_space_id=observation.information_space_id,
            items=tuple(
                item.model_dump(mode="python")
                for item in situation.space_for(
                    observation.observer_id, observation.game_state_id
                ).items
            ),
        )
        for observation in situation.observations
    )
    pairs = tuple(
        PairExample(
            entity_id=_pair_entity_id(game_id, transition.id),
            game_id=game_id,
            transition_id=transition.id,
            from_state_id=transition.from_state_id,
            to_state_id=transition.to_state_id,
            action_ids=tuple(transition.action_ids),
            kind=transition.kind,
        )
        for transition in situation.transitions
    )
    sequences = (
        SequenceExample(
            entity_id=_sequence_entity_id(game_id),
            game_id=game_id,
            group_id=game_id,
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


def _state_entity_id(game_id: str, state_id: str) -> str:
    return f"{game_id}/state/{state_id}"


def _observation_entity_id(game_id: str, observation_id: str) -> str:
    return f"{game_id}/obs/{observation_id}"


def _pair_entity_id(game_id: str, transition_id: str) -> str:
    return f"{game_id}/pair/{transition_id}"


def _sequence_entity_id(game_id: str) -> str:
    return f"{game_id}/seq"
