"""Explicit BGA interventions. Not callables and not a simulator."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ds_platform.hashing import payload_id
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from pydantic import BaseModel, ConfigDict, Field

from board_game_analysis.domain import Visibility
from board_game_analysis.modeling.examples import ObservationExample

Operation = Literal[
    "identity",
    "hide_information",
    "reveal_information",
    "change_observer",
    "substitute_action",
]
Origin = Literal["observed", "predicted", "counterfactual"]

INTERVENTION_FAMILY = "intervention-v0"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _IdBody(_FrozenModel):
    """Canonical intervention fields used only to hash a stable id."""

    operation: Operation
    game_id: str
    situation_id: str
    source_state_id: str
    observer_id: str | None = None
    item_id: str | None = None
    target_observer_id: str | None = None
    factual_action_ids: tuple[str, ...] = ()
    alternative_action_ids: tuple[str, ...] = ()
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    provenance: tuple[str, ...] = ()


class Intervention(_FrozenModel):
    """Serializable, inspectable intervention. Never an opaque callable."""

    intervention_id: str = ""
    operation: Operation
    game_id: str
    situation_id: str
    source_state_id: str
    observer_id: str | None = None
    item_id: str | None = None
    target_observer_id: str | None = None
    factual_action_ids: tuple[str, ...] = ()
    alternative_action_ids: tuple[str, ...] = ()
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    provenance: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class InterventionResult(_FrozenModel):
    """Functional result of applying an intervention. Origin is never observed."""

    intervention: Intervention
    origin: Origin
    source_entity_id: str
    result_entity_id: str
    source_observation: ObservationExample | None = None
    result_observation: ObservationExample | None = None
    notes: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class InformationScalars(_FrozenModel):
    volume: float
    visibility: float
    hidden: float


def identity_intervention(
    *,
    game_id: str,
    situation_id: str,
    source_state_id: str,
    observer_id: str,
) -> Intervention:
    return _with_id(
        Intervention(
            operation="identity",
            game_id=game_id,
            situation_id=situation_id,
            source_state_id=source_state_id,
            observer_id=observer_id,
        )
    )


def hide_information(
    *,
    game_id: str,
    situation_id: str,
    source_state_id: str,
    observer_id: str,
    item_id: str,
) -> Intervention:
    return _with_id(
        Intervention(
            operation="hide_information",
            game_id=game_id,
            situation_id=situation_id,
            source_state_id=source_state_id,
            observer_id=observer_id,
            item_id=item_id,
        )
    )


def reveal_information(
    *,
    game_id: str,
    situation_id: str,
    source_state_id: str,
    observer_id: str,
    item_id: str,
) -> Intervention:
    return _with_id(
        Intervention(
            operation="reveal_information",
            game_id=game_id,
            situation_id=situation_id,
            source_state_id=source_state_id,
            observer_id=observer_id,
            item_id=item_id,
        )
    )


def change_observer(
    *,
    game_id: str,
    situation_id: str,
    source_state_id: str,
    observer_id: str,
    target_observer_id: str,
) -> Intervention:
    return _with_id(
        Intervention(
            operation="change_observer",
            game_id=game_id,
            situation_id=situation_id,
            source_state_id=source_state_id,
            observer_id=observer_id,
            target_observer_id=target_observer_id,
        )
    )


def substitute_action(
    *,
    game_id: str,
    situation_id: str,
    source_state_id: str,
    factual_action_ids: Sequence[str],
    alternative_action_ids: Sequence[str],
) -> Intervention:
    return _with_id(
        Intervention(
            operation="substitute_action",
            game_id=game_id,
            situation_id=situation_id,
            source_state_id=source_state_id,
            factual_action_ids=tuple(factual_action_ids),
            alternative_action_ids=tuple(alternative_action_ids),
        )
    )


def apply_intervention(
    source: ObservationExample,
    intervention: Intervention,
    *,
    observations: Sequence[ObservationExample] = (),
) -> InterventionResult:
    """Return a new observation. The source example is not mutated."""
    _require_source_match(source, intervention)
    if intervention.operation == "identity":
        result = _retarget(source, intervention)
        return _result(source, result, intervention, "identity no-op")
    if intervention.operation == "hide_information":
        result = _set_item(
            source,
            intervention,
            visibility=Visibility.HIDDEN,
            content_known=False,
        )
        return _result(source, result, intervention, "hide_information")
    if intervention.operation == "reveal_information":
        result = _set_item(
            source,
            intervention,
            visibility=Visibility.PUBLIC,
            content_known=True,
        )
        return _result(source, result, intervention, "reveal_information")
    if intervention.operation == "change_observer":
        target = _observation_for(
            observations,
            intervention.source_state_id,
            intervention.target_observer_id,
        )
        result = _retarget(target, intervention)
        return _result(
            source,
            result,
            intervention,
            "change_observer uses the target observer's existing view",
        )
    raise ValueError(
        f"apply_intervention does not apply {intervention.operation!r} "
        "to an observation"
    )


def observation_scalars(example: ObservationExample) -> InformationScalars:
    items = [_copy_item(item) for item in example.items]
    n_items = float(len(items))
    known = float(sum(1 for item in items if item.get("content_known")))
    hidden = n_items - known
    visibility = (known / n_items) if n_items else 0.0
    return InformationScalars(volume=n_items, visibility=visibility, hidden=hidden)


def scalar_information_delta(
    actual: ObservationExample, counterfactual: ObservationExample
) -> InformationScalars:
    left = observation_scalars(actual)
    right = observation_scalars(counterfactual)
    return InformationScalars(
        volume=right.volume - left.volume,
        visibility=right.visibility - left.visibility,
        hidden=right.hidden - left.hidden,
    )


def counterfactual_entity_id(source_entity_id: str, intervention: Intervention) -> str:
    suffix = intervention.intervention_id.rsplit("/", 1)[-1][:16]
    return f"{source_entity_id}/cf/{intervention.operation}/{suffix}"


def _with_id(intervention: Intervention) -> Intervention:
    if intervention.intervention_id:
        return intervention
    body = intervention.model_dump(mode="python")
    body.pop("intervention_id", None)
    digest = payload_id(spec_canonical_json_bytes(_IdBody.model_validate(body)))
    return intervention.model_copy(update={"intervention_id": f"bga/iv/{digest}"})


def _require_source_match(
    source: ObservationExample, intervention: Intervention
) -> None:
    if source.game_id != intervention.game_id:
        raise ValueError("intervention game_id does not match observation")
    if source.state_id != intervention.source_state_id:
        raise ValueError("intervention source_state_id does not match observation")
    if (
        intervention.observer_id is not None
        and source.observer_id != intervention.observer_id
        and intervention.operation != "change_observer"
    ):
        raise ValueError("intervention observer_id does not match observation")


def _set_item(
    source: ObservationExample,
    intervention: Intervention,
    *,
    visibility: str,
    content_known: bool,
) -> ObservationExample:
    item_id = intervention.item_id
    if not item_id:
        raise ValueError("item_id is required")
    items: list[dict[str, Any]] = []
    found = False
    for item in source.items:
        copied = _copy_item(item)
        if copied.get("id") == item_id:
            copied["visibility"] = visibility
            copied["content_known"] = content_known
            found = True
        items.append(copied)
    if not found:
        raise KeyError(f"no information item {item_id!r}")
    return source.model_copy(
        update={
            "items": tuple(items),
            "entity_id": counterfactual_entity_id(source.entity_id, intervention),
        }
    )


def _retarget(
    source: ObservationExample, intervention: Intervention
) -> ObservationExample:
    copied = tuple(_copy_item(item) for item in source.items)
    return source.model_copy(
        update={
            "items": copied,
            "entity_id": counterfactual_entity_id(source.entity_id, intervention),
        }
    )


def _observation_for(
    observations: Sequence[ObservationExample],
    state_id: str,
    observer_id: str | None,
) -> ObservationExample:
    if observer_id is None:
        raise ValueError("target_observer_id is required")
    for example in observations:
        if example.state_id == state_id and example.observer_id == observer_id:
            return example
    raise KeyError(f"no observation for {observer_id!r} at {state_id!r}")


def _result(
    source: ObservationExample,
    result: ObservationExample,
    intervention: Intervention,
    note: str,
) -> InterventionResult:
    return InterventionResult(
        intervention=intervention,
        origin="counterfactual",
        source_entity_id=source.entity_id,
        result_entity_id=result.entity_id,
        source_observation=source,
        result_observation=result,
        notes=(note, "result is counterfactual, not an observed fixture state"),
    )


def _copy_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(item))
