"""Second-order structural perspectives. Not nested psychological beliefs."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from ds_platform import RunContext, Store
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.interventions import representation_delta
from ds_platform.modeling.perspectives import (
    PerspectivePath,
    PerspectiveTable,
    compose_perspectives,
)
from ds_platform.modeling.records import (
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import (
    RepresentationTable,
    vector_distance,
)
from ds_platform.modeling.spec import PerspectiveSpec, spec_config_hash
from pydantic import BaseModel, ConfigDict, Field, field_validator

from board_game_analysis.modeling._util import (
    composite_source_payload_id,
    known_item_ids,
)
from board_game_analysis.modeling.encoders.observation import ObservationBagEmbedder
from board_game_analysis.modeling.examples import ObservationExample
from board_game_analysis.modeling.interventions import (
    Intervention,
    InterventionResult,
    apply_intervention,
)
from board_game_analysis.modeling.logical_keys import (
    HIGHER_ORDER_MODEL_LOGICAL_KEY,
    HIGHER_ORDER_REPR_LOGICAL_KEY,
    PERSPECTIVE_EVAL_LOGICAL_KEY,
)
from board_game_analysis.modeling.perspectives import (
    perspective_table_from_observations,
)

HIGHER_ORDER_FAMILY = "higher-order-v0"

_CORPUS_NOTE = (
    "A_about_B is a structural difference of observed views, not a belief about beliefs"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OrderedConcatComposer:
    """Concatenate path vectors in observer order, plus last-minus-first."""

    family = HIGHER_ORDER_FAMILY

    def compose(self, chain: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
        if not chain:
            raise ValueError("perspective chain must not be empty")
        first = chain[0]
        last = chain[-1]
        if any(len(vector) != len(first) for vector in chain):
            raise ValueError("perspective chain vectors must share a dim")
        delta = tuple(a - b for a, b in zip(last, first, strict=True))
        parts: list[float] = []
        for vector in chain:
            parts.extend(vector)
        parts.extend(delta)
        return tuple(parts)


class ObserverPath(_FrozenModel):
    """Ordered observers of one state. Not a nested-belief claim."""

    state_id: str
    observers: tuple[str, ...] = Field(min_length=1)
    allow_cycles: bool = False

    @field_validator("observers")
    @classmethod
    def _observers_non_empty(cls, observers: tuple[str, ...]) -> tuple[str, ...]:
        if any(not observer for observer in observers):
            raise ValueError("observers must be non-empty strings")
        return observers

    def to_perspective_path(self) -> PerspectivePath:
        return PerspectivePath(steps=self.observers)


class HigherOrderPerspective(_FrozenModel):
    entity_id: str
    game_id: str
    situation_id: str
    state_id: str
    focal_observer_id: str
    target_observer_id: str
    relationship: Literal["about"] = "about"
    vector: tuple[float, ...]
    distance: float
    n_only_focal: int
    n_only_target: int
    n_shared_known: int
    n_hidden_both: int
    origin: Literal["observed", "counterfactual"]
    source_payload_id: str
    notes: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class HigherOrderEvaluation(_FrozenModel):
    representation: EvaluationReport
    n_pairs: int
    n_asymmetric: int
    n_paths: int
    notes: tuple[str, ...]


class HigherOrderRun(_FrozenModel):
    evaluation: HigherOrderEvaluation
    table: RepresentationTable
    pairs: tuple[HigherOrderPerspective, ...]
    representation_payload_id: str
    model_payload_id: str
    evaluation_payload_id: str


def higher_order_entity_id(
    situation_id: str, state_id: str, focal_id: str, target_id: str
) -> str:
    return f"{situation_id}/hop/{state_id}/{focal_id}>{target_id}"


def higher_order_perspective(
    focal: ObservationExample,
    target: ObservationExample,
    z_focal: tuple[float, ...],
    z_target: tuple[float, ...],
    *,
    source_payload_id: str,
    origin: Literal["observed", "counterfactual"] = "observed",
) -> HigherOrderPerspective:
    """Build A_about_B from two same-state observations.

    ``A_about_B`` is not defined to equal ``B_about_A``.
    """
    if focal.state_id != target.state_id:
        raise ValueError("higher-order perspectives require the same state_id")
    if focal.game_id != target.game_id:
        raise ValueError("higher-order perspectives require the same game_id")
    if len(z_focal) != len(z_target):
        raise ValueError("perspective vectors must have the same dim")
    known_focal = known_item_ids(focal)
    known_target = known_item_ids(target)
    all_ids = _item_ids(focal) | _item_ids(target)
    only_focal = known_focal - known_target
    only_target = known_target - known_focal
    shared = known_focal & known_target
    hidden_both = all_ids - known_focal - known_target
    distance = vector_distance(z_focal, z_target, metric="cosine")
    delta = tuple(a - b for a, b in zip(z_focal, z_target, strict=True))
    extras = (
        distance,
        float(len(only_focal)),
        float(len(only_target)),
        float(len(shared)),
        float(len(hidden_both)),
    )
    entity_id = higher_order_entity_id(
        focal.situation_id,
        focal.state_id,
        focal.observer_id,
        target.observer_id,
    )
    if origin == "counterfactual":
        entity_id = f"{entity_id}/cf"
    return HigherOrderPerspective(
        entity_id=entity_id,
        game_id=focal.game_id,
        situation_id=focal.situation_id,
        state_id=focal.state_id,
        focal_observer_id=focal.observer_id,
        target_observer_id=target.observer_id,
        vector=tuple([*z_focal, *z_target, *delta, *extras]),
        distance=distance,
        n_only_focal=len(only_focal),
        n_only_target=len(only_target),
        n_shared_known=len(shared),
        n_hidden_both=len(hidden_both),
        origin=origin,
        source_payload_id=source_payload_id,
        notes=(_CORPUS_NOTE,),
    )


def higher_order_pairs(
    observations: Sequence[ObservationExample],
    z_obs: RepresentationTable,
    *,
    origin: Literal["observed", "counterfactual"] = "observed",
) -> tuple[HigherOrderPerspective, ...]:
    """All ordered observer pairs that share a state."""
    by_key = {
        (example.situation_id, example.state_id, example.observer_id): example
        for example in observations
    }
    index = {entity_id: i for i, entity_id in enumerate(z_obs.entity_ids)}
    pairs: list[HigherOrderPerspective] = []
    seen_states: list[tuple[str, str]] = []
    for example in observations:
        bucket = (example.situation_id, example.state_id)
        if bucket not in seen_states:
            seen_states.append(bucket)
    for situation_id, state_id in seen_states:
        observers = [
            example.observer_id
            for example in observations
            if example.situation_id == situation_id and example.state_id == state_id
        ]
        unique: list[str] = []
        for observer_id in observers:
            if observer_id not in unique:
                unique.append(observer_id)
        for focal_id in unique:
            for target_id in unique:
                if focal_id == target_id:
                    continue
                focal = by_key[(situation_id, state_id, focal_id)]
                target = by_key[(situation_id, state_id, target_id)]
                try:
                    z_focal = z_obs.vectors[index[focal.entity_id]]
                    z_target = z_obs.vectors[index[target.entity_id]]
                except KeyError as exc:
                    raise KeyError(
                        f"missing observation representation for {state_id!r}"
                    ) from exc
                focal_payload = z_obs.source_payload_ids[index[focal.entity_id]]
                target_payload = z_obs.source_payload_ids[index[target.entity_id]]
                pairs.append(
                    higher_order_perspective(
                        focal,
                        target,
                        z_focal,
                        z_target,
                        source_payload_id=composite_source_payload_id(
                            [focal_payload, target_payload]
                        ),
                        origin=origin,
                    )
                )
    return tuple(pairs)


def count_asymmetric_pairs(pairs: Sequence[HigherOrderPerspective]) -> int:
    """Count unordered observer pairs whose inverse perspective differs."""
    inverses = {
        (
            pair.situation_id,
            pair.state_id,
            pair.focal_observer_id,
            pair.target_observer_id,
        ): pair
        for pair in pairs
    }
    seen: set[tuple[str, str, str, str]] = set()
    count = 0
    for pair in pairs:
        left_id, right_id = sorted((pair.focal_observer_id, pair.target_observer_id))
        bucket = (pair.situation_id, pair.state_id, left_id, right_id)
        if bucket in seen:
            continue
        other = inverses.get(
            (
                pair.situation_id,
                pair.state_id,
                pair.target_observer_id,
                pair.focal_observer_id,
            )
        )
        if other is not None and other.vector != pair.vector:
            count += 1
            seen.add(bucket)
    return count


def higher_order_table(
    pairs: Sequence[HigherOrderPerspective],
) -> RepresentationTable:
    if not pairs:
        return RepresentationTable(
            entity_ids=(),
            vectors=(),
            dim=0,
            source_payload_ids=(),
        )
    return RepresentationTable(
        entity_ids=tuple(pair.entity_id for pair in pairs),
        vectors=tuple(pair.vector for pair in pairs),
        dim=len(pairs[0].vector),
        source_payload_ids=tuple(pair.source_payload_id for pair in pairs),
    )


def require_observers(
    observations: Sequence[ObservationExample],
    state_id: str,
    observer_ids: Sequence[str],
) -> None:
    present = {
        example.observer_id for example in observations if example.state_id == state_id
    }
    for observer_id in observer_ids:
        if observer_id not in present:
            raise KeyError(f"no observer {observer_id!r} at {state_id!r}")


def compose_observer_path(
    table: PerspectiveTable,
    path: ObserverPath,
    *,
    composer: OrderedConcatComposer | None = None,
) -> RepresentationTable:
    """Compose A→B→C for one state. Cycles raise unless explicitly allowed."""
    if _has_cycle(path.observers) and not path.allow_cycles:
        raise ValueError("observer path contains a cycle")
    present = {
        perspective_id
        for subject_id, perspective_id in zip(
            table.subject_ids, table.perspective_ids, strict=True
        )
        if subject_id == path.state_id
    }
    for observer_id in path.observers:
        if observer_id not in present:
            raise KeyError(f"no observer {observer_id!r} at {path.state_id!r}")
    chosen = composer or OrderedConcatComposer()
    return compose_perspectives(
        table,
        [path.to_perspective_path()],
        composer=chosen,
        subject_ids=[path.state_id],
    )


def intervene_higher_order(
    focal: ObservationExample,
    target: ObservationExample,
    intervention: Intervention,
    encoder: ObservationBagEmbedder,
    z_obs: RepresentationTable,
    z_focal: tuple[float, ...],
    z_target: tuple[float, ...],
    *,
    observations: Sequence[ObservationExample] = (),
) -> tuple[HigherOrderPerspective, HigherOrderPerspective, RepresentationTable]:
    """Compare A_about_B before and after an intervention on B's view."""
    index = {entity_id: idx for idx, entity_id in enumerate(z_obs.entity_ids)}
    focal_payload = z_obs.source_payload_ids[index[focal.entity_id]]
    target_payload = z_obs.source_payload_ids[index[target.entity_id]]
    actual = higher_order_perspective(
        focal,
        target,
        z_focal,
        z_target,
        source_payload_id=composite_source_payload_id([focal_payload, target_payload]),
        origin="observed",
    )
    applied: InterventionResult = apply_intervention(
        target, intervention, observations=observations
    )
    if applied.result_observation is None:
        raise ValueError("intervention produced no observation")
    encoded = encoder.encode(
        [applied.result_observation.entity_id],
        [applied.result_observation.to_encoder_record()],
    )
    counterfactual = higher_order_perspective(
        focal,
        applied.result_observation,
        z_focal,
        encoded.vectors[0],
        source_payload_id=composite_source_payload_id(
            [focal_payload, encoded.source_payload_ids[0]]
        ),
        origin="counterfactual",
    )
    actual_table = higher_order_table(
        [actual.model_copy(update={"entity_id": "query"})]
    )
    cf_table = higher_order_table(
        [counterfactual.model_copy(update={"entity_id": "query"})]
    )
    return actual, counterfactual, representation_delta(actual_table, cf_table)


def run_higher_order_experiment(
    observations: Sequence[ObservationExample],
    z_obs: RepresentationTable,
    *,
    store: Store,
    run: RunContext,
    created_at: datetime | None = None,
) -> HigherOrderRun:
    pairs = higher_order_pairs(observations, z_obs)
    table = higher_order_table(pairs)
    n_asymmetric = count_asymmetric_pairs(pairs)
    n_paths = 0
    persp = perspective_table_from_observations(observations, z_obs)
    composer = OrderedConcatComposer()
    for state_id in _unique_states(observations):
        observers = _observers_at(observations, state_id)
        if len(observers) < 3:
            continue
        path = ObserverPath(state_id=state_id, observers=tuple(observers[:3]))
        compose_observer_path(persp, path, composer=composer)
        n_paths += 1
    evaluation = HigherOrderEvaluation(
        representation=EvaluationReport(
            metrics={
                "n_pairs": float(len(pairs)),
                "n_asymmetric": float(n_asymmetric),
                "n_paths": float(n_paths),
            },
            n=len(pairs),
            n_missing=0,
            notes=(_CORPUS_NOTE,),
        ),
        n_pairs=len(pairs),
        n_asymmetric=n_asymmetric,
        n_paths=n_paths,
        notes=(_CORPUS_NOTE,),
    )
    spec = PerspectiveSpec(
        subject_field="game_state_id",
        perspective_field="observer_id",
        params={"family": HIGHER_ORDER_FAMILY},
    )
    run_with_hash = run.model_copy(update={"config_hash": spec_config_hash(spec)})
    repr_pid, _repr_rid = put_feature_dataset(
        store,
        representation_payload_bytes(table),
        run=run_with_hash,
        inputs=[],
        media_type="application/json",
        logical_key=HIGHER_ORDER_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    model_pid, _model_rid = put_model_artifact(
        store,
        pickle.dumps(composer),
        run=run_with_hash,
        inputs=[repr_pid],
        media_type="application/octet-stream",
        logical_key=HIGHER_ORDER_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )
    eval_pid, _eval_rid = put_evaluation_artifact(
        store,
        evaluation.representation,
        run=run_with_hash,
        subject_payload_id=repr_pid,
        inputs=[repr_pid, model_pid],
        logical_key=PERSPECTIVE_EVAL_LOGICAL_KEY,
        created_at=created_at,
    )
    return HigherOrderRun(
        evaluation=evaluation,
        table=table,
        pairs=pairs,
        representation_payload_id=repr_pid,
        model_payload_id=model_pid,
        evaluation_payload_id=eval_pid,
    )


def _item_ids(example: ObservationExample) -> set[str]:
    return {str(item["id"]) for item in example.items if item.get("id")}


def _has_cycle(observers: Sequence[str]) -> bool:
    return len(observers) != len(set(observers))


def _unique_states(observations: Sequence[ObservationExample]) -> list[str]:
    seen: list[str] = []
    for example in observations:
        if example.state_id not in seen:
            seen.append(example.state_id)
    return seen


def _observers_at(
    observations: Sequence[ObservationExample], state_id: str
) -> list[str]:
    seen: list[str] = []
    for example in observations:
        if example.state_id != state_id:
            continue
        if example.observer_id not in seen:
            seen.append(example.observer_id)
    return seen


def _inverse_vector(
    pairs: Sequence[HigherOrderPerspective], pair: HigherOrderPerspective
) -> tuple[float, ...] | None:
    for other in pairs:
        if (
            other.state_id == pair.state_id
            and other.focal_observer_id == pair.target_observer_id
            and other.target_observer_id == pair.focal_observer_id
        ):
            return other.vector
    return None
