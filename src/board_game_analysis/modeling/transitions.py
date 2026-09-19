"""Action-conditioned transition evaluation against observed next states."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import datetime

from ds_platform import RunContext, Store
from ds_platform.modeling.evaluate import EvaluationReport, evaluate
from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.pairs import align_pairs
from ds_platform.modeling.records import (
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    put_prediction_artifact,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import (
    RepresentationTable,
    representation_mse,
    select_entities,
)
from ds_platform.modeling.spec import (
    ConditioningSpec,
    MetricSpec,
    SplitSpec,
    spec_config_hash,
)
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import DerivedMeasurement, PlaySituation
from board_game_analysis.modeling.encoders.action import ActionBagEmbedder
from board_game_analysis.modeling.encoders.conditioned import (
    CONDITIONED_FAMILY,
    CopyStateEncoder,
    LinearConditionedEncoder,
)
from board_game_analysis.modeling.encoders.state import StateBagEmbedder
from board_game_analysis.modeling.examples import PairExample, examples_from_situations
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.pairs import (
    pair_table_from_examples,
    transition_pairs_from_situations,
)
from board_game_analysis.modeling.split import split_entities_by_game

TRANSITION_MODEL_LOGICAL_KEY = "bga:model/transition:v0"
ACTION_MODEL_LOGICAL_KEY = "bga:model/action-embedder:v0"
ACTION_REPR_LOGICAL_KEY = "bga:repr/actions:v0"
TRANSITION_REPR_LOGICAL_KEY = "bga:repr/transitions:v0"
TRANSITION_EVAL_LOGICAL_KEY = "bga:evaluation/transition:v0"

SCALAR_COLUMNS = (
    "information_gain",
    "information_loss",
    "information_volume_delta",
    "available_decision_count_delta",
)


class TransitionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    representation: EvaluationReport
    scalars: dict[str, EvaluationReport]
    skipped: tuple[str, ...]
    n_pairs: int
    copy_mse: float
    conditioned_mse: float


class TransitionRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation: TransitionEvaluation
    predicted: RepresentationTable
    copy_predicted: RepresentationTable
    targets: RepresentationTable
    action_payload_id: str
    model_payload_id: str
    prediction_payload_id: str
    evaluation_payload_id: str


def target_states_for_pairs(
    z_states: RepresentationTable,
    pairs: Sequence[PairExample],
) -> RepresentationTable:
    """Actual to-state vectors, rekeyed by pair id. Missing targets raise."""
    to_ids = [pair.to_entity_id for pair in pairs]
    selected = select_entities(z_states, to_ids)
    return RepresentationTable(
        entity_ids=tuple(pair.entity_id for pair in pairs),
        vectors=selected.vectors,
        dim=selected.dim,
        source_payload_ids=selected.source_payload_ids,
    )


def transition_scalar_table(
    pairs: Sequence[PairExample],
    measurements: Sequence[DerivedMeasurement],
    *,
    source_payload_id: str,
) -> FeatureTable:
    """Observed deltas only. Missing sides stay None; nothing is imputed."""
    values: list[tuple[Scalar, ...]] = []
    for pair in pairs:
        values.append(
            (
                _mean_named(measurements, "information_gain", pair),
                _mean_named(measurements, "information_loss", pair),
                _volume_delta(measurements, pair),
                _decision_delta(measurements, pair),
            )
        )
    return FeatureTable(
        entity_ids=tuple(pair.entity_id for pair in pairs),
        columns=SCALAR_COLUMNS,
        values=tuple(values),
        source_payload_ids=tuple(source_payload_id for _ in pairs),
    )


def evaluate_transitions(
    predicted: RepresentationTable,
    copy_predicted: RepresentationTable,
    targets: RepresentationTable,
    *,
    scalar_true: FeatureTable | None = None,
    scalar_pred: FeatureTable | None = None,
    skipped: Sequence[str] = (),
) -> TransitionEvaluation:
    conditioned_mse = representation_mse(targets, predicted)
    copy_mse = representation_mse(targets, copy_predicted)
    notes = (
        "copy-state baseline compared against action-conditioned model",
        "targets are encoded observed to-states only",
    )
    representation = EvaluationReport(
        metrics={
            "representation_mse": conditioned_mse,
            "copy_mse": copy_mse,
        },
        n=len(targets.entity_ids),
        n_missing=0,
        notes=notes,
    )
    scalars: dict[str, EvaluationReport] = {}
    if scalar_true is not None and scalar_pred is not None:
        metrics = (MetricSpec(name="rmse"), MetricSpec(name="mae"))
        for column in scalar_true.columns:
            true_index = scalar_true.columns.index(column)
            pred_index = scalar_pred.columns.index(column)
            y_true = [row[true_index] for row in scalar_true.values]
            y_pred = [row[pred_index] for row in scalar_pred.values]
            scalars[column] = evaluate(y_true, y_pred, metrics)
    return TransitionEvaluation(
        representation=representation,
        scalars=scalars,
        skipped=tuple(skipped),
        n_pairs=len(targets.entity_ids),
        copy_mse=copy_mse,
        conditioned_mse=conditioned_mse,
    )


def run_transition_experiment(
    situations: Sequence[PlaySituation],
    *,
    store: Store,
    run: RunContext,
    split: SplitSpec,
    dim: int = 16,
    ridge: float = 1e-4,
    created_at: datetime | None = None,
) -> TransitionRun:
    """Fit and evaluate a game-held-out action-conditioned baseline."""
    bundle = transition_pairs_from_situations(situations)
    if not bundle.pairs:
        raise ValueError("no actionable transitions in the supplied situations")

    examples = examples_from_situations(situations)
    state_encoder = StateBagEmbedder(dim=dim)
    state_ids = [example.entity_id for example in examples.states]
    state_records = [example.to_encoder_record() for example in examples.states]
    state_encoder.fit(state_ids, state_records)
    z_states = state_encoder.encode(state_ids, state_records)

    action_encoder = ActionBagEmbedder(dim=dim)
    action_ids = [example.entity_id for example in bundle.actions]
    action_records = [example.to_encoder_record() for example in bundle.actions]
    action_encoder.fit(action_ids, action_records)
    z_actions = action_encoder.encode(action_ids, action_records)

    pairs_table = pair_table_from_examples(bundle.pairs)
    contexts, conditions = align_pairs(z_states, z_actions, pairs_table)
    targets = target_states_for_pairs(z_states, bundle.pairs)

    game_ids = [pair.game_id for pair in bundle.pairs]
    assignment = split_entities_by_game(
        [pair.entity_id for pair in bundle.pairs], game_ids, split
    )
    train_pairs = _select_pairs(bundle.pairs, assignment.train_ids)
    test_pairs = _select_pairs(bundle.pairs, assignment.test_ids)
    if not train_pairs or not test_pairs:
        raise ValueError("game split produced an empty transition partition")

    train_ctx, train_cond, train_tgt = _slice_aligned(
        contexts, conditions, targets, [pair.entity_id for pair in train_pairs]
    )
    test_ctx, test_cond, test_tgt = _slice_aligned(
        contexts, conditions, targets, [pair.entity_id for pair in test_pairs]
    )

    model = LinearConditionedEncoder(ridge=ridge)
    model.set_targets(train_tgt)
    model.fit(train_ctx, train_cond, [pair.entity_id for pair in train_pairs])
    predicted = model.encode(
        test_ctx, test_cond, [pair.entity_id for pair in test_pairs]
    )
    copy_model = CopyStateEncoder()
    copy_predicted = copy_model.encode(
        test_ctx, test_cond, [pair.entity_id for pair in test_pairs]
    )

    measurements = measure_situations(situations)
    scalar_true = transition_scalar_table(
        test_pairs, measurements, source_payload_id=test_tgt.source_payload_ids[0]
    )
    evaluation = evaluate_transitions(
        predicted,
        copy_predicted,
        test_tgt,
        scalar_true=scalar_true,
        scalar_pred=_zero_change_table(scalar_true),
        skipped=bundle.skipped,
    )

    spec = ConditioningSpec(
        family=CONDITIONED_FAMILY,
        params={"ridge": ridge, "dim": dim},
        seed=split.seed,
    )
    run_with_hash = run.model_copy(update={"config_hash": spec_config_hash(spec)})
    action_pid, _action_rid = put_feature_dataset(
        store,
        representation_payload_bytes(z_actions),
        run=run_with_hash,
        inputs=[],
        media_type="application/json",
        logical_key=ACTION_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    put_model_artifact(
        store,
        pickle.dumps(action_encoder),
        run=run_with_hash,
        inputs=[action_pid],
        media_type="application/octet-stream",
        logical_key=ACTION_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )
    model_pid, _model_rid = put_model_artifact(
        store,
        pickle.dumps(model),
        run=run_with_hash,
        inputs=[action_pid],
        media_type="application/octet-stream",
        logical_key=TRANSITION_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )
    pred_pid, _pred_rid = put_prediction_artifact(
        store,
        representation_payload_bytes(predicted),
        run=run_with_hash,
        inputs=[model_pid, action_pid],
        media_type="application/json",
        logical_key=TRANSITION_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    eval_pid, _eval_rid = put_evaluation_artifact(
        store,
        evaluation.representation,
        run=run_with_hash,
        subject_payload_id=pred_pid,
        inputs=[pred_pid, model_pid],
        logical_key=TRANSITION_EVAL_LOGICAL_KEY,
        created_at=created_at,
    )
    return TransitionRun(
        evaluation=evaluation,
        predicted=predicted,
        copy_predicted=copy_predicted,
        targets=test_tgt,
        action_payload_id=action_pid,
        model_payload_id=model_pid,
        prediction_payload_id=pred_pid,
        evaluation_payload_id=eval_pid,
    )


def _select_pairs(
    pairs: Sequence[PairExample], entity_ids: Sequence[str]
) -> tuple[PairExample, ...]:
    wanted = set(entity_ids)
    return tuple(pair for pair in pairs if pair.entity_id in wanted)


def _slice_aligned(
    contexts: RepresentationTable,
    conditions: RepresentationTable,
    targets: RepresentationTable,
    pair_ids: Sequence[str],
) -> tuple[RepresentationTable, RepresentationTable, RepresentationTable]:
    return (
        select_entities(contexts, pair_ids),
        select_entities(conditions, pair_ids),
        select_entities(targets, pair_ids),
    )


def _mean_named(
    measurements: Sequence[DerivedMeasurement], name: str, pair: PairExample
) -> float | None:
    values = [
        float(item.value)
        for item in measurements
        if item.name == name
        and item.game_id == pair.game_id
        and item.scope == "trajectory"
        and isinstance(item.value, (int, float))
        and item.id.startswith(f"{pair.situation_id}:{name}:{pair.transition_id}:")
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _volume_delta(
    measurements: Sequence[DerivedMeasurement], pair: PairExample
) -> float | None:
    deltas: list[float] = []
    for observer_id in pair.observer_ids:
        before = _scalar_at(
            measurements,
            "information_volume",
            pair.game_id,
            observer_id,
            pair.from_state_id,
        )
        after = _scalar_at(
            measurements,
            "information_volume",
            pair.game_id,
            observer_id,
            pair.to_state_id,
        )
        if before is None or after is None:
            continue
        deltas.append(after - before)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _decision_delta(
    measurements: Sequence[DerivedMeasurement], pair: PairExample
) -> float | None:
    deltas: list[float] = []
    players = {item.player_id for item in measurements if item.player_id}
    for player_id in sorted(player for player in players if player is not None):
        before = _player_count(
            measurements, pair.game_id, player_id, pair.from_state_id
        )
        after = _player_count(measurements, pair.game_id, player_id, pair.to_state_id)
        if before is None or after is None:
            continue
        deltas.append(after - before)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _scalar_at(
    measurements: Sequence[DerivedMeasurement],
    name: str,
    game_id: str,
    player_id: str,
    state_id: str,
) -> float | None:
    for item in measurements:
        if (
            item.name == name
            and item.game_id == game_id
            and item.player_id == player_id
            and item.state_id == state_id
            and isinstance(item.value, (int, float))
        ):
            return float(item.value)
    return None


def _zero_change_table(table: FeatureTable) -> FeatureTable:
    values = tuple(
        tuple(0.0 if isinstance(cell, (int, float)) else None for cell in row)
        for row in table.values
    )
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=table.columns,
        values=values,
        source_payload_ids=table.source_payload_ids,
    )


def _player_count(
    measurements: Sequence[DerivedMeasurement],
    game_id: str,
    player_id: str,
    state_id: str,
) -> float | None:
    return _scalar_at(
        measurements, "available_decision_count", game_id, player_id, state_id
    )
