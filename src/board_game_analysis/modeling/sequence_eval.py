"""Sequence-aware next-state evaluation against last-state and Phase 4."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import datetime

from ds_platform import RunContext, Store
from ds_platform.modeling.evaluate import EvaluationReport, evaluate
from ds_platform.modeling.features import FeatureTable, Scalar
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
    MetricSpec,
    SequenceSpec,
    SplitSpec,
    spec_config_hash,
)
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import DerivedMeasurement, PlaySituation
from board_game_analysis.modeling.encoders.action import ActionBagEmbedder
from board_game_analysis.modeling.encoders.conditioned import LinearConditionedEncoder
from board_game_analysis.modeling.encoders.sequence import (
    PREDICTOR_FAMILY,
    SEQUENCE_FAMILY,
    LastStatePredictor,
    LinearSequencePredictor,
    PrefixSequenceEncoder,
    encode_prefix_summaries,
    encode_trajectory_summaries,
    last_step_pairs,
    target_states_for_prefixes,
)
from board_game_analysis.modeling.encoders.state import StateBagEmbedder
from board_game_analysis.modeling.examples import examples_from_situations
from board_game_analysis.modeling.logical_keys import (
    SEQUENCE_ENCODER_LOGICAL_KEY,
    SEQUENCE_EVAL_LOGICAL_KEY,
    SEQUENCE_PREDICTOR_LOGICAL_KEY,
    SEQUENCE_REPR_LOGICAL_KEY,
)
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.pairs import transition_pairs_from_situations
from board_game_analysis.modeling.sequences import (
    PrefixExample,
    prefixes_from_situations,
    sequences_from_situations,
)
from board_game_analysis.modeling.split import split_sequences_by_game

SCALAR_COLUMNS = (
    "information_volume",
    "hidden_information",
    "available_decision_count",
)

_CORPUS_NOTE = (
    "fixture corpus is short authored trajectories; this does not establish "
    "general board-game dynamics"
)


class SequenceEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    representation: EvaluationReport
    last_state: EvaluationReport
    transition: EvaluationReport
    scalars: dict[str, EvaluationReport]
    skipped: tuple[str, ...]
    n_prefixes: int
    n_trajectories: int
    train_game_ids: tuple[str, ...]
    test_game_ids: tuple[str, ...]
    sequence_mse: float
    last_state_mse: float
    transition_mse: float
    notes: tuple[str, ...]


class SequenceRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation: SequenceEvaluation
    predicted: RepresentationTable
    last_state_predicted: RepresentationTable
    transition_predicted: RepresentationTable
    targets: RepresentationTable
    summary_payload_id: str
    encoder_payload_id: str
    model_payload_id: str
    prediction_payload_id: str
    evaluation_payload_id: str


def sequence_scalar_table(
    prefixes: Sequence[PrefixExample],
    measurements: Sequence[DerivedMeasurement],
    *,
    source_payload_id: str,
) -> FeatureTable:
    """Observed scalars at the target state. Missing sides stay None."""
    values: list[tuple[Scalar, ...]] = []
    for prefix in prefixes:
        values.append(
            (
                _mean_at(
                    measurements,
                    "information_volume",
                    prefix.situation_id,
                    prefix.game_id,
                    prefix.target_state_id,
                ),
                _mean_at(
                    measurements,
                    "hidden_information",
                    prefix.situation_id,
                    prefix.game_id,
                    prefix.target_state_id,
                ),
                _mean_at(
                    measurements,
                    "available_decision_count",
                    prefix.situation_id,
                    prefix.game_id,
                    prefix.target_state_id,
                ),
            )
        )
    return FeatureTable(
        entity_ids=tuple(prefix.entity_id for prefix in prefixes),
        columns=SCALAR_COLUMNS,
        values=tuple(values),
        source_payload_ids=tuple(source_payload_id for _ in prefixes),
    )


def evaluate_sequences(
    predicted: RepresentationTable,
    last_state_predicted: RepresentationTable,
    transition_predicted: RepresentationTable,
    targets: RepresentationTable,
    *,
    scalar_true: FeatureTable | None = None,
    scalar_pred: FeatureTable | None = None,
    skipped: Sequence[str] = (),
    n_trajectories: int = 0,
    train_game_ids: Sequence[str] = (),
    test_game_ids: Sequence[str] = (),
) -> SequenceEvaluation:
    sequence_mse = representation_mse(targets, predicted)
    last_mse = representation_mse(targets, last_state_predicted)
    transition_mse = representation_mse(targets, transition_predicted)
    notes = (
        "last-state and Phase 4 transition baselines compared against "
        "the prefix sequence model",
        "targets are encoded observed to-states of prefix steps only",
        _CORPUS_NOTE,
    )
    representation = EvaluationReport(
        metrics={
            "representation_mse": sequence_mse,
            "last_state_mse": last_mse,
            "transition_mse": transition_mse,
        },
        n=len(targets.entity_ids),
        n_missing=0,
        notes=notes,
    )
    last_report = EvaluationReport(
        metrics={"representation_mse": last_mse},
        n=len(targets.entity_ids),
        n_missing=0,
        notes=("last-state baseline", _CORPUS_NOTE),
    )
    transition_report = EvaluationReport(
        metrics={"representation_mse": transition_mse},
        n=len(transition_predicted.entity_ids),
        n_missing=0,
        notes=("Phase 4 immediate-transition baseline", _CORPUS_NOTE),
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
    return SequenceEvaluation(
        representation=representation,
        last_state=last_report,
        transition=transition_report,
        scalars=scalars,
        skipped=tuple(skipped),
        n_prefixes=len(targets.entity_ids),
        n_trajectories=n_trajectories,
        train_game_ids=tuple(train_game_ids),
        test_game_ids=tuple(test_game_ids),
        sequence_mse=sequence_mse,
        last_state_mse=last_mse,
        transition_mse=transition_mse,
        notes=notes,
    )


def run_sequence_experiment(
    situations: Sequence[PlaySituation],
    *,
    store: Store,
    run: RunContext,
    split: SplitSpec,
    dim: int = 16,
    ridge: float = 1e-4,
    created_at: datetime | None = None,
) -> SequenceRun:
    """Game-held-out sequence prediction versus last-state and Phase 4."""
    bundle = sequences_from_situations(situations)
    prefixes = prefixes_from_situations(situations)
    skipped = list(bundle.skipped)
    actionable = [
        prefix
        for prefix in prefixes
        if prefix.action_id_groups and prefix.action_id_groups[-1]
    ]
    skipped.extend(
        f"{prefix.entity_id}: no observed action"
        for prefix in prefixes
        if prefix not in actionable
    )
    if not bundle.sequences or not actionable:
        raise ValueError("no actionable sequence prefixes in the supplied situations")

    examples = examples_from_situations(situations)
    state_encoder = StateBagEmbedder(dim=dim)
    state_ids = [example.entity_id for example in examples.states]
    state_records = [example.to_encoder_record() for example in examples.states]
    state_encoder.fit(state_ids, state_records)
    z_states = state_encoder.encode(state_ids, state_records)

    pair_bundle = transition_pairs_from_situations(situations)
    action_encoder = ActionBagEmbedder(dim=dim)
    action_ids = [example.entity_id for example in pair_bundle.actions]
    action_records = [example.to_encoder_record() for example in pair_bundle.actions]
    action_encoder.fit(action_ids, action_records)
    z_actions = action_encoder.encode(action_ids, action_records)

    summaries = encode_trajectory_summaries(bundle.sequences, z_states, z_actions)
    prefix_reprs = encode_prefix_summaries(actionable, z_states, z_actions)
    targets = target_states_for_prefixes(z_states, actionable)

    assignment = split_sequences_by_game(
        [prefix.entity_id for prefix in actionable],
        [prefix.game_id for prefix in actionable],
        split,
    )
    train_prefixes = _select_prefixes(actionable, assignment.train_ids)
    test_prefixes = _select_prefixes(actionable, assignment.test_ids)
    if not train_prefixes or not test_prefixes:
        raise ValueError("game split produced an empty sequence partition")
    train_ids = [prefix.entity_id for prefix in train_prefixes]
    test_ids = [prefix.entity_id for prefix in test_prefixes]
    train_games = tuple(sorted({prefix.game_id for prefix in train_prefixes}))
    test_games = tuple(sorted({prefix.game_id for prefix in test_prefixes}))

    train_x = select_entities(prefix_reprs, train_ids)
    test_x = select_entities(prefix_reprs, test_ids)
    train_y = select_entities(targets, train_ids)
    test_y = select_entities(targets, test_ids)

    predictor = LinearSequencePredictor(ridge=ridge)
    predictor.set_targets(train_y)
    predictor.fit(train_x, train_ids)
    predicted = predictor.encode(test_x, test_ids)

    last_predicted = LastStatePredictor().encode(test_prefixes, z_states)
    transition_predicted = _phase4_baseline(
        train_prefixes, test_prefixes, z_states, z_actions, ridge
    )

    measurements = measure_situations(situations)
    scalar_true = sequence_scalar_table(
        test_prefixes, measurements, source_payload_id=test_y.source_payload_ids[0]
    )
    evaluation = evaluate_sequences(
        predicted,
        last_predicted,
        transition_predicted,
        test_y,
        scalar_true=scalar_true,
        scalar_pred=_zero_change_table(scalar_true),
        skipped=skipped,
        n_trajectories=len(bundle.sequences),
        train_game_ids=train_games,
        test_game_ids=test_games,
    )

    spec = SequenceSpec(
        family=PREDICTOR_FAMILY,
        params={"ridge": ridge, "dim": dim, "encoder": SEQUENCE_FAMILY},
        seed=split.seed,
    )
    run_with_hash = run.model_copy(update={"config_hash": spec_config_hash(spec)})
    summary_pid, _summary_rid = put_feature_dataset(
        store,
        representation_payload_bytes(summaries),
        run=run_with_hash,
        inputs=[],
        media_type="application/json",
        logical_key=SEQUENCE_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    encoder_pid, _encoder_rid = put_model_artifact(
        store,
        pickle.dumps(PrefixSequenceEncoder()),
        run=run_with_hash,
        inputs=[summary_pid],
        media_type="application/octet-stream",
        logical_key=SEQUENCE_ENCODER_LOGICAL_KEY,
        created_at=created_at,
    )
    model_pid, _model_rid = put_model_artifact(
        store,
        pickle.dumps(predictor),
        run=run_with_hash,
        inputs=[encoder_pid, summary_pid],
        media_type="application/octet-stream",
        logical_key=SEQUENCE_PREDICTOR_LOGICAL_KEY,
        created_at=created_at,
    )
    pred_pid, _pred_rid = put_prediction_artifact(
        store,
        representation_payload_bytes(predicted),
        run=run_with_hash,
        inputs=[model_pid, encoder_pid],
        media_type="application/json",
        created_at=created_at,
    )
    eval_pid, _eval_rid = put_evaluation_artifact(
        store,
        evaluation.representation,
        run=run_with_hash,
        subject_payload_id=pred_pid,
        inputs=[pred_pid, model_pid],
        logical_key=SEQUENCE_EVAL_LOGICAL_KEY,
        created_at=created_at,
    )
    return SequenceRun(
        evaluation=evaluation,
        predicted=predicted,
        last_state_predicted=last_predicted,
        transition_predicted=transition_predicted,
        targets=test_y,
        summary_payload_id=summary_pid,
        encoder_payload_id=encoder_pid,
        model_payload_id=model_pid,
        prediction_payload_id=pred_pid,
        evaluation_payload_id=eval_pid,
    )


def _phase4_baseline(
    train_prefixes: Sequence[PrefixExample],
    test_prefixes: Sequence[PrefixExample],
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
    ridge: float,
) -> RepresentationTable:
    train_ctx, train_cond, train_ids = _aligned_last_step(
        train_prefixes, z_states, z_actions
    )
    test_ctx, test_cond, test_ids = _aligned_last_step(
        test_prefixes, z_states, z_actions
    )
    train_targets = target_states_for_prefixes(z_states, train_prefixes)
    model = LinearConditionedEncoder(ridge=ridge)
    model.set_targets(train_targets)
    model.fit(train_ctx, train_cond, train_ids)
    return model.encode(test_ctx, test_cond, test_ids)


def _aligned_last_step(
    prefixes: Sequence[PrefixExample],
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
) -> tuple[RepresentationTable, RepresentationTable, list[str]]:
    from_ids, action_ids, pair_ids = last_step_pairs(prefixes)
    contexts = select_entities(z_states, from_ids)
    conditions = select_entities(z_actions, action_ids)
    rekeyed_ctx = RepresentationTable(
        entity_ids=pair_ids,
        vectors=contexts.vectors,
        dim=contexts.dim,
        source_payload_ids=contexts.source_payload_ids,
    )
    rekeyed_cond = RepresentationTable(
        entity_ids=pair_ids,
        vectors=conditions.vectors,
        dim=conditions.dim,
        source_payload_ids=conditions.source_payload_ids,
    )
    return rekeyed_ctx, rekeyed_cond, list(pair_ids)


def _select_prefixes(
    prefixes: Sequence[PrefixExample], entity_ids: Sequence[str]
) -> tuple[PrefixExample, ...]:
    wanted = set(entity_ids)
    return tuple(prefix for prefix in prefixes if prefix.entity_id in wanted)


def _mean_at(
    measurements: Sequence[DerivedMeasurement],
    name: str,
    situation_id: str,
    game_id: str,
    state_id: str,
) -> float | None:
    values = [
        float(item.value)
        for item in measurements
        if item.name == name
        and item.id.startswith(f"{situation_id}:{name}:")
        and item.game_id == game_id
        and item.state_id == state_id
        and isinstance(item.value, (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


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
