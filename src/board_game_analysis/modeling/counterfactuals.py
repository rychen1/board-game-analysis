"""Model-based counterfactual transitions and bounded representation rollout."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from ds_platform import RunContext, Store
from ds_platform.modeling.capabilities import ConditionedEncoder
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.interventions import (
    InterventionQuery,
    representation_delta,
)
from ds_platform.modeling.interventions import (
    apply_intervention as apply_conditioned_query,
)
from ds_platform.modeling.interventions import (
    rollout as platform_rollout,
)
from ds_platform.modeling.pairs import align_pairs
from ds_platform.modeling.records import (
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import RepresentationTable, select_entities
from ds_platform.modeling.sequences import SequenceTable
from ds_platform.modeling.spec import ConditioningSpec, SplitSpec, spec_config_hash
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import Action, PlaySituation
from board_game_analysis.modeling.encoders.action import ActionBagEmbedder
from board_game_analysis.modeling.encoders.conditioned import (
    CONDITIONED_FAMILY,
    LinearConditionedEncoder,
)
from board_game_analysis.modeling.encoders.observation import ObservationBagEmbedder
from board_game_analysis.modeling.encoders.state import StateBagEmbedder
from board_game_analysis.modeling.examples import (
    ObservationExample,
    PairExample,
    action_set_entity_id,
    examples_from_situations,
    situation_id_for,
    state_entity_id,
)
from board_game_analysis.modeling.interventions import (
    INTERVENTION_FAMILY,
    Intervention,
    InterventionSummary,
    Origin,
    apply_intervention,
    first_item_id,
    hide_information,
    identity_intervention,
    intervention_summaries_for_observations,
    reveal_information,
    scalar_information_delta,
    substitute_action,
)
from board_game_analysis.modeling.pairs import (
    ActionExample,
    pair_table_from_examples,
    transition_pairs_from_situations,
)
from board_game_analysis.modeling.sequences import (
    sequence_from_situation,
    sequence_table_from_examples,
)
from board_game_analysis.modeling.split import split_entities_by_game
from board_game_analysis.modeling.transitions import target_states_for_pairs

COUNTERFACTUAL_REPR_LOGICAL_KEY = "bga:repr/counterfactuals:v0"
INTERVENTION_MODEL_LOGICAL_KEY = "bga:model/intervention:v0"
COUNTERFACTUAL_EVAL_LOGICAL_KEY = "bga:evaluation/counterfactual:v0"

_CORPUS_NOTE = (
    "fixture corpus is short authored moments; counterfactuals are model "
    "queries, not causal effects in real games"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CounterfactualTransition(_FrozenModel):
    query_id: str
    pair_entity_id: str
    intervention: Intervention
    origin: Origin
    model_family: str
    factual_action_ids: tuple[str, ...]
    alternative_action_ids: tuple[str, ...]
    factual_predicted: RepresentationTable
    counterfactual_predicted: RepresentationTable
    delta: RepresentationTable
    notes: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class RolloutResult(_FrozenModel):
    origin: Origin
    max_steps: int
    n_steps: int
    predicted: RepresentationTable
    group_ids: tuple[str, ...]
    notes: tuple[str, ...]


class CounterfactualEvaluation(_FrozenModel):
    representation: EvaluationReport
    n_identity: int
    n_identity_preserved: int
    n_hide: int
    n_hide_visibility_ok: int
    n_reveal: int
    n_reveal_visibility_ok: int
    n_transitions: int
    n_rollout_steps: int
    train_game_ids: tuple[str, ...]
    test_game_ids: tuple[str, ...]
    notes: tuple[str, ...]


class CounterfactualRun(_FrozenModel):
    evaluation: CounterfactualEvaluation
    transitions: tuple[CounterfactualTransition, ...]
    intervention_summaries: tuple[InterventionSummary, ...]
    rollout: RolloutResult | None
    counterfactual_payload_id: str
    model_payload_id: str
    evaluation_payload_id: str


def counterfactual_transition(
    model: ConditionedEncoder,
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
    pair: PairExample,
    intervention: Intervention,
    *,
    model_family: str,
) -> CounterfactualTransition:
    """Predict next-state reps under factual and substitute actions.

    The observed to-state is not used as an input. The result is
    counterfactual, not a fixture state.
    """
    if intervention.operation != "substitute_action":
        raise ValueError("counterfactual_transition requires substitute_action")
    if not intervention.alternative_action_ids:
        raise KeyError("alternative action_ids are required")
    if not intervention.factual_action_ids:
        raise KeyError("factual action_ids are required")
    alt_entity = action_set_entity_id(
        pair.situation_id, intervention.alternative_action_ids
    )
    fact_entity = action_set_entity_id(
        pair.situation_id, intervention.factual_action_ids
    )
    if alt_entity not in z_actions.entity_ids:
        raise KeyError(f"no encoded alternative action {alt_entity!r}")
    if fact_entity not in z_actions.entity_ids:
        raise KeyError(f"no encoded factual action {fact_entity!r}")
    if pair.from_entity_id not in z_states.entity_ids:
        raise KeyError(f"no encoded from-state {pair.from_entity_id!r}")
    suffix = intervention.intervention_id.rsplit("/", 1)[-1][:16]
    query_id = f"{pair.entity_id}/cf/{suffix}"
    query = InterventionQuery(
        query_ids=(query_id,),
        context_ids=(pair.from_entity_id,),
        factual_condition_ids=(fact_entity,),
        alternative_condition_ids=(alt_entity,),
    )
    factual, alternative = apply_conditioned_query(model, z_states, z_actions, query)
    delta = representation_delta(factual, alternative)
    return CounterfactualTransition(
        query_id=query_id,
        pair_entity_id=pair.entity_id,
        intervention=intervention,
        origin="counterfactual",
        model_family=model_family,
        factual_action_ids=intervention.factual_action_ids,
        alternative_action_ids=intervention.alternative_action_ids,
        factual_predicted=factual,
        counterfactual_predicted=alternative,
        delta=delta,
        notes=(
            "predicted under the Phase 4 conditioned model",
            "not an observed fixture transition",
            _CORPUS_NOTE,
        ),
    )


def encode_action_set(
    situation: PlaySituation,
    action_ids: Sequence[str],
    *,
    pair_entity_id: str,
    encoder: ActionBagEmbedder,
) -> tuple[ActionExample, RepresentationTable]:
    """Encode an existing action set. Missing action ids raise."""
    if not action_ids:
        raise KeyError("action_ids must be non-empty")
    by_id = {action.id: action for action in situation.actions}
    resolved: list[Action] = []
    for action_id in action_ids:
        try:
            resolved.append(by_id[action_id])
        except KeyError as exc:
            raise KeyError(f"no action {action_id}") from exc
    situation_id = situation_id_for(situation)
    example = ActionExample(
        entity_id=action_set_entity_id(situation_id, action_ids),
        game_id=situation.game.id,
        pair_entity_id=pair_entity_id,
        action_ids=tuple(action_ids),
        action_types=tuple(action.action_type for action in resolved),
        parameter_keys=tuple(
            sorted({str(key) for action in resolved for key in action.parameters})
        ),
        n_actions=len(resolved),
        player_ids=tuple(action.player_id for action in resolved),
    )
    table = encoder.encode([example.entity_id], [example.to_encoder_record()])
    return example, table


def bound_sequence_table(table: SequenceTable, max_steps: int) -> SequenceTable:
    """Keep the first ``max_steps`` events per group. ``max_steps`` is required."""
    if max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    kept = [
        index for index, position in enumerate(table.positions) if position < max_steps
    ]
    return SequenceTable(
        group_ids=tuple(table.group_ids[index] for index in kept),
        event_ids=tuple(table.event_ids[index] for index in kept),
        positions=tuple(table.positions[index] for index in kept),
        actor_ids=tuple(table.actor_ids[index] for index in kept),
        source_payload_ids=tuple(table.source_payload_ids[index] for index in kept),
    )


def counterfactual_rollout(
    model: ConditionedEncoder,
    start: RepresentationTable,
    steps: SequenceTable,
    conditions: RepresentationTable,
    *,
    max_steps: int,
    origin: Origin = "predicted",
) -> RolloutResult:
    """Open-loop representation rollout. Bounded. Not a game simulator."""
    bounded = bound_sequence_table(steps, max_steps)
    predicted = platform_rollout(model, start, bounded, conditions)
    return RolloutResult(
        origin=origin,
        max_steps=max_steps,
        n_steps=len(predicted.entity_ids),
        predicted=predicted,
        group_ids=tuple(dict.fromkeys(bounded.group_ids)),
        notes=(
            f"stopped after at most {max_steps} step(s) per group",
            "predictions are not observed fixture states",
            _CORPUS_NOTE,
        ),
    )


def run_counterfactual_experiment(
    situations: Sequence[PlaySituation],
    *,
    store: Store,
    run: RunContext,
    split: SplitSpec,
    dim: int = 16,
    ridge: float = 1e-4,
    max_rollout_steps: int = 2,
    created_at: datetime | None = None,
) -> CounterfactualRun:
    """Game-held-out sanity evaluation of interventions and queries."""
    examples = examples_from_situations(situations)
    pair_bundle = transition_pairs_from_situations(situations)
    if not pair_bundle.pairs:
        raise ValueError("no actionable transitions")

    state_encoder = StateBagEmbedder(dim=dim)
    state_ids = [example.entity_id for example in examples.states]
    state_records = [example.to_encoder_record() for example in examples.states]
    z_states = state_encoder.encode(state_ids, state_records)

    action_encoder = ActionBagEmbedder(dim=dim)
    action_ids = [example.entity_id for example in pair_bundle.actions]
    action_records = [example.to_encoder_record() for example in pair_bundle.actions]
    z_actions = action_encoder.encode(action_ids, action_records)

    obs_encoder = ObservationBagEmbedder(dim=dim)
    assignment = split_entities_by_game(
        [pair.entity_id for pair in pair_bundle.pairs],
        [pair.game_id for pair in pair_bundle.pairs],
        split,
    )
    train_pairs = [
        pair for pair in pair_bundle.pairs if pair.entity_id in assignment.train_ids
    ]
    test_pairs = [
        pair for pair in pair_bundle.pairs if pair.entity_id in assignment.test_ids
    ]
    if not train_pairs or not test_pairs:
        raise ValueError("game split produced an empty partition")
    train_games = tuple(sorted({pair.game_id for pair in train_pairs}))
    test_games = tuple(sorted({pair.game_id for pair in test_pairs}))

    pairs_table = pair_table_from_examples(train_pairs)
    contexts, conditions = align_pairs(z_states, z_actions, pairs_table)
    targets = target_states_for_pairs(z_states, train_pairs)
    model = LinearConditionedEncoder(ridge=ridge)
    model.set_targets(targets)
    model.fit(contexts, conditions, [pair.entity_id for pair in train_pairs])

    by_game = {situation.game.id: situation for situation in situations}
    transitions: list[CounterfactualTransition] = []
    extra_actions: list[RepresentationTable] = [z_actions]
    for pair in test_pairs:
        situation = by_game[pair.game_id]
        alt_ids = _alternative_action_ids(situation, pair)
        if alt_ids is None:
            continue
        _example, alt_table = encode_action_set(
            situation, alt_ids, pair_entity_id=pair.entity_id, encoder=action_encoder
        )
        extra_actions.append(alt_table)
        merged = _merge_reprs(extra_actions)
        extra_actions = [merged]
        query = substitute_action(
            game_id=pair.game_id,
            situation_id=pair.situation_id,
            source_state_id=pair.from_state_id,
            factual_action_ids=pair.action_ids,
            alternative_action_ids=alt_ids,
        )
        transitions.append(
            counterfactual_transition(
                model,
                z_states,
                merged,
                pair,
                query,
                model_family=CONDITIONED_FAMILY,
            )
        )

    n_identity = 0
    n_identity_ok = 0
    n_hide = 0
    n_hide_ok = 0
    n_reveal = 0
    n_reveal_ok = 0
    cf_vectors: list[tuple[float, ...]] = []
    cf_ids: list[str] = []
    cf_sources: list[str] = []
    test_obs = [
        example
        for example in examples.observations
        if example.game_id in set(test_games)
    ]
    intervention_summaries = intervention_summaries_for_observations(test_obs)
    for example in test_obs:
        item_id = first_item_id(example)
        if item_id is None:
            continue
        identity = identity_intervention(
            game_id=example.game_id,
            situation_id=example.situation_id,
            source_state_id=example.state_id,
            observer_id=example.observer_id,
        )
        ident_result = apply_intervention(example, identity)
        n_identity += 1
        if _same_repr(obs_encoder, example, ident_result.result_observation):
            n_identity_ok += 1
        hide = hide_information(
            game_id=example.game_id,
            situation_id=example.situation_id,
            source_state_id=example.state_id,
            observer_id=example.observer_id,
            item_id=item_id,
        )
        hidden = apply_intervention(example, hide)
        n_hide += 1
        if hidden.result_observation is None:
            raise ValueError("hide_information produced no observation")
        delta_hide = scalar_information_delta(example, hidden.result_observation)
        if delta_hide.visibility <= 0.0:
            n_hide_ok += 1
        reveal = reveal_information(
            game_id=example.game_id,
            situation_id=example.situation_id,
            source_state_id=example.state_id,
            observer_id=example.observer_id,
            item_id=item_id,
        )
        revealed = apply_intervention(example, reveal)
        n_reveal += 1
        if revealed.result_observation is None:
            raise ValueError("reveal_information produced no observation")
        delta_reveal = scalar_information_delta(example, revealed.result_observation)
        if delta_reveal.visibility >= 0.0:
            n_reveal_ok += 1
        for result in (ident_result, hidden, revealed):
            observation = result.result_observation
            if observation is None:
                continue
            encoded = obs_encoder.encode(
                [observation.entity_id], [observation.to_encoder_record()]
            )
            cf_ids.append(observation.entity_id)
            cf_vectors.append(encoded.vectors[0])
            cf_sources.append(encoded.source_payload_ids[0])

    rollout_result = _maybe_rollout(
        situations,
        model,
        z_states,
        extra_actions[0],
        max_steps=max_rollout_steps,
    )

    cf_table = RepresentationTable(
        entity_ids=tuple(cf_ids),
        vectors=tuple(cf_vectors),
        dim=dim if cf_vectors else 0,
        source_payload_ids=tuple(cf_sources),
    )
    evaluation = CounterfactualEvaluation(
        representation=EvaluationReport(
            metrics={
                "identity_preserved": (
                    n_identity_ok / n_identity if n_identity else 0.0
                ),
                "hide_visibility_ok": (n_hide_ok / n_hide if n_hide else 0.0),
                "reveal_visibility_ok": (n_reveal_ok / n_reveal if n_reveal else 0.0),
                "n_counterfactual_transitions": float(len(transitions)),
            },
            n=n_identity + n_hide + n_reveal,
            n_missing=0,
            notes=(_CORPUS_NOTE, "sanity evaluation, not causal validation"),
        ),
        n_identity=n_identity,
        n_identity_preserved=n_identity_ok,
        n_hide=n_hide,
        n_hide_visibility_ok=n_hide_ok,
        n_reveal=n_reveal,
        n_reveal_visibility_ok=n_reveal_ok,
        n_transitions=len(transitions),
        n_rollout_steps=rollout_result.n_steps if rollout_result else 0,
        train_game_ids=train_games,
        test_game_ids=test_games,
        notes=(_CORPUS_NOTE,),
    )
    spec = ConditioningSpec(
        family=INTERVENTION_FAMILY,
        params={"ridge": ridge, "dim": dim, "max_rollout_steps": max_rollout_steps},
        seed=split.seed,
    )
    run_with_hash = run.model_copy(update={"config_hash": spec_config_hash(spec)})
    cf_pid, _cf_rid = put_feature_dataset(
        store,
        representation_payload_bytes(cf_table),
        run=run_with_hash,
        inputs=[],
        media_type="application/json",
        logical_key=COUNTERFACTUAL_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    model_pid, _model_rid = put_model_artifact(
        store,
        pickle.dumps({"family": INTERVENTION_FAMILY, "transition": model}),
        run=run_with_hash,
        inputs=[cf_pid],
        media_type="application/octet-stream",
        logical_key=INTERVENTION_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )
    eval_pid, _eval_rid = put_evaluation_artifact(
        store,
        evaluation.representation,
        run=run_with_hash,
        subject_payload_id=cf_pid,
        inputs=[cf_pid, model_pid],
        logical_key=COUNTERFACTUAL_EVAL_LOGICAL_KEY,
        created_at=created_at,
    )
    return CounterfactualRun(
        evaluation=evaluation,
        intervention_summaries=intervention_summaries,
        transitions=tuple(transitions),
        rollout=rollout_result,
        counterfactual_payload_id=cf_pid,
        model_payload_id=model_pid,
        evaluation_payload_id=eval_pid,
    )


def _maybe_rollout(
    situations: Sequence[PlaySituation],
    model: ConditionedEncoder,
    z_states: RepresentationTable,
    z_actions: RepresentationTable,
    *,
    max_steps: int,
) -> RolloutResult | None:
    for situation in situations:
        bundle = sequence_from_situation(situation)
        if not bundle.sequences:
            continue
        sequence = max(bundle.sequences, key=lambda item: len(item.steps))
        if not sequence.steps:
            continue
        table = sequence_table_from_examples([sequence])
        start_id = state_entity_id(
            sequence.situation_id, sequence.steps[0].from_state_id
        )
        if start_id not in z_states.entity_ids:
            continue
        start = select_entities(z_states, [start_id])
        start = RepresentationTable(
            entity_ids=(sequence.group_id,),
            vectors=start.vectors,
            dim=start.dim,
            source_payload_ids=start.source_payload_ids,
        )
        cond_ids = [step.pair_entity_id for step in sequence.steps]
        action_ids = [
            action_set_entity_id(sequence.situation_id, step.action_ids)
            for step in sequence.steps
            if step.action_ids
        ]
        if len(action_ids) != len(cond_ids):
            continue
        if any(action_id not in z_actions.entity_ids for action_id in action_ids):
            continue
        selected = select_entities(z_actions, action_ids)
        conditions = RepresentationTable(
            entity_ids=tuple(cond_ids),
            vectors=selected.vectors,
            dim=selected.dim,
            source_payload_ids=selected.source_payload_ids,
        )
        return counterfactual_rollout(
            model,
            start,
            table,
            conditions,
            max_steps=max_steps,
            origin="predicted",
        )
    return None


def _alternative_action_ids(
    situation: PlaySituation, pair: PairExample
) -> tuple[str, ...] | None:
    if len(pair.action_ids) > 1:
        return (pair.action_ids[0],)
    others = [
        action.id
        for action in situation.actions
        if action.id not in set(pair.action_ids)
    ]
    if not others:
        return None
    return (others[0],)


def _merge_reprs(tables: Sequence[RepresentationTable]) -> RepresentationTable:
    entity_ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    dim = tables[0].dim
    seen: set[str] = set()
    for table in tables:
        if table.dim != dim:
            raise ValueError("representation dims must match")
        for entity_id, vector, source in zip(
            table.entity_ids, table.vectors, table.source_payload_ids, strict=True
        ):
            if entity_id in seen:
                continue
            seen.add(entity_id)
            entity_ids.append(entity_id)
            vectors.append(vector)
            sources.append(source)
    return RepresentationTable(
        entity_ids=tuple(entity_ids),
        vectors=tuple(vectors),
        dim=dim,
        source_payload_ids=tuple(sources),
    )


def _same_repr(
    encoder: ObservationBagEmbedder,
    left: ObservationExample | None,
    right: ObservationExample | None,
) -> bool:
    if left is None or right is None:
        return False
    first = encoder.encode([left.entity_id], [left.to_encoder_record()])
    second = encoder.encode([right.entity_id], [right.to_encoder_record()])
    return first.vectors == second.vectors
