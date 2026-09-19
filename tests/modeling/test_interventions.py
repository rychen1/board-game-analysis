"""Phase 6: interventions, deltas, counterfactual transitions, rollouts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.modeling.pairs import align_pairs
from ds_platform.modeling.representations import RepresentationTable, select_entities
from ds_platform.modeling.spec import SplitSpec

from board_game_analysis.modeling.counterfactuals import (
    COUNTERFACTUAL_EVAL_LOGICAL_KEY,
    COUNTERFACTUAL_REPR_LOGICAL_KEY,
    INTERVENTION_MODEL_LOGICAL_KEY,
    bound_sequence_table,
    counterfactual_rollout,
    counterfactual_transition,
    encode_action_set,
    run_counterfactual_experiment,
)
from board_game_analysis.modeling.encoders import (
    ActionBagEmbedder,
    LinearConditionedEncoder,
    ObservationBagEmbedder,
    StateBagEmbedder,
)
from board_game_analysis.modeling.examples import (
    examples_from_situation,
    examples_from_situations,
)
from board_game_analysis.modeling.interventions import (
    apply_intervention,
    change_observer,
    hide_information,
    identity_intervention,
    observation_scalars,
    reveal_information,
    scalar_information_delta,
    substitute_action,
)
from board_game_analysis.modeling.pairs import (
    pair_table_from_examples,
    transition_pairs_from_situations,
)
from board_game_analysis.modeling.sequences import (
    sequence_from_situation,
    sequence_table_from_examples,
)
from board_game_analysis.modeling.split import split_entities_by_game
from board_game_analysis.modeling.transitions import target_states_for_pairs
from tests.fixtures.games import all_situations
from tests.fixtures.games.hanabi import ALICE, BOB, S0, make_hanabi_situation
from tests.fixtures.games.just_one import GUESSER, make_just_one_situation
from tests.fixtures.games.just_one import S1 as JUST_ONE_S1
from tests.fixtures.games.telestrations import make_telestrations_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-counterfactuals",
        project="bga",
        started_at=datetime(2026, 9, 19, 20, 10, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def _guesser_s1():
    examples = examples_from_situation(make_just_one_situation())
    return next(
        example
        for example in examples.observations
        if example.observer_id == GUESSER and example.state_id == JUST_ONE_S1
    )


def test_intervention_ids_are_deterministic() -> None:
    first = hide_information(
        game_id="just-one",
        situation_id="just-one:justone-s1",
        source_state_id=JUST_ONE_S1,
        observer_id=GUESSER,
        item_id="justone-visible-clues",
    )
    second = hide_information(
        game_id="just-one",
        situation_id="just-one:justone-s1",
        source_state_id=JUST_ONE_S1,
        observer_id=GUESSER,
        item_id="justone-visible-clues",
    )
    assert first.intervention_id == second.intervention_id
    assert first.intervention_id.startswith("bga/iv/")
    assert first.to_mapping() == second.to_mapping()


def test_hide_and_reveal_are_non_mutating() -> None:
    source = _guesser_s1()
    original = json.dumps(source.to_mapping())
    hide = hide_information(
        game_id=source.game_id,
        situation_id=f"{source.game_id}:{source.state_id}",
        source_state_id=source.state_id,
        observer_id=source.observer_id,
        item_id="justone-visible-clues",
    )
    hidden = apply_intervention(source, hide)
    assert hidden.origin == "counterfactual"
    assert "/cf/" in hidden.result_entity_id
    assert json.dumps(source.to_mapping()) == original
    assert hidden.result_observation is not None
    hidden_scalars = observation_scalars(hidden.result_observation)
    assert hidden_scalars.visibility <= observation_scalars(source).visibility
    assert hidden_scalars.hidden >= observation_scalars(source).hidden
    reveal = reveal_information(
        game_id=source.game_id,
        situation_id=f"{source.game_id}:{source.state_id}",
        source_state_id=source.state_id,
        observer_id=source.observer_id,
        item_id="justone-visible-clues",
    )
    restored = apply_intervention(hidden.result_observation, reveal)
    assert restored.origin == "counterfactual"
    assert restored.result_observation is not None
    encoder = ObservationBagEmbedder(dim=8)
    original_z = encoder.encode([source.entity_id], [source.to_encoder_record()])
    restored_z = encoder.encode(
        [restored.result_entity_id],
        [restored.result_observation.to_encoder_record()],
    )
    assert original_z.vectors == restored_z.vectors
    isolated = apply_intervention(source, hide)
    assert isolated.result_observation is not None
    with pytest.raises(TypeError, match="immutable"):
        isolated.result_observation.items[0]["content_known"] = True
    assert json.dumps(source.to_mapping()) == original


def test_identity_preserves_representation() -> None:
    source = _guesser_s1()
    intervention = identity_intervention(
        game_id=source.game_id,
        situation_id=f"{source.game_id}:{source.state_id}",
        source_state_id=source.state_id,
        observer_id=source.observer_id,
    )
    result = apply_intervention(source, intervention)
    assert result.result_observation is not None
    encoder = ObservationBagEmbedder(dim=8)
    left = encoder.encode([source.entity_id], [source.to_encoder_record()])
    right = encoder.encode(
        [result.result_entity_id], [result.result_observation.to_encoder_record()]
    )
    assert left.vectors == right.vectors
    assert result.origin == "counterfactual"


def test_missing_item_and_observer_raise() -> None:
    source = _guesser_s1()
    hide = hide_information(
        game_id=source.game_id,
        situation_id=f"{source.game_id}:{source.state_id}",
        source_state_id=source.state_id,
        observer_id=source.observer_id,
        item_id="missing-item",
    )
    with pytest.raises(KeyError, match="no information item"):
        apply_intervention(source, hide)
    switch = change_observer(
        game_id=source.game_id,
        situation_id=f"{source.game_id}:{source.state_id}",
        source_state_id=source.state_id,
        observer_id=source.observer_id,
        target_observer_id="nobody",
    )
    with pytest.raises(KeyError, match="no observation"):
        apply_intervention(source, switch, observations=(source,))


def test_change_observer_uses_existing_view() -> None:
    examples = examples_from_situation(make_hanabi_situation())
    alice = next(
        example
        for example in examples.observations
        if example.observer_id == ALICE and example.state_id == S0
    )
    bob = next(
        example
        for example in examples.observations
        if example.observer_id == BOB and example.state_id == S0
    )
    intervention = change_observer(
        game_id=alice.game_id,
        situation_id=f"{alice.game_id}:{S0}",
        source_state_id=S0,
        observer_id=ALICE,
        target_observer_id=BOB,
    )
    result = apply_intervention(alice, intervention, observations=examples.observations)
    assert result.origin == "counterfactual"
    assert result.result_observation is not None
    assert result.result_observation.observer_id == BOB
    encoder = ObservationBagEmbedder(dim=8)
    bob_z = encoder.encode([bob.entity_id], [bob.to_encoder_record()])
    result_z = encoder.encode(
        [result.result_entity_id], [result.result_observation.to_encoder_record()]
    )
    assert bob_z.vectors == result_z.vectors
    assert (
        alice.entity_id not in result.result_entity_id
        or "/cf/" in result.result_entity_id
    )


def test_scalar_delta_is_deterministic() -> None:
    source = _guesser_s1()
    hide = hide_information(
        game_id=source.game_id,
        situation_id=f"{source.game_id}:{source.state_id}",
        source_state_id=source.state_id,
        observer_id=source.observer_id,
        item_id="justone-visible-clues",
    )
    hidden = apply_intervention(source, hide)
    assert hidden.result_observation is not None
    first = scalar_information_delta(source, hidden.result_observation)
    second = scalar_information_delta(source, hidden.result_observation)
    assert first == second
    assert first.visibility <= 0.0
    identity = apply_intervention(
        source,
        identity_intervention(
            game_id=source.game_id,
            situation_id=f"{source.game_id}:{source.state_id}",
            source_state_id=source.state_id,
            observer_id=source.observer_id,
        ),
    )
    assert identity.result_observation is not None
    zero = scalar_information_delta(source, identity.result_observation)
    assert zero.volume == 0.0
    assert zero.visibility == 0.0
    assert zero.hidden == 0.0


def test_action_substitution_is_counterfactual() -> None:
    situations = all_situations()
    just_one = make_just_one_situation()
    examples = examples_from_situations(situations)
    pair = examples_from_situation(just_one).pairs[0]
    z_states, z_actions, model = _fit_transition_model(situations)
    alt_ids = (pair.action_ids[0],)
    _example, alt_table = encode_action_set(
        just_one,
        alt_ids,
        pair_entity_id=pair.entity_id,
        encoder=ActionBagEmbedder(dim=8),
    )
    merged = _merge(z_actions, alt_table)
    intervention = substitute_action(
        game_id=pair.game_id,
        situation_id=pair.situation_id,
        source_state_id=pair.from_state_id,
        factual_action_ids=pair.action_ids,
        alternative_action_ids=alt_ids,
    )
    result = counterfactual_transition(
        model, z_states, merged, pair, intervention, model_family="linear-concat-v0"
    )
    assert result.origin == "counterfactual"
    assert result.model_family == "linear-concat-v0"
    assert pair.to_entity_id not in result.counterfactual_predicted.entity_ids
    assert result.query_id != pair.entity_id
    authored = {item.entity_id for item in examples.pairs}
    assert result.query_id not in authored


def test_missing_action_is_not_fabricated() -> None:
    just_one = make_just_one_situation()
    pair = examples_from_situation(just_one).pairs[0]
    z_states, z_actions, model = _fit_transition_model(all_situations())
    intervention = substitute_action(
        game_id=pair.game_id,
        situation_id=pair.situation_id,
        source_state_id=pair.from_state_id,
        factual_action_ids=pair.action_ids,
        alternative_action_ids=("not-an-action",),
    )
    with pytest.raises(KeyError):
        encode_action_set(
            just_one,
            ("not-an-action",),
            pair_entity_id=pair.entity_id,
            encoder=ActionBagEmbedder(dim=8),
        )
    with pytest.raises(KeyError):
        counterfactual_transition(
            model,
            z_states,
            z_actions,
            pair,
            intervention,
            model_family="linear-concat-v0",
        )


def test_rollout_is_bounded_and_predicted() -> None:
    situation = make_telestrations_situation()
    situations = all_situations()
    z_states, z_actions, model = _fit_transition_model(situations)
    sequence = sequence_from_situation(situation).sequences[0]
    table = sequence_table_from_examples([sequence])
    from board_game_analysis.modeling.examples import (
        action_set_entity_id,
        state_entity_id,
    )

    start_id = state_entity_id(
        sequence.situation_id, sequence.steps[0].from_state_id
    )
    start = select_entities(z_states, [start_id])
    start = RepresentationTable(
        entity_ids=(sequence.group_id,),
        vectors=start.vectors,
        dim=start.dim,
        source_payload_ids=start.source_payload_ids,
    )
    action_ids = [
        action_set_entity_id(sequence.situation_id, step.action_ids)
        for step in sequence.steps
    ]
    selected = select_entities(z_actions, action_ids)
    conditions = RepresentationTable(
        entity_ids=tuple(step.pair_entity_id for step in sequence.steps),
        vectors=selected.vectors,
        dim=selected.dim,
        source_payload_ids=selected.source_payload_ids,
    )
    one = counterfactual_rollout(
        model, start, table, conditions, max_steps=1, origin="predicted"
    )
    two = counterfactual_rollout(
        model, start, table, conditions, max_steps=2, origin="predicted"
    )
    assert one.n_steps == 1
    assert two.n_steps == 2
    assert one.origin == "predicted"
    observed_states = {
        state_entity_id(sequence.situation_id, state_id)
        for state_id in sequence.event_ids
    }
    assert observed_states.isdisjoint(one.predicted.entity_ids)
    with pytest.raises(ValueError, match="max_steps"):
        bound_sequence_table(table, 0)


def test_counterfactual_split_is_game_safe() -> None:
    pairs = transition_pairs_from_situations(all_situations()).pairs
    assignment = split_entities_by_game(
        [pair.entity_id for pair in pairs],
        [pair.game_id for pair in pairs],
        SplitSpec(method="holdout", seed=0, test_size=0.3),
    )
    by_entity = {pair.entity_id: pair.game_id for pair in pairs}
    train = {by_entity[item] for item in assignment.train_ids}
    test = {by_entity[item] for item in assignment.test_ids}
    assert train.isdisjoint(test)


def test_run_counterfactual_experiment_persists(tmp_path) -> None:
    store = LocalStore(tmp_path)
    result = run_counterfactual_experiment(
        all_situations(),
        store=store,
        run=_run(),
        split=SplitSpec(method="holdout", seed=0, test_size=0.3),
        dim=8,
        max_rollout_steps=1,
    )
    assert result.evaluation.n_identity_preserved == result.evaluation.n_identity
    assert result.evaluation.n_hide_visibility_ok == result.evaluation.n_hide
    assert result.evaluation.n_reveal_visibility_ok == result.evaluation.n_reveal
    assert set(result.evaluation.train_game_ids).isdisjoint(
        result.evaluation.test_game_ids
    )
    keys = _logical_keys(tmp_path)
    assert COUNTERFACTUAL_REPR_LOGICAL_KEY in keys
    assert INTERVENTION_MODEL_LOGICAL_KEY in keys
    assert COUNTERFACTUAL_EVAL_LOGICAL_KEY in keys
    kinds = _kind_by_key(tmp_path)
    assert kinds.get(COUNTERFACTUAL_REPR_LOGICAL_KEY) == ArtifactKind.DATASET
    authored = {
        pair.entity_id for pair in examples_from_situations(all_situations()).pairs
    }
    for item in result.transitions:
        assert item.origin == "counterfactual"
        assert item.query_id not in authored


def _fit_transition_model(situations):
    examples = examples_from_situations(situations)
    states = StateBagEmbedder(dim=8)
    z_states = states.encode(
        [example.entity_id for example in examples.states],
        [example.to_encoder_record() for example in examples.states],
    )
    bundle = transition_pairs_from_situations(situations)
    actions = ActionBagEmbedder(dim=8)
    z_actions = actions.encode(
        [example.entity_id for example in bundle.actions],
        [example.to_encoder_record() for example in bundle.actions],
    )
    table = pair_table_from_examples(bundle.pairs)
    contexts, conditions = align_pairs(z_states, z_actions, table)
    targets = target_states_for_pairs(z_states, bundle.pairs)
    model = LinearConditionedEncoder(ridge=1e-3)
    model.set_targets(targets)
    model.fit(contexts, conditions, [pair.entity_id for pair in bundle.pairs])
    return z_states, z_actions, model


def _merge(left, right):
    ids = list(left.entity_ids)
    vectors = list(left.vectors)
    sources = list(left.source_payload_ids)
    seen = set(ids)
    for entity_id, vector, source in zip(
        right.entity_ids, right.vectors, right.source_payload_ids, strict=True
    ):
        if entity_id in seen:
            continue
        ids.append(entity_id)
        vectors.append(vector)
        sources.append(source)
    return RepresentationTable(
        entity_ids=tuple(ids),
        vectors=tuple(vectors),
        dim=left.dim,
        source_payload_ids=tuple(sources),
    )


def _logical_keys(root) -> set[str]:
    keys: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.read_bytes()[:1] != b"{":
            continue
        try:
            payload = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            continue
        key = payload.get("logical_key") if isinstance(payload, dict) else None
        if isinstance(key, str):
            keys.append(key)
    return set(keys)


def _kind_by_key(root) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.read_bytes()[:1] != b"{":
            continue
        try:
            payload = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("logical_key"):
            found[str(payload["logical_key"])] = str(payload.get("kind"))
    return found
