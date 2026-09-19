"""Phase 4: action encoding, pair examples, and conditioned transitions."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.modeling.pairs import align_pairs
from ds_platform.modeling.spec import SplitSpec

from board_game_analysis.modeling.encoders import (
    ActionBagEmbedder,
    CopyStateEncoder,
    LinearConditionedEncoder,
    StateBagEmbedder,
)
from board_game_analysis.modeling.examples import (
    examples_from_situation,
    examples_from_situations,
)
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.pairs import (
    ActionExample,
    pair_table_from_examples,
    transition_pairs_from_situation,
    transition_pairs_from_situations,
)
from board_game_analysis.modeling.run_encode import (
    ACTION_MODEL_LOGICAL_KEY,
    ACTION_REPR_LOGICAL_KEY,
    encode_actions,
)
from board_game_analysis.modeling.split import split_transitions_by_game
from board_game_analysis.modeling.transitions import (
    TRANSITION_EVAL_LOGICAL_KEY,
    TRANSITION_MODEL_LOGICAL_KEY,
    TRANSITION_REPR_LOGICAL_KEY,
    evaluate_transitions,
    run_transition_experiment,
    target_states_for_pairs,
    transition_scalar_table,
)
from tests.fixtures.games import all_situations
from tests.fixtures.games.just_one import make_just_one_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-transitions",
        project="bga",
        started_at=datetime(2026, 9, 19, 19, 10, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def test_all_fixture_transitions_convert_without_invention() -> None:
    situations = all_situations()
    bundle = transition_pairs_from_situations(situations)
    authored = sum(len(item.transitions) for item in situations)
    assert len(bundle.pairs) + len(bundle.skipped) == authored
    assert len(bundle.pairs) == authored
    assert bundle.skipped == ()
    for pair in bundle.pairs:
        situation = next(item for item in situations if item.game.id == pair.game_id)
        situation.state(pair.from_state_id)
        situation.state(pair.to_state_id)
        assert set(pair.action_ids) <= {action.id for action in situation.actions}


def test_just_one_keeps_both_action_ids() -> None:
    bundle = transition_pairs_from_situation(make_just_one_situation())
    assert len(bundle.pairs) == 1
    assert bundle.pairs[0].action_ids == (
        "justone-clue-star-a",
        "justone-clue-star-b",
    )
    assert bundle.actions[0].action_ids == bundle.pairs[0].action_ids
    assert bundle.actions[0].n_actions == 2
    assert "justone-clue-star-a" in bundle.actions[0].entity_id
    assert "justone-clue-star-b" in bundle.actions[0].entity_id


def test_action_encoding_is_deterministic_and_structural() -> None:
    first = ActionExample(
        entity_id="g/actions/a",
        game_id="g",
        pair_entity_id="g/pair/t",
        action_ids=("a",),
        action_types=("give_clue",),
        parameter_keys=("target", "rank"),
        n_actions=1,
    )
    second = first.model_copy(update={"entity_id": "g/actions/b", "action_ids": ("b",)})
    encoder = ActionBagEmbedder(dim=16)
    left = encoder.encode([first.entity_id], [first.to_encoder_record()])
    right = encoder.encode([second.entity_id], [second.to_encoder_record()])
    again = encoder.encode([first.entity_id], [first.to_encoder_record()])
    assert left.vectors == again.vectors
    assert left.vectors == right.vectors
    record = first.to_encoder_record()
    dumped = json.dumps(record)
    assert "title" not in record
    assert "mechanics" not in record
    assert "Hanabi" not in dumped
    assert set(record) == {"action_types", "parameter_keys", "n_actions"}


def test_action_encoder_records_omit_payload_values() -> None:
    bundle = transition_pairs_from_situation(make_just_one_situation())
    record = bundle.actions[0].to_encoder_record()
    blob = json.dumps(record)
    assert "STAR" not in blob
    assert "TELESCOPE" not in blob
    assert "Just One" not in blob


def test_empty_action_ids_are_skipped_not_fabricated() -> None:
    situation = make_just_one_situation()
    empty_transition = situation.transitions[0].model_copy(
        update={"action_id": None, "action_ids": []}
    )
    stripped = situation.model_copy(update={"transitions": [empty_transition]})
    bundle = transition_pairs_from_situation(stripped)
    assert bundle.pairs == ()
    assert bundle.actions == ()
    assert any(item.endswith("no action_ids") for item in bundle.skipped)


def test_missing_to_state_raises(tmp_path) -> None:
    del tmp_path
    situation = make_just_one_situation()
    examples = examples_from_situation(situation)
    encoder = StateBagEmbedder(dim=8)
    ids = [example.entity_id for example in examples.states]
    records = [example.to_encoder_record() for example in examples.states]
    table = encoder.encode(ids, records)
    broken = examples.pairs[0].model_copy(update={"to_state_id": "missing-state"})
    with pytest.raises(KeyError):
        target_states_for_pairs(table, [broken])


def test_transition_split_is_game_safe() -> None:
    bundle = transition_pairs_from_situations(all_situations())
    assignment = split_transitions_by_game(
        [pair.entity_id for pair in bundle.pairs],
        [pair.game_id for pair in bundle.pairs],
        SplitSpec(method="holdout", seed=0, test_size=0.3),
    )
    by_entity = {pair.entity_id: pair.game_id for pair in bundle.pairs}
    train_games = {by_entity[entity_id] for entity_id in assignment.train_ids}
    test_games = {by_entity[entity_id] for entity_id in assignment.test_ids}
    assert train_games.isdisjoint(test_games)
    by_game: dict[str, list[str]] = {}
    for pair in bundle.pairs:
        by_game.setdefault(pair.game_id, []).append(pair.entity_id)
    for entity_ids in by_game.values():
        sides = {
            "train" if entity_id in assignment.train_ids else "test"
            for entity_id in entity_ids
            if entity_id in assignment.train_ids or entity_id in assignment.test_ids
        }
        assert len(sides) == 1


def test_conditioned_model_and_copy_baseline_are_deterministic() -> None:
    situations = all_situations()
    bundle = transition_pairs_from_situations(situations)
    examples = examples_from_situations(situations)
    states = StateBagEmbedder(dim=8)
    z_states = states.encode(
        [example.entity_id for example in examples.states],
        [example.to_encoder_record() for example in examples.states],
    )
    actions = ActionBagEmbedder(dim=8)
    z_actions = actions.encode(
        [example.entity_id for example in bundle.actions],
        [example.to_encoder_record() for example in bundle.actions],
    )
    pairs = pair_table_from_examples(bundle.pairs)
    contexts, conditions = align_pairs(z_states, z_actions, pairs)
    targets = target_states_for_pairs(z_states, bundle.pairs)
    pair_ids = [pair.entity_id for pair in bundle.pairs]
    copy = CopyStateEncoder().encode(contexts, conditions, pair_ids)
    model = LinearConditionedEncoder(ridge=1e-3)
    model.set_targets(targets)
    model.fit(contexts, conditions, pair_ids)
    predicted = model.encode(contexts, conditions, pair_ids)
    again = model.encode(contexts, conditions, pair_ids)
    assert predicted.vectors == again.vectors
    assert copy.entity_ids == tuple(pair_ids)
    assert predicted.entity_ids == tuple(pair_ids)
    report = evaluate_transitions(predicted, copy, targets)
    assert report.n_pairs == len(bundle.pairs)
    assert report.representation.n_missing == 0
    assert "copy_mse" in report.representation.metrics
    assert "representation_mse" in report.representation.metrics


def test_fit_without_targets_fails_explicitly() -> None:
    encoder = LinearConditionedEncoder()
    with pytest.raises(RuntimeError, match="set_targets"):
        encoder.fit(
            _empty_repr(),
            _empty_repr(),
            (),
        )


def test_scalar_deltas_do_not_impute() -> None:
    situations = all_situations()
    bundle = transition_pairs_from_situations(situations)
    measurements = measure_situations(situations)
    table = transition_scalar_table(
        bundle.pairs, measurements, source_payload_id="a" * 64
    )
    assert table.columns == (
        "information_gain",
        "information_loss",
        "information_volume_delta",
        "available_decision_count_delta",
    )
    just_one = next(pair for pair in bundle.pairs if pair.game_id == "just-one")
    row = table.values[table.entity_ids.index(just_one.entity_id)]
    assert row[0] is not None
    assert row[3] is None


def test_run_transition_experiment_persists_artifacts(tmp_path) -> None:
    store = LocalStore(tmp_path)
    result = run_transition_experiment(
        all_situations(),
        store=store,
        run=_run(),
        split=SplitSpec(method="holdout", seed=0, test_size=0.3),
        dim=8,
    )
    assert result.evaluation.n_pairs >= 1
    assert result.predicted.dim == 8
    assert result.copy_predicted.dim == 8
    assert store.get(result.action_payload_id)
    assert store.get(result.model_payload_id)
    assert store.get(result.prediction_payload_id)
    keys = _logical_keys(tmp_path)
    assert ACTION_REPR_LOGICAL_KEY in keys
    assert ACTION_MODEL_LOGICAL_KEY in keys
    assert TRANSITION_MODEL_LOGICAL_KEY in keys
    assert TRANSITION_REPR_LOGICAL_KEY in keys
    assert TRANSITION_EVAL_LOGICAL_KEY in keys
    encode_actions(
        transition_pairs_from_situations(all_situations()).actions,
        store=store,
        run=_run(),
        dim=8,
    )


def test_each_fixture_with_transitions_runs_phase4_pipeline() -> None:
    skipped: list[str] = []
    ran = 0
    for situation in all_situations():
        bundle = transition_pairs_from_situation(situation)
        if not bundle.pairs:
            skipped.extend(bundle.skipped or (f"{situation.game.id}: no transitions",))
            continue
        examples = examples_from_situation(situation)
        states = StateBagEmbedder(dim=8)
        z_states = states.encode(
            [example.entity_id for example in examples.states],
            [example.to_encoder_record() for example in examples.states],
        )
        actions = ActionBagEmbedder(dim=8)
        z_actions = actions.encode(
            [example.entity_id for example in bundle.actions],
            [example.to_encoder_record() for example in bundle.actions],
        )
        pairs = pair_table_from_examples(bundle.pairs)
        contexts, conditions = align_pairs(z_states, z_actions, pairs)
        targets = target_states_for_pairs(z_states, bundle.pairs)
        pair_ids = [pair.entity_id for pair in bundle.pairs]
        copy = CopyStateEncoder().encode(contexts, conditions, pair_ids)
        model = LinearConditionedEncoder(ridge=1e-3)
        model.set_targets(targets)
        model.fit(contexts, conditions, pair_ids)
        predicted = model.encode(contexts, conditions, pair_ids)
        report = evaluate_transitions(predicted, copy, targets)
        assert report.n_pairs == len(bundle.pairs)
        ran += 1
    assert ran == 10
    assert skipped == []


def test_encode_actions_writes_dataset_kind(tmp_path) -> None:
    store = LocalStore(tmp_path)
    bundle = transition_pairs_from_situations(all_situations())
    encoded = encode_actions(bundle.actions, store=store, run=_run(), dim=8)
    record_keys = _kind_by_key(tmp_path)
    assert record_keys.get(ACTION_REPR_LOGICAL_KEY) == ArtifactKind.DATASET
    assert encoded.table.dim == 8


def _empty_repr():
    from ds_platform.modeling.representations import RepresentationTable

    return RepresentationTable(
        entity_ids=(),
        vectors=(),
        dim=0,
        source_payload_ids=(),
    )


def _logical_keys(root) -> set[str]:
    keys: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or not path.read_bytes()[:1] == b"{":
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
        if not path.is_file() or not path.read_bytes()[:1] == b"{":
            continue
        try:
            payload = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("logical_key"):
            found[str(payload["logical_key"])] = str(payload.get("kind"))
    return found
