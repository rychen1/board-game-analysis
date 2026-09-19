"""Phase 5: trajectories, prefix summaries, and sequence-safe evaluation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.modeling.representations import RepresentationTable, select_entities
from ds_platform.modeling.spec import SplitSpec

from board_game_analysis.modeling.encoders import (
    ActionBagEmbedder,
    LastStatePredictor,
    LinearConditionedEncoder,
    LinearSequencePredictor,
    PrefixSequenceEncoder,
    StateBagEmbedder,
    encode_prefix_summaries,
    encode_trajectory_summaries,
    target_states_for_prefixes,
)
from board_game_analysis.modeling.encoders.sequence import (
    last_step_pairs,
    prefix_event_tables,
)
from board_game_analysis.modeling.examples import (
    examples_from_situation,
    examples_from_situations,
)
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.pairs import transition_pairs_from_situations
from board_game_analysis.modeling.run_encode import (
    SEQUENCE_MODEL_LOGICAL_KEY,
    SEQUENCE_REPR_LOGICAL_KEY,
    encode_sequences,
)
from board_game_analysis.modeling.sequence_eval import (
    SEQUENCE_ENCODER_LOGICAL_KEY,
    SEQUENCE_EVAL_LOGICAL_KEY,
    SEQUENCE_PREDICTOR_LOGICAL_KEY,
    evaluate_sequences,
    run_sequence_experiment,
    sequence_scalar_table,
)
from board_game_analysis.modeling.sequences import (
    prefixes_from_situations,
    reverse_sequence,
    sequence_from_situation,
    sequence_table_from_examples,
    sequences_from_situations,
)
from board_game_analysis.modeling.split import split_sequences_by_game
from tests.fixtures.games import all_situations
from tests.fixtures.games.just_one import make_just_one_situation
from tests.fixtures.games.telestrations import S0, S1, S2, make_telestrations_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-sequences",
        project="bga",
        started_at=datetime(2026, 9, 19, 19, 40, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def _encode_tables(situations, dim: int = 8):
    examples = examples_from_situations(situations)
    states = StateBagEmbedder(dim=dim)
    z_states = states.encode(
        [example.entity_id for example in examples.states],
        [example.to_encoder_record() for example in examples.states],
    )
    actions = ActionBagEmbedder(dim=dim)
    bundle = transition_pairs_from_situations(situations)
    z_actions = actions.encode(
        [example.entity_id for example in bundle.actions],
        [example.to_encoder_record() for example in bundle.actions],
    )
    return z_states, z_actions


def test_fixture_sequences_preserve_order_and_identity() -> None:
    situations = all_situations()
    bundle = sequences_from_situations(situations)
    authored = sum(len(item.transitions) for item in situations)
    assert sum(len(item.steps) for item in bundle.sequences) == authored
    assert bundle.skipped == ()
    for sequence in bundle.sequences:
        situation = next(
            item for item in situations if item.game.id == sequence.game_id
        )
        assert sequence.situation_id.startswith(sequence.game_id)
        assert sequence.event_ids[0] == sequence.steps[0].from_state_id
        for index, step in enumerate(sequence.steps):
            assert step.index == index
            situation.state(step.from_state_id)
            situation.state(step.to_state_id)
            assert set(step.action_ids) <= {action.id for action in situation.actions}
            if index > 0:
                assert step.from_state_id == sequence.steps[index - 1].to_state_id


def test_telestrations_keeps_two_ordered_steps() -> None:
    bundle = sequence_from_situation(make_telestrations_situation())
    assert len(bundle.sequences) == 1
    sequence = bundle.sequences[0]
    assert sequence.event_ids == (S0, S1, S2)
    assert [step.from_state_id for step in sequence.steps] == [S0, S1]
    assert [step.to_state_id for step in sequence.steps] == [S1, S2]


def test_just_one_keeps_both_action_ids() -> None:
    bundle = sequence_from_situation(make_just_one_situation())
    assert len(bundle.sequences) == 1
    assert bundle.sequences[0].steps[0].action_ids == (
        "justone-clue-star-a",
        "justone-clue-star-b",
    )


def test_phase1_sequence_example_is_unchanged_state_list() -> None:
    situation = make_telestrations_situation()
    phase1 = examples_from_situation(situation).sequences[0]
    assert phase1.steps == ()
    assert phase1.event_ids == tuple(state.id for state in situation.states)
    assert phase1.situation_id.startswith(situation.game.id)


def test_empty_transitions_are_skipped() -> None:
    situation = make_just_one_situation().model_copy(update={"transitions": []})
    bundle = sequence_from_situation(situation)
    assert bundle.sequences == ()
    assert bundle.skipped == (f"{situation.game.id}: no authored transitions",)


def test_empty_action_ids_are_kept_not_filled() -> None:
    situation = make_just_one_situation()
    empty = situation.transitions[0].model_copy(
        update={"action_id": None, "action_ids": []}
    )
    stripped = situation.model_copy(update={"transitions": [empty]})
    bundle = sequence_from_situation(stripped)
    assert len(bundle.sequences) == 1
    assert bundle.sequences[0].steps[0].action_ids == ()


def test_missing_to_state_raises() -> None:
    situation = make_just_one_situation()
    broken = situation.transitions[0].model_copy(
        update={"to_state_id": "missing-state"}
    )
    missing = situation.model_copy(update={"transitions": [broken]})
    with pytest.raises(KeyError):
        sequence_from_situation(missing)


def test_disconnected_transitions_stay_separate() -> None:
    situation = make_telestrations_situation()
    first, second = situation.transitions
    split = second.model_copy(update={"from_state_id": S0, "to_state_id": S2})
    disconnected = situation.model_copy(update={"transitions": [first, split]})
    bundle = sequence_from_situation(disconnected)
    assert len(bundle.sequences) == 2
    assert all(len(item.steps) == 1 for item in bundle.sequences)


def test_cycle_is_skipped_not_unrolled() -> None:
    situation = make_telestrations_situation()
    first, second = situation.transitions
    cycled = second.model_copy(update={"from_state_id": S1, "to_state_id": S0})
    bundle = sequence_from_situation(
        situation.model_copy(update={"transitions": [first, cycled]})
    )
    assert all("cycle" in item for item in bundle.skipped)
    assert bundle.sequences == ()


def test_sequence_table_uses_platform_rows() -> None:
    bundle = sequences_from_situations(all_situations())
    table = sequence_table_from_examples(bundle.sequences)
    assert len(table.event_ids) == sum(len(item.steps) for item in bundle.sequences)
    assert len(set(table.event_ids)) == len(table.event_ids)


def test_prefixes_exclude_target_state() -> None:
    prefixes = prefixes_from_situations([make_telestrations_situation()])
    assert len(prefixes) == 2
    second = prefixes[1]
    assert second.state_ids == (S0, S1)
    assert second.target_state_id == S2
    assert S2 not in second.state_ids


def test_sequence_encoding_is_deterministic_and_order_sensitive() -> None:
    situations = [make_telestrations_situation()]
    z_states, z_actions = _encode_tables(situations)
    sequence = sequence_from_situation(situations[0]).sequences[0]
    first = encode_trajectory_summaries([sequence], z_states, z_actions)
    again = encode_trajectory_summaries([sequence], z_states, z_actions)
    assert first.vectors == again.vectors
    reversed_seq = reverse_sequence(sequence)
    reversed_states, reversed_actions = _encode_tables(situations)
    reversed_table = encode_trajectory_summaries(
        [reversed_seq], reversed_states, reversed_actions
    )
    dim = 8
    assert first.vectors[0][:dim] == reversed_table.vectors[0][dim : 2 * dim]
    assert first.vectors[0][dim : 2 * dim] == reversed_table.vectors[0][:dim]
    assert first.vectors != reversed_table.vectors


def test_prefix_summary_does_not_use_target_state() -> None:
    situations = [make_telestrations_situation()]
    z_states, z_actions = _encode_tables(situations)
    prefixes = prefixes_from_situations(situations)
    second = prefixes[1]
    summary = encode_prefix_summaries([second], z_states, z_actions)
    event_dim = 16
    last_event = summary.vectors[0][event_dim : 2 * event_dim]
    last_state = last_event[:8]
    target = target_states_for_prefixes(z_states, [second]).vectors[0]
    from_state = z_states.vectors[
        z_states.entity_ids.index(f"telestrations/state/{S1}")
    ]
    assert last_state == from_state
    assert last_state != target


def test_missing_prefix_target_raises() -> None:
    situations = [make_just_one_situation()]
    z_states, z_actions = _encode_tables(situations)
    del z_actions
    prefix = prefixes_from_situations(situations)[0]
    broken = prefix.model_copy(update={"target_state_id": "missing-state"})
    with pytest.raises(KeyError):
        target_states_for_prefixes(z_states, [broken])


def test_missing_action_is_zero_not_fabricated() -> None:
    situation = make_just_one_situation()
    empty = situation.transitions[0].model_copy(
        update={"action_id": None, "action_ids": []}
    )
    stripped = situation.model_copy(update={"transitions": [empty]})
    sequence = sequence_from_situation(stripped).sequences[0]
    assert sequence.steps[0].action_ids == ()
    z_states, z_actions = _encode_tables([stripped])
    prefix = prefixes_from_situations([stripped])[0]
    assert prefix.action_id_groups == ((),)
    summary = encode_prefix_summaries([prefix], z_states, z_actions)
    assert summary.dim > 0
    with pytest.raises(KeyError, match="no observed action"):
        last_step_pairs([prefix])


def test_one_step_prefix_has_zero_delta() -> None:
    situations = [make_just_one_situation()]
    z_states, z_actions = _encode_tables(situations)
    prefixes = prefixes_from_situations(situations)
    table, events = prefix_event_tables(prefixes, z_states, z_actions)
    contextual = PrefixSequenceEncoder().encode(table, events)
    event_dim = events.dim
    vector = contextual.vectors[0]
    first = vector[:event_dim]
    last = vector[event_dim : 2 * event_dim]
    delta = vector[3 * event_dim : 4 * event_dim]
    assert first == last
    assert delta == tuple(0.0 for _ in range(event_dim))


def test_titles_and_payload_values_are_absent() -> None:
    sequence = sequence_from_situation(make_telestrations_situation()).sequences[0]
    blob = json.dumps(sequence.to_mapping())
    assert "Telestrations" not in blob
    assert "octopus" not in blob
    assert "squid" not in blob
    assert "mechanics" not in blob


def test_sequence_split_is_game_safe() -> None:
    prefixes = prefixes_from_situations(all_situations())
    assignment = split_sequences_by_game(
        [prefix.entity_id for prefix in prefixes],
        [prefix.game_id for prefix in prefixes],
        SplitSpec(method="holdout", seed=0, test_size=0.3),
    )
    by_entity = {prefix.entity_id: prefix for prefix in prefixes}
    train_games = {by_entity[item].game_id for item in assignment.train_ids}
    test_games = {by_entity[item].game_id for item in assignment.test_ids}
    assert train_games.isdisjoint(test_games)
    by_traj: dict[str, list[str]] = {}
    for prefix in prefixes:
        by_traj.setdefault(prefix.trajectory_id, []).append(prefix.entity_id)
    for entity_ids in by_traj.values():
        sides = {
            "train" if entity_id in assignment.train_ids else "test"
            for entity_id in entity_ids
            if entity_id in assignment.train_ids or entity_id in assignment.test_ids
        }
        assert len(sides) == 1


def test_last_state_and_sequence_model_are_deterministic() -> None:
    situations = all_situations()
    z_states, z_actions = _encode_tables(situations)
    prefixes = prefixes_from_situations(situations)
    summaries = encode_prefix_summaries(prefixes, z_states, z_actions)
    targets = target_states_for_prefixes(z_states, prefixes)
    ids = [prefix.entity_id for prefix in prefixes]
    last = LastStatePredictor().encode(prefixes, z_states)
    model = LinearSequencePredictor(ridge=1e-3)
    model.set_targets(targets)
    model.fit(summaries, ids)
    predicted = model.encode(summaries, ids)
    again = model.encode(summaries, ids)
    assert predicted.vectors == again.vectors
    assert last.entity_ids == tuple(ids)
    report = evaluate_sequences(predicted, last, last, targets)
    assert report.n_prefixes == len(prefixes)
    assert report.representation.n_missing == 0
    assert "last_state_mse" in report.representation.metrics
    assert "transition_mse" in report.representation.metrics


def test_predictor_does_not_mutate_prefixes() -> None:
    situations = [make_telestrations_situation()]
    z_states, z_actions = _encode_tables(situations)
    prefixes = prefixes_from_situations(situations)
    before = tuple(prefix.model_dump() for prefix in prefixes)
    summaries = encode_prefix_summaries(prefixes, z_states, z_actions)
    targets = target_states_for_prefixes(z_states, prefixes)
    ids = [prefix.entity_id for prefix in prefixes]
    model = LinearSequencePredictor(ridge=1e-3)
    model.set_targets(targets)
    model.fit(summaries, ids)
    model.encode(summaries, ids)
    assert tuple(prefix.model_dump() for prefix in prefixes) == before


def test_fit_without_targets_fails() -> None:
    with pytest.raises(RuntimeError, match="set_targets"):
        LinearSequencePredictor().fit(
            encode_prefix_summaries((), _empty_repr(), _empty_repr()),
            (),
        )


def test_phase4_transition_baseline_is_reusable() -> None:
    situations = all_situations()
    z_states, z_actions = _encode_tables(situations)
    prefixes = prefixes_from_situations(situations)
    from_ids, action_ids, pair_ids = last_step_pairs(prefixes)
    contexts = select_entities(z_states, from_ids)
    conditions = select_entities(z_actions, action_ids)
    ctx = RepresentationTable(
        entity_ids=pair_ids,
        vectors=contexts.vectors,
        dim=contexts.dim,
        source_payload_ids=contexts.source_payload_ids,
    )
    cond = RepresentationTable(
        entity_ids=pair_ids,
        vectors=conditions.vectors,
        dim=conditions.dim,
        source_payload_ids=conditions.source_payload_ids,
    )
    targets = target_states_for_prefixes(z_states, prefixes)
    model = LinearConditionedEncoder(ridge=1e-3)
    model.set_targets(targets)
    model.fit(ctx, cond, list(pair_ids))
    predicted = model.encode(ctx, cond, list(pair_ids))
    assert predicted.entity_ids == pair_ids


def test_scalar_deltas_do_not_impute() -> None:
    situations = all_situations()
    prefixes = prefixes_from_situations(situations)
    table = sequence_scalar_table(
        prefixes, measure_situations(situations), source_payload_id="a" * 64
    )
    assert table.columns == (
        "information_volume",
        "hidden_information",
        "available_decision_count",
    )
    just_one = next(prefix for prefix in prefixes if prefix.game_id == "just-one")
    row = table.values[table.entity_ids.index(just_one.entity_id)]
    assert row[0] is not None
    ghost = just_one.model_copy(
        update={"entity_id": "ghost/prefix", "target_state_id": "missing-state"}
    )
    missing = sequence_scalar_table(
        [ghost], measure_situations(situations), source_payload_id="b" * 64
    )
    assert missing.values[0] == (None, None, None)


def test_run_sequence_experiment_persists_artifacts(tmp_path) -> None:
    store = LocalStore(tmp_path)
    result = run_sequence_experiment(
        all_situations(),
        store=store,
        run=_run(),
        split=SplitSpec(method="holdout", seed=0, test_size=0.3),
        dim=8,
    )
    assert result.evaluation.n_prefixes >= 1
    assert result.evaluation.train_game_ids
    assert result.evaluation.test_game_ids
    assert set(result.evaluation.train_game_ids).isdisjoint(
        result.evaluation.test_game_ids
    )
    assert store.get(result.summary_payload_id)
    assert store.get(result.model_payload_id)
    keys = _logical_keys(tmp_path)
    assert SEQUENCE_REPR_LOGICAL_KEY in keys
    assert SEQUENCE_ENCODER_LOGICAL_KEY in keys
    assert SEQUENCE_PREDICTOR_LOGICAL_KEY in keys
    assert SEQUENCE_EVAL_LOGICAL_KEY in keys
    assert SEQUENCE_MODEL_LOGICAL_KEY in keys


def test_encode_sequences_writes_dataset_kind(tmp_path) -> None:
    store = LocalStore(tmp_path)
    situations = all_situations()
    z_states, z_actions = _encode_tables(situations)
    bundle = sequences_from_situations(situations)
    encoded = encode_sequences(
        bundle.sequences, z_states, z_actions, store=store, run=_run()
    )
    encoded_ids = tuple(item.entity_id for item in bundle.sequences)
    assert encoded.table.entity_ids == encoded_ids
    kinds = _kind_by_key(tmp_path)
    assert kinds.get(SEQUENCE_REPR_LOGICAL_KEY) == ArtifactKind.DATASET


def test_each_fixture_with_transitions_runs_phase5_pipeline() -> None:
    skipped: list[str] = []
    ran = 0
    for situation in all_situations():
        bundle = sequence_from_situation(situation)
        if not bundle.sequences:
            skipped.extend(bundle.skipped or (f"{situation.game.id}: no transitions",))
            continue
        z_states, z_actions = _encode_tables([situation])
        prefixes = prefixes_from_situations([situation])
        summaries = encode_prefix_summaries(prefixes, z_states, z_actions)
        targets = target_states_for_prefixes(z_states, prefixes)
        last = LastStatePredictor().encode(prefixes, z_states)
        model = LinearSequencePredictor(ridge=1e-3)
        model.set_targets(targets)
        model.fit(summaries, [prefix.entity_id for prefix in prefixes])
        predicted = model.encode(summaries, [prefix.entity_id for prefix in prefixes])
        report = evaluate_sequences(predicted, last, last, targets)
        assert report.n_prefixes == len(prefixes)
        ran += 1
    assert ran == 10
    assert skipped == []


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
