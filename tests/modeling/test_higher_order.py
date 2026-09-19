"""Phase 6: higher-order structural perspectives and paths."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext

from board_game_analysis.modeling.encoders import ObservationBagEmbedder
from board_game_analysis.modeling.examples import examples_from_situation
from board_game_analysis.modeling.higher_order import (
    HIGHER_ORDER_MODEL_LOGICAL_KEY,
    HIGHER_ORDER_REPR_LOGICAL_KEY,
    PERSPECTIVE_EVAL_LOGICAL_KEY,
    ObserverPath,
    OrderedConcatComposer,
    compose_observer_path,
    higher_order_pairs,
    higher_order_perspective,
    higher_order_table,
    intervene_higher_order,
    require_observers,
    run_higher_order_experiment,
)
from board_game_analysis.modeling.interventions import hide_information
from board_game_analysis.modeling.perspectives import (
    perspective_table_from_observations,
)
from tests.fixtures.games.hanabi import ALICE, BOB, CAROL, S0, S1, make_hanabi_situation
from tests.fixtures.games.just_one import S1 as JUST_ONE_S1
from tests.fixtures.games.just_one import make_just_one_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-higher-order",
        project="bga",
        started_at=datetime(2026, 9, 19, 20, 20, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def _hanabi_obs():
    examples = examples_from_situation(make_hanabi_situation())
    encoder = ObservationBagEmbedder(dim=8)
    table = encoder.encode(
        [example.entity_id for example in examples.observations],
        [example.to_encoder_record() for example in examples.observations],
    )
    return examples.observations, table, encoder


def _at(observations, observer_id, state_id):
    return next(
        example
        for example in observations
        if example.observer_id == observer_id and example.state_id == state_id
    )


def test_higher_order_is_directional() -> None:
    observations, z_obs, _encoder = _hanabi_obs()
    alice = _at(observations, ALICE, S0)
    bob = _at(observations, BOB, S0)
    index = {entity_id: i for i, entity_id in enumerate(z_obs.entity_ids)}
    ab = higher_order_perspective(
        alice,
        bob,
        z_obs.vectors[index[alice.entity_id]],
        z_obs.vectors[index[bob.entity_id]],
    )
    ba = higher_order_perspective(
        bob,
        alice,
        z_obs.vectors[index[bob.entity_id]],
        z_obs.vectors[index[alice.entity_id]],
    )
    assert ab.focal_observer_id == ALICE
    assert ab.target_observer_id == BOB
    assert ba.focal_observer_id == BOB
    assert ba.target_observer_id == ALICE
    assert ab.vector != ba.vector
    assert ab.entity_id != ba.entity_id
    assert ab.origin == "observed"
    again = higher_order_perspective(
        alice,
        bob,
        z_obs.vectors[index[alice.entity_id]],
        z_obs.vectors[index[bob.entity_id]],
    )
    assert ab.vector == again.vector


def test_missing_observer_raises() -> None:
    observations, _z_obs, _encoder = _hanabi_obs()
    with pytest.raises(KeyError, match="no observer"):
        require_observers(observations, S0, (ALICE, "nobody"))


def test_same_state_pairs_only() -> None:
    observations, z_obs, _encoder = _hanabi_obs()
    pairs = higher_order_pairs(observations, z_obs)
    assert pairs
    assert all(pair.focal_observer_id != pair.target_observer_id for pair in pairs)
    s0 = [pair for pair in pairs if pair.state_id == S0]
    observers = {ALICE, BOB, CAROL}
    assert {pair.focal_observer_id for pair in s0} <= observers
    alice_s0 = _at(observations, ALICE, S0)
    alice_s1 = _at(observations, ALICE, S1)
    index = {entity_id: i for i, entity_id in enumerate(z_obs.entity_ids)}
    with pytest.raises(ValueError, match="same state_id"):
        higher_order_perspective(
            alice_s0,
            alice_s1,
            z_obs.vectors[index[alice_s0.entity_id]],
            z_obs.vectors[index[alice_s1.entity_id]],
        )


def test_paths_preserve_order_and_reject_cycles() -> None:
    observations, z_obs, _encoder = _hanabi_obs()
    table = perspective_table_from_observations(observations, z_obs)
    composer = OrderedConcatComposer()
    forward = compose_observer_path(
        table,
        ObserverPath(state_id=S0, observers=(ALICE, BOB, CAROL)),
        composer=composer,
    )
    backward = compose_observer_path(
        table,
        ObserverPath(state_id=S0, observers=(CAROL, BOB, ALICE)),
        composer=composer,
    )
    assert forward.vectors != backward.vectors
    with pytest.raises(ValueError, match="cycle"):
        compose_observer_path(
            table,
            ObserverPath(state_id=S0, observers=(ALICE, BOB, ALICE)),
            composer=composer,
        )
    cycled = compose_observer_path(
        table,
        ObserverPath(state_id=S0, observers=(ALICE, BOB, ALICE), allow_cycles=True),
        composer=composer,
    )
    assert cycled.entity_ids
    with pytest.raises(KeyError, match="no observer"):
        compose_observer_path(
            table,
            ObserverPath(state_id=S0, observers=(ALICE, "nobody")),
            composer=composer,
        )


def test_perspective_intervention_changes_a_about_b() -> None:
    observations, z_obs, encoder = _hanabi_obs()
    alice = _at(observations, ALICE, S0)
    bob = _at(observations, BOB, S0)
    index = {entity_id: i for i, entity_id in enumerate(z_obs.entity_ids)}
    item_id = next(str(item["id"]) for item in bob.items if item.get("id"))
    intervention = hide_information(
        game_id=bob.game_id,
        situation_id=f"{bob.game_id}:{S0}",
        source_state_id=S0,
        observer_id=BOB,
        item_id=item_id,
    )
    actual, counterfactual, delta = intervene_higher_order(
        alice,
        bob,
        intervention,
        encoder,
        z_obs.vectors[index[alice.entity_id]],
        z_obs.vectors[index[bob.entity_id]],
    )
    assert actual.origin == "observed"
    assert counterfactual.origin == "counterfactual"
    assert "/cf" in counterfactual.entity_id
    assert actual.vector != counterfactual.vector
    assert any(value != 0.0 for value in delta.vectors[0])
    assert S1 not in actual.entity_id
    blob = json.dumps(actual.to_mapping())
    assert "Hanabi" not in blob


def test_future_state_is_not_used_for_current_hop() -> None:
    observations, z_obs, _encoder = _hanabi_obs()
    s0_only = [example for example in observations if example.state_id == S0]
    pairs = higher_order_pairs(s0_only, z_obs)
    assert pairs
    assert all(pair.state_id == S0 for pair in pairs)
    s1_ids = {example.entity_id for example in observations if example.state_id == S1}
    used = {example.entity_id for example in s0_only}
    assert s1_ids.isdisjoint(used)


def test_run_higher_order_experiment_persists(tmp_path) -> None:
    store = LocalStore(tmp_path)
    observations, z_obs, _encoder = _hanabi_obs()
    result = run_higher_order_experiment(observations, z_obs, store=store, run=_run())
    assert result.evaluation.n_pairs >= 2
    assert result.evaluation.n_asymmetric >= 1
    keys = _logical_keys(tmp_path)
    assert HIGHER_ORDER_REPR_LOGICAL_KEY in keys
    assert HIGHER_ORDER_MODEL_LOGICAL_KEY in keys
    assert PERSPECTIVE_EVAL_LOGICAL_KEY in keys
    kinds = _kind_by_key(tmp_path)
    assert kinds.get(HIGHER_ORDER_REPR_LOGICAL_KEY) == ArtifactKind.DATASET
    table = higher_order_table(result.pairs)
    assert table.entity_ids == result.table.entity_ids


def test_just_one_hop_uses_same_state_only() -> None:
    examples = examples_from_situation(make_just_one_situation())
    encoder = ObservationBagEmbedder(dim=8)
    z_obs = encoder.encode(
        [example.entity_id for example in examples.observations],
        [example.to_encoder_record() for example in examples.observations],
    )
    pairs = higher_order_pairs(examples.observations, z_obs)
    s1 = [pair for pair in pairs if pair.state_id == JUST_ONE_S1]
    assert s1 == []
    assert pairs
    assert all(pair.state_id != JUST_ONE_S1 for pair in pairs)


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
