"""Same-state perspective tables versus ontology asymmetry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.modeling.perspectives import perspectives_for_subject

from board_game_analysis.modeling.examples import examples_from_situation
from board_game_analysis.modeling.measures import pairwise_information_asymmetry
from board_game_analysis.modeling.perspectives import (
    PERSPECTIVES_LOGICAL_KEY,
    aligned_distance_pairs,
    mean_pair_distance,
    perspective_distances,
    perspective_table_from_observations,
    store_perspective_distances,
    store_perspectives,
)
from board_game_analysis.modeling.run_encode import encode_observations
from tests.fixtures.games.chess import S0 as CHESS_S0
from tests.fixtures.games.chess import make_chess_situation
from tests.fixtures.games.hanabi import S0 as HANABI_S0
from tests.fixtures.games.hanabi import make_hanabi_situation
from tests.fixtures.games.just_one import make_just_one_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-perspectives",
        project="bga",
        started_at=datetime(2026, 9, 19, 18, 45, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def _table(store: LocalStore, situation):
    bundle = examples_from_situation(situation)
    encoded = encode_observations(bundle.observations, store=store, run=_run())
    return (
        perspective_table_from_observations(bundle.observations, encoded.table),
        encoded,
    )


def test_hanabi_s0_has_three_perspectives(tmp_path) -> None:
    table, _encoded = _table(LocalStore(tmp_path), make_hanabi_situation())
    s0 = perspectives_for_subject(table, HANABI_S0)
    assert len(s0.entity_ids) == 3


def test_chess_same_state_distance_is_near_zero(tmp_path) -> None:
    table, _encoded = _table(LocalStore(tmp_path), make_chess_situation())
    distances = perspective_distances(table, metric="cosine")
    assert mean_pair_distance(distances, CHESS_S0) == pytest.approx(0.0)


def test_hanabi_ranks_above_chess_and_tracks_asymmetry(tmp_path) -> None:
    store = LocalStore(tmp_path)
    hanabi = make_hanabi_situation()
    chess = make_chess_situation()
    hanabi_table, hanabi_enc = _table(store, hanabi)
    chess_table, _chess_enc = _table(store, chess)
    hanabi_dist = perspective_distances(hanabi_table)
    chess_dist = perspective_distances(chess_table)
    assert mean_pair_distance(hanabi_dist, HANABI_S0) > mean_pair_distance(
        chess_dist, CHESS_S0
    )
    pairs = aligned_distance_pairs(hanabi_dist, pairwise_information_asymmetry(hanabi))
    assert pairs
    assert all(distance > 0 for distance, _asymmetry in pairs)
    store_perspectives(
        store,
        hanabi_table,
        run=_run(),
        inputs=[hanabi_enc.result.representation_payload_id],
    )
    pid, record_id = store_perspective_distances(
        store,
        hanabi_dist,
        run=_run(),
        inputs=[hanabi_enc.result.representation_payload_id],
    )
    record = json.loads(store.get(record_id).decode("utf-8"))
    assert record["kind"] == ArtifactKind.DATASET
    assert record["logical_key"] in {
        PERSPECTIVES_LOGICAL_KEY,
        "bga:repr/perspective-distances:v0",
    }
    assert pid


def test_missing_observer_raises_and_unknown_to_self_does_not_crash(tmp_path) -> None:
    table, _encoded = _table(LocalStore(tmp_path), make_hanabi_situation())
    with pytest.raises(KeyError):
        perspectives_for_subject(table, "missing-state")
    just_one = make_just_one_situation()
    just_table, _encoded = _table(LocalStore(tmp_path), just_one)
    assert just_table.dim > 0
    assert len(just_table.subject_ids) == len(just_one.observations)
