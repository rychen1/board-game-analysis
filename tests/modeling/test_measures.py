"""Ontology-v0 measurement engine on the ten play fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.hashing import payload_id
from ds_platform.modeling.spec import spec_config_hash

from board_game_analysis.analysis.spec import spec_by_id
from board_game_analysis.domain import DerivedMeasurement
from board_game_analysis.modeling.artifacts import (
    MEASUREMENTS_LOGICAL_KEY,
    measurements_payload_bytes,
    scalar_measurement_table,
    store_measurements,
)
from board_game_analysis.modeling.measures import (
    MEASURE_METHOD,
    MEASURE_VERSION,
    SUPPORTED_METRIC_IDS,
    MeasureSpec,
    default_measure_spec,
    measure_situation,
    measure_situations,
    pairwise_information_asymmetry,
)
from tests.fixtures.games import all_situations
from tests.fixtures.games.chess import WHITE as CHESS_WHITE
from tests.fixtures.games.chess import make_chess_situation
from tests.fixtures.games.codenames import SPYMASTER, make_codenames_situation
from tests.fixtures.games.hanabi import ALICE as HANABI_ALICE
from tests.fixtures.games.hanabi import S0 as HANABI_S0
from tests.fixtures.games.hanabi import make_hanabi_situation
from tests.fixtures.games.just_one import make_just_one_situation
from tests.fixtures.games.pandemic import ALICE as P_ALICE
from tests.fixtures.games.pandemic import make_pandemic_situation


def _by_name(
    measurements: list[DerivedMeasurement], name: str
) -> list[DerivedMeasurement]:
    return [item for item in measurements if item.name == name]


def test_supported_metrics_match_computable_catalog() -> None:
    for metric_id in SUPPORTED_METRIC_IDS:
        assert spec_by_id(metric_id).computable_from_ontology_v0 is True
    assert "information_uncertainty" not in SUPPORTED_METRIC_IDS
    assert "legal_decision_count" not in SUPPORTED_METRIC_IDS


def test_every_fixture_emits_supported_metrics() -> None:
    for situation in all_situations():
        measurements = measure_situation(situation)
        names = {item.name for item in measurements}
        assert "information_volume" in names
        assert "legal_action_type_count" in names
        assert "structural_interaction" in names
        for item in measurements:
            assert item.method == MEASURE_METHOD
            assert item.version == MEASURE_VERSION
            assert item.value is not None


def test_hanabi_self_knowledge_asymmetry_is_inverted() -> None:
    measurements = measure_situation(make_hanabi_situation())
    self_knowledge = _by_name(measurements, "self_knowledge_asymmetry")
    assert self_knowledge
    alice_s0 = next(
        item
        for item in self_knowledge
        if item.player_id == HANABI_ALICE and item.state_id == HANABI_S0
    )
    assert isinstance(alice_s0.value, float)
    assert alice_s0.value > 0
    pairs = pairwise_information_asymmetry(make_hanabi_situation())
    s0_pairs = [item for item in pairs if item.state_id == HANABI_S0]
    assert len(s0_pairs) == 3
    assert all(item.value > 0 for item in s0_pairs)


def test_pandemic_constraint_counts_legal_unavailable_charter() -> None:
    measurements = measure_situation(make_pandemic_situation())
    constraints = _by_name(measurements, "decision_constraint")
    alice = next(item for item in constraints if item.player_id == P_ALICE)
    assert alice.value == 1.0
    available = _by_name(measurements, "available_decision_count")
    assert available[0].value == 1.0


def test_chess_and_codenames_skip_incomplete_legal_counts() -> None:
    chess = measure_situation(make_chess_situation())
    chess_available = _by_name(chess, "available_decision_count")
    assert all(item.player_id != CHESS_WHITE for item in chess_available)
    assert _by_name(chess, "decision_branching_factor") == []

    codenames = measure_situation(make_codenames_situation())
    spy_available = [
        item
        for item in _by_name(codenames, "available_decision_count")
        if item.player_id == SPYMASTER
    ]
    assert spy_available == []
    spy_branching = [
        item
        for item in _by_name(codenames, "decision_branching_factor")
        if item.player_id == SPYMASTER
    ]
    assert spy_branching == []


def test_chess_same_state_asymmetry_is_zero() -> None:
    pairs = pairwise_information_asymmetry(make_chess_situation())
    assert pairs
    assert all(item.value == 0.0 for item in pairs)


def test_measurements_are_deterministic() -> None:
    situation = make_hanabi_situation()
    first = measurements_payload_bytes(measure_situation(situation))
    second = measurements_payload_bytes(measure_situation(situation))
    assert first == second
    assert payload_id(first) == payload_id(second)


def test_no_silent_imputation_for_missing_paired_observations() -> None:
    measurements = measure_situation(make_just_one_situation())
    gains = _by_name(measurements, "information_gain")
    observers = {item.player_id for item in gains}
    assert "justone-guesser" in observers
    assert "justone-giver-a" not in observers


def test_unsupported_metric_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        measure_situation(
            make_chess_situation(),
            MeasureSpec(metric_ids=("information_uncertainty",)),
        )


def test_scalar_table_excludes_dict_valued_metrics() -> None:
    measurements = measure_situations(all_situations())
    table = scalar_measurement_table(measurements, source_payload_id="a" * 64)
    assert "information_ownership" not in table.columns
    assert all(
        row[0] not in {"information_ownership", "structural_interaction"}
        for row in table.values
    )
    names = {row[0] for row in table.values}
    assert "information_volume" in names


def test_store_measurements_sets_spec_hash(tmp_path) -> None:
    store = LocalStore(tmp_path)
    spec = default_measure_spec()
    run = RunContext(
        run_id="run-measures",
        project="bga",
        started_at=datetime(2026, 9, 19, 18, 30, tzinfo=UTC),
        environment=Environment.LOCAL,
    )
    measurements = measure_situations(all_situations(), spec)
    payload = measurements_payload_bytes(measurements)
    pid, record_id = store_measurements(store, measurements, run=run, spec=spec)
    assert pid == payload_id(payload)
    record = json.loads(store.get(record_id).decode("utf-8"))
    assert record["kind"] == ArtifactKind.DATASET
    assert record["logical_key"] == MEASUREMENTS_LOGICAL_KEY
    assert record["produced_by"]["config_hash"] == spec_config_hash(spec)
    restored = [
        DerivedMeasurement.model_validate(json.loads(line))
        for line in payload.decode("utf-8").splitlines()
        if line
    ]
    assert [item.id for item in restored] == sorted(item.id for item in measurements)
