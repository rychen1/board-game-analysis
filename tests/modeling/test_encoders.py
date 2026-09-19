"""State and observation bag embedders, game splits, and probes."""

from __future__ import annotations

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.encode import run_encoding
from ds_platform.modeling.representations import as_feature_table
from ds_platform.modeling.spec import SplitSpec

from board_game_analysis.modeling.artifacts import labels_for_observations
from board_game_analysis.modeling.encoders import (
    ObservationBagEmbedder,
    StateBagEmbedder,
)
from board_game_analysis.modeling.examples import examples_from_situations
from board_game_analysis.modeling.measures import measure_situations
from board_game_analysis.modeling.probes import probe_column
from board_game_analysis.modeling.run_encode import (
    OBS_MODEL_LOGICAL_KEY,
    OBS_REPR_LOGICAL_KEY,
    STATE_MODEL_LOGICAL_KEY,
    STATE_REPR_LOGICAL_KEY,
    bag_encoding_spec,
    encode_observations,
    encode_states,
    store_probe_evaluation,
)
from board_game_analysis.modeling.split import split_entities_by_game
from tests.fixtures.games import all_situations
from tests.fixtures.games.chess import make_chess_situation
from tests.fixtures.games.hanabi import make_hanabi_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-encode",
        project="bga",
        started_at=datetime(2026, 9, 19, 18, 40, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def test_bag_embedder_goes_through_run_encoding(tmp_path) -> None:
    bundle = examples_from_situations(all_situations())
    encoder = ObservationBagEmbedder(dim=16)
    result = run_encoding(
        bag_encoding_spec(dim=16),
        encoder=encoder,
        entity_ids=[example.entity_id for example in bundle.observations],
        records=[example.to_encoder_record() for example in bundle.observations],
        store=LocalStore(tmp_path),
        run=_run(),
        serialize_model=pickle.dumps,
    )
    assert result.representation_payload_id
    assert result.model_payload_id
    reloaded = pickle.loads(LocalStore(tmp_path).get(result.model_payload_id))
    assert isinstance(reloaded, ObservationBagEmbedder)
    assert reloaded.dim == 16


def test_encode_all_fixtures_and_persist_logical_keys(tmp_path) -> None:
    store = LocalStore(tmp_path)
    bundle = examples_from_situations(all_situations())
    obs = encode_observations(bundle.observations, store=store, run=_run())
    states = encode_states(bundle.states, store=store, run=_run())
    assert obs.table.dim == 16
    assert len(obs.table.entity_ids) == len(bundle.observations)
    assert len(states.table.entity_ids) == len(bundle.states)
    assert len(set(obs.table.vectors)) > 1
    found_keys = _logical_keys(tmp_path)
    assert OBS_REPR_LOGICAL_KEY in found_keys
    assert OBS_MODEL_LOGICAL_KEY in found_keys
    assert STATE_REPR_LOGICAL_KEY in found_keys
    assert STATE_MODEL_LOGICAL_KEY in found_keys


def test_encoder_records_exclude_title_and_values() -> None:
    bundle = examples_from_situations([make_hanabi_situation(), make_chess_situation()])
    for example in bundle.observations:
        record = example.to_encoder_record()
        dumped = json.dumps(record)
        assert "Hanabi" not in dumped
        assert "Chess" not in dumped
        assert "title" not in record
        assert "mechanics" not in record
        items = record["items"]
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            assert "payload" not in item
    for example in bundle.states:
        record = example.to_encoder_record()
        assert "title" not in record
        tokens_source = StateBagEmbedder(dim=8)
        table = tokens_source.encode([example.entity_id], [record])
        assert table.dim == 8


def test_game_split_does_not_leak_entities() -> None:
    bundle = examples_from_situations(all_situations())
    entity_ids = [example.entity_id for example in bundle.observations]
    game_ids = [example.game_id for example in bundle.observations]
    assignment = split_entities_by_game(
        entity_ids,
        game_ids,
        SplitSpec(method="holdout", seed=0, test_size=0.3),
    )
    by_entity = dict(zip(entity_ids, game_ids, strict=True))
    train_games = {by_entity[entity_id] for entity_id in assignment.train_ids}
    test_games = {by_entity[entity_id] for entity_id in assignment.test_ids}
    assert train_games.isdisjoint(test_games)
    assert assignment.train_ids
    assert assignment.test_ids


def test_structural_probes_beat_mean_baseline_on_two_labels(tmp_path) -> None:
    store = LocalStore(tmp_path)
    situations = all_situations()
    bundle = examples_from_situations(situations)
    measurements = measure_situations(situations)
    encoded = encode_observations(bundle.observations, store=store, run=_run())
    features = as_feature_table(encoded.table)
    labels = labels_for_observations(
        measurements,
        bundle.observations,
        source_payload_ids=encoded.table.source_payload_ids,
    )
    assignment = split_entities_by_game(
        features.entity_ids,
        [example.game_id for example in bundle.observations],
        SplitSpec(method="holdout", seed=0, test_size=0.3),
    )
    wins = 0
    last_report = None
    probe_columns = (
        "information_volume",
        "hidden_information",
        "available_decision_count",
    )
    for column in probe_columns:
        report, baseline = probe_column(features, labels, column, assignment)
        last_report = report
        if report.metrics["rmse"] < baseline:
            wins += 1
    assert last_report is not None
    store_probe_evaluation(
        store,
        last_report,
        run=_run(),
        subject_payload_id=encoded.result.representation_payload_id,
        inputs=[encoded.result.representation_payload_id],
    )
    assert wins >= 2


def _logical_keys(root: Path) -> set[str]:
    keys: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if not raw.startswith(b"{"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        logical_key = payload.get("logical_key") if isinstance(payload, dict) else None
        if isinstance(logical_key, str):
            keys.append(logical_key)
    return set(keys)
