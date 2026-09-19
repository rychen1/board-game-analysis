"""PlaySituation example records for all ten fixtures."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.hashing import payload_id

from board_game_analysis.modeling.artifacts import (
    EXAMPLES_LOGICAL_KEY,
    examples_payload_bytes,
    store_examples,
)
from board_game_analysis.modeling.examples import (
    ExampleBundle,
    ObservationExample,
    examples_from_situation,
    examples_from_situations,
)
from tests.fixtures.games import all_situations
from tests.fixtures.games.just_one import make_just_one_situation


def test_all_fixtures_produce_typed_example_records() -> None:
    situations = all_situations()
    assert len(situations) == 10
    bundle = examples_from_situations(situations)
    assert len(bundle.states) == sum(len(item.states) for item in situations)
    assert len(bundle.observations) == sum(
        len(item.observations) for item in situations
    )
    assert len(bundle.pairs) == sum(len(item.transitions) for item in situations)
    assert len(bundle.sequences) == 10
    assert len({example.entity_id for example in bundle.states}) == len(bundle.states)
    assert len({example.entity_id for example in bundle.observations}) == len(
        bundle.observations
    )


def test_just_one_pair_keeps_both_simultaneous_action_ids() -> None:
    bundle = examples_from_situation(make_just_one_situation())
    assert len(bundle.pairs) == 1
    assert set(bundle.pairs[0].action_ids) == {
        "justone-clue-star-a",
        "justone-clue-star-b",
    }


def test_state_data_is_not_flattened() -> None:
    bundle = examples_from_situation(make_just_one_situation())
    assert isinstance(bundle.states[0].data, dict)
    assert "target" in bundle.states[0].data


def test_encoder_records_omit_title_and_payload_values() -> None:
    bundle = examples_from_situation(make_just_one_situation())
    state_record = bundle.states[0].to_encoder_record()
    assert "title" not in state_record
    assert "mechanics" not in state_record
    assert "game_id" not in state_record
    obs_record = bundle.observations[0].to_encoder_record()
    assert set(obs_record) == {"state_id", "observer_id", "items"}
    items = obs_record["items"]
    assert isinstance(items, list)
    for item in items:
        assert isinstance(item, dict)
        assert set(item) == {
            "visibility",
            "content_known",
            "about",
            "holder_id",
            "payload_keys",
        }
        assert "payload" not in item


def test_example_bundle_round_trips_through_json() -> None:
    original = examples_from_situations(all_situations())
    payload = examples_payload_bytes(original)
    loaded = json.loads(payload)
    restored = ExampleBundle(
        states=tuple(original.states[0].__class__(**row) for row in loaded["states"]),
        observations=tuple(ObservationExample(**row) for row in loaded["observations"]),
        pairs=tuple(original.pairs[0].__class__(**row) for row in loaded["pairs"]),
        sequences=tuple(
            original.sequences[0].__class__(**row) for row in loaded["sequences"]
        ),
    )
    assert restored.to_mapping() == original.to_mapping()
    assert examples_payload_bytes(restored) == payload


def test_store_examples_writes_dataset_artifact(tmp_path) -> None:
    store = LocalStore(tmp_path)
    run = RunContext(
        run_id="run-examples",
        project="bga",
        started_at=datetime(2026, 9, 19, 18, 30, tzinfo=UTC),
        environment=Environment.LOCAL,
    )
    bundle = examples_from_situations(all_situations())
    payload = examples_payload_bytes(bundle)
    pid, record_id = store_examples(store, bundle, run=run)
    assert pid == payload_id(payload)
    record = json.loads(store.get(record_id).decode("utf-8"))
    assert record["kind"] == ArtifactKind.DATASET
    assert record["logical_key"] == EXAMPLES_LOGICAL_KEY
    assert record["inputs"] == []
