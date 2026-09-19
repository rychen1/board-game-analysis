"""Persist modeling examples and ontology-v0 measurements as datasets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

from ds_platform import RunContext, Store
from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.records import put_feature_dataset
from ds_platform.modeling.spec import spec_config_hash

from board_game_analysis.domain import DerivedMeasurement
from board_game_analysis.modeling.examples import ExampleBundle, ObservationExample
from board_game_analysis.modeling.measures import MeasureSpec, default_measure_spec

MEASUREMENTS_LOGICAL_KEY = "bga:measurements/ontology-v0:v0"
EXAMPLES_LOGICAL_KEY = "bga:examples/play-fixtures:v0"
MEASUREMENTS_MEDIA_TYPE = "application/json"
EXAMPLES_MEDIA_TYPE = "application/json"

_DICT_METRICS = {"information_ownership", "structural_interaction"}


def measurements_payload_bytes(measurements: Sequence[DerivedMeasurement]) -> bytes:
    """Deterministic JSONL of derived measurements, sorted by id."""
    lines = [
        json.dumps(
            measurement.model_dump(mode="python", exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for measurement in sorted(measurements, key=lambda item: item.id)
    ]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def examples_payload_bytes(bundle: ExampleBundle) -> bytes:
    """Deterministic JSON document of an example bundle."""
    return (json.dumps(bundle.to_mapping(), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def scalar_measurement_table(
    measurements: Sequence[DerivedMeasurement],
    *,
    source_payload_id: str,
) -> FeatureTable:
    """Long FeatureTable of scalar measurements. Dict-valued metrics are omitted."""
    entity_ids: list[str] = []
    values: list[tuple[Scalar, ...]] = []
    for measurement in measurements:
        value = measurement.value
        if measurement.name in _DICT_METRICS or not isinstance(value, (int, float)):
            continue
        entity_ids.append(measurement.id)
        values.append(
            (measurement.name, float(value), measurement.scope, measurement.game_id)
        )
    return FeatureTable(
        entity_ids=tuple(entity_ids),
        columns=("name", "value", "scope", "game_id"),
        values=tuple(values),
        source_payload_ids=tuple(source_payload_id for _ in entity_ids),
    )


def observation_label_table(
    measurements: Sequence[DerivedMeasurement],
    *,
    entity_ids: Sequence[str],
    game_ids: Sequence[str],
    observation_ids: Sequence[str],
    observer_ids: Sequence[str],
    state_ids: Sequence[str],
    source_payload_ids: Sequence[str],
) -> FeatureTable:
    """Wide observation-scoped labels aligned to example entity ids.

    ``available_decision_count`` is attached when a player-scoped measurement
    exists for the same observer and state; otherwise the cell is ``None``.
    """
    if not (
        len(entity_ids)
        == len(game_ids)
        == len(observation_ids)
        == len(observer_ids)
        == len(state_ids)
        == len(source_payload_ids)
    ):
        raise ValueError("observation label alignments must have equal length")

    obs_values: dict[tuple[str, str | None, str | None], dict[str, float]] = {}
    player_values: dict[tuple[str, str | None, str | None], dict[str, float]] = {}
    for measurement in measurements:
        if not isinstance(measurement.value, (int, float)):
            continue
        value = float(measurement.value)
        if measurement.scope == "observation":
            key = (measurement.game_id, measurement.player_id, measurement.state_id)
            obs_values.setdefault(key, {})[measurement.name] = value
        elif measurement.scope == "player":
            key = (measurement.game_id, measurement.player_id, measurement.state_id)
            player_values.setdefault(key, {})[measurement.name] = value

    columns = (
        "information_volume",
        "information_visibility",
        "hidden_information",
        "self_knowledge_asymmetry",
        "available_decision_count",
    )
    values: list[tuple[Scalar, ...]] = []
    aligned = zip(game_ids, observer_ids, state_ids, strict=True)
    for game_id, observer_id, state_id in aligned:
        obs = obs_values.get((game_id, observer_id, state_id), {})
        player = player_values.get((game_id, observer_id, state_id), {})
        values.append(
            (
                obs.get("information_volume"),
                obs.get("information_visibility"),
                obs.get("hidden_information"),
                obs.get("self_knowledge_asymmetry"),
                player.get("available_decision_count"),
            )
        )
    return FeatureTable(
        entity_ids=tuple(entity_ids),
        columns=columns,
        values=tuple(values),
        source_payload_ids=tuple(source_payload_ids),
    )


def labels_for_observations(
    measurements: Sequence[DerivedMeasurement],
    examples: Sequence[ObservationExample],
    *,
    source_payload_ids: Sequence[str],
) -> FeatureTable:
    return observation_label_table(
        measurements,
        entity_ids=[example.entity_id for example in examples],
        game_ids=[example.game_id for example in examples],
        observation_ids=[example.observation_id for example in examples],
        observer_ids=[example.observer_id for example in examples],
        state_ids=[example.state_id for example in examples],
        source_payload_ids=source_payload_ids,
    )


def store_measurements(
    store: Store,
    measurements: Sequence[DerivedMeasurement],
    *,
    run: RunContext,
    spec: MeasureSpec | None = None,
    created_at: datetime | None = None,
    logical_key: str = MEASUREMENTS_LOGICAL_KEY,
) -> tuple[str, str]:
    chosen = spec or default_measure_spec()
    run_with_hash = run.model_copy(update={"config_hash": spec_config_hash(chosen)})
    payload = measurements_payload_bytes(measurements)
    return put_feature_dataset(
        store,
        payload,
        run=run_with_hash,
        inputs=[],
        media_type=MEASUREMENTS_MEDIA_TYPE,
        logical_key=logical_key,
        created_at=created_at,
    )


def store_examples(
    store: Store,
    bundle: ExampleBundle,
    *,
    run: RunContext,
    created_at: datetime | None = None,
    logical_key: str = EXAMPLES_LOGICAL_KEY,
) -> tuple[str, str]:
    payload = examples_payload_bytes(bundle)
    return put_feature_dataset(
        store,
        payload,
        run=run,
        inputs=[],
        media_type=EXAMPLES_MEDIA_TYPE,
        logical_key=logical_key,
        created_at=created_at,
    )
