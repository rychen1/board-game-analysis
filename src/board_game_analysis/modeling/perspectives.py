"""Build platform PerspectiveTables from observation embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from ds_platform import RunContext, Store
from ds_platform.modeling._spec_json import spec_canonical_json_bytes
from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.perspectives import (
    PerspectiveTable,
    pairwise_perspective_distances,
    to_representation_table,
)
from ds_platform.modeling.records import (
    put_feature_dataset,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import PerspectiveSpec, spec_config_hash

from board_game_analysis.modeling.examples import ObservationExample
from board_game_analysis.modeling.measures import PairAsymmetry

PERSPECTIVES_LOGICAL_KEY = "bga:repr/perspectives:v0"
PERSPECTIVE_DISTANCES_LOGICAL_KEY = "bga:repr/perspective-distances:v0"

BGA_PERSPECTIVE_SPEC = PerspectiveSpec(
    subject_field="game_state_id",
    perspective_field="observer_id",
)


def perspective_table_from_observations(
    examples: Sequence[ObservationExample],
    z_obs: RepresentationTable,
) -> PerspectiveTable:
    """Stack ``z_obs`` as ``(game_state_id, observer_id)`` perspectives."""
    index_by_id = {entity_id: index for index, entity_id in enumerate(z_obs.entity_ids)}
    subject_ids: list[str] = []
    perspective_ids: list[str] = []
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    for example in examples:
        try:
            index = index_by_id[example.entity_id]
        except KeyError as exc:
            raise KeyError(
                f"missing observation representation: {example.entity_id!r}"
            ) from exc
        subject_ids.append(example.state_id)
        perspective_ids.append(example.observer_id)
        vectors.append(z_obs.vectors[index])
        sources.append(z_obs.source_payload_ids[index])
    return PerspectiveTable(
        subject_ids=tuple(subject_ids),
        perspective_ids=tuple(perspective_ids),
        vectors=tuple(vectors),
        dim=z_obs.dim,
        source_payload_ids=tuple(sources),
    )


def perspective_distances(
    table: PerspectiveTable,
    *,
    metric: Literal["cosine", "l2"] = "cosine",
) -> FeatureTable:
    return pairwise_perspective_distances(table, metric=metric)


def mean_pair_distance(distances: FeatureTable, subject_id: str) -> float:
    try:
        row_index = distances.entity_ids.index(subject_id)
    except ValueError as exc:
        raise KeyError(f"subject not found: {subject_id!r}") from exc
    cells = [
        float(cell)
        for cell in distances.values[row_index]
        if isinstance(cell, (int, float))
    ]
    if not cells:
        raise KeyError(f"no pairwise distances for subject {subject_id!r}")
    return sum(cells) / len(cells)


def aligned_distance_pairs(
    distances: FeatureTable,
    asymmetries: Sequence[PairAsymmetry],
) -> list[tuple[float, float]]:
    """Pairs of (embedding distance, ontology asymmetry) with matching columns."""
    column_index = {name: index for index, name in enumerate(distances.columns)}
    row_index = {
        entity_id: index for index, entity_id in enumerate(distances.entity_ids)
    }
    paired: list[tuple[float, float]] = []
    for score in asymmetries:
        if score.state_id not in row_index or score.pair_column not in column_index:
            continue
        cell = distances.values[row_index[score.state_id]][
            column_index[score.pair_column]
        ]
        if isinstance(cell, (int, float)):
            paired.append((float(cell), score.value))
    return paired


def store_perspectives(
    store: Store,
    table: PerspectiveTable,
    *,
    run: RunContext,
    inputs: Sequence[str] = (),
    created_at: datetime | None = None,
    logical_key: str = PERSPECTIVES_LOGICAL_KEY,
) -> tuple[str, str]:
    exported = to_representation_table(table, entity_key="pair")
    run_with_hash = run.model_copy(
        update={"config_hash": spec_config_hash(BGA_PERSPECTIVE_SPEC)}
    )
    return put_feature_dataset(
        store,
        representation_payload_bytes(exported),
        run=run_with_hash,
        inputs=list(inputs),
        media_type="application/json",
        logical_key=logical_key,
        created_at=created_at,
    )


def store_perspective_distances(
    store: Store,
    distances: FeatureTable,
    *,
    run: RunContext,
    inputs: Sequence[str] = (),
    created_at: datetime | None = None,
    logical_key: str = PERSPECTIVE_DISTANCES_LOGICAL_KEY,
) -> tuple[str, str]:
    return put_feature_dataset(
        store,
        spec_canonical_json_bytes(distances),
        run=run,
        inputs=list(inputs),
        media_type="application/json",
        logical_key=logical_key,
        created_at=created_at,
    )
