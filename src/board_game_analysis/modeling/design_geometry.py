"""Design-space geometry: normalize, project, neighbors, clusters, novelty."""

from __future__ import annotations

import pickle
import random
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from ds_platform import RunContext, Store
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.geometry import knn, novelty_scores, pairwise_distances
from ds_platform.modeling.records import (
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import (
    DistanceMetric,
    RepresentationTable,
    select_entities,
    vector_distance,
)
from ds_platform.modeling.spec import GeometrySpec, SplitSpec, spec_config_hash
from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import PlaySituation
from board_game_analysis.modeling.design_space import (
    DESIGN_SPACE_FAMILY,
    SCHEMA_VERSION,
    FeatureSchema,
    GameRepresentation,
    PlayLayer,
    SituationRepresentation,
    ViewName,
    build_play_layer,
    default_feature_schema,
    external_metadata,
    game_table,
    represent_games,
    represent_situations,
    situation_table,
)
from board_game_analysis.modeling.split import split_entities_by_game

SITUATION_REPR_LOGICAL_KEY = "bga:repr/situations:v0"
GAME_REPR_LOGICAL_KEY = "bga:repr/games:v0"
DESIGN_SPACE_MODEL_LOGICAL_KEY = "bga:model/design-space:v0"
NORMALIZER_LOGICAL_KEY = "bga:model/design-space-normalizer:v0"
PROJECTION_LOGICAL_KEY = "bga:model/design-space-projection:v0"
NOVELTY_EVAL_LOGICAL_KEY = "bga:evaluation/novelty:v0"
CLUSTER_EVAL_LOGICAL_KEY = "bga:evaluation/clusters:v0"

_CORPUS_NOTE = (
    "structural position in a learned representation space; not game "
    "quality, fun, commercial potential, or objective originality"
)
_SMALL_CORPUS_NOTE = (
    "fixture corpus is about ten authored moments; clusters and novelty "
    "are exploratory geometry, not discovered game categories"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FamilyStandardizer(_FrozenModel):
    """Per-coordinate mean/std fit on training games only."""

    means: tuple[float, ...]
    stds: tuple[float, ...]
    zero_variance_columns: tuple[str, ...]
    train_entity_ids: tuple[str, ...]
    column_names: tuple[str, ...]

    def transform(self, table: RepresentationTable) -> RepresentationTable:
        if table.dim != len(self.means):
            raise ValueError("table dim does not match fitted normalizer")
        vectors = tuple(
            tuple(
                (value - mean) / std
                for value, mean, std in zip(vector, self.means, self.stds, strict=True)
            )
            for vector in table.vectors
        )
        return RepresentationTable(
            entity_ids=table.entity_ids,
            vectors=vectors,
            dim=table.dim,
            source_payload_ids=table.source_payload_ids,
        )


class LeadingVarianceProjection(_FrozenModel):
    """Keep the highest-variance training coordinates. Not the canonical identity."""

    n_components: int
    keep_indexes: tuple[int, ...]
    train_variances: tuple[float, ...]
    train_entity_ids: tuple[str, ...]

    def transform(self, table: RepresentationTable) -> RepresentationTable:
        if table.dim != len(self.train_variances):
            raise ValueError("table dim does not match fitted projection")
        vectors = tuple(
            tuple(vector[index] for index in self.keep_indexes)
            for vector in table.vectors
        )
        return RepresentationTable(
            entity_ids=table.entity_ids,
            vectors=vectors,
            dim=len(self.keep_indexes),
            source_payload_ids=table.source_payload_ids,
        )


class DeterministicKMeans:
    """Exploratory k-means. Assignments are reproducible for a seed."""

    family = "kmeans-v0"

    def __init__(
        self,
        n_clusters: int,
        *,
        seed: int = 0,
        max_iter: int = 50,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        self.n_clusters = n_clusters
        self.seed = seed
        self.max_iter = max_iter
        self.centroids: tuple[tuple[float, ...], ...] = ()

    def fit(self, table: RepresentationTable) -> None:
        if len(table.entity_ids) < self.n_clusters:
            raise ValueError("not enough entities to form the requested clusters")
        centroids = list(_initial_centroids(table, self.n_clusters, self.seed))
        for _ in range(self.max_iter):
            labels = _assign(table.vectors, centroids)
            updated = _update_centroids(table.vectors, labels, centroids)
            if updated == centroids:
                break
            centroids = updated
        self.centroids = tuple(centroids)

    def predict(self, table: RepresentationTable) -> tuple[int, ...]:
        if not self.centroids:
            raise ValueError("clusterer has not been fit")
        if table.dim != len(self.centroids[0]):
            raise ValueError("table dim does not match fitted clusterer")
        return tuple(_assign(table.vectors, list(self.centroids)))


class NeighborHit(_FrozenModel):
    entity_id: str
    distance: float


class Neighborhood(_FrozenModel):
    target_id: str
    neighbors: tuple[NeighborHit, ...]
    metric: DistanceMetric
    representation: Literal["game", "situation"]
    view: ViewName
    schema_version: str
    notes: tuple[str, ...] = ()


class NoveltyScore(_FrozenModel):
    entity_id: str
    k: int
    structural_novelty: float
    local_isolation: float
    nearest_distance: float


class ClusterAssignment(_FrozenModel):
    entity_id: str
    cluster: int
    n_clusters: int
    seed: int
    family: str


class DesignSpace(_FrozenModel):
    feature_schema: FeatureSchema
    layer: PlayLayer
    situations: tuple[SituationRepresentation, ...]
    games: tuple[GameRepresentation, ...]
    canonical_situations: RepresentationTable
    canonical_games: RepresentationTable
    normalized_situations: RepresentationTable
    normalized_games: RepresentationTable
    projected_situations: RepresentationTable | None = None
    projected_games: RepresentationTable | None = None
    normalizer: FamilyStandardizer
    projection: LeadingVarianceProjection | None = None
    metric: DistanceMetric
    train_game_ids: tuple[str, ...]
    test_game_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def game(self, game_id: str) -> GameRepresentation:
        for item in self.games:
            if item.game_id == game_id:
                return item
        raise KeyError(f"no game {game_id!r}")

    def situation(self, situation_id: str) -> SituationRepresentation:
        for item in self.situations:
            if item.situation_id == situation_id:
                return item
        raise KeyError(f"no situation {situation_id!r}")


class DesignSpaceEvaluation(_FrozenModel):
    novelty: EvaluationReport
    clusters: EvaluationReport
    n_games: int
    n_situations: int
    train_game_ids: tuple[str, ...]
    test_game_ids: tuple[str, ...]
    notes: tuple[str, ...]


class DesignSpaceRun(_FrozenModel):
    space: DesignSpace
    evaluation: DesignSpaceEvaluation
    novelty: tuple[NoveltyScore, ...]
    assignments: tuple[ClusterAssignment, ...]
    situation_payload_id: str
    game_payload_id: str
    model_payload_id: str
    normalizer_payload_id: str
    projection_payload_id: str | None
    novelty_payload_id: str
    cluster_payload_id: str


def fit_standardizer(
    table: RepresentationTable,
    train_ids: Sequence[str],
    column_names: Sequence[str],
) -> FamilyStandardizer:
    """Fit mean/std on ``train_ids`` only. Test rows must not be included."""
    if not train_ids:
        raise ValueError("normalizer requires at least one training entity")
    train = select_entities(table, train_ids)
    if train.dim != len(column_names):
        raise ValueError("column_names length must match table dim")
    means: list[float] = []
    stds: list[float] = []
    zero: list[str] = []
    n_rows = float(len(train.entity_ids))
    for index in range(train.dim):
        column = [vector[index] for vector in train.vectors]
        mean = sum(column) / n_rows
        variance = sum((value - mean) ** 2 for value in column) / n_rows
        if variance == 0.0:
            stds.append(1.0)
            zero.append(column_names[index])
        else:
            stds.append(variance**0.5)
        means.append(mean)
    return FamilyStandardizer(
        means=tuple(means),
        stds=tuple(stds),
        zero_variance_columns=tuple(zero),
        train_entity_ids=tuple(train_ids),
        column_names=tuple(column_names),
    )


def fit_projection(
    table: RepresentationTable,
    train_ids: Sequence[str],
    n_components: int,
) -> LeadingVarianceProjection:
    """Fit a coordinate mask on training variance only."""
    if n_components < 1:
        raise ValueError("n_components must be a positive integer")
    train = select_entities(table, train_ids)
    n_rows = float(len(train.entity_ids))
    variances: list[float] = []
    for index in range(train.dim):
        column = [vector[index] for vector in train.vectors]
        mean = sum(column) / n_rows
        variances.append(sum((value - mean) ** 2 for value in column) / n_rows)
    ranked = sorted(
        range(train.dim),
        key=lambda index: (-variances[index], index),
    )
    keep = tuple(ranked[: min(n_components, train.dim)])
    return LeadingVarianceProjection(
        n_components=min(n_components, train.dim),
        keep_indexes=keep,
        train_variances=tuple(variances),
        train_entity_ids=tuple(train_ids),
    )


def build_design_space(
    situations: Sequence[PlaySituation],
    *,
    train_game_ids: Sequence[str],
    test_game_ids: Sequence[str] = (),
    dim: int = 16,
    metric: DistanceMetric = "l2",
    n_components: int | None = None,
) -> DesignSpace:
    """Aggregate play structure and fit leakage-safe views on training games."""
    train = tuple(train_game_ids)
    test = tuple(test_game_ids)
    if set(train) & set(test):
        raise ValueError("train and test game ids must be disjoint")
    schema = default_feature_schema()
    layer = build_play_layer(situations, dim=dim)
    situation_reps = represent_situations(layer, schema)
    game_reps = represent_games(situation_reps)
    games_table = game_table(game_reps)
    sits_table = situation_table(situation_reps)
    missing_train = [
        game_id for game_id in train if game_id not in games_table.entity_ids
    ]
    if missing_train:
        raise KeyError(
            f"training game missing from representations: {missing_train[0]}"
        )
    normalizer = fit_standardizer(games_table, train, schema.column_names)
    normalized_games = normalizer.transform(games_table)
    normalized_sits = normalizer.transform(sits_table)
    projection: LeadingVarianceProjection | None = None
    projected_games: RepresentationTable | None = None
    projected_sits: RepresentationTable | None = None
    if n_components is not None:
        projection = fit_projection(normalized_games, train, n_components)
        projected_games = projection.transform(normalized_games)
        projected_sits = projection.transform(normalized_sits)
    return DesignSpace(
        feature_schema=schema,
        layer=layer,
        situations=situation_reps,
        games=game_reps,
        canonical_situations=sits_table,
        canonical_games=games_table,
        normalized_situations=normalized_sits,
        normalized_games=normalized_games,
        projected_situations=projected_sits,
        projected_games=projected_games,
        normalizer=normalizer,
        projection=projection,
        metric=metric,
        train_game_ids=train,
        test_game_ids=test,
        notes=(_CORPUS_NOTE, _SMALL_CORPUS_NOTE),
    )


def view_table(
    space: DesignSpace,
    *,
    level: Literal["game", "situation"],
    view: ViewName = "normalized",
) -> RepresentationTable:
    if level == "game":
        if view == "canonical":
            return space.canonical_games
        if view == "normalized":
            return space.normalized_games
        if space.projected_games is None:
            raise ValueError("no projection was fit")
        return space.projected_games
    if view == "canonical":
        return space.canonical_situations
    if view == "normalized":
        return space.normalized_situations
    if space.projected_situations is None:
        raise ValueError("no projection was fit")
    return space.projected_situations


def game_distance(
    space: DesignSpace,
    left_id: str,
    right_id: str,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> float:
    return _pair_distance(space, left_id, right_id, "game", view, metric)


def situation_distance(
    space: DesignSpace,
    left_id: str,
    right_id: str,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> float:
    return _pair_distance(space, left_id, right_id, "situation", view, metric)


def pairwise_game_distances(
    space: DesignSpace,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> tuple[tuple[float, ...], ...]:
    table = view_table(space, level="game", view=view)
    return pairwise_distances(table, metric=metric or space.metric)


def pairwise_situation_distances(
    space: DesignSpace,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> tuple[tuple[float, ...], ...]:
    table = view_table(space, level="situation", view=view)
    return pairwise_distances(table, metric=metric or space.metric)


def nearest_games(
    space: DesignSpace,
    game_id: str,
    k: int,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> Neighborhood:
    return _neighborhood(space, game_id, k, "game", view, metric)


def nearest_situations(
    space: DesignSpace,
    situation_id: str,
    k: int,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> Neighborhood:
    return _neighborhood(space, situation_id, k, "situation", view, metric)


def novelty_score(
    space: DesignSpace,
    game_id: str,
    k: int,
    *,
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> NoveltyScore:
    for row in novelty_table(space, ks=(k,), view=view, metric=metric):
        if row.entity_id == game_id:
            return row
    raise KeyError(f"no game {game_id!r}")


def novelty_table(
    space: DesignSpace,
    *,
    ks: Sequence[int] = (1, 3),
    view: ViewName = "normalized",
    metric: DistanceMetric | None = None,
) -> tuple[NoveltyScore, ...]:
    table = view_table(space, level="game", view=view)
    chosen = metric or space.metric
    n_entities = len(table.entity_ids)
    if n_entities < 2:
        raise ValueError("novelty requires at least two games")
    scores: list[NoveltyScore] = []
    nearest = novelty_scores(table, k=1, metric=chosen)
    nearest_by_id = {
        entity_id: _as_float(row[0])
        for entity_id, row in zip(nearest.entity_ids, nearest.values, strict=True)
    }
    for k in ks:
        if k < 1:
            raise ValueError("k must be a positive integer")
        if k >= n_entities:
            continue
        table_k = novelty_scores(table, k=k, metric=chosen)
        for entity_id, row in zip(table_k.entity_ids, table_k.values, strict=True):
            value = _as_float(row[0])
            scores.append(
                NoveltyScore(
                    entity_id=entity_id,
                    k=k,
                    structural_novelty=value,
                    local_isolation=value,
                    nearest_distance=nearest_by_id[entity_id],
                )
            )
    return tuple(scores)


def cluster_games(
    space: DesignSpace,
    n_clusters: int,
    *,
    seed: int = 0,
    view: ViewName = "normalized",
) -> tuple[ClusterAssignment, ...]:
    table = view_table(space, level="game", view=view)
    clusterer = DeterministicKMeans(n_clusters, seed=seed)
    clusterer.fit(table)
    labels = clusterer.predict(table)
    return tuple(
        ClusterAssignment(
            entity_id=entity_id,
            cluster=int(label),
            n_clusters=n_clusters,
            seed=seed,
            family=clusterer.family,
        )
        for entity_id, label in zip(table.entity_ids, labels, strict=True)
    )


def within_game_dispersion(space: DesignSpace, game_id: str) -> float:
    return space.game(game_id).situation_dispersion


def between_game_dispersion(
    space: DesignSpace, *, view: ViewName = "normalized"
) -> float:
    table = view_table(space, level="game", view=view)
    if len(table.vectors) < 2:
        return 0.0
    centroid = _mean_vector(table.vectors)
    total = sum(
        vector_distance(vector, centroid, metric="l2") for vector in table.vectors
    )
    return total / len(table.vectors)


def attach_external_metadata(
    situations: Sequence[PlaySituation],
) -> dict[str, dict[str, Any]]:
    """Post-hoc catalog fields keyed by game id. Not representation inputs."""
    attached: dict[str, dict[str, Any]] = {}
    for situation in situations:
        attached.setdefault(situation.game.id, external_metadata(situation))
    return attached


def run_design_space_experiment(
    situations: Sequence[PlaySituation],
    *,
    store: Store,
    run: RunContext,
    split: SplitSpec,
    dim: int = 16,
    n_clusters: int = 3,
    n_components: int | None = None,
    novelty_ks: Sequence[int] = (1, 3),
    metric: DistanceMetric = "l2",
    seed: int = 0,
    created_at: datetime | None = None,
) -> DesignSpaceRun:
    """Game-held-out design-space sanity evaluation."""
    if not situations:
        raise ValueError("design-space experiment requires situations")
    game_ids = [situation.game.id for situation in situations]
    assignment = split_entities_by_game(game_ids, game_ids, split)
    if set(assignment.train_ids) & set(assignment.test_ids):
        raise ValueError("game split leaked a game into both partitions")
    if not assignment.train_ids or not assignment.test_ids:
        raise ValueError("game split produced an empty partition")
    space = build_design_space(
        situations,
        train_game_ids=assignment.train_ids,
        test_game_ids=assignment.test_ids,
        dim=dim,
        metric=metric,
        n_components=n_components,
    )
    novelty = novelty_table(space, ks=novelty_ks)
    assignments = cluster_games(space, n_clusters, seed=seed)
    again = cluster_games(space, n_clusters, seed=seed)
    repeatable = 1.0 if assignments == again else 0.0
    evaluation = DesignSpaceEvaluation(
        novelty=EvaluationReport(
            metrics=_novelty_metrics(novelty, novelty_ks),
            n=len(space.games),
            n_missing=0,
            notes=(_CORPUS_NOTE, _SMALL_CORPUS_NOTE),
        ),
        clusters=EvaluationReport(
            metrics={
                "n_clusters": float(n_clusters),
                "repeatable": repeatable,
                "n_games": float(len(space.games)),
            },
            n=len(space.games),
            n_missing=0,
            notes=(_SMALL_CORPUS_NOTE, "clusters are not ground-truth categories"),
        ),
        n_games=len(space.games),
        n_situations=len(space.situations),
        train_game_ids=space.train_game_ids,
        test_game_ids=space.test_game_ids,
        notes=(_CORPUS_NOTE, _SMALL_CORPUS_NOTE),
    )
    spec = GeometrySpec(
        metric=metric,
        k=min(novelty_ks) if novelty_ks else 1,
        family=DESIGN_SPACE_FAMILY,
        params={
            "schema": SCHEMA_VERSION,
            "dim": dim,
            "n_clusters": n_clusters,
            "n_components": n_components,
            "novelty_ks": list(novelty_ks),
            "seed": seed,
        },
    )
    run_with_hash = run.model_copy(update={"config_hash": spec_config_hash(spec)})
    sit_pid, _sit_rid = put_feature_dataset(
        store,
        representation_payload_bytes(space.canonical_situations),
        run=run_with_hash,
        inputs=[],
        media_type="application/json",
        logical_key=SITUATION_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    game_pid, _game_rid = put_feature_dataset(
        store,
        representation_payload_bytes(space.canonical_games),
        run=run_with_hash,
        inputs=[sit_pid],
        media_type="application/json",
        logical_key=GAME_REPR_LOGICAL_KEY,
        created_at=created_at,
    )
    model_pid, _model_rid = put_model_artifact(
        store,
        pickle.dumps(
            {
                "family": DESIGN_SPACE_FAMILY,
                "schema": space.feature_schema,
                "metric": metric,
                "train_game_ids": space.train_game_ids,
            }
        ),
        run=run_with_hash,
        inputs=[game_pid, sit_pid],
        media_type="application/octet-stream",
        logical_key=DESIGN_SPACE_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )
    norm_pid, _norm_rid = put_model_artifact(
        store,
        pickle.dumps(space.normalizer),
        run=run_with_hash,
        inputs=[model_pid],
        media_type="application/octet-stream",
        logical_key=NORMALIZER_LOGICAL_KEY,
        created_at=created_at,
    )
    projection_pid: str | None = None
    if space.projection is not None:
        projection_pid, _proj_rid = put_model_artifact(
            store,
            pickle.dumps(space.projection),
            run=run_with_hash,
            inputs=[norm_pid],
            media_type="application/octet-stream",
            logical_key=PROJECTION_LOGICAL_KEY,
            created_at=created_at,
        )
    novelty_pid, _nov_rid = put_evaluation_artifact(
        store,
        evaluation.novelty,
        run=run_with_hash,
        subject_payload_id=game_pid,
        inputs=[game_pid, norm_pid],
        logical_key=NOVELTY_EVAL_LOGICAL_KEY,
        created_at=created_at,
    )
    cluster_inputs = [game_pid, norm_pid]
    if projection_pid is not None:
        cluster_inputs.append(projection_pid)
    cluster_pid, _cl_rid = put_evaluation_artifact(
        store,
        evaluation.clusters,
        run=run_with_hash,
        subject_payload_id=game_pid,
        inputs=cluster_inputs,
        logical_key=CLUSTER_EVAL_LOGICAL_KEY,
        created_at=created_at,
    )
    return DesignSpaceRun(
        space=space,
        evaluation=evaluation,
        novelty=novelty,
        assignments=tuple(assignments),
        situation_payload_id=sit_pid,
        game_payload_id=game_pid,
        model_payload_id=model_pid,
        normalizer_payload_id=norm_pid,
        projection_payload_id=projection_pid,
        novelty_payload_id=novelty_pid,
        cluster_payload_id=cluster_pid,
    )


def _pair_distance(
    space: DesignSpace,
    left_id: str,
    right_id: str,
    level: Literal["game", "situation"],
    view: ViewName,
    metric: DistanceMetric | None,
) -> float:
    table = view_table(space, level=level, view=view)
    chosen = metric or space.metric
    index = {entity_id: i for i, entity_id in enumerate(table.entity_ids)}
    try:
        left = table.vectors[index[left_id]]
        right = table.vectors[index[right_id]]
    except KeyError as exc:
        raise KeyError(f"missing {level} id in {exc.args[0]}") from exc
    return vector_distance(left, right, metric=chosen)


def _neighborhood(
    space: DesignSpace,
    target_id: str,
    k: int,
    level: Literal["game", "situation"],
    view: ViewName,
    metric: DistanceMetric | None,
) -> Neighborhood:
    table = view_table(space, level=level, view=view)
    chosen = metric or space.metric
    if k < 1:
        raise ValueError("k must be a positive integer")
    if k >= len(table.entity_ids):
        raise ValueError("k must be smaller than the number of entities")
    neighbors = knn(table, k=k, metric=chosen)
    try:
        row_index = neighbors.entity_ids.index(target_id)
    except ValueError as exc:
        raise KeyError(f"no {level} {target_id!r}") from exc
    row = neighbors.values[row_index]
    hits = tuple(
        NeighborHit(entity_id=str(row[index]), distance=_as_float(row[k + index]))
        for index in range(k)
    )
    if any(hit.entity_id == target_id for hit in hits):
        raise ValueError("target appeared as its own neighbor")
    return Neighborhood(
        target_id=target_id,
        neighbors=hits,
        metric=chosen,
        representation=level,
        view=view,
        schema_version=space.feature_schema.version,
        notes=(
            "nearest under this representation and distance metric",
            "not a claim of human or psychological similarity",
        ),
    )


def _novelty_metrics(
    rows: Sequence[NoveltyScore], ks: Sequence[int]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        values = [row.structural_novelty for row in rows if row.k == k]
        if values:
            metrics[f"mean_structural_novelty_k{k}"] = sum(values) / len(values)
    metrics["n_novelty_rows"] = float(len(rows))
    return metrics


def _initial_centroids(
    table: RepresentationTable, n_clusters: int, seed: int
) -> tuple[tuple[float, ...], ...]:
    order = list(range(len(table.entity_ids)))
    if seed == 0:
        ranked = sorted(order, key=lambda index: table.entity_ids[index])
        step = max(len(ranked) // n_clusters, 1)
        chosen = [
            ranked[min(index * step, len(ranked) - 1)] for index in range(n_clusters)
        ]
    else:
        rng = random.Random(seed)
        rng.shuffle(order)
        chosen = order[:n_clusters]
    unique: list[int] = []
    for index in chosen:
        if index not in unique:
            unique.append(index)
    next_index = 0
    while len(unique) < n_clusters:
        if next_index not in unique:
            unique.append(next_index)
        next_index += 1
    return tuple(table.vectors[index] for index in unique)


def _assign(
    vectors: Sequence[tuple[float, ...]],
    centroids: Sequence[tuple[float, ...]],
) -> list[int]:
    labels: list[int] = []
    for vector in vectors:
        distances = [
            (vector_distance(vector, centroid, metric="l2"), index)
            for index, centroid in enumerate(centroids)
        ]
        distances.sort()
        labels.append(distances[0][1])
    return labels


def _update_centroids(
    vectors: Sequence[tuple[float, ...]],
    labels: Sequence[int],
    previous: Sequence[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    updated: list[tuple[float, ...]] = []
    for cluster in range(len(previous)):
        members = [
            vectors[index] for index, label in enumerate(labels) if label == cluster
        ]
        if not members:
            updated.append(previous[cluster])
            continue
        updated.append(_mean_vector(members))
    return updated


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {value!r}")
    return float(value)


def _mean_vector(vectors: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(vectors))
    return tuple(total / count for total in totals)
