"""Phase 7: structural design-space representations and geometry."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from ds_platform import ArtifactKind, Environment, LocalStore, RunContext
from ds_platform.modeling.geometry import novelty_scores
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import SplitSpec

from board_game_analysis.domain import Mechanic
from board_game_analysis.modeling.design_geometry import (
    CLUSTER_EVAL_LOGICAL_KEY,
    DESIGN_SPACE_MODEL_LOGICAL_KEY,
    GAME_REPR_LOGICAL_KEY,
    NORMALIZER_LOGICAL_KEY,
    NOVELTY_EVAL_LOGICAL_KEY,
    PROJECTION_LOGICAL_KEY,
    SITUATION_REPR_LOGICAL_KEY,
    DeterministicKMeans,
    attach_external_metadata,
    between_game_dispersion,
    build_design_space,
    cluster_games,
    fit_projection,
    fit_standardizer,
    game_distance,
    nearest_games,
    nearest_situations,
    novelty_score,
    novelty_table,
    pairwise_game_distances,
    run_design_space_experiment,
    situation_distance,
    view_table,
    within_game_dispersion,
)
from board_game_analysis.modeling.design_space import (
    FORBIDDEN_METADATA_FIELDS,
    assert_schema_excludes_metadata,
    build_play_layer,
    default_feature_schema,
    represent_game,
    represent_games,
    represent_situation,
    represent_situations,
)
from tests.fixtures.games import all_situations
from tests.fixtures.games.chess import make_chess_situation
from tests.fixtures.games.hanabi import make_hanabi_situation


def _run() -> RunContext:
    return RunContext(
        run_id="run-design-space",
        project="bga",
        started_at=datetime(2026, 9, 19, 21, 0, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def _split() -> SplitSpec:
    return SplitSpec(method="holdout", seed=0, test_size=0.3)


def _space(*, n_components: int | None = None):
    situations = all_situations()
    game_ids = [situation.game.id for situation in situations]
    train = tuple(game_ids[:7])
    test = tuple(game_ids[7:])
    return build_design_space(
        situations,
        train_game_ids=train,
        test_game_ids=test,
        dim=8,
        n_components=n_components,
    )


def test_feature_ordering_is_stable() -> None:
    first = default_feature_schema()
    second = default_feature_schema()
    assert first.column_names == second.column_names
    assert first.column_names[0].endswith("__present")
    families = [family.name for family in first.families]
    assert families == [
        "state",
        "perspective",
        "action",
        "sequence",
        "intervention",
        "higher_order",
        "region",
    ]


def test_schema_excludes_metadata_tokens() -> None:
    schema = default_feature_schema()
    names = set(schema.column_names)
    assert names.isdisjoint(FORBIDDEN_METADATA_FIELDS)
    for name in names:
        assert name.split("__")[-1] not in FORBIDDEN_METADATA_FIELDS
    assert_schema_excludes_metadata(schema)


def test_aggregation_is_deterministic_and_keeps_sources() -> None:
    layer = build_play_layer([make_hanabi_situation()], dim=8)
    first = represent_situation(layer.situations[0], layer)
    second = represent_situation(layer.situations[0], layer)
    assert first.vector == second.vector
    assert first.source_artifacts
    assert first.game_id == "hanabi"
    assert all(not artifact.startswith("title:") for artifact in first.source_artifacts)
    game = represent_game([first])
    assert game.vector == first.vector
    assert first.situation_id in game.situation_ids
    assert first.entity_id in game.source_artifacts
    assert first.family("higher_order").present is True
    assert game.situation_range == 0.0


def test_metadata_is_not_a_representation_input() -> None:
    original = make_chess_situation()
    altered_game = original.game.model_copy(
        update={
            "title": "Not Chess",
            "release_year": 1999,
            "rating": 9.9,
            "popularity": 100.0,
            "publishers": ["Nobody"],
            "designers": ["Nobody"],
            "categories": ["Fake"],
            "mechanics": [Mechanic(id="m", name="Invented")],
        }
    )
    altered = original.model_copy(update={"game": altered_game})
    left = represent_situation(original, build_play_layer([original], dim=8))
    right = represent_situation(altered, build_play_layer([altered], dim=8))
    assert left.vector == right.vector
    blob = json.dumps(left.to_mapping())
    assert "Not Chess" not in blob
    assert "Invented" not in blob
    assert "Nobody" not in blob


def test_missing_families_are_explicit() -> None:
    chess = make_chess_situation()
    stripped = chess.model_copy(
        update={
            "observations": [],
            "information_spaces": [],
            "transitions": [],
            "actions": [],
        }
    )
    layer = build_play_layer([stripped], dim=8)
    represented = represent_situation(stripped, layer)
    assert represented.family("state").present is True
    assert represented.family("perspective").present is False
    assert represented.family("sequence").present is False
    assert represented.family("intervention").present is False
    schema = represented.feature_schema
    present_index = schema.column_names.index("perspective__present")
    assert represented.vector[present_index] == 0.0


def test_normalization_fits_only_training_games() -> None:
    space = _space()
    assert set(space.train_game_ids).isdisjoint(space.test_game_ids)
    assert set(space.normalizer.train_entity_ids) == set(space.train_game_ids)
    assert set(space.normalizer.train_entity_ids).isdisjoint(space.test_game_ids)
    train_only = fit_standardizer(
        view_table(space, level="game", view="canonical"),
        space.train_game_ids,
        space.feature_schema.column_names,
    )
    assert train_only.means == space.normalizer.means
    assert train_only.stds == space.normalizer.stds
    all_ids = space.canonical_games.entity_ids
    leaked = fit_standardizer(
        space.canonical_games, all_ids, space.feature_schema.column_names
    )
    assert (
        leaked.means != space.normalizer.means or leaked.stds != space.normalizer.stds
    )


def test_zero_variance_is_explicit() -> None:
    table = RepresentationTable(
        entity_ids=("a", "b"),
        vectors=((4.0, 1.0), (4.0, 3.0)),
        dim=2,
        source_payload_ids=("a", "b"),
    )
    fitted = fit_standardizer(table, ("a", "b"), ("const", "var"))
    assert fitted.zero_variance_columns == ("const",)
    assert fitted.stds[0] == 1.0
    transformed = fitted.transform(table)
    assert transformed.vectors[0][0] == 0.0
    assert transformed.vectors[1][0] == 0.0


def test_normalization_is_deterministic() -> None:
    space = _space()
    again = _space()
    assert space.normalizer == again.normalizer
    assert space.normalized_games.vectors == again.normalized_games.vectors


def test_projection_is_optional_and_train_only() -> None:
    space = _space()
    assert space.projection is None
    assert space.projected_games is None
    with pytest.raises(ValueError, match="no projection"):
        view_table(space, level="game", view="projected")
    projected = _space(n_components=2)
    assert projected.projection is not None
    assert projected.projection.n_components == 2
    assert set(projected.projection.train_entity_ids) == set(projected.train_game_ids)
    assert projected.projected_games is not None
    assert projected.projected_games.dim == 2
    assert projected.canonical_games.dim > 2
    train_only = fit_projection(projected.normalized_games, projected.train_game_ids, 2)
    assert train_only.keep_indexes == projected.projection.keep_indexes


def test_distances_and_neighbors_are_reproducible() -> None:
    space = _space()
    chess = "chess"
    hanabi = "hanabi"
    first = game_distance(space, chess, hanabi)
    second = game_distance(space, chess, hanabi)
    assert first == second
    assert game_distance(space, chess, chess) == 0.0
    assert game_distance(space, chess, hanabi) == game_distance(space, hanabi, chess)
    cosine = game_distance(space, chess, hanabi, metric="cosine")
    assert 0.0 <= cosine <= 2.0
    neighbors = nearest_games(space, chess, k=3)
    again = nearest_games(space, chess, k=3)
    assert neighbors == again
    assert chess not in [hit.entity_id for hit in neighbors.neighbors]
    assert neighbors.metric == "l2"
    assert neighbors.view == "normalized"
    assert neighbors.schema_version == space.feature_schema.version
    matrix = pairwise_game_distances(space)
    chess_index = space.normalized_games.entity_ids.index(chess)
    hanabi_index = space.normalized_games.entity_ids.index(hanabi)
    assert matrix[chess_index][hanabi_index] == first
    with pytest.raises(ValueError, match="k must be a positive integer"):
        nearest_games(space, chess, 0)
    with pytest.raises(ValueError, match="smaller than the number of entities"):
        nearest_games(space, chess, len(space.games))


def test_situation_geometry_preserves_game_ids() -> None:
    space = _space()
    chess_sit = next(
        item.situation_id for item in space.situations if item.game_id == "chess"
    )
    hanabi_sit = next(
        item.situation_id for item in space.situations if item.game_id == "hanabi"
    )
    assert situation_distance(space, chess_sit, chess_sit) == 0.0
    assert situation_distance(space, chess_sit, hanabi_sit) > 0.0
    neighbors = nearest_situations(space, chess_sit, k=2)
    assert chess_sit not in [hit.entity_id for hit in neighbors.neighbors]
    chess_game = space.game("chess")
    assert chess_sit in chess_game.situation_ids
    assert chess_game.n_situations == 1
    assert within_game_dispersion(space, "chess") == 0.0
    assert between_game_dispersion(space) > 0.0
    assert "chess" in space.canonical_games.entity_ids
    assert "chess" not in space.canonical_situations.entity_ids
    assert chess_sit in space.canonical_situations.entity_ids
    assert chess_sit not in space.canonical_games.entity_ids


def test_game_aggregation_is_order_independent() -> None:
    layer = build_play_layer(all_situations(), dim=8)
    reps = represent_situations(layer)
    forward = {item.game_id: item.vector for item in represent_games(reps)}
    backward = {item.game_id: item.vector for item in represent_games(reps[::-1])}
    assert forward == backward


def test_novelty_is_structural_and_multiscale() -> None:
    space = _space()
    rows = novelty_table(space, ks=(1, 3))
    ks = {row.k for row in rows}
    assert ks == {1, 3}
    one = novelty_score(space, "chess", 1)
    three = novelty_score(space, "chess", 3)
    assert one.k == 1
    assert three.k == 3
    assert one.structural_novelty == one.local_isolation
    assert one.nearest_distance == one.structural_novelty
    blob = json.dumps(one.model_dump(mode="python"))
    assert "quality" not in blob
    assert "originality" not in blob
    isolated = RepresentationTable(
        entity_ids=("dense-a", "dense-b", "far"),
        vectors=((0.0, 0.0), (0.1, 0.0), (8.0, 8.0)),
        dim=2,
        source_payload_ids=("dense-a", "dense-b", "far"),
    )
    scores = novelty_scores(isolated, k=1, metric="l2")
    by_id: dict[str, float] = {}
    for entity_id, row in zip(scores.entity_ids, scores.values, strict=True):
        value = row[0]
        assert isinstance(value, int | float) and not isinstance(value, bool)
        by_id[entity_id] = float(value)
    assert by_id["far"] > by_id["dense-a"]
    assert by_id["far"] > by_id["dense-b"]


def test_clustering_is_reproducible_and_configurable() -> None:
    space = _space()
    first = cluster_games(space, 3, seed=0)
    second = cluster_games(space, 3, seed=0)
    assert first == second
    assert {item.n_clusters for item in first} == {3}
    other_k = cluster_games(space, 2, seed=0)
    assert {item.n_clusters for item in other_k} == {2}
    table = view_table(space, level="game", view="normalized")
    clusterer = DeterministicKMeans(3, seed=0)
    clusterer.fit(table)
    assert clusterer.predict(table) == tuple(item.cluster for item in first)
    with pytest.raises(ValueError, match="not enough entities"):
        DeterministicKMeans(len(space.games) + 1).fit(table)
    with pytest.raises(ValueError, match="n_clusters"):
        DeterministicKMeans(0)


def test_metadata_does_not_enter_clusters() -> None:
    space = _space()
    attached = attach_external_metadata(all_situations())
    assert "title" in attached["chess"]
    assignments = cluster_games(space, 3, seed=0)
    blob = json.dumps([item.model_dump(mode="python") for item in assignments])
    assert "Chess" not in blob
    assert "title" not in blob
    assert attached["chess"]["title"] == "Chess"


def test_run_persists_and_stays_game_safe(tmp_path) -> None:
    store = LocalStore(tmp_path)
    result = run_design_space_experiment(
        all_situations(),
        store=store,
        run=_run(),
        split=_split(),
        dim=8,
        n_clusters=3,
        n_components=2,
        novelty_ks=(1, 3),
        seed=0,
    )
    assert set(result.evaluation.train_game_ids).isdisjoint(
        result.evaluation.test_game_ids
    )
    assert set(result.space.normalizer.train_entity_ids).isdisjoint(
        result.evaluation.test_game_ids
    )
    assert result.space.projection is not None
    assert set(result.space.projection.train_entity_ids).isdisjoint(
        result.evaluation.test_game_ids
    )
    assert result.evaluation.clusters.metrics["repeatable"] == 1.0
    keys = _logical_keys(tmp_path)
    assert SITUATION_REPR_LOGICAL_KEY in keys
    assert GAME_REPR_LOGICAL_KEY in keys
    assert DESIGN_SPACE_MODEL_LOGICAL_KEY in keys
    assert NORMALIZER_LOGICAL_KEY in keys
    assert PROJECTION_LOGICAL_KEY in keys
    assert NOVELTY_EVAL_LOGICAL_KEY in keys
    assert CLUSTER_EVAL_LOGICAL_KEY in keys
    kinds = _kind_by_key(tmp_path)
    assert kinds.get(GAME_REPR_LOGICAL_KEY) == ArtifactKind.DATASET
    assert kinds.get(NORMALIZER_LOGICAL_KEY) == ArtifactKind.MODEL
    situations = represent_situations(result.space.layer)
    assert [item.situation_id for item in situations] == [
        item.situation_id for item in result.space.situations
    ]


def test_bgg_metadata_corpus_is_not_used() -> None:
    space = _space()
    blob = json.dumps([item.to_mapping() for item in space.games])
    assert "bgg_boardgames" not in blob
    assert "CATAN" not in blob


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
