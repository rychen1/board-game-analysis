"""Regression tests for reproducibility audit fixes R-C1, R-H1, R-H2, R-H3."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from ds_platform import Environment, LocalStore, RunContext
from ds_platform.modeling.evaluate import evaluate, rmse
from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import MetricSpec, SplitSpec

from board_game_analysis.modeling.design_geometry import (
    DesignSpaceManifest,
    build_design_space,
    run_design_space_experiment,
)
from board_game_analysis.modeling.design_space import (
    build_phase6_context,
    build_play_layer,
    represent_situations,
)
from board_game_analysis.modeling.higher_order import higher_order_table
from board_game_analysis.modeling.scalar_eval import (
    scalar_predictions_from_representations,
    zero_change_scalar_table,
)
from board_game_analysis.modeling.split import split_entities_by_game
from board_game_analysis.modeling.transitions import (
    SCALAR_COLUMNS,
    evaluate_transitions,
)
from tests.fixtures.games import all_situations

_SOURCE = "a" * 64
_SPLIT = SplitSpec(method="holdout", seed=0, test_size=0.3)


def _run() -> RunContext:
    return RunContext(
        run_id="repro-fix",
        project="bga",
        started_at=datetime(2026, 9, 19, 21, 0, tzinfo=UTC),
        environment=Environment.LOCAL,
    )


def _valid_payload_id(payload_id: str) -> None:
    assert len(payload_id) == 64
    assert payload_id.islower()
    assert all(ch in "0123456789abcdef" for ch in payload_id)


def _scalar_table(
    entity_ids: tuple[str, ...],
    values: tuple[tuple[float, ...], ...],
) -> FeatureTable:
    return FeatureTable(
        entity_ids=entity_ids,
        columns=SCALAR_COLUMNS[:1],
        values=tuple((value[0],) for value in values),
        source_payload_ids=tuple(_SOURCE for _ in entity_ids),
    )


def test_phase7_tables_use_encoder_payload_lineage() -> None:
    situations = all_situations()
    layer = build_play_layer(situations, dim=8)
    phase6 = build_phase6_context(layer)
    space = build_design_space(
        situations,
        train_game_ids=("chess",),
        test_game_ids=("go",),
        dim=8,
    )
    for table in (
        space.canonical_situations,
        space.canonical_games,
        space.normalized_situations,
        space.normalized_games,
    ):
        for payload_id in table.source_payload_ids:
            _valid_payload_id(payload_id)
    situations_repr = represent_situations(layer, phase6=phase6)
    for item in situations_repr:
        _valid_payload_id(item.source_payload_id)
        assert item.source_payload_id not in (item.entity_id, item.game_id)
    hop = higher_order_table(phase6.higher_order_pairs)
    for payload_id in hop.source_payload_ids:
        _valid_payload_id(payload_id)


def test_split_entities_by_game_is_order_invariant() -> None:
    situations = all_situations()
    forward_games = [situation.game.id for situation in situations]
    reverse_games = list(reversed(forward_games))
    entity_ids = [f"e{index}" for index in range(len(forward_games))]
    forward = split_entities_by_game(entity_ids, forward_games, _SPLIT)
    reverse = split_entities_by_game(entity_ids, reverse_games, _SPLIT)
    forward_train_games = {
        game
        for entity, game in zip(entity_ids, forward_games, strict=True)
        if entity in forward.train_ids
    }
    reverse_train_games = {
        game
        for entity, game in zip(entity_ids, reverse_games, strict=True)
        if entity in reverse.train_ids
    }
    assert forward_train_games == reverse_train_games


def test_split_entities_by_game_is_stable_under_python_hash_seed() -> None:
    script = """
from ds_platform.modeling.spec import SplitSpec
from board_game_analysis.modeling.split import split_entities_by_game

entity_ids = [f"e{index}" for index in range(6)]
game_ids = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
spec = SplitSpec(method="holdout", seed=0, test_size=0.3)
assignment = split_entities_by_game(entity_ids, game_ids, spec)
train_games = sorted(
    {
        game
        for entity, game in zip(entity_ids, game_ids)
        if entity in assignment.train_ids
    }
)
print(",".join(train_games))
"""
    outputs = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            cwd="/home/r-chen/Projects/board-game-analysis",
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("0", "1", "424242")
    }
    assert len(set(outputs)) == 1


def test_design_space_manifest_records_audit_fields(tmp_path) -> None:
    store = LocalStore(tmp_path)
    result = run_design_space_experiment(
        all_situations(),
        store=store,
        run=_run(),
        split=_SPLIT,
        dim=8,
        n_clusters=3,
        seed=0,
    )
    manifest = pickle.loads(store.get(result.model_payload_id))
    assert isinstance(manifest, DesignSpaceManifest)
    assert manifest.split_spec == _SPLIT
    assert manifest.config_hash
    assert manifest.input_examples_payload_id
    assert manifest.novelty_eval_payload_id == result.novelty_payload_id
    assert manifest.cluster_eval_payload_id == result.cluster_payload_id
    assert manifest.software_versions
    assert any(name == "python" for name, _version in manifest.software_versions)


def test_scalar_eval_perfect_model_matches_hand_computed_rmse() -> None:
    train_predicted = RepresentationTable(
        entity_ids=("t1", "t2"),
        vectors=((1.0,), (2.0,)),
        dim=1,
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    test_predicted = RepresentationTable(
        entity_ids=("e1",),
        vectors=((3.0,),),
        dim=1,
        source_payload_ids=(_SOURCE,),
    )
    train_scalar = _scalar_table(("t1", "t2"), ((2.0,), (4.0,)))
    test_scalar = _scalar_table(("e1",), ((6.0,),))
    scalar_pred = scalar_predictions_from_representations(
        train_predicted,
        test_predicted,
        train_scalar,
        test_scalar,
        ridge=1e-8,
    )
    report = evaluate(
        [6.0],
        [scalar_pred.values[0][0]],
        (MetricSpec(name="rmse"),),
    )
    assert report.metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_scalar_eval_wrong_model_has_high_error() -> None:
    train_predicted = RepresentationTable(
        entity_ids=("t1", "t2"),
        vectors=((1.0,), (2.0,)),
        dim=1,
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    test_predicted = RepresentationTable(
        entity_ids=("e1",),
        vectors=((-50.0,),),
        dim=1,
        source_payload_ids=(_SOURCE,),
    )
    train_scalar = _scalar_table(("t1", "t2"), ((2.0,), (4.0,)))
    test_scalar = _scalar_table(("e1",), ((6.0,),))
    scalar_pred = scalar_predictions_from_representations(
        train_predicted,
        test_predicted,
        train_scalar,
        test_scalar,
        ridge=1e-8,
    )
    pred_value = scalar_pred.values[0][0]
    assert isinstance(pred_value, float)
    assert abs(pred_value - 6.0) > 1.0


def test_zero_change_baseline_scalar_semantics() -> None:
    true_table = _scalar_table(("e1", "e2"), ((0.0,), (5.0,)))
    zero_pred = zero_change_scalar_table(true_table)
    perfect = evaluate([0.0], [0.0], (MetricSpec(name="rmse"),))
    wrong = evaluate([5.0], [0.0], (MetricSpec(name="rmse"),))
    assert perfect.metrics["rmse"] == pytest.approx(0.0)
    assert wrong.metrics["rmse"] == pytest.approx(5.0)
    assert rmse([5.0], [zero_pred.values[1][0]]) == pytest.approx(5.0)


def test_evaluate_transitions_scalar_block_uses_model_predictions() -> None:
    targets = RepresentationTable(
        entity_ids=("p1",),
        vectors=((1.0, 0.0),),
        dim=2,
        source_payload_ids=(_SOURCE,),
    )
    predicted = RepresentationTable(
        entity_ids=("p1",),
        vectors=((1.0, 0.0),),
        dim=2,
        source_payload_ids=(_SOURCE,),
    )
    copy = RepresentationTable(
        entity_ids=("p1",),
        vectors=((0.0, 1.0),),
        dim=2,
        source_payload_ids=(_SOURCE,),
    )
    train_predicted = RepresentationTable(
        entity_ids=("t1", "t2"),
        vectors=((2.0,), (3.0,)),
        dim=1,
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    test_predicted = RepresentationTable(
        entity_ids=("p1",),
        vectors=((5.0,),),
        dim=1,
        source_payload_ids=(_SOURCE,),
    )
    train_scalar = FeatureTable(
        entity_ids=("t1", "t2"),
        columns=("information_gain",),
        values=((4.0,), (6.0,)),
        source_payload_ids=(_SOURCE, _SOURCE),
    )
    test_scalar = FeatureTable(
        entity_ids=("p1",),
        columns=("information_gain",),
        values=((10.0,),),
        source_payload_ids=(_SOURCE,),
    )
    scalar_pred = scalar_predictions_from_representations(
        train_predicted,
        test_predicted,
        train_scalar,
        test_scalar,
        ridge=1e-8,
    )
    report = evaluate_transitions(
        predicted,
        copy,
        targets,
        scalar_true=test_scalar,
        scalar_pred=scalar_pred,
    )
    assert "information_gain" in report.scalars
    rmse_value = report.scalars["information_gain"].metrics["rmse"]
    assert rmse_value == pytest.approx(0.0, abs=1e-5)
    probe_note = "scalar metrics score train-fit ridge probes"
    assert probe_note in report.representation.notes[2]
