"""Fixture tests for corpus-descriptive robustness checks."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ds_platform import ArtifactKind, payload_id

from board_game_analysis.analysis.artifacts import report_json_bytes
from board_game_analysis.analysis.descriptive import build_rows, game_row
from board_game_analysis.analysis.robustness import (
    EXTREME_MAX_PLAYERS,
    EXTREME_PLAYER_SPAN,
    build_robustness_tables,
    extreme_player_games,
    log_positive,
    mechanic_robustness,
    player_interval_analysis,
    player_midpoint,
    time_weight_sensitivity,
)
from board_game_analysis.analysis.robustness_report import (
    build_robustness_report,
    render_robustness_findings,
)
from board_game_analysis.analysis.run import run_corpus_robustness
from board_game_analysis.config import Settings
from board_game_analysis.domain.game import Game
from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.ingestion.raw_artifacts import artifact_store


def _mechanic(name: str) -> Mechanic:
    slug = name.lower().replace(" ", "-")
    return Mechanic(id=f"bgg-{slug}", name=name)


def _game(**overrides: object) -> Game:
    payload: dict[str, object] = {
        "id": "bgg-1",
        "title": "Fixture",
        "release_year": 2010,
        "min_players": 2,
        "max_players": 4,
        "min_play_time_minutes": 30,
        "max_play_time_minutes": 45,
        "complexity": 2.0,
        "mechanics": [_mechanic("Hand Management")],
    }
    payload.update(overrides)
    return Game.model_validate(payload)


def _fixture_games() -> list[Game]:
    return [
        _game(
            id="bgg-a",
            title="Duel",
            min_players=2,
            max_players=2,
            min_play_time_minutes=30,
            max_play_time_minutes=30,
            complexity=1.5,
            mechanics=[_mechanic("Hand Management"), _mechanic("Duel")],
        ),
        _game(
            id="bgg-b",
            title="Duel Long",
            min_players=2,
            max_players=2,
            min_play_time_minutes=45,
            max_play_time_minutes=60,
            complexity=2.0,
            mechanics=[_mechanic("Hand Management")],
        ),
        _game(
            id="bgg-c",
            title="Family",
            min_players=2,
            max_players=4,
            min_play_time_minutes=60,
            max_play_time_minutes=90,
            complexity=2.8,
            mechanics=[_mechanic("Hand Management"), _mechanic("Drafting")],
        ),
        _game(
            id="bgg-d",
            title="Party",
            min_players=3,
            max_players=8,
            min_play_time_minutes=15,
            max_play_time_minutes=15,
            complexity=1.1,
            mechanics=[_mechanic("Duel"), _mechanic("Set Collection")],
        ),
        _game(
            id="bgg-e",
            title="Rare Mech",
            min_players=2,
            max_players=4,
            min_play_time_minutes=40,
            max_play_time_minutes=40,
            complexity=2.2,
            mechanics=[_mechanic("Rare Label")],
        ),
        _game(
            id="bgg-f",
            title="No Weight",
            min_players=2,
            max_players=4,
            min_play_time_minutes=40,
            max_play_time_minutes=40,
            complexity=None,
            mechanics=[_mechanic("Hand Management")],
        ),
        _game(
            id="bgg-g",
            title="No Time",
            min_players=2,
            max_players=2,
            min_play_time_minutes=None,
            max_play_time_minutes=None,
            complexity=2.2,
            mechanics=[_mechanic("Hand Management")],
        ),
        _game(
            id="bgg-h",
            title="Epic",
            min_players=2,
            max_players=5,
            min_play_time_minutes=120,
            max_play_time_minutes=180,
            complexity=4.0,
            mechanics=[_mechanic("Drafting")],
        ),
    ]


def _extreme_games() -> list[Game]:
    base = _fixture_games()
    base.append(
        _game(
            id="bgg-cart",
            title="Cartographers",
            min_players=1,
            max_players=100,
            min_play_time_minutes=30,
            max_play_time_minutes=45,
            complexity=1.8,
            mechanics=[_mechanic("Hand Management")],
        )
    )
    return base


def test_player_midpoint_is_labeled_proxy() -> None:
    assert player_midpoint(2, 4) == 3.0
    assert player_midpoint(None, 4) is None
    assert player_midpoint(1, 100) == 50.5


def test_log_positive_requires_positive_values() -> None:
    assert log_positive(60) is not None
    assert log_positive(0) is None
    assert log_positive(-5) is None
    assert log_positive(None) is None


def test_time_weight_sensitivity_reports_all_representations() -> None:
    rows = build_rows(_extreme_games())
    payload = cast(dict[str, Any], time_weight_sensitivity(rows))
    raw = [item for item in payload["correlations"] if item.get("transform") == "raw"]
    assert len(raw) == 4
    attrs = {str(item["x"]) for item in raw}
    assert attrs == {
        "min_play_time_minutes",
        "max_play_time_minutes",
        "play_time_midpoint",
        "play_time_span",
    }
    for item in raw:
        assert int(item["n"]) >= 1
        assert "missing_x" in item
        assert "missing_y" in item


def test_extreme_player_games_flags_without_removing() -> None:
    rows = build_rows(_extreme_games())
    flagged = extreme_player_games(rows)
    cart = next(item for item in flagged if item["game_id"] == "bgg-cart")
    assert cart["max_players"] == 100
    assert "max_players=100" in str(cart["reason"])


def test_player_interval_subset_excludes_extreme_max_players() -> None:
    rows = build_rows(_extreme_games())
    payload = cast(dict[str, Any], player_interval_analysis(rows))
    full_n = len(rows)
    subset = payload["without_extreme_max_players"]
    assert int(subset["n"]) < full_n
    assert int(subset["n"]) == sum(
        1
        for row in rows
        if row.max_players is not None and row.max_players <= EXTREME_MAX_PLAYERS
    )


def test_player_interval_subset_excludes_extreme_span() -> None:
    rows = build_rows(_extreme_games())
    payload = cast(dict[str, Any], player_interval_analysis(rows))
    subset = payload["without_extreme_player_span"]
    assert int(subset["n"]) == sum(
        1
        for row in rows
        if row.player_span is not None and row.player_span <= EXTREME_PLAYER_SPAN
    )


def test_mechanic_threshold_survival_counts_labels() -> None:
    rows = build_rows(_fixture_games())
    payload = cast(dict[str, Any], mechanic_robustness(rows))
    survival = {
        int(item["min_games"]): int(item["n_labels"])
        for item in payload["threshold_survival"]
    }
    assert survival[1] == payload["n_distinct_labels"]
    assert survival[5] <= survival[3] <= survival[1]
    assert payload["n_labels_occurrence_1"] >= 1


def test_mechanic_lift_is_threshold_sensitive() -> None:
    rows = build_rows(_fixture_games())
    payload = cast(dict[str, Any], mechanic_robustness(rows))
    by_threshold = {
        int(item["min_games"]): int(item["n_labels_compared"])
        for item in payload["lift_by_threshold"]
    }
    assert by_threshold[20] <= by_threshold[10] <= by_threshold[5] <= by_threshold[3]


def test_robustness_tables_are_deterministic() -> None:
    games = _extreme_games()
    assert build_robustness_tables(games) == build_robustness_tables(games)


def test_leave_one_out_quantifies_cartographers_influence() -> None:
    rows = build_rows(_extreme_games())
    payload = cast(dict[str, Any], time_weight_sensitivity(rows))
    loo = cast(list[dict[str, Any]], payload["leave_one_out"])
    cart_rows = [item for item in loo if item["game_id"] == "bgg-cart"]
    assert cart_rows
    for item in cart_rows:
        assert item["n_full"] is not None
        assert item["n_without"] is not None
        assert int(item["n_without"]) == int(item["n_full"]) - 1


def test_robustness_report_and_artifacts(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus_v1.jsonl"
    games = _extreme_games()
    lines = [game.model_dump_json() for game in games]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = Settings.model_validate({"data_dir": tmp_path})
    result = run_corpus_robustness(
        settings,
        corpus_id="bgg_boardgames_v1",
        jsonl_path=jsonl,
        output_dir=tmp_path / "robustness",
    )
    assert result.corpus_payload_id == payload_id(jsonl.read_bytes())
    report = result.report_path.read_bytes()
    assert result.report_payload_id == payload_id(report)
    store = artifact_store(tmp_path)
    record = _record_for_payload(store, result.report_payload_id)
    assert record is not None
    assert record.kind is ArtifactKind.DATASET
    assert record.inputs == [result.corpus_payload_id]
    assert record.logical_key == "bga:analysis/corpus-descriptive-robustness:v1"
    findings = result.findings_path.read_text(encoding="utf-8")
    assert "Robust observations" in findings
    assert "Representation-sensitive" in findings
    assert "Sparse-data" in findings
    assert len(result.figure_paths) == 6
    assert all(path.is_file() for path in result.figure_paths)
    frozen = datetime(2026, 1, 1, tzinfo=UTC)
    tables = build_robustness_tables(games)
    payload = build_robustness_report(
        games,
        tables,
        corpus_id="bgg_boardgames_v1",
        jsonl_name="corpus_v1.jsonl",
        corpus_payload_id=result.corpus_payload_id,
        started_at=frozen,
        load_errors=0,
    )
    assert report_json_bytes(payload) == report_json_bytes(payload)
    assert "Play time vs weight" in render_robustness_findings(payload)


def _record_for_payload(store, payload_id_value: str):
    from ds_platform import ArtifactRecord, LocalStore

    root = cast(LocalStore, store)._root  # noqa: SLF001
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        try:
            record = ArtifactRecord.model_validate_json(path.read_bytes())
        except (ValueError, KeyError, TypeError):
            continue
        if record.payload_id == payload_id_value:
            return record
    return None


def test_game_row_span_matches_interval() -> None:
    row = game_row(
        _game(
            min_players=1,
            max_players=100,
            min_play_time_minutes=30,
            max_play_time_minutes=90,
        )
    )
    assert row.player_span == 99
    assert row.play_time_span == 60
