"""Fixture tests for corpus-descriptive transforms. No live JSONL or HTTP."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ds_platform import ArtifactKind, ArtifactRecord, LocalStore, payload_id

from board_game_analysis.analysis.artifacts import report_json_bytes
from board_game_analysis.analysis.descriptive import (
    MIN_MECHANIC_FOR_PROFILE,
    build_tables,
    game_row,
    play_time_kind,
    play_time_midpoint,
    player_profile,
    spearman_rho,
)
from board_game_analysis.analysis.report import build_report, render_findings
from board_game_analysis.analysis.run import run_corpus_descriptive
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
            title="Duel Light",
            min_players=2,
            max_players=2,
            min_play_time_minutes=20,
            max_play_time_minutes=20,
            complexity=1.2,
            mechanics=[_mechanic("Hand Management"), _mechanic("Set Collection")],
        ),
        _game(
            id="bgg-d",
            title="Family",
            min_players=2,
            max_players=4,
            min_play_time_minutes=60,
            max_play_time_minutes=90,
            complexity=2.8,
            mechanics=[_mechanic("Hand Management"), _mechanic("Drafting")],
        ),
        _game(
            id="bgg-e",
            title="Family Solo",
            min_players=1,
            max_players=4,
            min_play_time_minutes=30,
            max_play_time_minutes=45,
            complexity=2.1,
            mechanics=[_mechanic("Hand Management"), _mechanic("Duel")],
        ),
        _game(
            id="bgg-f",
            title="Family Heavy",
            min_players=2,
            max_players=4,
            min_play_time_minutes=90,
            max_play_time_minutes=120,
            complexity=3.5,
            mechanics=[_mechanic("Drafting")],
        ),
        _game(
            id="bgg-g",
            title="Party",
            min_players=2,
            max_players=6,
            min_play_time_minutes=45,
            max_play_time_minutes=90,
            complexity=2.4,
            mechanics=[_mechanic("Duel")],
        ),
        _game(
            id="bgg-h",
            title="Party Light",
            min_players=3,
            max_players=8,
            min_play_time_minutes=15,
            max_play_time_minutes=15,
            complexity=1.1,
            mechanics=[_mechanic("Duel"), _mechanic("Set Collection")],
        ),
        _game(
            id="bgg-i",
            title="Unknown Players",
            min_players=None,
            max_players=None,
            min_play_time_minutes=30,
            max_play_time_minutes=30,
            complexity=2.0,
            mechanics=[_mechanic("Hand Management")],
        ),
        _game(
            id="bgg-j",
            title="No Weight",
            min_players=2,
            max_players=4,
            min_play_time_minutes=40,
            max_play_time_minutes=40,
            complexity=None,
            mechanics=[_mechanic("Rare Label")],
        ),
        _game(
            id="bgg-k",
            title="No Time",
            min_players=2,
            max_players=2,
            min_play_time_minutes=None,
            max_play_time_minutes=None,
            complexity=2.2,
            mechanics=[_mechanic("Hand Management")],
        ),
        _game(
            id="bgg-l",
            title="Epic",
            min_players=2,
            max_players=5,
            min_play_time_minutes=120,
            max_play_time_minutes=180,
            complexity=4.0,
            mechanics=[_mechanic("Drafting")],
        ),
    ]


def test_player_profile_partitions_reported_interval() -> None:
    assert player_profile(2, 2) == "two_only"
    assert player_profile(1, 4) == "upto_four"
    assert player_profile(2, 4) == "upto_four"
    assert player_profile(3, 4) == "upto_four"
    assert player_profile(2, 5) == "five_plus"
    assert player_profile(1, 8) == "five_plus"
    assert player_profile(None, 4) == "unknown"
    assert player_profile(2, None) == "unknown"


def test_play_time_preserves_range_and_labels_midpoint() -> None:
    assert play_time_kind(30, 30) == "point"
    assert play_time_kind(30, 45) == "range"
    assert play_time_kind(None, 45) == "unknown"
    assert play_time_midpoint(30, 90) == 60.0
    row = game_row(_game(min_play_time_minutes=30, max_play_time_minutes=90))
    assert row.play_time_span == 60
    assert row.play_time_midpoint == 60.0
    assert row.play_time_kind == "range"


def test_spearman_is_monotone_without_pvalue() -> None:
    xs = [float(value) for value in range(1, 10)]
    assert spearman_rho(xs, xs) == 1.0
    assert spearman_rho(xs, list(reversed(xs))) == -1.0
    assert spearman_rho([1.0, 2.0], [2.0, 1.0]) is None


def test_mechanic_counts_are_multilabel() -> None:
    tables = build_tables(_fixture_games())
    mechanics = cast(dict[str, Any], tables["mechanics"])
    assert mechanics["n_empty_mechanic_list"] == 0
    prevalence = {item["name"]: item["games"] for item in mechanics["prevalence"]}
    assert prevalence["Hand Management"] == 7
    assert prevalence["Duel"] == 4
    assert prevalence["Rare Label"] == 1
    shares = sum(item["share_of_games"] for item in mechanics["prevalence"])
    assert shares > 1.0


def test_rare_mechanics_are_excluded_from_profile_lift() -> None:
    tables = build_tables(_fixture_games())
    mechanics = cast(dict[str, Any], tables["mechanics"])
    lifts = mechanics["profile_lifts"]
    names = {item["name"] for item in lifts}
    assert "Rare Label" not in names
    assert "Hand Management" in names
    hand = next(item for item in lifts if item["name"] == "Hand Management")
    assert hand["games"] >= MIN_MECHANIC_FOR_PROFILE
    two_only = hand["by_profile"]["two_only"]
    assert two_only["games_with_mechanic"] >= 1
    assert two_only["lift"] is not None


def test_missing_fields_are_counted_not_imputed() -> None:
    tables = build_tables(_fixture_games())
    coverage_rows = cast(list[dict[str, Any]], tables["coverage"])
    coverage = {row["field"]: row for row in coverage_rows}
    assert coverage["popularity"]["missing_null"] == 12
    assert coverage["complexity"]["missing_null"] == 1
    assert coverage["min_play_time_minutes"]["missing_null"] == 1
    profiles = cast(dict[str, Any], tables["player_profiles"])
    assert profiles["counts"]["unknown"] == 1
    complexity = cast(dict[str, Any], tables["complexity"])["overall"]
    assert complexity["n"] == 11
    assert complexity["n_missing"] == 1


def test_tables_are_deterministic_for_identical_games() -> None:
    first = build_tables(_fixture_games())
    second = build_tables(_fixture_games())
    assert first == second


def test_report_and_artifacts_cite_corpus_payload(tmp_path: Path) -> None:
    jsonl = tmp_path / "corpus_v1.jsonl"
    lines = [game.model_dump_json() for game in _fixture_games()]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = Settings.model_validate({"data_dir": tmp_path})
    result = run_corpus_descriptive(
        settings,
        corpus_id="bgg_boardgames_v1",
        jsonl_path=jsonl,
        output_dir=tmp_path / "out",
    )
    assert result.corpus_payload_id == payload_id(jsonl.read_bytes())
    report = result.report_path.read_bytes()
    assert result.report_payload_id == payload_id(report)
    store = artifact_store(tmp_path)
    assert store.get(result.corpus_payload_id) == jsonl.read_bytes()
    assert store.get(result.report_payload_id) == report
    record = _record_for_payload(store, result.report_payload_id)
    assert record is not None
    assert record.kind is ArtifactKind.DATASET
    assert record.inputs == [result.corpus_payload_id]
    assert record.logical_key == "bga:analysis/corpus-descriptive:v1"
    assert record.contract_ref is not None
    findings = result.findings_path.read_text(encoding="utf-8")
    assert "convenience / coverage corpus" in findings
    assert "causal" in findings.lower()
    assert len(result.figure_paths) == 5
    assert all(path.is_file() for path in result.figure_paths)
    frozen = datetime(2026, 1, 1, tzinfo=UTC)
    tables = build_tables(_fixture_games())
    payload = build_report(
        _fixture_games(),
        tables,
        corpus_id="bgg_boardgames_v1",
        jsonl_name="corpus_v1.jsonl",
        corpus_payload_id=result.corpus_payload_id,
        started_at=frozen,
        load_errors=0,
    )
    assert report_json_bytes(payload) == report_json_bytes(payload)
    assert "Q1" in render_findings(payload)


def _record_for_payload(
    store: LocalStore, payload_id_value: str
) -> ArtifactRecord | None:
    root = store._root  # noqa: SLF001
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
