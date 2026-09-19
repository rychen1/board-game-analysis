"""Unit tests for corpus-quality helpers. Do not require the live JSONL."""

from datetime import UTC, datetime
from pathlib import Path

from board_game_analysis.domain.game import Game
from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.domain.source import SourceReference
from board_game_analysis.ingestion.quality import (
    CorpusLoad,
    flag_anomalies,
    integrity_report,
    list_length_summaries,
    load_games_jsonl,
    missingness_rows,
    numeric_summaries,
)


def _source(identifier: str = "13") -> SourceReference:
    return SourceReference(
        source="boardgamegeek",
        source_type="xmlapi2",
        url=f"https://boardgamegeek.com/xmlapi2/thing?id={identifier}&stats=1",
        source_identifier=identifier,
        retrieved_at=datetime(2026, 9, 19, 18, 0, tzinfo=UTC),
    )


def _game(**overrides: object) -> Game:
    payload: dict[str, object] = {
        "id": "bgg-13",
        "title": "Catan",
        "release_year": 1995,
        "min_players": 3,
        "max_players": 4,
        "min_play_time_minutes": 60,
        "max_play_time_minutes": 120,
        "popularity": None,
        "rating": 7.1,
        "rating_count": 100,
        "complexity": 2.3,
        "designers": ["Klaus Teuber"],
        "publishers": ["KOSMOS"],
        "categories": ["Economic"],
        "mechanics": [Mechanic(id="bgg-2072", name="Dice Rolling")],
        "sources": [_source()],
    }
    payload.update(overrides)
    return Game.model_validate(payload)


def test_load_games_jsonl_keeps_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    good = _game().model_dump_json()
    path.write_text(
        "\n".join([good, "{", '{"id": "bad", "title": ""}']) + "\n",
        encoding="utf-8",
    )
    loaded = load_games_jsonl(path)
    assert len(loaded.games) == 1
    assert len(loaded.errors) == 2
    assert loaded.errors[0].line_number == 2
    assert loaded.errors[1].raw_id == "bad"


def test_integrity_and_missingness_distinguish_empty_lists() -> None:
    empty_lists = _game(
        id="bgg-2",
        title="Sparse",
        designers=[],
        publishers=[],
        categories=[],
        mechanics=[],
        sources=[_source("2")],
    )
    loaded_games = [_game(), empty_lists]
    rows = {row.field: row for row in missingness_rows(loaded_games)}
    assert rows["popularity"].missing_null == 2
    assert rows["popularity"].pct_missing_null == 100.0
    assert rows["designers"].missing_null == 0
    assert rows["designers"].empty_list == 1
    assert rows["designers"].nonempty_list == 1
    report = integrity_report(
        CorpusLoad(games=loaded_games, errors=[], raw_objects=[{}, {}]),
        manifest={
            "requested_ids": ["13", "2"],
            "ok": [
                {"game_id": "bgg-13", "item_type": "boardgame"},
                {"game_id": "bgg-2", "item_type": "boardgame"},
            ],
            "skipped": [],
            "errors": [],
        },
    )
    assert report.n_jsonl_records == 2
    assert report.n_unique_ids == 2
    assert report.duplicate_ids == []
    assert report.n_manifest_ok == 2
    assert report.provenance_issues == []
    assert report.anomaly_counts == {}
    assert report.anomaly_examples == []
    assert report.empty_taxonomy == ["bgg-2: Sparse"]
    assert report.duplicate_mechanic_ids == []
    assert report.non_boardgame_ok_rows == []


def test_numeric_and_list_summaries() -> None:
    games = [
        _game(),
        _game(
            id="bgg-3",
            title="Other",
            rating=8.0,
            rating_count=10,
            publishers=["A", "B"],
        ),
    ]
    numeric = {row.field: row for row in numeric_summaries(games)}
    assert numeric["rating"].count == 2
    assert numeric["rating"].minimum == 7.1
    assert numeric["rating"].maximum == 8.0
    lengths = {row.field: row for row in list_length_summaries(games)}
    assert lengths["publishers"].minimum == 1
    assert lengths["publishers"].maximum == 2


def test_flag_anomalies_are_heuristic_only() -> None:
    flagged = _game(
        id="bgg-9",
        title="Weird",
        release_year=1500,
        rating=12.0,
        rating_count=0,
        publishers=["p"] * 30,
    )
    flags = flag_anomalies([_game(), flagged], current_year=2026)
    codes = {item.code for item in flags}
    assert "implausible_year" in codes
    assert "rating_off_scale" in codes
    assert "rating_without_count" in codes
    assert "large_publisher_list" in codes
    assert all(item.game_id == "bgg-9" for item in flags)
    persisted = integrity_report(
        CorpusLoad(games=[_game(), flagged], errors=[], raw_objects=[{}, {}]),
        current_year=2026,
    )
    assert persisted.anomaly_counts["implausible_year"] == 1
    assert persisted.anomaly_counts["rating_off_scale"] == 1
    assert persisted.anomaly_examples
    assert persisted.anomaly_examples[0].game_id == "bgg-9"


def test_integrity_detects_taxonomy_mechanic_and_item_type_issues() -> None:
    empty = _game(
        id="bgg-2",
        title="Sparse",
        categories=[],
        mechanics=[],
        sources=[_source("2")],
    )
    duped = _game(
        id="bgg-3",
        title="Duped",
        mechanics=[
            Mechanic(id="bgg-2072", name="Dice Rolling"),
            Mechanic(id="bgg-2072", name="Dice Rolling again"),
        ],
        sources=[_source("3")],
    )
    report = integrity_report(
        CorpusLoad(games=[empty, duped], errors=[], raw_objects=[{}, {}]),
        manifest={
            "ok": [
                {
                    "source_id": "2",
                    "game_id": "bgg-2",
                    "item_type": "boardgame",
                },
                {
                    "source_id": "99",
                    "game_id": "bgg-99",
                    "item_type": "boardgameexpansion",
                },
            ]
        },
        current_year=2026,
    )
    assert report.empty_taxonomy == ["bgg-2: Sparse"]
    assert report.duplicate_mechanic_ids == ["bgg-3: bgg-2072"]
    assert report.non_boardgame_ok_rows == ["99: boardgameexpansion"]
