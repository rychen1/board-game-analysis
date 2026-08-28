"""Fixture-based BGG ingestion tests. No live HTTP."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from board_game_analysis.config import Settings
from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.client import BggClient
from board_game_analysis.ingestion.bgg.normalizer import normalize_bgg_artifact
from board_game_analysis.ingestion.bgg.parser import parse_thing_xml
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.errors import (
    BggAuthenticationError,
    BggNotFoundError,
)
from board_game_analysis.ingestion.pipeline import ingest_bgg_game

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
RETRIEVED = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
CATAN_URL = "https://boardgamegeek.com/xmlapi2/thing?id=13&stats=1"


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _artifact(body: str, source_id: str = "13") -> RawArtifact:
    return RawArtifact(
        source="boardgamegeek",
        source_identifier=source_id,
        retrieved_at=RETRIEVED,
        request_url=f"https://boardgamegeek.com/xmlapi2/thing?id={source_id}&stats=1",
        content_type="text/xml",
        http_status=200,
        body=body,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings.model_validate(
        {
            "data_dir": tmp_path,
            "bgg_token": "test-token",
            "bgg_rate_limit_seconds": 0,
            "bgg_max_retries": 3,
            "bgg_retry_backoff_seconds": 0,
        }
    )


def test_source_response_parses() -> None:
    thing = parse_thing_xml(_xml("bgg_thing_13.xml"))
    assert thing.bgg_id == "13"
    assert thing.primary_name == "CATAN"


def test_canonical_fields_populate() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert game.id == "bgg-13"
    assert game.title == "CATAN"
    assert game.release_year == 1995
    assert game.categories == ["Economic", "Negotiation"]
    assert [mechanic.name for mechanic in game.mechanics] == [
        "Dice Rolling",
        "Network and Route Building",
    ]


def test_missing_values_remain_missing() -> None:
    game = normalize_bgg_artifact(
        _artifact(_xml("bgg_thing_sparse.xml"), source_id="999001")
    )
    assert game.release_year is None
    assert game.min_players is None
    assert game.max_players is None
    assert game.min_play_time_minutes is None
    assert game.max_play_time_minutes is None
    assert game.rating is None
    assert game.complexity is None
    assert game.popularity is None
    assert game.designers == []
    assert game.publishers == []
    assert game.categories == []
    assert game.mechanics == []


def test_player_counts_normalize() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert game.min_players == 3
    assert game.max_players == 4


def test_play_times_normalize() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert game.min_play_time_minutes == 60
    assert game.max_play_time_minutes == 120


def test_rating_and_rating_count_remain_distinct() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert game.rating == 7.135
    assert game.rating_count == 128000
    sparse = normalize_bgg_artifact(
        _artifact(_xml("bgg_thing_sparse.xml"), source_id="999001")
    )
    assert sparse.rating is None
    assert sparse.rating_count == 0


def test_designers_and_publishers_normalize() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert game.designers == ["Klaus Teuber"]
    assert game.publishers == ["KOSMOS", "Catan Studio"]


def test_source_provenance_is_retained() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert len(game.sources) == 1
    source = game.sources[0]
    assert source.source == "boardgamegeek"
    assert source.source_type == "xmlapi2"
    assert source.source_identifier == "13"
    assert source.url == CATAN_URL
    assert source.retrieved_at == RETRIEVED


def test_normalization_is_deterministic() -> None:
    body = _xml("bgg_thing_13.xml")
    first = normalize_bgg_artifact(_artifact(body))
    second = normalize_bgg_artifact(_artifact(body))
    assert first.model_dump() == second.model_dump()


def test_empty_items_is_not_found() -> None:
    with pytest.raises(BggNotFoundError):
        parse_thing_xml(_xml("bgg_thing_empty.xml"))


def test_missing_token_does_not_call_http() -> None:
    settings = Settings.model_validate({"bgg_token": None, "bgg_rate_limit_seconds": 0})

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http:
        client = BggClient(settings, http_client=http)
        with pytest.raises(BggAuthenticationError):
            client.fetch_game("13")


def test_retries_on_accepted_then_succeeds() -> None:
    settings = _settings(Path("/tmp"))
    body = _xml("bgg_thing_13.xml")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(202, text="queued")
        return httpx.Response(200, text=body, headers={"content-type": "text/xml"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        artifact = client.fetch_game("13")
    assert calls["n"] == 2
    assert parse_thing_xml(artifact.body).bgg_id == "13"


def test_client_parses_mocked_http() -> None:
    settings = _settings(Path("/tmp"))
    body = _xml("bgg_thing_13.xml")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        assert "board-game-analysis" in request.headers["User-Agent"]
        assert str(request.url).endswith("/thing?id=13&stats=1")
        return httpx.Response(200, text=body, headers={"content-type": "text/xml"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        artifact = client.fetch_game("13")
    assert artifact.source_identifier == "13"
    assert parse_thing_xml(artifact.body).primary_name == "CATAN"


def test_pipeline_uses_raw_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml")))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="should not be called")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        game = ingest_bgg_game("13", settings, client=client)
    assert calls["n"] == 0
    assert game.title == "CATAN"
    processed = tmp_path / "processed" / "boardgamegeek" / "13.json"
    assert processed.is_file()
    loaded = Game.model_validate_json(processed.read_text(encoding="utf-8"))
    assert loaded.title == "CATAN"


def test_categories_are_not_interpretations() -> None:
    game = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml")))
    assert "Negotiation" in game.categories
    assert not hasattr(game, "interpretations")
    dumped = game.model_dump()
    assert "cooperative" not in dumped
    assert "hidden_information" not in dumped
