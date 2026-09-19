"""ds-platform boundary for normalized Game documents. No live HTTP."""

import json
from datetime import UTC, datetime
from pathlib import Path

from ds_platform import ArtifactKind, LocalStore, payload_id, record_id, sha256_hex

from board_game_analysis.config import Settings
from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.normalizer import normalize_bgg_artifact
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.document_artifacts import (
    default_game_schema_path,
    game_contract_ref,
    game_document_logical_key,
    store_game_document,
)
from board_game_analysis.ingestion.pipeline import (
    ingest_bgg_game,
    processed_game_json_bytes,
    write_processed_game,
)
from board_game_analysis.ingestion.raw_artifacts import (
    artifact_store,
    bga_run_context,
    raw_payload_bytes,
    store_raw_bgg_artifact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
RETRIEVED = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _xml_text() -> str:
    return (FIXTURES / "bgg_thing_13.xml").read_text(encoding="utf-8")


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
        }
    )


def _game_and_raw(tmp_path: Path) -> tuple[bytes, str, Game]:
    artifact = _artifact(_xml_text())
    raw_pid, _, _ = store_raw_bgg_artifact(
        LocalStore(tmp_path / "raw"),
        artifact,
        bga_run_context(started_at=RETRIEVED),
    )
    game = normalize_bgg_artifact(artifact)
    return processed_game_json_bytes(game), raw_pid, game


def test_identical_game_json_bytes_produce_identical_payload_id() -> None:
    artifact = _artifact(_xml_text())
    game = normalize_bgg_artifact(artifact)
    first = processed_game_json_bytes(game)
    second = processed_game_json_bytes(game)
    assert first == second
    assert payload_id(first) == payload_id(second)


def test_json_formatting_change_changes_document_payload_id() -> None:
    artifact = _artifact(_xml_text())
    game = normalize_bgg_artifact(artifact)
    written = processed_game_json_bytes(game)
    compact = json.dumps(game.model_dump(mode="json")).encode("utf-8")
    assert written != compact
    assert payload_id(written) != payload_id(compact)


def test_document_record_cites_raw_and_game_schema(tmp_path: Path) -> None:
    game_json, raw_pid, game = _game_and_raw(tmp_path)
    store = LocalStore(tmp_path / "cas")
    store.put(
        raw_pid,
        raw_payload_bytes(_artifact(_xml_text())),
        media_type="application/xml",
    )
    pid, _rid, record = store_game_document(
        store,
        game_json,
        raw_payload_id=raw_pid,
        run=bga_run_context(started_at=RETRIEVED),
        game_id=game.id,
    )
    assert record.kind is ArtifactKind.DOCUMENT
    assert list(record.inputs) == [raw_pid]
    assert record.payload_id == pid == payload_id(game_json)
    assert record.logical_key == game_document_logical_key("bgg-13")
    schema_path = default_game_schema_path()
    assert record.contract_ref is not None
    assert record.contract_ref == game_contract_ref(schema_path)
    assert record.contract_ref.schema_hash == sha256_hex(schema_path.read_bytes())
    assert record.contract_ref.uri == (
        "https://github.com/rychen1/board-game-analysis/schemas/game.schema.json"
    )
    assert store.get(raw_pid) == raw_payload_bytes(_artifact(_xml_text()))
    assert store.get(pid) == game_json


def test_schema_hash_change_changes_record_id_not_payload_id(tmp_path: Path) -> None:
    game_json, raw_pid, game = _game_and_raw(tmp_path)
    store = LocalStore(tmp_path / "cas")
    run = bga_run_context(started_at=RETRIEVED)
    _pid, first_rid, first = store_game_document(
        store,
        game_json,
        raw_payload_id=raw_pid,
        run=run,
        game_id=game.id,
    )
    other_schema = tmp_path / "other.schema.json"
    other_schema.write_bytes(default_game_schema_path().read_bytes() + b"\n")
    _pid2, second_rid, second = store_game_document(
        LocalStore(tmp_path / "other"),
        game_json,
        raw_payload_id=raw_pid,
        run=run,
        game_id=game.id,
        schema_path=other_schema,
    )
    assert first.payload_id == second.payload_id == payload_id(game_json)
    assert first_rid != second_rid
    assert record_id(first) != record_id(second)
    assert first.contract_ref is not None
    assert second.contract_ref is not None
    assert first.contract_ref.schema_hash != second.contract_ref.schema_hash


def test_document_record_has_no_game_fields_or_path(tmp_path: Path) -> None:
    game_json, raw_pid, game = _game_and_raw(tmp_path)
    _pid, _rid, record = store_game_document(
        LocalStore(tmp_path / "cas"),
        game_json,
        raw_payload_id=raw_pid,
        run=bga_run_context(started_at=RETRIEVED),
        game_id=game.id,
    )
    dumped = record.model_dump()
    assert "location" not in dumped
    assert "path" not in dumped
    for field in (
        "title",
        "release_year",
        "mechanics",
        "sources",
        "designers",
        "publishers",
        "categories",
        "min_players",
        "rating",
    ):
        assert field not in dumped


def test_pipeline_writes_document_and_keeps_processed_json(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    artifact = _artifact(_xml_text())
    RawArchive(tmp_path).save(artifact)
    game = ingest_bgg_game("13", settings)
    processed = tmp_path / "processed" / "boardgamegeek" / "13.json"
    assert processed.is_file()
    assert processed.read_bytes() == processed_game_json_bytes(game)
    raw = raw_payload_bytes(artifact)
    store = artifact_store(tmp_path)
    assert store.get(payload_id(raw)) == raw
    document_pid = payload_id(processed.read_bytes())
    assert store.get(document_pid) == processed.read_bytes()
    write_processed_game(tmp_path / "again", "13", game)
    assert (tmp_path / "again" / "processed" / "boardgamegeek" / "13.json").is_file()
