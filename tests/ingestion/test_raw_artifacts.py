"""ds-platform boundary for raw BGG artifacts. No live HTTP."""

from datetime import UTC, datetime
from pathlib import Path

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    LocalStore,
    canonical_json_bytes,
    payload_id,
    record_id,
    sha256_hex,
)

from board_game_analysis.config import Settings
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.pipeline import ingest_bgg_game
from board_game_analysis.ingestion.raw_artifacts import (
    artifact_store,
    bga_run_context,
    raw_logical_key,
    raw_payload_bytes,
    store_raw_bgg_artifact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
RETRIEVED = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _xml_bytes() -> bytes:
    return (FIXTURES / "bgg_thing_13.xml").read_bytes()


def _xml_text() -> str:
    return _xml_bytes().decode("utf-8")


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


def test_raw_payload_identity_round_trips_exact_bytes(tmp_path: Path) -> None:
    raw = _xml_bytes()
    pid = payload_id(raw)
    store = LocalStore(tmp_path / "cas")
    store.put(pid, raw, media_type="application/xml")
    retrieved = store.get(pid)
    assert retrieved == raw
    assert payload_id(retrieved) == pid
    assert sha256_hex(retrieved) == pid


def test_raw_artifact_record_is_generic(tmp_path: Path) -> None:
    artifact = _artifact(_xml_text())
    raw = raw_payload_bytes(artifact)
    assert raw == _xml_bytes()
    pid, _rid, record = store_raw_bgg_artifact(
        LocalStore(tmp_path),
        artifact,
        bga_run_context(started_at=RETRIEVED),
    )
    assert record.kind is ArtifactKind.RAW
    assert record.media_type == "application/xml"
    assert record.payload_id == pid == payload_id(raw)
    assert record.logical_key == raw_logical_key("13") == "bga:raw/bgg/thing-13"
    assert record.produced_by.project == "bga"
    dumped = record.model_dump()
    assert "location" not in dumped
    assert "record_id" not in dumped
    for field in (
        "source",
        "source_identifier",
        "request_url",
        "http_status",
        "bgg_id",
        "content_type",
    ):
        assert field not in dumped


def test_record_id_changes_with_metadata_not_payload(tmp_path: Path) -> None:
    artifact = _artifact(_xml_text())
    run = bga_run_context(started_at=RETRIEVED)
    _pid, first_rid, first = store_raw_bgg_artifact(
        LocalStore(tmp_path / "a"),
        artifact,
        run,
    )
    changed = first.model_copy(update={"logical_key": "bga:raw/bgg/thing-13-alias"})
    assert first.payload_id == changed.payload_id
    assert record_id(changed) != first_rid
    named = first.model_copy(update={"name": "Catan raw dump"})
    assert named.payload_id == first.payload_id
    assert record_id(named) != first_rid


def test_store_root_does_not_change_payload_identity(tmp_path: Path) -> None:
    raw = _xml_bytes()
    pid = payload_id(raw)
    left = LocalStore(tmp_path / "one")
    right = LocalStore(tmp_path / "two")
    left_loc = left.put(pid, raw, media_type="application/xml")
    right_loc = right.put(pid, raw, media_type="application/xml")
    assert left.get(pid) == right.get(pid) == raw
    assert left_loc.uri != right_loc.uri


def test_record_round_trip_through_same_store(tmp_path: Path) -> None:
    artifact = _artifact(_xml_text())
    store = LocalStore(tmp_path)
    _pid, rid, record = store_raw_bgg_artifact(
        store,
        artifact,
        bga_run_context(started_at=RETRIEVED),
    )
    loaded = store.get(rid)
    assert loaded == canonical_json_bytes(record)
    assert sha256_hex(loaded) == rid
    assert record_id(ArtifactRecord.model_validate_json(loaded)) == rid


def test_pipeline_writes_cas_without_replacing_raw_cache(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    archive = RawArchive(tmp_path)
    artifact = _artifact(_xml_text())
    archive.save(artifact)
    game = ingest_bgg_game("13", settings)
    assert game.title == "CATAN"
    assert game.sources[0].source == "boardgamegeek"
    assert archive.path_for("13").is_file()
    raw = raw_payload_bytes(artifact)
    store = artifact_store(tmp_path)
    assert store.get(payload_id(raw)) == raw
