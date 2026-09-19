"""ds-platform boundary for the corpus JSONL dataset. No live HTTP."""

from datetime import UTC, datetime
from pathlib import Path

from ds_platform import ArtifactKind, LocalStore, payload_id

from board_game_analysis.config import Settings
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.normalizer import normalize_bgg_artifact
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.dataset_artifacts import (
    CORPUS_LOGICAL_KEY,
    store_corpus_dataset,
)
from board_game_analysis.ingestion.document_artifacts import (
    game_document_logical_key,
    store_game_document,
)
from board_game_analysis.ingestion.pipeline import (
    ingest_corpus,
    processed_game_json_bytes,
    write_corpus_jsonl,
)
from board_game_analysis.ingestion.raw_artifacts import (
    artifact_store,
    bga_run_context,
    raw_payload_bytes,
    store_raw_bgg_artifact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"
RETRIEVED = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _artifact(body: str, source_id: str) -> RawArtifact:
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


def _write_ids(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_identical_jsonl_bytes_produce_identical_payload_id(tmp_path: Path) -> None:
    games = [normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml"), "13"))]
    first = write_corpus_jsonl(tmp_path / "a", games).read_bytes()
    second = write_corpus_jsonl(tmp_path / "b", games).read_bytes()
    assert first == second
    assert payload_id(first) == payload_id(second)


def test_membership_change_changes_dataset_payload_id(tmp_path: Path) -> None:
    catan = normalize_bgg_artifact(_artifact(_xml("bgg_thing_13.xml"), "13"))
    ticket = normalize_bgg_artifact(_artifact(_xml("bgg_thing_9209.xml"), "9209"))
    one = write_corpus_jsonl(tmp_path / "one", [catan]).read_bytes()
    two = write_corpus_jsonl(tmp_path / "two", [catan, ticket]).read_bytes()
    assert payload_id(one) != payload_id(two)


def test_dataset_record_inputs_are_document_ids_not_raw(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "cas")
    run = bga_run_context(started_at=RETRIEVED)
    artifact = _artifact(_xml("bgg_thing_13.xml"), "13")
    raw_pid, _, _ = store_raw_bgg_artifact(store, artifact, run)
    game = normalize_bgg_artifact(artifact)
    game_json = processed_game_json_bytes(game)
    document_pid, _, document = store_game_document(
        store,
        game_json,
        raw_payload_id=raw_pid,
        run=run,
        game_id=game.id,
    )
    jsonl = write_corpus_jsonl(tmp_path / "jsonl", [game]).read_bytes()
    dataset_pid, _rid, record = store_corpus_dataset(
        store,
        jsonl,
        game_payload_ids=[document_pid],
        run=run,
    )
    assert record.kind is ArtifactKind.DATASET
    assert record.logical_key == CORPUS_LOGICAL_KEY == "bga:corpus:v0"
    assert list(record.inputs) == [document_pid]
    assert raw_pid not in record.inputs
    assert record.payload_id == dataset_pid == payload_id(jsonl)
    assert document.kind is ArtifactKind.DOCUMENT
    assert document.logical_key == game_document_logical_key("bgg-13")
    assert document.logical_key != record.logical_key
    assert store.get(raw_pid) == raw_payload_bytes(artifact)
    assert store.get(document_pid) == game_json
    assert store.get(dataset_pid) == jsonl


def test_store_root_does_not_change_dataset_payload_id(tmp_path: Path) -> None:
    jsonl = b'{"id":"bgg-13"}\n'
    pid = payload_id(jsonl)
    left = LocalStore(tmp_path / "one")
    right = LocalStore(tmp_path / "two")
    left.put(pid, jsonl, media_type="application/jsonl")
    right.put(pid, jsonl, media_type="application/jsonl")
    assert left.get(pid) == right.get(pid) == jsonl


def test_dataset_record_has_no_path_or_game_fields(tmp_path: Path) -> None:
    run = bga_run_context(started_at=RETRIEVED)
    _pid, _rid, record = store_corpus_dataset(
        LocalStore(tmp_path),
        b'{"id":"bgg-13"}\n',
        game_payload_ids=[payload_id(b"doc")],
        run=run,
    )
    dumped = record.model_dump()
    assert "location" not in dumped
    assert "path" not in dumped
    assert "title" not in dumped
    assert "sources" not in dumped
    assert "jsonl" not in dumped


def test_corpus_ingest_stores_dataset_and_keeps_jsonl(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    RawArchive(tmp_path).save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    result = ingest_corpus(settings, ids_file=ids_file)
    assert result.jsonl_path.is_file()
    assert result.manifest_path.is_file()
    jsonl = result.jsonl_path.read_bytes()
    store = artifact_store(tmp_path)
    assert store.get(payload_id(jsonl)) == jsonl
    game_json = (tmp_path / "processed" / "boardgamegeek" / "13.json").read_bytes()
    assert store.get(payload_id(game_json)) == game_json
    assert payload_id(jsonl) != payload_id(game_json)
