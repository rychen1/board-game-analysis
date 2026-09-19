"""ds-platform boundary for corpus quality artifacts. No live HTTP."""

from datetime import UTC, datetime
from pathlib import Path

from ds_platform import ArtifactKind, LocalStore, RelationType, payload_id

from board_game_analysis.config import Settings
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.dataset_artifacts import store_corpus_dataset
from board_game_analysis.ingestion.pipeline import ingest_corpus
from board_game_analysis.ingestion.quality import (
    CorpusLoad,
    IntegrityReport,
    integrity_report,
    load_games_jsonl,
    load_manifest,
)
from board_game_analysis.ingestion.quality_artifacts import (
    QUALITY_LOGICAL_KEY,
    integrity_report_json_bytes,
    store_quality_report,
)
from board_game_analysis.ingestion.raw_artifacts import (
    artifact_store,
    bga_run_context,
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


def _empty_report() -> IntegrityReport:
    return integrity_report(CorpusLoad(games=[], errors=[], raw_objects=[]))


def test_identical_quality_bytes_produce_identical_payload_id() -> None:
    report = _empty_report()
    first = integrity_report_json_bytes(report)
    second = integrity_report_json_bytes(report)
    assert first == second
    assert payload_id(first) == payload_id(second)


def test_quality_record_cites_corpus_and_not_the_dataset_envelope(
    tmp_path: Path,
) -> None:
    run = bga_run_context(started_at=RETRIEVED)
    store = LocalStore(tmp_path)
    jsonl = b'{"id":"bgg-13"}\n'
    corpus_pid, _corpus_rid, dataset = store_corpus_dataset(
        store,
        jsonl,
        game_payload_ids=[payload_id(b"doc")],
        run=run,
    )
    report_bytes = integrity_report_json_bytes(_empty_report())
    quality_pid, _rid, record = store_quality_report(
        store,
        report_bytes,
        subject_payload_id=corpus_pid,
        run=run,
    )
    assert record.kind is ArtifactKind.QUALITY
    assert record.logical_key == QUALITY_LOGICAL_KEY
    assert record.payload_id == quality_pid == payload_id(report_bytes)
    assert quality_pid != corpus_pid
    assert len(record.related) == 1
    assert record.related[0].rel is RelationType.QUALITY_FOR
    assert record.related[0].payload_id == corpus_pid
    dumped_quality = record.model_dump()
    assert "location" not in dumped_quality
    assert "duplicate_ids" not in dumped_quality
    assert "n_jsonl_records" not in dumped_quality
    dumped_dataset = dataset.model_dump()
    assert dumped_dataset["related"] == []
    assert "quality" not in dumped_dataset
    assert store.get(corpus_pid) == jsonl
    assert store.get(quality_pid) == report_bytes


def test_report_change_changes_quality_id_not_corpus_id(tmp_path: Path) -> None:
    run = bga_run_context(started_at=RETRIEVED)
    store = LocalStore(tmp_path)
    jsonl = b'{"id":"bgg-13"}\n'
    corpus_pid, _, _ = store_corpus_dataset(
        store,
        jsonl,
        game_payload_ids=[],
        run=run,
    )
    first_bytes = integrity_report_json_bytes(_empty_report())
    first_pid, _, _ = store_quality_report(
        store,
        first_bytes,
        subject_payload_id=corpus_pid,
        run=run,
    )
    changed = integrity_report(
        CorpusLoad(games=[], errors=[], raw_objects=[{}, {}]),
    )
    second_bytes = integrity_report_json_bytes(changed)
    second_pid, _, _ = store_quality_report(
        LocalStore(tmp_path / "other"),
        second_bytes,
        subject_payload_id=corpus_pid,
        run=run,
    )
    assert first_bytes != second_bytes
    assert first_pid != second_pid
    assert payload_id(jsonl) == corpus_pid
    assert store.get(corpus_pid) == jsonl


def test_corpus_ingest_stores_quality_beside_dataset(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    RawArchive(tmp_path).save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    result = ingest_corpus(settings, ids_file=ids_file)
    store = artifact_store(tmp_path)
    jsonl = result.jsonl_path.read_bytes()
    corpus_pid = payload_id(jsonl)
    assert store.get(corpus_pid) == jsonl
    report = integrity_report(
        load_games_jsonl(result.jsonl_path),
        load_manifest(result.manifest_path),
    )
    quality_bytes = integrity_report_json_bytes(report)
    assert store.get(payload_id(quality_bytes)) == quality_bytes
    assert payload_id(quality_bytes) != corpus_pid
