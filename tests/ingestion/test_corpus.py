"""Frozen corpus loader and resume-friendly corpus ingest. No live HTTP."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from ds_platform import ArtifactKind, ArtifactRecord, LocalStore

from board_game_analysis.config import Settings
from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.client import BggClient
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.cli import main
from board_game_analysis.ingestion.corpus import (
    DEFAULT_CORPUS_ID,
    corpus_path,
    default_corpus_path,
    load_source_ids,
)
from board_game_analysis.ingestion.errors import IngestionError
from board_game_analysis.ingestion.pipeline import ingest_corpus
from board_game_analysis.ingestion.raw_artifacts import artifact_store
from board_game_analysis.ingestion.util import try_git_code_ref

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ingestion"


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _settings(tmp_path: Path, *, batch_size: int = 20) -> Settings:
    return Settings.model_validate(
        {
            "data_dir": tmp_path,
            "bgg_token": "test-token",
            "bgg_rate_limit_seconds": 0,
            "bgg_max_retries": 2,
            "bgg_retry_backoff_seconds": 0,
            "bgg_batch_size": batch_size,
        }
    )


def _artifact(body: str, source_id: str) -> RawArtifact:
    from datetime import UTC, datetime

    return RawArtifact(
        source="boardgamegeek",
        source_identifier=source_id,
        retrieved_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        request_url=f"https://boardgamegeek.com/xmlapi2/thing?id={source_id}&stats=1",
        content_type="text/xml",
        http_status=200,
        body=body,
    )


def _write_ids(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_default_corpus_is_frozen_and_unique() -> None:
    ids = load_source_ids()
    assert default_corpus_path().name == "bgg_boardgames_v0.txt"
    assert DEFAULT_CORPUS_ID == "bgg_boardgames_v0"
    assert len(ids) == 50
    assert len(set(ids)) == 50
    assert ids[0] == "13"
    assert all(item.isdigit() for item in ids)


def test_v1_corpus_is_200_unique_ids_prefixed_by_v0() -> None:
    v0 = load_source_ids()
    v1 = load_source_ids(corpus_id="bgg_boardgames_v1")
    assert corpus_path("bgg_boardgames_v1").name == "bgg_boardgames_v1.txt"
    assert len(v1) == 200
    assert len(set(v1)) == 200
    assert v1[:50] == v0
    assert all(item.isdigit() for item in v1)


def test_corpus_loader_skips_comments_and_duplicates(tmp_path: Path) -> None:
    path = _write_ids(
        tmp_path / "ids.txt",
        [
            "# header",
            "13  # Catan",
            "",
            "9209",
            "13",
            "822",
        ],
    )
    assert load_source_ids(path) == ["13", "9209", "822"]
    assert load_source_ids(path, offset=1, limit=1) == ["9209"]
    assert load_source_ids(path, offset=10, limit=5) == []


def test_corpus_skips_expansions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13", "999003"])
    body = _xml("bgg_thing_batch_mixed.xml")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "id=13,999003" in str(request.url)
        return httpx.Response(200, text=body, headers={"content-type": "text/xml"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        result = ingest_corpus(settings, ids_file=ids_file, client=client)

    assert [game.title for game in result.games] == ["CATAN"]
    assert [record.source_id for record in result.ok] == ["13"]
    assert result.skipped[0].source_id == "999003"
    assert result.skipped[0].reason == "not_boardgame"
    assert result.skipped[0].item_type == "boardgameexpansion"
    processed_expansion = tmp_path / "processed" / "boardgamegeek" / "999003.json"
    assert not processed_expansion.exists()


def test_corpus_continues_on_http_and_not_found_errors(tmp_path: Path) -> None:
    settings = _settings(tmp_path, batch_size=1)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13", "999999", "9209"])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "id=13&" in url:
            return httpx.Response(500, text="nope")
        if "id=999999&" in url:
            return httpx.Response(
                200,
                text=_xml("bgg_thing_empty.xml"),
                headers={"content-type": "text/xml"},
            )
        if "id=9209&" in url:
            return httpx.Response(
                200,
                text=_xml("bgg_thing_9209.xml"),
                headers={"content-type": "text/xml"},
            )
        raise AssertionError(url)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        result = ingest_corpus(settings, ids_file=ids_file, client=client)

    assert len(calls) == 3
    assert [game.title for game in result.games] == ["Ticket to Ride"]
    reasons = {record.source_id: record.reason for record in result.errors}
    assert "13" in reasons
    assert reasons["999999"] == "not_found"
    assert result.ok[0].source_id == "9209"


def test_corpus_resume_skips_cached_ids(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13", "9209"])
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert "id=9209" in str(request.url)
        assert "id=13" not in str(request.url).split("id=")[1]
        return httpx.Response(
            200,
            text=_xml("bgg_thing_9209.xml"),
            headers={"content-type": "text/xml"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        result = ingest_corpus(settings, ids_file=ids_file, client=client)

    assert calls["n"] == 1
    assert [game.id for game in result.games] == ["bgg-13", "bgg-9209"]
    by_id = {record.source_id: record for record in result.ok}
    assert by_id["13"].cached is True
    assert by_id["9209"].cached is False


def test_corpus_limit_and_offset(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13", "9209", "822"])
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_9209.xml"), "9209"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        result = ingest_corpus(
            settings,
            ids_file=ids_file,
            offset=1,
            limit=1,
            client=client,
        )

    assert result.requested_ids == ["9209"]
    assert [game.title for game in result.games] == ["Ticket to Ride"]


def test_corpus_writes_manifest_and_jsonl(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml"), "13"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        result = ingest_corpus(settings, ids_file=ids_file, client=client)

    assert result.corpus_id == DEFAULT_CORPUS_ID
    assert result.jsonl_path.name == "corpus_v0.jsonl"
    assert result.manifest_path.is_file()
    assert result.jsonl_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["ok"] == 1
    assert manifest["counts"]["requested"] == 1
    assert manifest["ok"][0]["game_id"] == "bgg-13"
    assert manifest["ok"][0]["cached"] is True
    assert (
        manifest["ids_file_sha256"] == hashlib.sha256(ids_file.read_bytes()).hexdigest()
    )
    assert manifest["run_id"]
    assert "code_ref" not in manifest or {
        "git_sha",
        "dirty",
    } <= set(manifest["code_ref"])
    line = result.jsonl_path.read_text(encoding="utf-8").strip()
    loaded = Game.model_validate_json(line)
    assert loaded.title == "CATAN"


def test_cli_explicit_id_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BGG_TOKEN", "test-token")
    monkeypatch.setenv("BGG_RATE_LIMIT_SECONDS", "0")
    code = main(["13"])
    assert code == 0
    processed = tmp_path / "processed" / "boardgamegeek" / "13.json"
    assert processed.is_file()


def test_cli_corpus_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BGG_TOKEN", "test-token")
    monkeypatch.setenv("BGG_RATE_LIMIT_SECONDS", "0")
    ids_file = _write_ids(tmp_path / "ids.txt", ["13", "9209"])
    code = main(["--corpus", "--ids-file", str(ids_file), "--limit", "1"])
    assert code == 0
    processed = tmp_path / "processed" / "boardgamegeek" / "13.json"
    assert processed.is_file()
    jsonl = tmp_path / "processed" / "boardgamegeek" / "corpus_v0.jsonl"
    assert jsonl.is_file()


def test_cli_corpus_id_requires_corpus() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--corpus-id", "bgg_boardgames_v1"])
    assert exc.value.code == 2


def test_cli_corpus_id_defaults_to_v0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BGG_TOKEN", "test-token")
    monkeypatch.setenv("BGG_RATE_LIMIT_SECONDS", "0")
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    assert main(["--corpus", "--ids-file", str(ids_file)]) == 0
    assert (tmp_path / "processed" / "boardgamegeek" / "corpus_v0.jsonl").is_file()
    assert not (tmp_path / "processed" / "boardgamegeek" / "corpus_v1.jsonl").exists()


def test_v1_ingest_uses_v1_jsonl_and_logical_keys(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    archive = RawArchive(tmp_path)
    archive.save(_artifact(_xml("bgg_thing_13.xml"), "13"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = BggClient(settings, http_client=http)
        result = ingest_corpus(
            settings,
            ids_file=ids_file,
            client=client,
            corpus_id="bgg_boardgames_v1",
        )

    assert result.corpus_id == "bgg_boardgames_v1"
    assert result.jsonl_path.name == "corpus_v1.jsonl"
    assert result.jsonl_path.is_file()
    assert not (tmp_path / "processed" / "boardgamegeek" / "corpus_v0.jsonl").exists()
    store = artifact_store(tmp_path)
    records = _artifact_records(store)
    dataset_keys = [
        record.logical_key for record in records if record.kind is ArtifactKind.DATASET
    ]
    quality_keys = [
        record.logical_key for record in records if record.kind is ArtifactKind.QUALITY
    ]
    assert dataset_keys == ["bga:corpus:v1"]
    assert quality_keys == ["bga:quality/corpus:v1"]
    assert "bga:corpus:v0" not in dataset_keys
    assert "bga:quality/corpus:v0" not in quality_keys


def test_unknown_corpus_id_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(IngestionError, match="unknown corpus_id"):
        ingest_corpus(settings, corpus_id="not-a-corpus")


def test_ids_file_overrides_packaged_v1_list(tmp_path: Path) -> None:
    path = _write_ids(tmp_path / "ids.txt", ["13"])
    assert load_source_ids(path, corpus_id="bgg_boardgames_v1") == ["13"]


def test_manifest_omits_code_ref_when_git_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "board_game_analysis.ingestion.pipeline.try_git_code_ref",
        lambda: None,
    )
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    RawArchive(tmp_path).save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    result = ingest_corpus(settings, ids_file=ids_file)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "code_ref" not in manifest
    assert manifest["run_id"]
    assert len(manifest["ids_file_sha256"]) == 64


def test_manifest_includes_code_ref_when_git_available(tmp_path: Path) -> None:
    ref = try_git_code_ref()
    if ref is None:
        pytest.skip("git is not available")
    settings = _settings(tmp_path)
    ids_file = _write_ids(tmp_path / "ids.txt", ["13"])
    RawArchive(tmp_path).save(_artifact(_xml("bgg_thing_13.xml"), "13"))
    result = ingest_corpus(settings, ids_file=ids_file)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["code_ref"]["git_sha"] == ref["git_sha"]
    assert manifest["code_ref"]["dirty"] is ref["dirty"]


def _artifact_records(store: LocalStore) -> list[ArtifactRecord]:
    root = store._root  # noqa: SLF001
    records: list[ArtifactRecord] = []
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        try:
            records.append(ArtifactRecord.model_validate_json(path.read_bytes()))
        except (ValueError, KeyError, TypeError):
            continue
    return records
