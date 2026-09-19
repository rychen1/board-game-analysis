"""Fetch → archive raw XML → normalize → write processed JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from board_game_analysis.config import Settings
from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.client import BggClient
from board_game_analysis.ingestion.bgg.normalizer import (
    normalize_bgg_artifact,
    normalize_bgg_thing,
)
from board_game_analysis.ingestion.bgg.parser import parse_thing_xml
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.corpus.ids import (
    DEFAULT_CORPUS_ID,
    DEFAULT_CORPUS_VERSION,
    load_source_ids,
)
from board_game_analysis.ingestion.dataset_artifacts import store_corpus_dataset
from board_game_analysis.ingestion.document_artifacts import store_game_document
from board_game_analysis.ingestion.errors import (
    BggAuthenticationError,
    BggHttpError,
    BggNotFoundError,
    IngestionError,
)
from board_game_analysis.ingestion.quality import (
    integrity_report,
    load_games_jsonl,
    load_manifest,
)
from board_game_analysis.ingestion.quality_artifacts import (
    integrity_report_json_bytes,
    store_quality_report,
)
from board_game_analysis.ingestion.raw_artifacts import (
    artifact_store,
    bga_run_context,
    store_raw_bgg_artifact,
)
from board_game_analysis.ingestion.util import chunked

BOARDGAME_ITEM_TYPE = "boardgame"
CORPUS_JSONL_NAME = "corpus_v0.jsonl"


@dataclass
class CorpusRecord:
    source_id: str
    status: Literal["ok", "skipped", "error"]
    reason: str | None = None
    item_type: str | None = None
    game_id: str | None = None
    title: str | None = None
    cached: bool = False


@dataclass
class CorpusRunResult:
    corpus_id: str
    corpus_version: str
    requested_ids: list[str]
    games: list[Game]
    records: list[CorpusRecord]
    manifest_path: Path
    jsonl_path: Path
    started_at: datetime
    finished_at: datetime

    @property
    def ok(self) -> list[CorpusRecord]:
        return [record for record in self.records if record.status == "ok"]

    @property
    def skipped(self) -> list[CorpusRecord]:
        return [record for record in self.records if record.status == "skipped"]

    @property
    def errors(self) -> list[CorpusRecord]:
        return [record for record in self.records if record.status == "error"]


def ingest_bgg_game(
    source_id: str,
    settings: Settings,
    *,
    force: bool = False,
    client: BggClient | None = None,
) -> Game:
    """Ingest one BGG thing id. Uses the raw cache unless `force` is set."""
    archive = RawArchive(settings.data_dir)
    artifact = None if force else archive.load(source_id)
    if artifact is None:
        owned = client is None
        bgg = client or BggClient(settings)
        try:
            artifact = bgg.fetch_game(source_id)
        finally:
            if owned:
                bgg.close()
        archive.save(artifact, overwrite=force)
    run = bga_run_context()
    store = artifact_store(settings.data_dir)
    raw_payload_id, _, _ = store_raw_bgg_artifact(store, artifact, run)
    game = normalize_bgg_artifact(artifact)
    processed_path = write_processed_game(settings.data_dir, source_id, game)
    store_game_document(
        store,
        processed_path.read_bytes(),
        raw_payload_id=raw_payload_id,
        run=run,
        game_id=game.id,
    )
    return game


def ingest_bgg_games(
    source_ids: list[str],
    settings: Settings,
    *,
    force: bool = False,
    client: BggClient | None = None,
) -> list[Game]:
    """Ingest explicit ids. Fail-fast. Prefetches missing ids in BGG batches."""
    archive = RawArchive(settings.data_dir)
    owned = client is None
    bgg = client or BggClient(settings)
    try:
        _prefetch_missing(
            source_ids,
            settings,
            archive=archive,
            client=bgg,
            force=force,
            continue_on_error=False,
        )
        return [
            ingest_bgg_game(source_id, settings, force=False, client=bgg)
            for source_id in source_ids
        ]
    finally:
        if owned:
            bgg.close()


def ingest_corpus(
    settings: Settings,
    *,
    ids_file: Path | None = None,
    offset: int = 0,
    limit: int | None = None,
    force: bool = False,
    client: BggClient | None = None,
    corpus_id: str = DEFAULT_CORPUS_ID,
    corpus_version: str = DEFAULT_CORPUS_VERSION,
) -> CorpusRunResult:
    """Ingest a frozen id list. Resume from cache; continue on per-id errors."""
    started_at = datetime.now(UTC)
    run = bga_run_context(started_at=started_at)
    store = artifact_store(settings.data_dir)
    source_ids = load_source_ids(ids_file, offset=offset, limit=limit)
    archive = RawArchive(settings.data_dir)
    cached_before = {
        source_id
        for source_id in source_ids
        if not force and archive.load(source_id) is not None
    }
    owned = client is None
    bgg = client or BggClient(settings)
    fetch_errors: dict[str, str] = {}
    try:
        fetch_errors = _prefetch_missing(
            source_ids,
            settings,
            archive=archive,
            client=bgg,
            force=force,
            continue_on_error=True,
        )
    finally:
        if owned:
            bgg.close()

    records: list[CorpusRecord] = []
    games: list[Game] = []
    game_payload_ids: list[str] = []
    for source_id in source_ids:
        cached = source_id in cached_before
        if source_id in fetch_errors:
            records.append(
                CorpusRecord(
                    source_id=source_id,
                    status="error",
                    reason=fetch_errors[source_id],
                    cached=cached,
                )
            )
            continue
        artifact = archive.load(source_id)
        if artifact is None:
            records.append(
                CorpusRecord(
                    source_id=source_id,
                    status="error",
                    reason="not_found",
                    cached=cached,
                )
            )
            continue
        raw_payload_id, _, _ = store_raw_bgg_artifact(store, artifact, run)
        record, game = _normalize_corpus_item(source_id, artifact, cached=cached)
        records.append(record)
        if game is not None:
            processed_path = write_processed_game(settings.data_dir, source_id, game)
            document_payload_id, _, _ = store_game_document(
                store,
                processed_path.read_bytes(),
                raw_payload_id=raw_payload_id,
                run=run,
                game_id=game.id,
            )
            games.append(game)
            game_payload_ids.append(document_payload_id)

    finished_at = datetime.now(UTC)
    jsonl_path = write_corpus_jsonl(settings.data_dir, games)
    dataset_payload_id, _, _ = store_corpus_dataset(
        store,
        jsonl_path.read_bytes(),
        game_payload_ids=game_payload_ids,
        run=run,
    )
    manifest_path = write_corpus_manifest(
        settings.data_dir,
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        requested_ids=source_ids,
        records=records,
        started_at=started_at,
        finished_at=finished_at,
        jsonl_path=jsonl_path,
    )
    report = integrity_report(
        load_games_jsonl(jsonl_path),
        load_manifest(manifest_path),
    )
    store_quality_report(
        store,
        integrity_report_json_bytes(report),
        subject_payload_id=dataset_payload_id,
        run=run,
    )
    return CorpusRunResult(
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        requested_ids=source_ids,
        games=games,
        records=records,
        manifest_path=manifest_path,
        jsonl_path=jsonl_path,
        started_at=started_at,
        finished_at=finished_at,
    )


def processed_game_json_bytes(game: Game) -> bytes:
    """Exact processed JSON bytes written for a Game. Not platform-canonical."""
    payload = game.model_dump(mode="json")
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def write_processed_game(data_dir: Path, source_id: str, game: Game) -> Path:
    directory = data_dir / "processed" / "boardgamegeek"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{source_id}.json"
    path.write_bytes(processed_game_json_bytes(game))
    return path


def write_corpus_jsonl(data_dir: Path, games: list[Game]) -> Path:
    directory = data_dir / "processed" / "boardgamegeek"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CORPUS_JSONL_NAME
    lines = [
        json.dumps(game.model_dump(mode="json"), separators=(",", ":"))
        for game in games
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_corpus_manifest(
    data_dir: Path,
    *,
    corpus_id: str,
    corpus_version: str,
    requested_ids: list[str],
    records: list[CorpusRecord],
    started_at: datetime,
    finished_at: datetime,
    jsonl_path: Path,
) -> Path:
    directory = data_dir / "derived" / "corpus"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{corpus_id}.manifest.json"
    ok = [record for record in records if record.status == "ok"]
    skipped = [record for record in records if record.status == "skipped"]
    errors = [record for record in records if record.status == "error"]
    payload = {
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "source": "boardgamegeek",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "jsonl": str(jsonl_path),
        "requested_ids": requested_ids,
        "ok": [_record_payload(record) for record in ok],
        "skipped": [_record_payload(record) for record in skipped],
        "errors": [_record_payload(record) for record in errors],
        "counts": {
            "requested": len(requested_ids),
            "ok": len(ok),
            "skipped": len(skipped),
            "errors": len(errors),
            "cached": sum(1 for record in records if record.cached),
            "fetched": sum(1 for record in records if not record.cached),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _record_payload(record: CorpusRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_id": record.source_id,
        "status": record.status,
        "cached": record.cached,
    }
    if record.reason is not None:
        payload["reason"] = record.reason
    if record.item_type is not None:
        payload["item_type"] = record.item_type
    if record.game_id is not None:
        payload["game_id"] = record.game_id
    if record.title is not None:
        payload["title"] = record.title
    return payload


def _normalize_corpus_item(
    source_id: str,
    artifact: RawArtifact,
    *,
    cached: bool,
) -> tuple[CorpusRecord, Game | None]:
    try:
        thing = parse_thing_xml(artifact.body)
    except IngestionError as exc:
        return (
            CorpusRecord(
                source_id=source_id,
                status="error",
                reason=str(exc),
                cached=cached,
            ),
            None,
        )
    if thing.item_type != BOARDGAME_ITEM_TYPE:
        return (
            CorpusRecord(
                source_id=source_id,
                status="skipped",
                reason="not_boardgame",
                item_type=thing.item_type,
                cached=cached,
            ),
            None,
        )
    try:
        game = normalize_bgg_thing(thing, artifact)
    except (IngestionError, ValueError) as exc:
        return (
            CorpusRecord(
                source_id=source_id,
                status="error",
                reason=str(exc),
                item_type=thing.item_type,
                cached=cached,
            ),
            None,
        )
    return (
        CorpusRecord(
            source_id=source_id,
            status="ok",
            item_type=thing.item_type,
            game_id=game.id,
            title=game.title,
            cached=cached,
        ),
        game,
    )


def _prefetch_missing(
    source_ids: list[str],
    settings: Settings,
    *,
    archive: RawArchive,
    client: BggClient,
    force: bool,
    continue_on_error: bool,
) -> dict[str, str]:
    """Fetch uncached ids in batches. Returns source_id → error reason."""
    errors: dict[str, str] = {}
    missing = [
        source_id
        for source_id in source_ids
        if force or archive.load(source_id) is None
    ]
    if not missing:
        return errors
    for chunk in chunked(missing, min(settings.bgg_batch_size, 20)):
        try:
            artifacts = client.fetch_games(chunk)
        except BggAuthenticationError:
            raise
        except (BggHttpError, IngestionError) as exc:
            if not continue_on_error:
                raise
            reason = str(exc)
            for source_id in chunk:
                errors[source_id] = reason
            continue
        found = {artifact.source_identifier: artifact for artifact in artifacts}
        for source_id in chunk:
            artifact = found.get(source_id)
            if artifact is None:
                if continue_on_error:
                    errors[source_id] = "not_found"
                    continue
                msg = f"BGG thing id {source_id} was not found"
                raise BggNotFoundError(msg)
            archive.save(artifact, overwrite=force)
    return errors
