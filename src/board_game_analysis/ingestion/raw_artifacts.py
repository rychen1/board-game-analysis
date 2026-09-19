"""Thin ds-platform adapter for raw BGG response artifacts.

BGG fetch, cache, parse, and Game/SourceReference stay in existing BGA
modules. This module only hashes exact archived XML bytes and writes a
generic ArtifactRecord through LocalStore.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    Environment,
    LocalStore,
    RunContext,
    Store,
    new_run_id,
    payload_id,
    put_record,
)

from board_game_analysis.ingestion.bgg.types import RawArtifact

PROJECT = "bga"
RAW_MEDIA_TYPE = "application/xml"
ARTIFACT_STORE_DIRNAME = "artifacts"


def artifact_store(data_dir: Path) -> LocalStore:
    return LocalStore(data_dir / ARTIFACT_STORE_DIRNAME)


def raw_logical_key(source_id: str) -> str:
    return f"bga:raw/bgg/thing-{source_id}"


def raw_payload_bytes(artifact: RawArtifact) -> bytes:
    """Exact archived XML bytes. No parse, pretty-print, or re-encode pass."""
    return artifact.body.encode("utf-8")


def bga_run_context(*, started_at: datetime | None = None) -> RunContext:
    """Minimal run annotation. Not an execution backend."""
    return RunContext(
        run_id=new_run_id(),
        project=PROJECT,
        started_at=started_at or datetime.now(UTC),
        environment=_environment(),
    )


def store_raw_bgg_artifact(
    store: Store,
    artifact: RawArtifact,
    run: RunContext,
) -> tuple[str, str, ArtifactRecord]:
    """Store raw XML bytes and a slim ArtifactRecord in the same store."""
    raw = raw_payload_bytes(artifact)
    pid = payload_id(raw)
    store.put(pid, raw, media_type=RAW_MEDIA_TYPE)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=RAW_MEDIA_TYPE,
        kind=ArtifactKind.RAW,
        produced_by=run,
        created_at=artifact.retrieved_at,
        logical_key=raw_logical_key(artifact.source_identifier),
    )
    rid = put_record(store, record)
    return pid, rid, record


def _environment() -> Environment:
    if os.environ.get("CI"):
        return Environment.CI
    return Environment.LOCAL
