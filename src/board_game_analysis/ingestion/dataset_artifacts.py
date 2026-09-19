"""Thin ds-platform adapter for the BGA corpus JSONL dataset.

The corpus file format and Game documents stay in this project. This module
only stores the exact JSONL bytes BGA already writes.
"""

from __future__ import annotations

from datetime import datetime

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    RunContext,
    Store,
    payload_id,
    put_record,
)

DATASET_MEDIA_TYPE = "application/jsonl"
CORPUS_LOGICAL_KEY = "bga:corpus:v0"


def store_corpus_dataset(
    store: Store,
    jsonl_bytes: bytes,
    *,
    game_payload_ids: list[str],
    run: RunContext,
    created_at: datetime | None = None,
    logical_key: str = CORPUS_LOGICAL_KEY,
) -> tuple[str, str, ArtifactRecord]:
    """Store exact corpus JSONL bytes and a slim dataset ArtifactRecord."""
    pid = payload_id(jsonl_bytes)
    store.put(pid, jsonl_bytes, media_type=DATASET_MEDIA_TYPE)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=DATASET_MEDIA_TYPE,
        kind=ArtifactKind.DATASET,
        produced_by=run,
        created_at=created_at or run.started_at,
        logical_key=logical_key,
        inputs=list(game_payload_ids),
    )
    rid = put_record(store, record)
    return pid, rid, record
