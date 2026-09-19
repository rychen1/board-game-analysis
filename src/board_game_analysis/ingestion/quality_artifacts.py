"""Thin ds-platform adapter for corpus quality reports.

Quality metrics stay BGA-owned. This module stores exact report JSON bytes
and a slim ArtifactRecord that cites the subject corpus dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    RelatedRef,
    RelationType,
    RunContext,
    Store,
    payload_id,
    put_record,
)

from board_game_analysis.ingestion.quality import IntegrityReport

QUALITY_MEDIA_TYPE = "application/json"
QUALITY_LOGICAL_KEY = "bga:quality/corpus:v0"


def integrity_report_json_bytes(report: IntegrityReport) -> bytes:
    """Exact BGA quality payload. Not platform-canonical domain JSON."""
    return (json.dumps(asdict(report), indent=2, sort_keys=True) + "\n").encode("utf-8")


def store_quality_report(
    store: Store,
    report_bytes: bytes,
    *,
    subject_payload_id: str,
    run: RunContext,
    created_at: datetime | None = None,
    logical_key: str = QUALITY_LOGICAL_KEY,
) -> tuple[str, str, ArtifactRecord]:
    """Store exact quality JSON and a record with related quality_for."""
    pid = payload_id(report_bytes)
    store.put(pid, report_bytes, media_type=QUALITY_MEDIA_TYPE)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=QUALITY_MEDIA_TYPE,
        kind=ArtifactKind.QUALITY,
        produced_by=run,
        created_at=created_at or run.started_at,
        logical_key=logical_key,
        related=[
            RelatedRef(rel=RelationType.QUALITY_FOR, payload_id=subject_payload_id)
        ],
    )
    rid = put_record(store, record)
    return pid, rid, record
