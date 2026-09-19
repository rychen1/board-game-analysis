"""Store the descriptive report as a ds-platform dataset artifact."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ds_platform import (
    ArtifactKind,
    ArtifactRecord,
    ContractRef,
    RunContext,
    Store,
    payload_id,
    put_record,
    sha256_hex,
)

from board_game_analysis.analysis.descriptive import ANALYSIS_LOGICAL_KEY

REPORT_MEDIA_TYPE = "application/json"
REPORT_SCHEMA_FILENAME = "corpus-descriptive-report.schema.json"
ROBUSTNESS_SCHEMA_FILENAME = "corpus-descriptive-robustness-report.schema.json"


def default_report_schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / REPORT_SCHEMA_FILENAME


def default_robustness_schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "schemas" / ROBUSTNESS_SCHEMA_FILENAME


def report_json_bytes(report: dict[str, object]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def report_contract_ref(schema_path: Path | None = None) -> ContractRef:
    path = schema_path or default_report_schema_path()
    data = path.read_bytes()
    uri = json.loads(data)["$id"]
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"Report schema is missing $id: {path}")
    return ContractRef(uri=uri, schema_hash=sha256_hex(data))


def store_descriptive_report(
    store: Store,
    report_bytes: bytes,
    *,
    corpus_payload_id: str,
    run: RunContext,
    created_at: datetime | None = None,
    logical_key: str = ANALYSIS_LOGICAL_KEY,
    schema_path: Path | None = None,
) -> tuple[str, str, ArtifactRecord]:
    pid = payload_id(report_bytes)
    store.put(pid, report_bytes, media_type=REPORT_MEDIA_TYPE)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=REPORT_MEDIA_TYPE,
        kind=ArtifactKind.DATASET,
        produced_by=run,
        created_at=created_at or run.started_at,
        logical_key=logical_key,
        contract_ref=report_contract_ref(schema_path),
        inputs=[corpus_payload_id],
    )
    rid = put_record(store, record)
    return pid, rid, record
