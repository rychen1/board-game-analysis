"""Thin ds-platform adapter for normalized Game document artifacts.

Game, SourceReference, and BGA schemas stay in this project. This module
only stores the exact processed JSON bytes BGA already writes.
"""

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

DOCUMENT_MEDIA_TYPE = "application/json"
GAME_SCHEMA_FILENAME = "game.schema.json"


def default_game_schema_path() -> Path:
    """Committed BGA Game schema. Not packaged in ds-platform."""
    return Path(__file__).resolve().parents[3] / "schemas" / GAME_SCHEMA_FILENAME


def game_document_logical_key(game_id: str) -> str:
    return f"bga:document/game/{game_id}"


def game_contract_ref(schema_path: Path | None = None) -> ContractRef:
    path = schema_path or default_game_schema_path()
    data = path.read_bytes()
    uri = json.loads(data)["$id"]
    if not isinstance(uri, str) or not uri:
        raise ValueError(f"Game schema is missing $id: {path}")
    return ContractRef(uri=uri, schema_hash=sha256_hex(data))


def store_game_document(
    store: Store,
    game_json: bytes,
    *,
    raw_payload_id: str,
    run: RunContext,
    game_id: str,
    created_at: datetime | None = None,
    schema_path: Path | None = None,
) -> tuple[str, str, ArtifactRecord]:
    """Store exact Game JSON bytes and a slim document ArtifactRecord."""
    pid = payload_id(game_json)
    store.put(pid, game_json, media_type=DOCUMENT_MEDIA_TYPE)
    record = ArtifactRecord(
        payload_id=pid,
        media_type=DOCUMENT_MEDIA_TYPE,
        kind=ArtifactKind.DOCUMENT,
        produced_by=run,
        created_at=created_at or run.started_at,
        logical_key=game_document_logical_key(game_id),
        contract_ref=game_contract_ref(schema_path),
        inputs=[raw_payload_id],
    )
    rid = put_record(store, record)
    return pid, rid, record
