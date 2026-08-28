"""Local raw-response archive. Raw files are not overwritten unless forced."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from board_game_analysis.ingestion.bgg.types import RawArtifact

SOURCE = "boardgamegeek"


class RawArchive:
    """Filesystem cache under `data/raw/<source>/`."""

    def __init__(self, data_dir: Path) -> None:
        self._root = data_dir / "raw" / SOURCE

    def path_for(self, source_id: str) -> Path:
        return self._root / f"{source_id}.xml"

    def meta_path_for(self, source_id: str) -> Path:
        return self._root / f"{source_id}.meta.json"

    def load(self, source_id: str) -> RawArtifact | None:
        body_path = self.path_for(source_id)
        meta_path = self.meta_path_for(source_id)
        if not body_path.is_file() or not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return RawArtifact(
            source=str(meta["source"]),
            source_identifier=str(meta["source_identifier"]),
            retrieved_at=datetime.fromisoformat(meta["retrieved_at"]),
            request_url=str(meta["request_url"]),
            content_type=str(meta["content_type"]),
            http_status=int(meta["http_status"]),
            body=body_path.read_text(encoding="utf-8"),
        )

    def save(self, artifact: RawArtifact, *, overwrite: bool = False) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        body_path = self.path_for(artifact.source_identifier)
        meta_path = self.meta_path_for(artifact.source_identifier)
        if body_path.exists() and not overwrite:
            return body_path
        body_path.write_text(artifact.body, encoding="utf-8")
        retrieved = artifact.retrieved_at
        if retrieved.tzinfo is None:
            retrieved = retrieved.replace(tzinfo=UTC)
        meta = {
            "source": artifact.source,
            "source_identifier": artifact.source_identifier,
            "retrieved_at": retrieved.isoformat(),
            "request_url": artifact.request_url,
            "content_type": artifact.content_type,
            "http_status": artifact.http_status,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return body_path
