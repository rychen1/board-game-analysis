"""Fetch → archive raw XML → normalize → write processed JSON."""

from __future__ import annotations

import json
from pathlib import Path

from board_game_analysis.config import Settings
from board_game_analysis.domain.game import Game
from board_game_analysis.ingestion.archive import RawArchive
from board_game_analysis.ingestion.bgg.client import BggClient
from board_game_analysis.ingestion.bgg.normalizer import normalize_bgg_artifact


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
    game = normalize_bgg_artifact(artifact)
    write_processed_game(settings.data_dir, source_id, game)
    return game


def ingest_bgg_games(
    source_ids: list[str],
    settings: Settings,
    *,
    force: bool = False,
    client: BggClient | None = None,
) -> list[Game]:
    owned = client is None
    bgg = client or BggClient(settings)
    try:
        return [
            ingest_bgg_game(source_id, settings, force=force, client=bgg)
            for source_id in source_ids
        ]
    finally:
        if owned:
            bgg.close()


def write_processed_game(data_dir: Path, source_id: str, game: Game) -> Path:
    directory = data_dir / "processed" / "boardgamegeek"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{source_id}.json"
    payload = game.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
