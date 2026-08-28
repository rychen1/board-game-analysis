"""Minimal CLI: ingest one or more BoardGameGeek thing ids."""

from __future__ import annotations

import argparse
import sys

from board_game_analysis.config import Settings
from board_game_analysis.ingestion.errors import IngestionError
from board_game_analysis.ingestion.pipeline import ingest_bgg_games


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="board-game-ingest",
        description="Ingest BoardGameGeek XML API2 thing records (v0).",
    )
    parser.add_argument(
        "source_ids",
        nargs="+",
        help="BGG thing ids (for example 13 for CATAN)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch from BGG even if a raw archive file exists",
    )
    args = parser.parse_args(argv)
    settings = Settings()
    try:
        games = ingest_bgg_games(args.source_ids, settings, force=args.force)
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for game in games:
        identifier = game.sources[0].source_identifier if game.sources else game.id
        print(f"{identifier}\t{game.title}\t{game.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
