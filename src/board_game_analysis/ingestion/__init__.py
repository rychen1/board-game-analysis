"""Ingestion of board-game metadata from public APIs.

v0 supports BoardGameGeek XML API2 only. This package must not be imported
by `board_game_analysis.domain`.
"""

from board_game_analysis.ingestion.pipeline import (
    ingest_bgg_game,
    ingest_bgg_games,
    ingest_corpus,
)

__all__ = ["ingest_bgg_game", "ingest_bgg_games", "ingest_corpus"]
