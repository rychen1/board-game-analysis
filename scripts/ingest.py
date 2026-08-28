#!/usr/bin/env python3
"""Ingest BoardGameGeek thing ids. Wrapper around `board-game-ingest`."""

from board_game_analysis.ingestion.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
