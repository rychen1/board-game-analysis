"""BGG XML API2 ingestion (source-specific)."""

from board_game_analysis.ingestion.bgg.client import BggClient
from board_game_analysis.ingestion.bgg.normalizer import (
    normalize_bgg_artifact,
    normalize_bgg_thing,
)
from board_game_analysis.ingestion.bgg.parser import parse_thing_xml

__all__ = [
    "BggClient",
    "normalize_bgg_artifact",
    "normalize_bgg_thing",
    "parse_thing_xml",
]
