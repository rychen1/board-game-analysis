"""Ingestion must not leak into the domain package."""

from pathlib import Path

DOMAIN = Path(__file__).resolve().parents[2] / "src" / "board_game_analysis" / "domain"


def test_domain_has_no_ingestion_or_bgg_imports() -> None:
    for path in DOMAIN.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "board_game_analysis.ingestion" not in text
        assert "ds_platform" not in text
        assert "boardgamegeek" not in text.lower()
        assert "xmlapi" not in text.lower()
