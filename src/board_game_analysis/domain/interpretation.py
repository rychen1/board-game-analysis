"""Extracted interpretations derived from rules or other evidence.

These are classifications, not observed catalog facts and not quantitative
analysis outputs. Do not store them on `Game`.
"""

from pydantic import BaseModel, ConfigDict, Field

from board_game_analysis.domain.source import SourceReference


class ExtractedInterpretation(BaseModel):
    """A labeled interpretation of a game, with optional provenance."""

    model_config = ConfigDict(extra="forbid")

    id: str
    game_id: str
    label: str = Field(min_length=1)
    description: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sources: list[SourceReference] = Field(default_factory=list)
    extractor: str | None = None
