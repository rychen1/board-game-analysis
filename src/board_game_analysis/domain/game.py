"""Observed game metadata.

Fields here are facts reported by sources (title, player count, ratings, and
so on). Interpretations and analytical measurements must not be stored on
this model.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.domain.source import SourceReference


class Game(BaseModel):
    """Catalog-level description of a board game as observed from sources."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1)
    release_year: int | None = None
    min_players: int | None = None
    max_players: int | None = None
    min_play_time_minutes: int | None = None
    max_play_time_minutes: int | None = None
    popularity: float | None = None
    rating: float | None = None
    rating_count: int | None = Field(default=None, ge=0)
    complexity: float | None = None
    designers: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    mechanics: list[Mechanic] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        if (
            self.min_players is not None
            and self.max_players is not None
            and self.min_players > self.max_players
        ):
            msg = "min_players cannot exceed max_players"
            raise ValueError(msg)
        if (
            self.min_play_time_minutes is not None
            and self.max_play_time_minutes is not None
            and self.min_play_time_minutes > self.max_play_time_minutes
        ):
            msg = "min_play_time_minutes cannot exceed max_play_time_minutes"
            raise ValueError(msg)
        return self
