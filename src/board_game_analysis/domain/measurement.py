"""Derived analytical measurements produced by analysis, not by ingestion.

Do not store these on `Game` or treat them as source-reported facts.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DerivedMeasurement(BaseModel):
    """A named quantitative or structured measurement for a game."""

    model_config = ConfigDict(extra="forbid")

    id: str
    game_id: str
    name: str = Field(min_length=1)
    value: float | dict[str, Any]
    unit: str | None = None
    method: str | None = None
