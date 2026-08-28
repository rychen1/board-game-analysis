"""Derived analytical measurements produced by analysis, not by ingestion.

Do not store these on `Game` or treat them as source-reported facts.
A measurement records a result (or a placeholder). Proposed metrics without
values belong in `board_game_analysis.analysis.spec`, not here.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DerivedMeasurement(BaseModel):
    """One recorded measurement of a named metric on a game or scope.

    Example identity (value may be omitted until a later engine computes it):

        game_id = hanabi
        name = available_decision_count
        scope = player
        player_id = hanabi-alice
        version = analysis-spec-v0
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    game_id: str
    name: str = Field(min_length=1)
    value: float | dict[str, Any] | None = None
    unit: str | None = None
    method: str | None = None
    scope: str | None = None
    player_id: str | None = None
    state_id: str | None = None
    version: str | None = None
    model_ref: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
