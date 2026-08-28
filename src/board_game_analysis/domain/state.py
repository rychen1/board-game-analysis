"""A snapshot of play at a particular point.

`data` is the underlying, omniscient snapshot. Player-specific views belong
on `Observation` / `InformationSpace`, not here.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GameState(BaseModel):
    """An identified snapshot of a game."""

    model_config = ConfigDict(extra="forbid")

    id: str
    game_id: str | None = None
    turn_number: int | None = None
    active_player_id: str | None = None
    phase: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
