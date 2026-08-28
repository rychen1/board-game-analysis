"""Decisions available to a player at a point in play.

`legal` means the rules permit this action type in principle.
`available` means it is currently possible given the game state.
Whether options are strategically distinct is a later analysis concern.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionOption(BaseModel):
    """One option in a decision space."""

    model_config = ConfigDict(extra="forbid")

    id: str
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    legal: bool = True
    available: bool = True
    notes: str | None = None


class DecisionSpace(BaseModel):
    """The set of decisions available to a player at a particular point."""

    model_config = ConfigDict(extra="forbid")

    id: str
    player_id: str
    game_state_id: str | None = None
    options: list[DecisionOption] = Field(default_factory=list)
    complete: bool = True
