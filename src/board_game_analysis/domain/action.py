"""An action taken by a player."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from board_game_analysis.domain.source import SourceReference


class Action(BaseModel):
    """A recorded player action, distinct from a merely available option."""

    model_config = ConfigDict(extra="forbid")

    id: str
    player_id: str
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    turn_number: int | None = None
    sources: list[SourceReference] = Field(default_factory=list)
