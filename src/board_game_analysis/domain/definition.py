"""Static rules-level structure, distinct from catalog metadata and from state."""

from pydantic import BaseModel, ConfigDict, Field


class GameDefinition(BaseModel):
    """What the rules permit and how play is structured.

    This is not `Game` (catalog facts) and not `GameState` (what is true now).
    Interaction and decision timing are open strings; see `vocabulary.py`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    game_id: str
    interaction: str
    decision_timing: str
    legal_action_types: list[str] = Field(default_factory=list)
    player_roles: list[str] = Field(default_factory=list)
    notes: str | None = None
