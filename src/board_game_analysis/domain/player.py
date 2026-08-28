"""Players as participants in a game, independent of a specific ruleset."""

from pydantic import BaseModel, ConfigDict


class Player(BaseModel):
    """A participant in a game or recorded play sequence."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str | None = None
    role: str | None = None
