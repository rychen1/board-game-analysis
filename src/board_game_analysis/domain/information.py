"""Information available to a player at a point in play.

`GameState` is the underlying (omniscient) snapshot. An `Observation` is one
player's view of that snapshot. `InformationSpace` holds the items in that
view.

`visibility` is an open string. Documented v0 values live on
`vocabulary.Visibility`.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InformationItem(BaseModel):
    """One atom of information in a player's information space."""

    model_config = ConfigDict(extra="forbid")

    id: str
    visibility: str
    about: str | None = None
    known_to: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    content_known: bool = True
    holder_id: str | None = None
    communicated_by: str | None = None
    predecessor_id: str | None = None


class InformationSpace(BaseModel):
    """The information available to a player at a particular point."""

    model_config = ConfigDict(extra="forbid")

    id: str
    player_id: str
    items: list[InformationItem] = Field(default_factory=list)


class Observation(BaseModel):
    """A player's view of an underlying `GameState`.

    Two observations may share a `game_state_id` and differ in their
    information spaces. That is the v0 distinction between state and
    observation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    game_state_id: str
    observer_id: str
    information_space_id: str
