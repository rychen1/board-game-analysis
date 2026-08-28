"""Extensible mechanic labels. This is not a taxonomy."""

from pydantic import BaseModel, ConfigDict, Field


class Mechanic(BaseModel):
    """A named mechanic associated with a game.

    `category` is a free-form grouping, not a closed ontology. A later phase
    may introduce a controlled vocabulary without changing this shape.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1)
    category: str | None = None
    description: str | None = None
