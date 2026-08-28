"""Temporal structure of play."""

from pydantic import BaseModel, ConfigDict


class Turn(BaseModel):
    """One turn (or turn-like step) in a play sequence.

    Related objects are referenced by id so the domain graph stays acyclic.
    `active_player_id` is optional so simultaneous phases can be represented.
    Per-player observations of the same state live on `PlaySituation`, not
    as a single information space on the turn.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    number: int
    active_player_id: str | None = None
    phase: str | None = None
    game_state_id: str | None = None
    information_space_id: str | None = None
    decision_space_id: str | None = None
    action_id: str | None = None
    resulting_state_id: str | None = None
