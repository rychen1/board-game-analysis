"""State change produced by one or more actions, or by chance/system."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from board_game_analysis.domain.vocabulary import TransitionKind


class StateTransition(BaseModel):
    """GameState → Action(s) → GameState, linked by ids."""

    model_config = ConfigDict(extra="forbid")

    id: str
    from_state_id: str
    to_state_id: str
    action_id: str | None = None
    action_ids: list[str] = Field(default_factory=list)
    kind: str = TransitionKind.PLAYER

    @model_validator(mode="after")
    def collect_action_ids(self) -> Self:
        ids = list(self.action_ids)
        if self.action_id is not None and self.action_id not in ids:
            ids.insert(0, self.action_id)
        self.action_ids = ids
        return self
