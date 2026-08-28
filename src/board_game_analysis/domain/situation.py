"""A recorded fragment of play: definition, state, observations, decisions."""

from pydantic import BaseModel, ConfigDict, Field

from board_game_analysis.domain.action import Action
from board_game_analysis.domain.decision import DecisionSpace
from board_game_analysis.domain.definition import GameDefinition
from board_game_analysis.domain.game import Game
from board_game_analysis.domain.information import InformationSpace, Observation
from board_game_analysis.domain.interpretation import ExtractedInterpretation
from board_game_analysis.domain.player import Player
from board_game_analysis.domain.state import GameState
from board_game_analysis.domain.transition import StateTransition
from board_game_analysis.domain.turn import Turn


class PlaySituation(BaseModel):
    """A small, typed moment (or short chain) of play.

    Multiple observations may share a state id. That is how v0 represents
    different players seeing the same underlying state differently.
    """

    model_config = ConfigDict(extra="forbid")

    game: Game
    definition: GameDefinition
    players: list[Player]
    states: list[GameState]
    observations: list[Observation] = Field(default_factory=list)
    information_spaces: list[InformationSpace] = Field(default_factory=list)
    decision_spaces: list[DecisionSpace] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    transitions: list[StateTransition] = Field(default_factory=list)
    interpretations: list[ExtractedInterpretation] = Field(default_factory=list)

    def player(self, player_id: str) -> Player:
        for player in self.players:
            if player.id == player_id:
                return player
        msg = f"no player {player_id}"
        raise KeyError(msg)

    def state(self, state_id: str) -> GameState:
        for game_state in self.states:
            if game_state.id == state_id:
                return game_state
        msg = f"no state {state_id}"
        raise KeyError(msg)

    def observation_for(
        self, observer_id: str, state_id: str | None = None
    ) -> Observation:
        target_state = state_id if state_id is not None else self.states[0].id
        for observation in self.observations:
            if (
                observation.observer_id == observer_id
                and observation.game_state_id == target_state
            ):
                return observation
        msg = f"no observation for {observer_id} at {target_state}"
        raise KeyError(msg)

    def space_for(
        self, observer_id: str, state_id: str | None = None
    ) -> InformationSpace:
        observation = self.observation_for(observer_id, state_id)
        for space in self.information_spaces:
            if space.id == observation.information_space_id:
                return space
        msg = f"no information space for {observer_id}"
        raise KeyError(msg)

    def decision_space_for(
        self, player_id: str, state_id: str | None = None
    ) -> DecisionSpace:
        for space in self.decision_spaces:
            if space.player_id != player_id:
                continue
            if state_id is None or space.game_state_id in {None, state_id}:
                return space
        msg = f"no decision space for {player_id}"
        raise KeyError(msg)
