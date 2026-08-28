"""v0 additions: definition, observation, legal vs available."""

import pytest
from pydantic import ValidationError

from board_game_analysis.domain import (
    DecisionOption,
    DecisionTiming,
    Game,
    GameDefinition,
    GameState,
    InformationItem,
    InformationSpace,
    Interaction,
    Observation,
    Player,
    PlaySituation,
    StateTransition,
    TransitionKind,
    Visibility,
)


def test_game_definition_is_not_catalog_metadata() -> None:
    game = Game(id="g", title="G")
    definition = GameDefinition(
        id="d",
        game_id=game.id,
        interaction=Interaction.COOPERATIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["play_card"],
    )
    assert not hasattr(game, "legal_action_types")
    assert definition.game_id == game.id


def test_legal_option_may_be_unavailable() -> None:
    option = DecisionOption(
        id="clue",
        action_type="give_clue",
        legal=True,
        available=False,
    )
    assert option.legal is True
    assert option.available is False


def test_observation_binds_player_to_underlying_state() -> None:
    state = GameState(id="s", data={"secret": 1})
    space = InformationSpace(
        id="info",
        player_id="p1",
        items=[
            InformationItem(
                id="gap",
                visibility=Visibility.UNKNOWN_TO_SELF,
                content_known=False,
            )
        ],
    )
    observation = Observation(
        id="obs",
        game_state_id=state.id,
        observer_id="p1",
        information_space_id=space.id,
    )
    situation = PlaySituation(
        game=Game(id="g", title="G"),
        definition=GameDefinition(
            id="d",
            game_id="g",
            interaction=Interaction.COMPETITIVE,
            decision_timing=DecisionTiming.SEQUENTIAL,
        ),
        players=[Player(id="p1")],
        states=[state],
        observations=[observation],
        information_spaces=[space],
    )
    assert situation.space_for("p1").items[0].content_known is False
    assert situation.observation_for("p1").game_state_id == "s"


def test_simultaneous_transition_collects_action_ids() -> None:
    transition = StateTransition(
        id="tr",
        from_state_id="a",
        to_state_id="b",
        action_ids=["x", "y"],
        kind=TransitionKind.PLAYER,
    )
    assert transition.action_id is None
    assert transition.action_ids == ["x", "y"]


def test_transition_action_id_is_folded_into_action_ids() -> None:
    transition = StateTransition(
        id="tr",
        from_state_id="a",
        to_state_id="b",
        action_id="x",
    )
    assert transition.action_ids == ["x"]


def test_game_still_rejects_mixed_in_interpretation() -> None:
    with pytest.raises(ValidationError):
        Game(id="g", title="G", cooperative=True)  # type: ignore[call-arg]
