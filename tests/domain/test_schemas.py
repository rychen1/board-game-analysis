"""Instantiation and validation of foundational domain models."""

import pytest
from pydantic import ValidationError

from board_game_analysis.domain import (
    Action,
    DecisionOption,
    DecisionSpace,
    DerivedMeasurement,
    ExtractedInterpretation,
    Game,
    GameState,
    InformationItem,
    InformationSpace,
    Mechanic,
    Player,
    SourceReference,
    StateTransition,
    Turn,
)


def test_game_requires_only_id_and_title() -> None:
    game = Game(id="g1", title="Minimal")
    assert game.release_year is None
    assert game.mechanics == []
    assert game.sources == []


def test_game_rejects_empty_title() -> None:
    with pytest.raises(ValidationError):
        Game(id="g1", title="")


def test_game_rejects_player_count_inversion() -> None:
    with pytest.raises(ValidationError):
        Game(id="g1", title="X", min_players=5, max_players=2)


def test_game_rejects_play_time_inversion() -> None:
    with pytest.raises(ValidationError):
        Game(id="g1", title="X", min_play_time_minutes=60, max_play_time_minutes=15)


def test_game_rejects_negative_rating_count() -> None:
    with pytest.raises(ValidationError):
        Game(id="g1", title="X", rating_count=-1)


def test_game_rejects_interpretation_fields() -> None:
    with pytest.raises(ValidationError):
        Game(id="g1", title="X", cooperative=True)  # type: ignore[call-arg]


def test_game_accepts_equal_player_and_time_bounds() -> None:
    game = Game(
        id="g1",
        title="Duel",
        min_players=2,
        max_players=2,
        min_play_time_minutes=30,
        max_play_time_minutes=30,
        rating_count=0,
    )
    assert game.min_players == game.max_players


def test_source_reference_and_mechanic_instantiate() -> None:
    source = SourceReference(source="bgg", source_identifier="123")
    mechanic = Mechanic(id="m1", name="Drafting")
    assert source.page_number is None
    assert mechanic.category is None


def test_player_state_turn_action_and_transition_instantiate() -> None:
    player = Player(id="p1", name="A")
    state = GameState(id="s1", turn_number=1, active_player_id=player.id)
    action = Action(id="a1", player_id=player.id, action_type="pass")
    turn = Turn(id="t1", number=1, active_player_id=player.id, action_id=action.id)
    transition = StateTransition(
        id="tr1",
        from_state_id=state.id,
        action_id=action.id,
        to_state_id="s2",
    )
    assert turn.action_id == action.id
    assert transition.from_state_id == state.id


def test_information_and_decision_spaces_instantiate() -> None:
    item = InformationItem(id="i1", visibility="public", payload={"score": 0})
    info = InformationSpace(id="info1", player_id="p1", items=[item])
    option = DecisionOption(id="o1", action_type="pass")
    space = DecisionSpace(id="d1", player_id="p1", options=[option])
    assert info.items[0].visibility == "public"
    assert space.options[0].legal is True


def test_interpretation_and_measurement_are_separate_from_game() -> None:
    interpretation = ExtractedInterpretation(
        id="int1",
        game_id="g1",
        label="hidden_information",
        extractor="human",
    )
    measurement = DerivedMeasurement(
        id="meas1",
        game_id="g1",
        name="decision_space_size",
        value=4.0,
        method="example",
    )
    game = Game(id="g1", title="Minimal")
    assert not hasattr(game, "label")
    assert interpretation.game_id == game.id
    assert measurement.game_id == game.id
