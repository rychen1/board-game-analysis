"""The example scenario constructs a consistent id graph."""

from tests.fixtures.example_scenario import make_example_scenario


def test_example_scenario_links_game_turn_player_spaces_action() -> None:
    scenario = make_example_scenario()
    alice = scenario.players[0]

    assert scenario.game.title == "Example Card Game"
    assert scenario.turn.active_player_id == alice.id
    assert scenario.information_space.player_id == alice.id
    assert scenario.decision_space.player_id == alice.id
    assert scenario.action.player_id == alice.id

    assert scenario.turn.information_space_id == scenario.information_space.id
    assert scenario.turn.decision_space_id == scenario.decision_space.id
    assert scenario.turn.action_id == scenario.action.id
    assert scenario.turn.resulting_state_id == scenario.to_state.id

    assert scenario.transition.from_state_id == scenario.from_state.id
    assert scenario.transition.action_id == scenario.action.id
    assert scenario.transition.to_state_id == scenario.to_state.id

    option_types = {option.action_type for option in scenario.decision_space.options}
    assert scenario.action.action_type in option_types
    visibilities = {item.visibility for item in scenario.information_space.items}
    assert {"public", "private", "hidden", "inferred"} <= visibilities
