"""Structural questions for poker."""

from board_game_analysis.domain import Visibility
from tests.fixtures.games.poker import ALICE, BOB, make_poker_situation


def test_poker_private_public_hidden_and_inferred() -> None:
    situation = make_poker_situation()
    space = situation.space_for(ALICE)
    by_vis = {item.visibility: item for item in space.items}
    assert by_vis[Visibility.PRIVATE].payload["cards"] == ["Ah", "Kd"]
    assert by_vis[Visibility.PUBLIC].payload["cards"] == ["As", "7c", "2d"]
    assert by_vis[Visibility.OTHER_PRIVATE].content_known is False
    assert by_vis[Visibility.OTHER_PRIVATE].holder_id == BOB
    inferred = by_vis[Visibility.INFERRED]
    assert inferred.about == BOB
    assert "likely_range" in inferred.payload


def test_poker_bob_does_not_see_alice_hole() -> None:
    situation = make_poker_situation()
    hidden = next(
        item for item in situation.space_for(BOB).items if item.holder_id == ALICE
    )
    assert hidden.content_known is False


def test_poker_betting_decision_and_pot_change() -> None:
    situation = make_poker_situation()
    types = {
        option.action_type for option in situation.decision_space_for(ALICE).options
    }
    assert {"check", "bet", "fold"} <= types
    assert situation.actions[0].action_type == "bet"
    before, after = situation.states
    assert before.data["pot"] == 12
    assert after.data["pot"] == 20
