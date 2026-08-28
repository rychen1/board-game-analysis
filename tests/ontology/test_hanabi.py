"""Structural questions for the Hanabi situation."""

from board_game_analysis.domain import Visibility
from tests.fixtures.games.hanabi import ALICE, BOB, S0, make_hanabi_situation


def test_hanabi_alice_cannot_see_own_card_identity() -> None:
    situation = make_hanabi_situation()
    own = [item for item in situation.space_for(ALICE).items if item.holder_id == ALICE]
    assert own
    assert all(not item.content_known for item in own)
    assert all(item.visibility == Visibility.UNKNOWN_TO_SELF for item in own)


def test_hanabi_alice_can_see_bob_card_identity() -> None:
    situation = make_hanabi_situation()
    bobs = [item for item in situation.space_for(ALICE).items if item.holder_id == BOB]
    assert bobs
    assert all(item.content_known for item in bobs)
    assert bobs[0].payload["rank"] == 3


def test_hanabi_same_state_observed_differently() -> None:
    situation = make_hanabi_situation()
    alice_obs = situation.observation_for(ALICE, S0)
    bob_obs = situation.observation_for(BOB, S0)
    assert alice_obs.game_state_id == bob_obs.game_state_id == S0
    alice_own = next(
        item for item in situation.space_for(ALICE, S0).items if item.holder_id == ALICE
    )
    bob_view_of_alice = next(
        item for item in situation.space_for(BOB, S0).items if item.holder_id == ALICE
    )
    assert alice_own.content_known is False
    assert bob_view_of_alice.content_known is True
    assert bob_view_of_alice.payload["color"] == "red"


def test_hanabi_clue_is_communicated_to_target() -> None:
    situation = make_hanabi_situation()
    action = situation.actions[0]
    assert action.action_type == "give_clue"
    assert action.parameters["target"] == BOB
    bob_after = situation.space_for(BOB, "hanabi-s1")
    communicated = [
        item for item in bob_after.items if item.visibility == Visibility.COMMUNICATED
    ]
    assert communicated
    assert communicated[0].communicated_by == ALICE
    assert communicated[0].payload["rank"] == 3
