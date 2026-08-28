"""Structural questions for Telestrations."""

from board_game_analysis.domain import Visibility
from tests.fixtures.games.telestrations import (
    ALICE,
    BOB,
    CAROL,
    DRAWING,
    S0,
    S1,
    WORD,
    make_telestrations_situation,
)


def test_telestrations_information_is_transformed_along_predecessor_chain() -> None:
    situation = make_telestrations_situation()
    bob_view = situation.space_for(BOB, S0).items[0]
    assert bob_view.predecessor_id is not None
    assert bob_view.payload["medium"] == "text"
    assert bob_view.payload["content"] == WORD
    carol_drawing = next(
        item
        for item in situation.space_for(CAROL, S1).items
        if item.payload.get("medium") == "drawing"
    )
    assert carol_drawing.predecessor_id is not None
    assert carol_drawing.payload["content"] == DRAWING
    assert carol_drawing.visibility == Visibility.COMMUNICATED


def test_telestrations_each_player_sees_only_part_of_the_chain() -> None:
    situation = make_telestrations_situation()
    carol_s0 = situation.space_for(CAROL, S0)
    assert all(item.visibility == Visibility.HIDDEN for item in carol_s0.items)
    alice_s1 = situation.space_for(ALICE, S1)
    assert any(item.payload.get("content") == WORD for item in alice_s1.items)
    assert any(
        item.about == "drawing" and not item.content_known for item in alice_s1.items
    )
    carol_s1 = situation.space_for(CAROL, S1)
    assert any(
        item.about == "original_word" and not item.content_known
        for item in carol_s1.items
    )


def test_telestrations_actions_pass_information_between_players() -> None:
    situation = make_telestrations_situation()
    assert [action.player_id for action in situation.actions] == [BOB, CAROL]
    assert [tr.from_state_id for tr in situation.transitions] == [S0, S1]
    assert situation.states[-1].data["chain"][-1]["content"] == "squid"
