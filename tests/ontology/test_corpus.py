"""Structural questions for the remaining corpus games."""

from board_game_analysis.domain import Interaction, Visibility
from tests.fixtures.games.codenames import (
    OPERATIVE,
    S0,
    SPYMASTER,
    make_codenames_situation,
)
from tests.fixtures.games.pandemic import ALICE as P_ALICE
from tests.fixtures.games.pandemic import BOB as P_BOB
from tests.fixtures.games.pandemic import make_pandemic_situation
from tests.fixtures.games.the_crew import ALICE as C_ALICE
from tests.fixtures.games.the_crew import BOB as C_BOB
from tests.fixtures.games.the_crew import make_the_crew_situation
from tests.fixtures.games.the_gang import ALICE as G_ALICE
from tests.fixtures.games.the_gang import BOB as G_BOB
from tests.fixtures.games.the_gang import make_the_gang_situation
from tests.fixtures.games.wingspan import ALICE as W_ALICE
from tests.fixtures.games.wingspan import BOB as W_BOB
from tests.fixtures.games.wingspan import make_wingspan_situation


def test_the_crew_own_hand_known_opponents_unknown_communication_constrained() -> None:
    situation = make_the_crew_situation()
    alice = situation.space_for(C_ALICE)
    own = next(item for item in alice.items if item.holder_id == C_ALICE)
    other = next(item for item in alice.items if item.holder_id == C_BOB)
    assert own.content_known is True
    assert other.content_known is False
    assert situation.actions[0].action_type == "communicate_card"
    labels = {item.label for item in situation.interpretations}
    assert "constrained_communication" in labels


def test_the_gang_private_holes_public_board_cooperative() -> None:
    situation = make_the_gang_situation()
    assert situation.definition.interaction == Interaction.COOPERATIVE
    alice = situation.space_for(G_ALICE)
    private = next(
        item for item in alice.items if item.visibility == Visibility.PRIVATE
    )
    assert private.content_known
    other = next(item for item in alice.items if item.holder_id == G_BOB)
    assert other.content_known is False
    board = next(item for item in alice.items if item.about == "board")
    assert board.visibility == Visibility.PUBLIC


def test_codenames_spymaster_sees_key_operative_does_not() -> None:
    situation = make_codenames_situation()
    spy_key = next(
        item for item in situation.space_for(SPYMASTER, S0).items if item.about == "key"
    )
    op_key = next(
        item for item in situation.space_for(OPERATIVE, S0).items if item.about == "key"
    )
    assert spy_key.content_known is True
    assert op_key.content_known is False
    clue = situation.actions[0]
    assert clue.parameters == {"clue": "WATER", "count": 2}
    guess_types = {
        option.action_type for option in situation.decision_space_for(OPERATIVE).options
    }
    assert guess_types == {"guess_word"}


def test_wingspan_private_hand_public_tableau() -> None:
    situation = make_wingspan_situation()
    alice = situation.space_for(W_ALICE)
    hand = next(item for item in alice.items if item.holder_id == W_ALICE)
    bob_hand = next(item for item in alice.items if item.holder_id == W_BOB)
    mats = next(item for item in alice.items if item.about == "mats")
    assert hand.content_known is True
    assert bob_hand.content_known is False
    assert mats.visibility == Visibility.PUBLIC
    action_types = {
        option.action_type for option in situation.decision_space_for(W_ALICE).options
    }
    assert action_types >= {
        "play_bird",
        "gain_food",
    }
    after = situation.states[1]
    assert "Steller's Jay" in after.data["mats"][W_ALICE]["forest"]


def test_pandemic_shared_board_and_legal_but_unavailable_action() -> None:
    situation = make_pandemic_situation()
    alice_board = next(
        item for item in situation.space_for(P_ALICE).items if item.about == "board"
    )
    bob_board = next(
        item for item in situation.space_for(P_BOB).items if item.about == "board"
    )
    assert alice_board.payload == bob_board.payload
    hidden = next(
        item
        for item in situation.space_for(P_ALICE).items
        if item.about == "infection_deck"
    )
    assert hidden.visibility == Visibility.HIDDEN
    options = {
        option.action_type: option
        for option in situation.decision_space_for(P_ALICE).options
    }
    assert options["treat_disease"].available is True
    assert options["charter_flight"].legal is True
    assert options["charter_flight"].available is False
    assert situation.states[0].data["cubes"]["Atlanta"]["blue"] == 3
    assert situation.states[1].data["cubes"]["Atlanta"]["blue"] == 0
