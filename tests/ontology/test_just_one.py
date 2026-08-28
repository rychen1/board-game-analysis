"""Structural questions for Just One."""

from board_game_analysis.domain import DecisionTiming, Visibility
from tests.fixtures.games.just_one import (
    GIVER_A,
    GUESSER,
    S0,
    S1,
    TARGET,
    make_just_one_situation,
)


def test_just_one_givers_know_target_guesser_does_not() -> None:
    situation = make_just_one_situation()
    giver = next(
        item
        for item in situation.space_for(GIVER_A, S0).items
        if item.about == "target"
    )
    guesser = next(
        item
        for item in situation.space_for(GUESSER, S0).items
        if item.about == "target"
    )
    assert giver.content_known is True
    assert giver.payload["word"] == TARGET
    assert guesser.content_known is False
    assert guesser.visibility == Visibility.HIDDEN


def test_just_one_simultaneous_clue_submission_and_duplicate_removal() -> None:
    situation = make_just_one_situation()
    assert situation.definition.decision_timing == DecisionTiming.SIMULTANEOUS
    assert {action.action_type for action in situation.actions} == {"submit_clue"}
    transition = situation.transitions[0]
    assert set(transition.action_ids) == {action.id for action in situation.actions}
    guesser_after = situation.space_for(GUESSER, S1)
    clues = next(item for item in guesser_after.items if item.about == "clues")
    assert clues.payload["submitted"] == ["STAR", "STAR"]
    assert clues.payload["visible_clues"] == []
    assert clues.payload["removed_duplicates"] == ["STAR"]


def test_just_one_guesser_decides_after_clues() -> None:
    situation = make_just_one_situation()
    guess_space = situation.decision_space_for(GUESSER, S1)
    assert [option.action_type for option in guess_space.options] == ["guess"]
    target = next(
        item
        for item in situation.space_for(GUESSER, S1).items
        if item.about == "target"
    )
    assert target.content_known is False
