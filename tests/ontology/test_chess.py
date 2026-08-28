"""Structural questions for chess."""

from tests.fixtures.games.chess import BLACK, S0, WHITE, make_chess_situation


def test_chess_state_is_public_to_both_players() -> None:
    situation = make_chess_situation()
    white_board = situation.space_for(WHITE).items[0]
    black_board = situation.space_for(BLACK).items[0]
    assert white_board.content_known is True
    assert black_board.content_known is True
    assert white_board.payload["fen"] == black_board.payload["fen"]
    assert situation.observation_for(WHITE).game_state_id == S0
    assert situation.observation_for(BLACK).game_state_id == S0


def test_chess_only_side_to_move_has_options() -> None:
    situation = make_chess_situation()
    white = situation.decision_space_for(WHITE)
    black = situation.decision_space_for(BLACK)
    assert white.options
    assert white.complete is False
    assert all(option.action_type == "move" for option in white.options)
    assert black.options == []


def test_chess_move_transitions_fen() -> None:
    situation = make_chess_situation()
    assert situation.actions[0].parameters == {"from": "e2", "to": "e4"}
    assert situation.states[0].data["fen"].endswith("w KQkq - 0 1")
    assert "4P3" in situation.states[1].data["fen"]
