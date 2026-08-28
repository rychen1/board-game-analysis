"""Chess: public state, player-specific legal moves, a sample of a large space."""

from board_game_analysis.domain import (
    Action,
    DecisionOption,
    DecisionSpace,
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
    Turn,
    Visibility,
)

WHITE = "chess-white"
BLACK = "chess-black"
S0 = "chess-s0"
S1 = "chess-s1"
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _public_space(player_id: str) -> InformationSpace:
    return InformationSpace(
        id=f"chess-info-{player_id}",
        player_id=player_id,
        items=[
            InformationItem(
                id=f"chess-board-{player_id}",
                visibility=Visibility.PUBLIC,
                about="board",
                content_known=True,
                known_to=[WHITE, BLACK],
                payload={"fen": FEN, "side_to_move": "white"},
            )
        ],
    )


def make_chess_situation() -> PlaySituation:
    """Starting position: both players see the board; only white chooses."""
    game = Game(id="chess", title="Chess", min_players=2, max_players=2)
    definition = GameDefinition(
        id="chess-def",
        game_id=game.id,
        interaction=Interaction.COMPETITIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["move"],
    )
    white = Player(id=WHITE, name="White", role="white")
    black = Player(id=BLACK, name="Black", role="black")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=1,
        active_player_id=WHITE,
        phase="move",
        data={"fen": FEN},
    )
    white_space = _public_space(WHITE)
    black_space = _public_space(BLACK)
    white_decisions = DecisionSpace(
        id="chess-white-dec",
        player_id=WHITE,
        game_state_id=S0,
        complete=False,
        options=[
            DecisionOption(
                id="chess-e4",
                action_type="move",
                parameters={"from": "e2", "to": "e4"},
            ),
            DecisionOption(
                id="chess-d4",
                action_type="move",
                parameters={"from": "d2", "to": "d4"},
            ),
            DecisionOption(
                id="chess-nf3",
                action_type="move",
                parameters={"from": "g1", "to": "f3"},
            ),
        ],
    )
    black_decisions = DecisionSpace(
        id="chess-black-dec",
        player_id=BLACK,
        game_state_id=S0,
        options=[],
    )
    action = Action(
        id="chess-move-e4",
        player_id=WHITE,
        action_type="move",
        parameters={"from": "e2", "to": "e4"},
        turn_number=1,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=1,
        active_player_id=BLACK,
        data={"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"},
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[white, black],
        states=[s0, s1],
        observations=[
            Observation(
                id="chess-obs-white",
                game_state_id=S0,
                observer_id=WHITE,
                information_space_id=white_space.id,
            ),
            Observation(
                id="chess-obs-black",
                game_state_id=S0,
                observer_id=BLACK,
                information_space_id=black_space.id,
            ),
        ],
        information_spaces=[white_space, black_space],
        decision_spaces=[white_decisions, black_decisions],
        turns=[
            Turn(
                id="chess-turn-1",
                number=1,
                active_player_id=WHITE,
                phase="move",
                game_state_id=S0,
                decision_space_id=white_decisions.id,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="chess-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
    )
