"""Poker: private holes, public board, hidden opponent, inferred range, bet."""

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

ALICE = "poker-alice"
BOB = "poker-bob"
S0 = "poker-s0"
S1 = "poker-s1"
BOARD = ["As", "7c", "2d"]


def make_poker_situation() -> PlaySituation:
    """Texas Hold'em flop: Alice knows her cards and infers Bob's range."""
    game = Game(id="poker", title="Texas Hold'em Poker", min_players=2, max_players=10)
    definition = GameDefinition(
        id="poker-def",
        game_id=game.id,
        interaction=Interaction.COMPETITIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["check", "bet", "call", "raise", "fold"],
    )
    alice = Player(id=ALICE, name="Alice")
    bob = Player(id=BOB, name="Bob")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=3,
        active_player_id=ALICE,
        phase="flop",
        data={
            "holes": {ALICE: ["Ah", "Kd"], BOB: ["9s", "9h"]},
            "board": BOARD,
            "pot": 12,
        },
    )
    alice_space = InformationSpace(
        id="poker-info-alice",
        player_id=ALICE,
        items=[
            InformationItem(
                id="poker-alice-hole",
                visibility=Visibility.PRIVATE,
                about=ALICE,
                holder_id=ALICE,
                content_known=True,
                known_to=[ALICE],
                payload={"cards": ["Ah", "Kd"]},
            ),
            InformationItem(
                id="poker-bob-hole",
                visibility=Visibility.OTHER_PRIVATE,
                about=BOB,
                holder_id=BOB,
                content_known=False,
                payload={"card_count": 2},
            ),
            InformationItem(
                id="poker-board",
                visibility=Visibility.PUBLIC,
                about="board",
                content_known=True,
                known_to=[ALICE, BOB],
                payload={"cards": BOARD, "pot": 12},
            ),
            InformationItem(
                id="poker-alice-inference",
                visibility=Visibility.INFERRED,
                about=BOB,
                holder_id=BOB,
                content_known=False,
                payload={"likely_range": ["mid_pair", "ace_x"], "confidence": 0.3},
            ),
        ],
    )
    bob_space = InformationSpace(
        id="poker-info-bob",
        player_id=BOB,
        items=[
            InformationItem(
                id="poker-bob-own",
                visibility=Visibility.PRIVATE,
                about=BOB,
                holder_id=BOB,
                content_known=True,
                known_to=[BOB],
                payload={"cards": ["9s", "9h"]},
            ),
            InformationItem(
                id="poker-alice-hidden",
                visibility=Visibility.OTHER_PRIVATE,
                about=ALICE,
                holder_id=ALICE,
                content_known=False,
                payload={"card_count": 2},
            ),
            InformationItem(
                id="poker-board-bob",
                visibility=Visibility.PUBLIC,
                about="board",
                content_known=True,
                known_to=[ALICE, BOB],
                payload={"cards": BOARD},
            ),
        ],
    )
    decisions = DecisionSpace(
        id="poker-alice-dec",
        player_id=ALICE,
        game_state_id=S0,
        options=[
            DecisionOption(id="poker-check", action_type="check"),
            DecisionOption(id="poker-bet", action_type="bet", parameters={"amount": 8}),
            DecisionOption(id="poker-fold", action_type="fold"),
        ],
    )
    action = Action(
        id="poker-alice-bet",
        player_id=ALICE,
        action_type="bet",
        parameters={"amount": 8},
        turn_number=3,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        phase="flop",
        data={**s0.data, "pot": 20, "to_call": 8, "last_bet": ALICE},
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob],
        states=[s0, s1],
        observations=[
            Observation(
                id="poker-obs-alice",
                game_state_id=S0,
                observer_id=ALICE,
                information_space_id=alice_space.id,
            ),
            Observation(
                id="poker-obs-bob",
                game_state_id=S0,
                observer_id=BOB,
                information_space_id=bob_space.id,
            ),
        ],
        information_spaces=[alice_space, bob_space],
        decision_spaces=[decisions],
        turns=[
            Turn(
                id="poker-turn-3",
                number=3,
                active_player_id=ALICE,
                phase="flop",
                game_state_id=S0,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="poker-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
    )
