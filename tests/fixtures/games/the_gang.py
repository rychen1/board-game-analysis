"""The Gang: cooperative poker-like private hands and public board."""

from board_game_analysis.domain import (
    Action,
    DecisionOption,
    DecisionSpace,
    DecisionTiming,
    ExtractedInterpretation,
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

ALICE = "gang-alice"
BOB = "gang-bob"
S0 = "gang-s0"
S1 = "gang-s1"


def make_the_gang_situation() -> PlaySituation:
    """Private hole cards, public community cards, cooperative bet."""
    game = Game(
        id="the-gang",
        title="The Gang",
        release_year=2024,
        min_players=2,
        max_players=6,
    )
    definition = GameDefinition(
        id="gang-def",
        game_id=game.id,
        interaction=Interaction.COOPERATIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["check", "raise", "fold"],
        notes="Players share a goal against the house and may not discuss hole cards.",
    )
    alice = Player(id=ALICE, name="Alice")
    bob = Player(id=BOB, name="Bob")
    board = ["As", "Kh", "7d"]
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=4,
        active_player_id=ALICE,
        phase="flop_betting",
        data={"holes": {ALICE: ["Ad", "Kc"], BOB: ["Qh", "2c"]}, "board": board},
    )

    def space(observer: str, hole: list[str], other: str) -> InformationSpace:
        return InformationSpace(
            id=f"gang-info-{observer}",
            player_id=observer,
            items=[
                InformationItem(
                    id=f"gang-{observer}-hole",
                    visibility=Visibility.PRIVATE,
                    about=observer,
                    holder_id=observer,
                    content_known=True,
                    known_to=[observer],
                    payload={"cards": hole},
                ),
                InformationItem(
                    id=f"gang-{observer}-other",
                    visibility=Visibility.OTHER_PRIVATE,
                    about=other,
                    holder_id=other,
                    content_known=False,
                    payload={"card_count": 2},
                ),
                InformationItem(
                    id=f"gang-{observer}-board",
                    visibility=Visibility.PUBLIC,
                    about="board",
                    content_known=True,
                    known_to=[ALICE, BOB],
                    payload={"cards": board},
                ),
            ],
        )

    alice_space = space(ALICE, ["Ad", "Kc"], BOB)
    bob_space = space(BOB, ["Qh", "2c"], ALICE)
    decisions = DecisionSpace(
        id="gang-alice-decisions",
        player_id=ALICE,
        game_state_id=S0,
        options=[
            DecisionOption(id="gang-check", action_type="check"),
            DecisionOption(
                id="gang-raise",
                action_type="raise",
                parameters={"amount": 2},
            ),
            DecisionOption(id="gang-fold", action_type="fold"),
        ],
    )
    action = Action(
        id="gang-alice-raise",
        player_id=ALICE,
        action_type="raise",
        parameters={"amount": 2},
        turn_number=4,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        phase="flop_betting",
        data={**s0.data, "current_bet": 2, "last_actor": ALICE},
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob],
        states=[s0, s1],
        observations=[
            Observation(
                id="gang-obs-alice",
                game_state_id=S0,
                observer_id=ALICE,
                information_space_id=alice_space.id,
            ),
            Observation(
                id="gang-obs-bob",
                game_state_id=S0,
                observer_id=BOB,
                information_space_id=bob_space.id,
            ),
        ],
        information_spaces=[alice_space, bob_space],
        decision_spaces=[decisions],
        turns=[
            Turn(
                id="gang-turn-4",
                number=4,
                active_player_id=ALICE,
                phase="flop_betting",
                game_state_id=S0,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="gang-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
        interpretations=[
            ExtractedInterpretation(
                id="gang-interp",
                game_id=game.id,
                label="constrained_communication",
                extractor="human",
            )
        ],
    )
