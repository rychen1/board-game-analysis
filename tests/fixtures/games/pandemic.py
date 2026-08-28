"""Pandemic: shared public board, hidden decks, cooperative treat action."""

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

ALICE = "pandemic-alice"
BOB = "pandemic-bob"
S0 = "pandemic-s0"
S1 = "pandemic-s1"


def _shared_items(observer_id: str) -> list[InformationItem]:
    return [
        InformationItem(
            id=f"pandemic-board-{observer_id}",
            visibility=Visibility.PUBLIC,
            about="board",
            content_known=True,
            known_to=[ALICE, BOB],
            payload={"atlanta_cubes": {"blue": 3}, "outbreaks": 0},
        ),
        InformationItem(
            id=f"pandemic-hands-{observer_id}",
            visibility=Visibility.PUBLIC,
            about="player_hands",
            content_known=True,
            known_to=[ALICE, BOB],
            payload={"hands": {ALICE: ["Atlanta", "Chicago"], BOB: ["Paris"]}},
        ),
        InformationItem(
            id=f"pandemic-infection-deck-{observer_id}",
            visibility=Visibility.HIDDEN,
            about="infection_deck",
            content_known=False,
            payload={"remaining": 44},
        ),
    ]


def make_pandemic_situation() -> PlaySituation:
    """Both players share board knowledge; treating Atlanta changes public state."""
    game = Game(
        id="pandemic",
        title="Pandemic",
        release_year=2008,
        min_players=2,
        max_players=4,
    )
    definition = GameDefinition(
        id="pandemic-def",
        game_id=game.id,
        interaction=Interaction.COOPERATIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=[
            "drive_ferry",
            "treat_disease",
            "share_knowledge",
            "charter_flight",
        ],
    )
    alice = Player(id=ALICE, name="Alice", role="medic")
    bob = Player(id=BOB, name="Bob", role="scientist")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=2,
        active_player_id=ALICE,
        phase="actions",
        data={
            "locations": {ALICE: "Atlanta", BOB: "Paris"},
            "cubes": {"Atlanta": {"blue": 3}},
            "hands": {ALICE: ["Atlanta", "Chicago"], BOB: ["Paris"]},
        },
    )
    alice_space = InformationSpace(
        id="pandemic-info-alice",
        player_id=ALICE,
        items=_shared_items(ALICE),
    )
    bob_space = InformationSpace(
        id="pandemic-info-bob",
        player_id=BOB,
        items=_shared_items(BOB),
    )
    decisions = DecisionSpace(
        id="pandemic-alice-dec",
        player_id=ALICE,
        game_state_id=S0,
        options=[
            DecisionOption(
                id="pandemic-treat",
                action_type="treat_disease",
                parameters={"city": "Atlanta", "color": "blue"},
                available=True,
            ),
            DecisionOption(
                id="pandemic-charter",
                action_type="charter_flight",
                parameters={"to": "Tokyo"},
                legal=True,
                available=False,
                notes=(
                    "Requires discarding the city card matching the current location."
                ),
            ),
        ],
    )
    action = Action(
        id="pandemic-treat-atlanta",
        player_id=ALICE,
        action_type="treat_disease",
        parameters={"city": "Atlanta", "color": "blue"},
        turn_number=2,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=2,
        active_player_id=ALICE,
        phase="actions",
        data={
            "locations": {ALICE: "Atlanta", BOB: "Paris"},
            "cubes": {"Atlanta": {"blue": 0}},
        },
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob],
        states=[s0, s1],
        observations=[
            Observation(
                id="pandemic-obs-alice",
                game_state_id=S0,
                observer_id=ALICE,
                information_space_id=alice_space.id,
            ),
            Observation(
                id="pandemic-obs-bob",
                game_state_id=S0,
                observer_id=BOB,
                information_space_id=bob_space.id,
            ),
        ],
        information_spaces=[alice_space, bob_space],
        decision_spaces=[decisions],
        turns=[
            Turn(
                id="pandemic-turn-2",
                number=2,
                active_player_id=ALICE,
                phase="actions",
                game_state_id=S0,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="pandemic-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
        interpretations=[
            ExtractedInterpretation(
                id="pandemic-interp",
                game_id=game.id,
                label="shared_public_information",
                extractor="human",
            )
        ],
    )
