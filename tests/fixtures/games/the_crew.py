"""The Crew: private hands, public tasks, constrained communication."""

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

ALICE = "crew-alice"
BOB = "crew-bob"
S0 = "crew-s0"
S1 = "crew-s1"


def make_the_crew_situation() -> PlaySituation:
    """Alice knows her hand, not Bob's, and communicates one card."""
    game = Game(
        id="the-crew",
        title="The Crew",
        release_year=2019,
        min_players=3,
        max_players=5,
    )
    definition = GameDefinition(
        id="crew-def",
        game_id=game.id,
        interaction=Interaction.COOPERATIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["play_card", "communicate_card"],
    )
    alice = Player(id=ALICE, name="Alice")
    bob = Player(id=BOB, name="Bob")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=2,
        active_player_id=ALICE,
        phase="communication",
        data={
            "hands": {ALICE: ["pink-9", "blue-3"], BOB: ["green-5", "pink-1"]},
            "tasks": ["win-pink-9"],
            "communication_used": {ALICE: False, BOB: False},
        },
    )
    alice_space = InformationSpace(
        id="crew-info-alice-s0",
        player_id=ALICE,
        items=[
            InformationItem(
                id="crew-alice-hand",
                visibility=Visibility.PRIVATE,
                about=ALICE,
                holder_id=ALICE,
                content_known=True,
                known_to=[ALICE],
                payload={"cards": ["pink-9", "blue-3"]},
            ),
            InformationItem(
                id="crew-bob-hand-hidden",
                visibility=Visibility.OTHER_PRIVATE,
                about=BOB,
                holder_id=BOB,
                content_known=False,
                payload={"card_count": 2},
            ),
            InformationItem(
                id="crew-tasks",
                visibility=Visibility.PUBLIC,
                about="tasks",
                content_known=True,
                known_to=[ALICE, BOB],
                payload={"tasks": ["win-pink-9"]},
            ),
        ],
    )
    bob_space = InformationSpace(
        id="crew-info-bob-s0",
        player_id=BOB,
        items=[
            InformationItem(
                id="crew-bob-own-hand",
                visibility=Visibility.PRIVATE,
                about=BOB,
                holder_id=BOB,
                content_known=True,
                known_to=[BOB],
                payload={"cards": ["green-5", "pink-1"]},
            ),
            InformationItem(
                id="crew-alice-hand-hidden",
                visibility=Visibility.OTHER_PRIVATE,
                about=ALICE,
                holder_id=ALICE,
                content_known=False,
                payload={"card_count": 2},
            ),
            InformationItem(
                id="crew-tasks-bob",
                visibility=Visibility.PUBLIC,
                about="tasks",
                content_known=True,
                known_to=[ALICE, BOB],
                payload={"tasks": ["win-pink-9"]},
            ),
        ],
    )
    decisions = DecisionSpace(
        id="crew-alice-decisions",
        player_id=ALICE,
        game_state_id=S0,
        options=[
            DecisionOption(
                id="crew-communicate",
                action_type="communicate_card",
                parameters={"card": "pink-9", "position": "highest"},
            ),
            DecisionOption(
                id="crew-play",
                action_type="play_card",
                parameters={"card": "blue-3"},
            ),
        ],
    )
    action = Action(
        id="crew-alice-communicate",
        player_id=ALICE,
        action_type="communicate_card",
        parameters={"card": "pink-9", "position": "highest"},
        turn_number=2,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=2,
        data={**s0.data, "communication_used": {ALICE: True, BOB: False}},
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob],
        states=[s0, s1],
        observations=[
            Observation(
                id="crew-obs-alice",
                game_state_id=S0,
                observer_id=ALICE,
                information_space_id=alice_space.id,
            ),
            Observation(
                id="crew-obs-bob",
                game_state_id=S0,
                observer_id=BOB,
                information_space_id=bob_space.id,
            ),
        ],
        information_spaces=[alice_space, bob_space],
        decision_spaces=[decisions],
        turns=[
            Turn(
                id="crew-turn-2",
                number=2,
                active_player_id=ALICE,
                phase="communication",
                game_state_id=S0,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="crew-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
        interpretations=[
            ExtractedInterpretation(
                id="crew-interp-comm",
                game_id=game.id,
                label="constrained_communication",
                extractor="human",
            )
        ],
    )
