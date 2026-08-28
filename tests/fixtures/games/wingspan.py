"""Wingspan: private hand, public tableau, sequential engine-building choice."""

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

ALICE = "wingspan-alice"
BOB = "wingspan-bob"
S0 = "wingspan-s0"
S1 = "wingspan-s1"


def make_wingspan_situation() -> PlaySituation:
    """Alice's birds in hand are private; mats are public."""
    game = Game(
        id="wingspan",
        title="Wingspan",
        release_year=2019,
        min_players=1,
        max_players=5,
    )
    definition = GameDefinition(
        id="wingspan-def",
        game_id=game.id,
        interaction=Interaction.COMPETITIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["play_bird", "gain_food", "lay_eggs", "draw_birds"],
    )
    alice = Player(id=ALICE, name="Alice")
    bob = Player(id=BOB, name="Bob")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=3,
        active_player_id=ALICE,
        phase="action",
        data={
            "hands": {ALICE: ["Steller's Jay"], BOB: ["Bald Eagle"]},
            "mats": {ALICE: {"forest": []}, BOB: {"forest": ["Northern Cardinal"]}},
            "food": {ALICE: ["invertebrate", "seed"], BOB: ["fish"]},
        },
    )
    alice_space = InformationSpace(
        id="wingspan-info-alice",
        player_id=ALICE,
        items=[
            InformationItem(
                id="wingspan-alice-hand",
                visibility=Visibility.PRIVATE,
                about=ALICE,
                holder_id=ALICE,
                content_known=True,
                known_to=[ALICE],
                payload={"cards": ["Steller's Jay"]},
            ),
            InformationItem(
                id="wingspan-bob-hand",
                visibility=Visibility.OTHER_PRIVATE,
                about=BOB,
                holder_id=BOB,
                content_known=False,
                payload={"card_count": 1},
            ),
            InformationItem(
                id="wingspan-mats",
                visibility=Visibility.PUBLIC,
                about="mats",
                content_known=True,
                known_to=[ALICE, BOB],
                payload={"mats": s0.data["mats"], "food": s0.data["food"]},
            ),
        ],
    )
    bob_space = InformationSpace(
        id="wingspan-info-bob",
        player_id=BOB,
        items=[
            InformationItem(
                id="wingspan-bob-own-hand",
                visibility=Visibility.PRIVATE,
                about=BOB,
                holder_id=BOB,
                content_known=True,
                known_to=[BOB],
                payload={"cards": ["Bald Eagle"]},
            ),
            InformationItem(
                id="wingspan-alice-hand-hidden",
                visibility=Visibility.OTHER_PRIVATE,
                about=ALICE,
                holder_id=ALICE,
                content_known=False,
                payload={"card_count": 1},
            ),
            InformationItem(
                id="wingspan-mats-bob",
                visibility=Visibility.PUBLIC,
                about="mats",
                content_known=True,
                known_to=[ALICE, BOB],
                payload={"mats": s0.data["mats"]},
            ),
        ],
    )
    decisions = DecisionSpace(
        id="wingspan-alice-dec",
        player_id=ALICE,
        game_state_id=S0,
        options=[
            DecisionOption(
                id="wingspan-play",
                action_type="play_bird",
                parameters={"bird": "Steller's Jay", "habitat": "forest"},
            ),
            DecisionOption(id="wingspan-food", action_type="gain_food"),
            DecisionOption(id="wingspan-eggs", action_type="lay_eggs"),
            DecisionOption(id="wingspan-draw", action_type="draw_birds"),
        ],
    )
    action = Action(
        id="wingspan-play-jay",
        player_id=ALICE,
        action_type="play_bird",
        parameters={"bird": "Steller's Jay", "habitat": "forest"},
        turn_number=3,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=3,
        data={
            "hands": {ALICE: [], BOB: ["Bald Eagle"]},
            "mats": {
                ALICE: {"forest": ["Steller's Jay"]},
                BOB: {"forest": ["Northern Cardinal"]},
            },
        },
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob],
        states=[s0, s1],
        observations=[
            Observation(
                id="wingspan-obs-alice",
                game_state_id=S0,
                observer_id=ALICE,
                information_space_id=alice_space.id,
            ),
            Observation(
                id="wingspan-obs-bob",
                game_state_id=S0,
                observer_id=BOB,
                information_space_id=bob_space.id,
            ),
        ],
        information_spaces=[alice_space, bob_space],
        decision_spaces=[decisions],
        turns=[
            Turn(
                id="wingspan-turn-3",
                number=3,
                active_player_id=ALICE,
                phase="action",
                game_state_id=S0,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="wingspan-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
    )
