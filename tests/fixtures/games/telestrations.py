"""Telestrations: information transforms as it passes along a chain."""

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

ALICE = "tele-alice"
BOB = "tele-bob"
CAROL = "tele-carol"
S0 = "tele-s0"
S1 = "tele-s1"
S2 = "tele-s2"
WORD = "octopus"
DRAWING = "drawing:round-animal-tentacles"
GUESS = "squid"


def make_telestrations_situation() -> PlaySituation:
    """Word → drawing → guess; each player sees only their adjacent step."""
    game = Game(
        id="telestrations",
        title="Telestrations",
        release_year=2009,
        min_players=4,
        max_players=8,
    )
    definition = GameDefinition(
        id="tele-def",
        game_id=game.id,
        interaction=Interaction.MIXED,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["write_word", "draw", "guess_word"],
        notes=(
            "Party game: scoring is comparative; "
            "the chain is the structure of interest."
        ),
    )
    alice = Player(id=ALICE, name="Alice")
    bob = Player(id=BOB, name="Bob")
    carol = Player(id=CAROL, name="Carol")
    word_item = InformationItem(
        id="tele-word",
        visibility=Visibility.PRIVATE,
        about="prompt",
        holder_id=ALICE,
        content_known=True,
        known_to=[ALICE],
        payload={"medium": "text", "content": WORD},
    )
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=1,
        active_player_id=BOB,
        phase="draw",
        data={"chain": [{"player": ALICE, "medium": "text", "content": WORD}]},
    )
    alice_s0 = InformationSpace(
        id="tele-info-alice-s0",
        player_id=ALICE,
        items=[word_item],
    )
    bob_s0 = InformationSpace(
        id="tele-info-bob-s0",
        player_id=BOB,
        items=[
            InformationItem(
                id="tele-bob-sees-word",
                visibility=Visibility.COMMUNICATED,
                about="prompt",
                communicated_by=ALICE,
                predecessor_id=word_item.id,
                content_known=True,
                known_to=[BOB],
                payload={"medium": "text", "content": WORD},
            )
        ],
    )
    carol_s0 = InformationSpace(
        id="tele-info-carol-s0",
        player_id=CAROL,
        items=[
            InformationItem(
                id="tele-carol-gap-s0",
                visibility=Visibility.HIDDEN,
                about="chain",
                content_known=False,
                payload={"known_steps": []},
            )
        ],
    )
    bob_decisions = DecisionSpace(
        id="tele-bob-dec",
        player_id=BOB,
        game_state_id=S0,
        options=[
            DecisionOption(
                id="tele-draw",
                action_type="draw",
                parameters={"from_word": WORD},
            )
        ],
    )
    draw = Action(
        id="tele-bob-draw",
        player_id=BOB,
        action_type="draw",
        parameters={"content": DRAWING},
        turn_number=1,
    )
    drawing_item = InformationItem(
        id="tele-drawing",
        visibility=Visibility.COMMUNICATED,
        about="prompt",
        communicated_by=BOB,
        predecessor_id=word_item.id,
        holder_id=BOB,
        content_known=True,
        known_to=[BOB],
        payload={"medium": "drawing", "content": DRAWING},
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=2,
        active_player_id=CAROL,
        phase="guess",
        data={
            "chain": [
                {"player": ALICE, "medium": "text", "content": WORD},
                {"player": BOB, "medium": "drawing", "content": DRAWING},
            ]
        },
    )
    carol_s1 = InformationSpace(
        id="tele-info-carol-s1",
        player_id=CAROL,
        items=[
            InformationItem(
                id="tele-carol-sees-drawing",
                visibility=Visibility.COMMUNICATED,
                about="prompt",
                communicated_by=BOB,
                predecessor_id=drawing_item.id,
                content_known=True,
                known_to=[CAROL],
                payload={"medium": "drawing", "content": DRAWING},
            ),
            InformationItem(
                id="tele-carol-no-word",
                visibility=Visibility.HIDDEN,
                about="original_word",
                content_known=False,
            ),
        ],
    )
    alice_s1 = InformationSpace(
        id="tele-info-alice-s1",
        player_id=ALICE,
        items=[
            word_item.model_copy(update={"id": "tele-word-s1"}),
            InformationItem(
                id="tele-alice-no-drawing",
                visibility=Visibility.HIDDEN,
                about="drawing",
                content_known=False,
            ),
        ],
    )
    carol_decisions = DecisionSpace(
        id="tele-carol-dec",
        player_id=CAROL,
        game_state_id=S1,
        options=[
            DecisionOption(
                id="tele-guess",
                action_type="guess_word",
                parameters={"word": GUESS},
            )
        ],
    )
    guess = Action(
        id="tele-carol-guess",
        player_id=CAROL,
        action_type="guess_word",
        parameters={"word": GUESS},
        turn_number=2,
    )
    s2 = GameState(
        id=S2,
        game_id=game.id,
        phase="reveal",
        data={
            "chain": [
                {"player": ALICE, "medium": "text", "content": WORD},
                {"player": BOB, "medium": "drawing", "content": DRAWING},
                {"player": CAROL, "medium": "text", "content": GUESS},
            ]
        },
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob, carol],
        states=[s0, s1, s2],
        observations=[
            Observation(
                id="tele-obs-alice-s0",
                game_state_id=S0,
                observer_id=ALICE,
                information_space_id=alice_s0.id,
            ),
            Observation(
                id="tele-obs-bob-s0",
                game_state_id=S0,
                observer_id=BOB,
                information_space_id=bob_s0.id,
            ),
            Observation(
                id="tele-obs-carol-s0",
                game_state_id=S0,
                observer_id=CAROL,
                information_space_id=carol_s0.id,
            ),
            Observation(
                id="tele-obs-carol-s1",
                game_state_id=S1,
                observer_id=CAROL,
                information_space_id=carol_s1.id,
            ),
            Observation(
                id="tele-obs-alice-s1",
                game_state_id=S1,
                observer_id=ALICE,
                information_space_id=alice_s1.id,
            ),
        ],
        information_spaces=[alice_s0, bob_s0, carol_s0, carol_s1, alice_s1],
        decision_spaces=[bob_decisions, carol_decisions],
        turns=[
            Turn(
                id="tele-turn-draw",
                number=1,
                active_player_id=BOB,
                phase="draw",
                game_state_id=S0,
                action_id=draw.id,
                resulting_state_id=S1,
            ),
            Turn(
                id="tele-turn-guess",
                number=2,
                active_player_id=CAROL,
                phase="guess",
                game_state_id=S1,
                action_id=guess.id,
                resulting_state_id=S2,
            ),
        ],
        actions=[draw, guess],
        transitions=[
            StateTransition(
                id="tele-tr-draw",
                from_state_id=S0,
                to_state_id=S1,
                action_id=draw.id,
            ),
            StateTransition(
                id="tele-tr-guess",
                from_state_id=S1,
                to_state_id=S2,
                action_id=guess.id,
            ),
        ],
        interpretations=[
            ExtractedInterpretation(
                id="tele-interp-transform",
                game_id=game.id,
                label="information_transformation",
                extractor="human",
            )
        ],
    )
