"""Hanabi: inverted hands, clues, and per-player views of one state."""

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
    TransitionKind,
    Turn,
    Visibility,
)

ALICE = "hanabi-alice"
BOB = "hanabi-bob"
CAROL = "hanabi-carol"
S0 = "hanabi-s0"
S1 = "hanabi-s1"


def _hand_item(
    item_id: str,
    *,
    observer_id: str,
    holder_id: str,
    slot: int,
    color: str,
    rank: int,
) -> InformationItem:
    own = holder_id == observer_id
    return InformationItem(
        id=item_id,
        visibility=Visibility.UNKNOWN_TO_SELF if own else Visibility.OTHER_PRIVATE,
        about=holder_id,
        holder_id=holder_id,
        content_known=not own,
        known_to=[] if own else [observer_id],
        payload=(
            {"slot": slot} if own else {"slot": slot, "color": color, "rank": rank}
        ),
    )


def _space(
    observer_id: str, suffix: str, extra: list[InformationItem]
) -> InformationSpace:
    hands = [
        _hand_item(
            f"hanabi-{observer_id}-alice-{suffix}",
            observer_id=observer_id,
            holder_id=ALICE,
            slot=0,
            color="red",
            rank=1,
        ),
        _hand_item(
            f"hanabi-{observer_id}-bob-{suffix}",
            observer_id=observer_id,
            holder_id=BOB,
            slot=0,
            color="blue",
            rank=3,
        ),
        _hand_item(
            f"hanabi-{observer_id}-carol-{suffix}",
            observer_id=observer_id,
            holder_id=CAROL,
            slot=0,
            color="white",
            rank=4,
        ),
        InformationItem(
            id=f"hanabi-{observer_id}-clues-{suffix}",
            visibility=Visibility.PUBLIC,
            about="clue_tokens",
            content_known=True,
            known_to=[ALICE, BOB, CAROL],
            payload={"remaining": 8},
        ),
    ]
    return InformationSpace(
        id=f"hanabi-info-{observer_id}-{suffix}",
        player_id=observer_id,
        items=hands + extra,
    )


def make_hanabi_situation() -> PlaySituation:
    """Alice clues Bob about a rank-3 card she can see and he cannot."""
    game = Game(
        id="hanabi",
        title="Hanabi",
        release_year=2010,
        min_players=2,
        max_players=5,
    )
    definition = GameDefinition(
        id="hanabi-def",
        game_id=game.id,
        interaction=Interaction.COOPERATIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["play_card", "discard_card", "give_clue"],
    )
    alice = Player(id=ALICE, name="Alice")
    bob = Player(id=BOB, name="Bob")
    carol = Player(id=CAROL, name="Carol")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=1,
        active_player_id=ALICE,
        phase="main",
        data={
            "hands": {
                ALICE: [{"slot": 0, "color": "red", "rank": 1}],
                BOB: [{"slot": 0, "color": "blue", "rank": 3}],
                CAROL: [{"slot": 0, "color": "white", "rank": 4}],
            },
            "clue_tokens": 8,
        },
    )
    spaces_s0 = [
        _space(ALICE, "s0", []),
        _space(BOB, "s0", []),
        _space(CAROL, "s0", []),
    ]
    clue = Action(
        id="hanabi-clue-bob-3",
        player_id=ALICE,
        action_type="give_clue",
        parameters={"target": BOB, "rank": 3, "slots": [0]},
        turn_number=1,
    )
    communicated = InformationItem(
        id="hanabi-bob-clue-rank-3",
        visibility=Visibility.COMMUNICATED,
        about=BOB,
        holder_id=BOB,
        communicated_by=ALICE,
        content_known=True,
        known_to=[ALICE, BOB, CAROL],
        payload={"slots": [0], "rank": 3},
    )
    spaces_s1 = [
        _space(ALICE, "s1", []),
        _space(BOB, "s1", [communicated]),
        _space(CAROL, "s1", []),
    ]
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=1,
        active_player_id=BOB,
        phase="main",
        data={**s0.data, "clue_tokens": 7},
    )
    alice_decisions = DecisionSpace(
        id="hanabi-alice-decisions",
        player_id=ALICE,
        game_state_id=S0,
        options=[
            DecisionOption(
                id="hanabi-play",
                action_type="play_card",
                parameters={"slot": 0},
            ),
            DecisionOption(
                id="hanabi-discard",
                action_type="discard_card",
                parameters={"slot": 0},
            ),
            DecisionOption(
                id="hanabi-clue-opt",
                action_type="give_clue",
                parameters={"target": BOB, "rank": 3},
                available=True,
            ),
        ],
    )
    observations = [
        Observation(
            id=f"hanabi-obs-{space.player_id}-{S0 if 's0' in space.id else S1}",
            game_state_id=S0 if "s0" in space.id else S1,
            observer_id=space.player_id,
            information_space_id=space.id,
        )
        for space in spaces_s0 + spaces_s1
    ]
    turn = Turn(
        id="hanabi-turn-1",
        number=1,
        active_player_id=ALICE,
        phase="main",
        game_state_id=S0,
        information_space_id=spaces_s0[0].id,
        decision_space_id=alice_decisions.id,
        action_id=clue.id,
        resulting_state_id=S1,
    )
    transition = StateTransition(
        id="hanabi-tr-clue",
        from_state_id=S0,
        to_state_id=S1,
        action_id=clue.id,
        kind=TransitionKind.PLAYER,
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[alice, bob, carol],
        states=[s0, s1],
        observations=observations,
        information_spaces=spaces_s0 + spaces_s1,
        decision_spaces=[alice_decisions],
        turns=[turn],
        actions=[clue],
        transitions=[transition],
        interpretations=[
            ExtractedInterpretation(
                id="hanabi-interp-asymmetric",
                game_id=game.id,
                label="asymmetric_information",
                description="Players see others' cards, not their own.",
                extractor="human",
            )
        ],
    )
