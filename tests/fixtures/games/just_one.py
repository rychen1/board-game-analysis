"""Just One: role-asymmetric knowledge and simultaneous clues."""

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

GIVER_A = "justone-giver-a"
GIVER_B = "justone-giver-b"
GUESSER = "justone-guesser"
S0 = "justone-s0"
S1 = "justone-s1"
TARGET = "TELESCOPE"


def _target_item(observer_id: str, *, known: bool) -> InformationItem:
    return InformationItem(
        id=f"justone-target-{observer_id}",
        visibility=Visibility.PRIVATE if known else Visibility.HIDDEN,
        about="target",
        content_known=known,
        known_to=[GIVER_A, GIVER_B] if known else [],
        payload={"word": TARGET} if known else {"word": None},
    )


def make_just_one_situation() -> PlaySituation:
    """Givers know the target; guesser does not. Duplicate clues are removed."""
    game = Game(
        id="just-one",
        title="Just One",
        release_year=2018,
        min_players=3,
        max_players=7,
    )
    definition = GameDefinition(
        id="justone-def",
        game_id=game.id,
        interaction=Interaction.COOPERATIVE,
        decision_timing=DecisionTiming.SIMULTANEOUS,
        legal_action_types=["submit_clue", "guess"],
        player_roles=["clue_giver", "guesser"],
    )
    giver_a = Player(id=GIVER_A, name="Avery", role="clue_giver")
    giver_b = Player(id=GIVER_B, name="Blair", role="clue_giver")
    guesser = Player(id=GUESSER, name="Gwen", role="guesser")
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=1,
        phase="clue_submission",
        data={"target": TARGET, "clues": {}},
    )
    spaces_s0 = [
        InformationSpace(
            id=f"justone-info-{pid}-s0",
            player_id=pid,
            items=[_target_item(pid, known=pid != GUESSER)],
        )
        for pid in (GIVER_A, GIVER_B, GUESSER)
    ]
    giver_options = [
        DecisionOption(id=f"justone-clue-{pid}", action_type="submit_clue")
        for pid in (GIVER_A, GIVER_B)
    ]
    decisions_s0 = [
        DecisionSpace(
            id=f"justone-dec-{pid}",
            player_id=pid,
            game_state_id=S0,
            options=[giver_options[i]],
        )
        for i, pid in enumerate((GIVER_A, GIVER_B))
    ]
    actions = [
        Action(
            id="justone-clue-star-a",
            player_id=GIVER_A,
            action_type="submit_clue",
            parameters={"clue": "STAR"},
            turn_number=1,
        ),
        Action(
            id="justone-clue-star-b",
            player_id=GIVER_B,
            action_type="submit_clue",
            parameters={"clue": "STAR"},
            turn_number=1,
        ),
    ]
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=1,
        phase="guess",
        data={"target": TARGET, "submitted": ["STAR", "STAR"], "visible_clues": []},
    )
    guesser_s1 = InformationSpace(
        id="justone-info-guesser-s1",
        player_id=GUESSER,
        items=[
            _target_item(GUESSER, known=False),
            InformationItem(
                id="justone-visible-clues",
                visibility=Visibility.PUBLIC,
                about="clues",
                content_known=True,
                payload={
                    "submitted": ["STAR", "STAR"],
                    "visible_clues": [],
                    "removed_duplicates": ["STAR"],
                },
            ),
        ],
    )
    guess_space = DecisionSpace(
        id="justone-guess-dec",
        player_id=GUESSER,
        game_state_id=S1,
        options=[DecisionOption(id="justone-guess", action_type="guess")],
    )
    observations = [
        Observation(
            id=f"justone-obs-{space.player_id}-s0",
            game_state_id=S0,
            observer_id=space.player_id,
            information_space_id=space.id,
        )
        for space in spaces_s0
    ] + [
        Observation(
            id="justone-obs-guesser-s1",
            game_state_id=S1,
            observer_id=GUESSER,
            information_space_id=guesser_s1.id,
        )
    ]
    return PlaySituation(
        game=game,
        definition=definition,
        players=[giver_a, giver_b, guesser],
        states=[s0, s1],
        observations=observations,
        information_spaces=[*spaces_s0, guesser_s1],
        decision_spaces=[*decisions_s0, guess_space],
        turns=[
            Turn(
                id="justone-turn-clues",
                number=1,
                phase="clue_submission",
                game_state_id=S0,
                resulting_state_id=S1,
            ),
            Turn(
                id="justone-turn-guess",
                number=1,
                active_player_id=GUESSER,
                phase="guess",
                game_state_id=S1,
                decision_space_id=guess_space.id,
            ),
        ],
        actions=actions,
        transitions=[
            StateTransition(
                id="justone-tr-clues",
                from_state_id=S0,
                to_state_id=S1,
                action_ids=[action.id for action in actions],
                kind=TransitionKind.PLAYER,
            )
        ],
        interpretations=[
            ExtractedInterpretation(
                id="justone-interp-roles",
                game_id=game.id,
                label="role_asymmetric_information",
                extractor="human",
            )
        ],
    )
