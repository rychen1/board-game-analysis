"""Codenames: spymaster key vs operative view, then a clue."""

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

SPYMASTER = "codenames-spymaster"
OPERATIVE = "codenames-operative"
S0 = "codenames-s0"
S1 = "codenames-s1"
WORDS = ["OCEAN", "WAVE", "MOON", "BOOT"]


def make_codenames_situation() -> PlaySituation:
    """Spymaster sees the key; operative sees words only, then a clue."""
    game = Game(
        id="codenames",
        title="Codenames",
        release_year=2015,
        min_players=4,
        max_players=8,
    )
    definition = GameDefinition(
        id="codenames-def",
        game_id=game.id,
        interaction=Interaction.COMPETITIVE,
        decision_timing=DecisionTiming.SEQUENTIAL,
        legal_action_types=["give_clue", "guess_word"],
        player_roles=["spymaster", "operative"],
    )
    spymaster = Player(id=SPYMASTER, name="Sam", role="spymaster")
    operative = Player(id=OPERATIVE, name="Oak", role="operative")
    key = {"OCEAN": "red", "WAVE": "red", "MOON": "blue", "BOOT": "neutral"}
    s0 = GameState(
        id=S0,
        game_id=game.id,
        turn_number=1,
        active_player_id=SPYMASTER,
        phase="clue",
        data={"words": WORDS, "key": key},
    )
    board_item = InformationItem(
        id="codenames-board",
        visibility=Visibility.PUBLIC,
        about="board",
        content_known=True,
        known_to=[SPYMASTER, OPERATIVE],
        payload={"words": WORDS},
    )
    spy_space = InformationSpace(
        id="codenames-info-spy-s0",
        player_id=SPYMASTER,
        items=[
            board_item,
            InformationItem(
                id="codenames-key",
                visibility=Visibility.PRIVATE,
                about="key",
                holder_id=SPYMASTER,
                content_known=True,
                known_to=[SPYMASTER],
                payload={"key": key},
            ),
        ],
    )
    op_space_s0 = InformationSpace(
        id="codenames-info-op-s0",
        player_id=OPERATIVE,
        items=[
            InformationItem(
                id="codenames-board-op",
                visibility=Visibility.PUBLIC,
                about="board",
                content_known=True,
                known_to=[SPYMASTER, OPERATIVE],
                payload={"words": WORDS},
            ),
            InformationItem(
                id="codenames-key-hidden",
                visibility=Visibility.HIDDEN,
                about="key",
                holder_id=SPYMASTER,
                content_known=False,
            ),
        ],
    )
    spy_decisions = DecisionSpace(
        id="codenames-spy-dec",
        player_id=SPYMASTER,
        game_state_id=S0,
        options=[
            DecisionOption(
                id="codenames-clue-ocean",
                action_type="give_clue",
                parameters={"clue": "WATER", "count": 2},
            )
        ],
        complete=False,
    )
    action = Action(
        id="codenames-clue",
        player_id=SPYMASTER,
        action_type="give_clue",
        parameters={"clue": "WATER", "count": 2},
        turn_number=1,
    )
    s1 = GameState(
        id=S1,
        game_id=game.id,
        turn_number=1,
        active_player_id=OPERATIVE,
        phase="guess",
        data={**s0.data, "clue": {"word": "WATER", "count": 2}},
    )
    op_space_s1 = InformationSpace(
        id="codenames-info-op-s1",
        player_id=OPERATIVE,
        items=[
            op_space_s0.items[0].model_copy(update={"id": "codenames-board-op-s1"}),
            InformationItem(
                id="codenames-clue-seen",
                visibility=Visibility.COMMUNICATED,
                about="clue",
                communicated_by=SPYMASTER,
                content_known=True,
                known_to=[SPYMASTER, OPERATIVE],
                payload={"clue": "WATER", "count": 2},
            ),
        ],
    )
    op_decisions = DecisionSpace(
        id="codenames-op-dec",
        player_id=OPERATIVE,
        game_state_id=S1,
        options=[
            DecisionOption(
                id="codenames-guess-ocean",
                action_type="guess_word",
                parameters={"word": "OCEAN"},
            ),
            DecisionOption(
                id="codenames-guess-wave",
                action_type="guess_word",
                parameters={"word": "WAVE"},
            ),
        ],
    )
    return PlaySituation(
        game=game,
        definition=definition,
        players=[spymaster, operative],
        states=[s0, s1],
        observations=[
            Observation(
                id="codenames-obs-spy-s0",
                game_state_id=S0,
                observer_id=SPYMASTER,
                information_space_id=spy_space.id,
            ),
            Observation(
                id="codenames-obs-op-s0",
                game_state_id=S0,
                observer_id=OPERATIVE,
                information_space_id=op_space_s0.id,
            ),
            Observation(
                id="codenames-obs-op-s1",
                game_state_id=S1,
                observer_id=OPERATIVE,
                information_space_id=op_space_s1.id,
            ),
        ],
        information_spaces=[spy_space, op_space_s0, op_space_s1],
        decision_spaces=[spy_decisions, op_decisions],
        turns=[
            Turn(
                id="codenames-turn-clue",
                number=1,
                active_player_id=SPYMASTER,
                phase="clue",
                game_state_id=S0,
                action_id=action.id,
                resulting_state_id=S1,
            )
        ],
        actions=[action],
        transitions=[
            StateTransition(
                id="codenames-tr",
                from_state_id=S0,
                to_state_id=S1,
                action_id=action.id,
            )
        ],
        interpretations=[
            ExtractedInterpretation(
                id="codenames-interp",
                game_id=game.id,
                label="role_asymmetric_information",
                extractor="human",
            )
        ],
    )
