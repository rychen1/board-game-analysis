"""A single generic play fragment used by tests.

This is not a rules implementation of any commercial game.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from board_game_analysis.domain import (
    Action,
    DecisionOption,
    DecisionSpace,
    Game,
    GameState,
    InformationItem,
    InformationSpace,
    Mechanic,
    Player,
    SourceReference,
    StateTransition,
    Turn,
)


class ExampleScenario(BaseModel):
    """Linked objects demonstrating Game → Turn → Player → spaces → Action."""

    model_config = ConfigDict(extra="forbid")

    game: Game
    players: list[Player]
    from_state: GameState
    to_state: GameState
    information_space: InformationSpace
    decision_space: DecisionSpace
    action: Action
    turn: Turn
    transition: StateTransition


def make_example_scenario() -> ExampleScenario:
    """Build one draw/play-a-card moment with generic example data."""
    source = SourceReference(
        source="example-catalog",
        source_type="fixture",
        source_identifier="example-card-game",
        retrieved_at=datetime(2026, 1, 15, tzinfo=UTC),
    )
    mechanic = Mechanic(
        id="mech-hand-management",
        name="Hand management",
        category="cards",
        description="Players hold and play cards from a private hand.",
    )
    game = Game(
        id="game-example-card",
        title="Example Card Game",
        release_year=2020,
        min_players=2,
        max_players=4,
        min_play_time_minutes=20,
        max_play_time_minutes=40,
        popularity=12.0,
        rating=7.2,
        rating_count=48,
        complexity=2.1,
        designers=["Example Designer"],
        publishers=["Example Publisher"],
        categories=["card game"],
        mechanics=[mechanic],
        sources=[source],
    )
    alice = Player(id="player-alice", name="Alice")
    bob = Player(id="player-bob", name="Bob")
    from_state = GameState(
        id="state-before-play",
        turn_number=3,
        active_player_id=alice.id,
        phase="main",
        data={"deck_size": 17, "hands": {alice.id: 4, bob.id: 5}},
    )
    information_space = InformationSpace(
        id="info-alice-turn-3",
        player_id=alice.id,
        items=[
            InformationItem(
                id="info-public-discard",
                visibility="public",
                about="discard",
                known_to=[alice.id, bob.id],
                payload={"top_card": "red-3"},
            ),
            InformationItem(
                id="info-private-hand",
                visibility="private",
                about="self",
                known_to=[alice.id],
                payload={"cards": ["blue-1", "green-4", "red-7", "yellow-2"]},
            ),
            InformationItem(
                id="info-hidden-deck",
                visibility="hidden",
                about="deck",
                known_to=[],
                payload={"known": False},
            ),
            InformationItem(
                id="info-inferred-bob-plays",
                visibility="inferred",
                about=bob.id,
                known_to=[alice.id],
                payload={"likely_color": "green"},
            ),
        ],
    )
    play_option = DecisionOption(
        id="opt-play-blue-1",
        action_type="play_card",
        parameters={"card": "blue-1"},
        category="play",
        legal=True,
    )
    draw_option = DecisionOption(
        id="opt-draw",
        action_type="draw_card",
        parameters={},
        category="draw",
        legal=True,
    )
    decision_space = DecisionSpace(
        id="decision-alice-turn-3",
        player_id=alice.id,
        options=[play_option, draw_option],
    )
    action = Action(
        id="action-alice-play-blue-1",
        player_id=alice.id,
        action_type="play_card",
        parameters={"card": "blue-1"},
        turn_number=3,
        sources=[source],
    )
    to_state = GameState(
        id="state-after-play",
        turn_number=3,
        active_player_id=alice.id,
        phase="main",
        data={
            "deck_size": 17,
            "hands": {alice.id: 3, bob.id: 5},
            "discard_top": "blue-1",
        },
    )
    turn = Turn(
        id="turn-3",
        number=3,
        active_player_id=alice.id,
        phase="main",
        information_space_id=information_space.id,
        decision_space_id=decision_space.id,
        action_id=action.id,
        resulting_state_id=to_state.id,
    )
    transition = StateTransition(
        id="transition-play-blue-1",
        from_state_id=from_state.id,
        action_id=action.id,
        to_state_id=to_state.id,
    )
    return ExampleScenario(
        game=game,
        players=[alice, bob],
        from_state=from_state,
        to_state=to_state,
        information_space=information_space,
        decision_space=decision_space,
        action=action,
        turn=turn,
        transition=transition,
    )
