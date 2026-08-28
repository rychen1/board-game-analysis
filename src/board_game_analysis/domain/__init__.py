"""Observed-fact, interpretation, measurement, and play-representation models.

These models are independent of FastAPI, storage, and analysis implementations.
"""

from board_game_analysis.domain.action import Action
from board_game_analysis.domain.decision import DecisionOption, DecisionSpace
from board_game_analysis.domain.definition import GameDefinition
from board_game_analysis.domain.game import Game
from board_game_analysis.domain.information import (
    InformationItem,
    InformationSpace,
    Observation,
)
from board_game_analysis.domain.interpretation import ExtractedInterpretation
from board_game_analysis.domain.measurement import DerivedMeasurement
from board_game_analysis.domain.mechanic import Mechanic
from board_game_analysis.domain.player import Player
from board_game_analysis.domain.situation import PlaySituation
from board_game_analysis.domain.source import SourceReference
from board_game_analysis.domain.state import GameState
from board_game_analysis.domain.transition import StateTransition
from board_game_analysis.domain.turn import Turn
from board_game_analysis.domain.vocabulary import (
    DecisionTiming,
    Interaction,
    TransitionKind,
    Visibility,
)

__all__ = [
    "Action",
    "DecisionOption",
    "DecisionSpace",
    "DecisionTiming",
    "DerivedMeasurement",
    "ExtractedInterpretation",
    "Game",
    "GameDefinition",
    "GameState",
    "InformationItem",
    "InformationSpace",
    "Interaction",
    "Mechanic",
    "Observation",
    "PlaySituation",
    "Player",
    "SourceReference",
    "StateTransition",
    "TransitionKind",
    "Turn",
    "Visibility",
]
