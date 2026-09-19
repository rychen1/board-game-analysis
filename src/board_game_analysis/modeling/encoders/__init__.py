"""BGA Embedder implementations. No torch or sklearn."""

from board_game_analysis.modeling.encoders.action import ActionBagEmbedder
from board_game_analysis.modeling.encoders.bag import BAG_HASH_FAMILY, DEFAULT_DIM
from board_game_analysis.modeling.encoders.conditioned import (
    CONDITIONED_FAMILY,
    CopyStateEncoder,
    LinearConditionedEncoder,
)
from board_game_analysis.modeling.encoders.observation import ObservationBagEmbedder
from board_game_analysis.modeling.encoders.sequence import (
    PREDICTOR_FAMILY,
    SEQUENCE_FAMILY,
    LastStatePredictor,
    LinearSequencePredictor,
    PrefixSequenceEncoder,
    encode_prefix_summaries,
    encode_trajectory_summaries,
    target_states_for_prefixes,
)
from board_game_analysis.modeling.encoders.state import StateBagEmbedder

__all__ = [
    "BAG_HASH_FAMILY",
    "CONDITIONED_FAMILY",
    "DEFAULT_DIM",
    "PREDICTOR_FAMILY",
    "SEQUENCE_FAMILY",
    "ActionBagEmbedder",
    "CopyStateEncoder",
    "LastStatePredictor",
    "LinearConditionedEncoder",
    "LinearSequencePredictor",
    "ObservationBagEmbedder",
    "PrefixSequenceEncoder",
    "StateBagEmbedder",
    "encode_prefix_summaries",
    "encode_trajectory_summaries",
    "target_states_for_prefixes",
]
