"""BGA research modeling: examples, encodings, and representation-space geometry.

Stable v0 surface is listed in ``__all__``. Experiment runners, internal
helpers, and duplicate imports remain on submodules (for example
``board_game_analysis.modeling.design_geometry``).
"""

# Re-export the full logical-key catalog for ``from … import *`` on keys only.
from board_game_analysis.modeling import logical_keys as logical_keys
from board_game_analysis.modeling.design_geometry import (
    FamilyStandardizer,
    LeadingVarianceProjection,
    build_design_space,
    cluster_games,
    game_distance,
    nearest_games,
    nearest_situations,
    novelty_score,
    novelty_table,
    situation_distance,
)
from board_game_analysis.modeling.design_space import (
    FeatureSchema,
    GameRepresentation,
    PlayLayer,
    SituationRepresentation,
    build_play_layer,
    default_feature_schema,
    represent_game,
    represent_situation,
    represent_situations,
)
from board_game_analysis.modeling.encoders import (
    ActionBagEmbedder,
    ObservationBagEmbedder,
    StateBagEmbedder,
)
from board_game_analysis.modeling.examples import (
    ExampleBundle,
    ObservationExample,
    PairExample,
    SequenceExample,
    StateExample,
    action_set_entity_id,
    examples_from_situation,
    examples_from_situations,
    situation_id_for,
    state_entity_id,
)
from board_game_analysis.modeling.higher_order import (
    ObserverPath,
    compose_observer_path,
    higher_order_pairs,
    higher_order_perspective,
)
from board_game_analysis.modeling.interventions import (
    Intervention,
    apply_intervention,
    counterfactual_entity_id,
    hide_information,
    identity_intervention,
    reveal_information,
    substitute_action,
)
from board_game_analysis.modeling.logical_keys import (
    ACTION_MODEL_LOGICAL_KEY,
    ACTION_REPR_LOGICAL_KEY,
    CLUSTER_EVAL_LOGICAL_KEY,
    COUNTERFACTUAL_EVAL_LOGICAL_KEY,
    COUNTERFACTUAL_REPR_LOGICAL_KEY,
    DESIGN_SPACE_MODEL_LOGICAL_KEY,
    EXAMPLES_LOGICAL_KEY,
    GAME_REPR_LOGICAL_KEY,
    HIGHER_ORDER_MODEL_LOGICAL_KEY,
    HIGHER_ORDER_REPR_LOGICAL_KEY,
    INTERVENTION_MODEL_LOGICAL_KEY,
    MEASUREMENTS_LOGICAL_KEY,
    NORMALIZER_LOGICAL_KEY,
    NOVELTY_EVAL_LOGICAL_KEY,
    OBS_MODEL_LOGICAL_KEY,
    OBS_REPR_LOGICAL_KEY,
    PERSPECTIVE_DISTANCES_LOGICAL_KEY,
    PERSPECTIVE_EVAL_LOGICAL_KEY,
    PERSPECTIVES_LOGICAL_KEY,
    PROBE_EVAL_LOGICAL_KEY,
    PROJECTION_LOGICAL_KEY,
    SEQUENCE_ENCODER_LOGICAL_KEY,
    SEQUENCE_EVAL_LOGICAL_KEY,
    SEQUENCE_MODEL_LOGICAL_KEY,
    SEQUENCE_PREDICTOR_LOGICAL_KEY,
    SEQUENCE_REPR_LOGICAL_KEY,
    SITUATION_REPR_LOGICAL_KEY,
    STATE_MODEL_LOGICAL_KEY,
    STATE_REPR_LOGICAL_KEY,
    TRAJECTORY_REPR_LOGICAL_KEY,
    TRANSITION_EVAL_LOGICAL_KEY,
    TRANSITION_MODEL_LOGICAL_KEY,
    TRANSITION_REPR_LOGICAL_KEY,
)
from board_game_analysis.modeling.pairs import (
    ActionExample,
    transition_pairs_from_situation,
    transition_pairs_from_situations,
)
from board_game_analysis.modeling.perspectives import (
    perspective_distances,
    perspective_table_from_observations,
)
from board_game_analysis.modeling.sequences import (
    prefixes_from_situations,
    sequence_from_situation,
    sequences_from_situations,
)
from board_game_analysis.modeling.split import split_entities_by_game

__all__ = [
    # Artifact keys (canonical module: logical_keys)
    "ACTION_MODEL_LOGICAL_KEY",
    "ACTION_REPR_LOGICAL_KEY",
    "CLUSTER_EVAL_LOGICAL_KEY",
    "COUNTERFACTUAL_EVAL_LOGICAL_KEY",
    "COUNTERFACTUAL_REPR_LOGICAL_KEY",
    "DESIGN_SPACE_MODEL_LOGICAL_KEY",
    "EXAMPLES_LOGICAL_KEY",
    "GAME_REPR_LOGICAL_KEY",
    "HIGHER_ORDER_MODEL_LOGICAL_KEY",
    "HIGHER_ORDER_REPR_LOGICAL_KEY",
    "INTERVENTION_MODEL_LOGICAL_KEY",
    "MEASUREMENTS_LOGICAL_KEY",
    "NORMALIZER_LOGICAL_KEY",
    "NOVELTY_EVAL_LOGICAL_KEY",
    "OBS_MODEL_LOGICAL_KEY",
    "OBS_REPR_LOGICAL_KEY",
    "PERSPECTIVE_DISTANCES_LOGICAL_KEY",
    "PERSPECTIVE_EVAL_LOGICAL_KEY",
    "PERSPECTIVES_LOGICAL_KEY",
    "PROBE_EVAL_LOGICAL_KEY",
    "PROJECTION_LOGICAL_KEY",
    "SEQUENCE_ENCODER_LOGICAL_KEY",
    "SEQUENCE_EVAL_LOGICAL_KEY",
    "SEQUENCE_MODEL_LOGICAL_KEY",
    "SEQUENCE_PREDICTOR_LOGICAL_KEY",
    "SEQUENCE_REPR_LOGICAL_KEY",
    "SITUATION_REPR_LOGICAL_KEY",
    "STATE_MODEL_LOGICAL_KEY",
    "STATE_REPR_LOGICAL_KEY",
    "TRAJECTORY_REPR_LOGICAL_KEY",
    "TRANSITION_EVAL_LOGICAL_KEY",
    "TRANSITION_MODEL_LOGICAL_KEY",
    "TRANSITION_REPR_LOGICAL_KEY",
    # Phase 1 — examples
    "ExampleBundle",
    "ObservationExample",
    "PairExample",
    "SequenceExample",
    "StateExample",
    "action_set_entity_id",
    "examples_from_situation",
    "examples_from_situations",
    "situation_id_for",
    "state_entity_id",
    # Phase 2 — bag encodings
    "ActionBagEmbedder",
    "ObservationBagEmbedder",
    "StateBagEmbedder",
    # Phase 3 — perspectives
    "perspective_distances",
    "perspective_table_from_observations",
    # Phase 4 — authored-transition model
    "ActionExample",
    "transition_pairs_from_situation",
    "transition_pairs_from_situations",
    # Phase 5 — trajectories
    "prefixes_from_situations",
    "sequence_from_situation",
    "sequences_from_situations",
    # Phase 6 — interventions and higher-order
    "Intervention",
    "ObserverPath",
    "apply_intervention",
    "compose_observer_path",
    "counterfactual_entity_id",
    "hide_information",
    "higher_order_pairs",
    "higher_order_perspective",
    "identity_intervention",
    "reveal_information",
    "substitute_action",
    # Phase 7 — representation space
    "FeatureSchema",
    "FamilyStandardizer",
    "GameRepresentation",
    "LeadingVarianceProjection",
    "PlayLayer",
    "SituationRepresentation",
    "build_design_space",
    "build_play_layer",
    "cluster_games",
    "default_feature_schema",
    "game_distance",
    "nearest_games",
    "nearest_situations",
    "novelty_score",
    "novelty_table",
    "represent_game",
    "represent_situation",
    "represent_situations",
    "situation_distance",
    # Split helper
    "split_entities_by_game",
    # Submodule for key discovery
    "logical_keys",
]
