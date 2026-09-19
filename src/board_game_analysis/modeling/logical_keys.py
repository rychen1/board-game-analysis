"""Canonical logical keys for BGA modeling artifacts.

Import keys from this module only. Phase modules re-export for backward
compatibility but must not define duplicate strings.
"""

from __future__ import annotations

# Phase 1 — examples and measurements
EXAMPLES_LOGICAL_KEY = "bga:examples/play-fixtures:v0"
MEASUREMENTS_LOGICAL_KEY = "bga:measurements/ontology-v0:v0"

# Phase 2 — bag encodings and trajectory summaries
OBS_MODEL_LOGICAL_KEY = "bga:model/obs-embedder:v0"
STATE_MODEL_LOGICAL_KEY = "bga:model/state-embedder:v0"
OBS_REPR_LOGICAL_KEY = "bga:repr/observations:v0"
STATE_REPR_LOGICAL_KEY = "bga:repr/states:v0"
PROBE_EVAL_LOGICAL_KEY = "bga:evaluation/obs-probes:v0"
ACTION_MODEL_LOGICAL_KEY = "bga:model/action-embedder:v0"
ACTION_REPR_LOGICAL_KEY = "bga:repr/actions:v0"
TRAJECTORY_REPR_LOGICAL_KEY = "bga:repr/trajectory-summaries:v0"

# Phase 3 — first-order perspectives
PERSPECTIVES_LOGICAL_KEY = "bga:repr/perspectives:v0"
PERSPECTIVE_DISTANCES_LOGICAL_KEY = "bga:repr/perspective-distances:v0"

# Phase 4 — authored-transition model
TRANSITION_MODEL_LOGICAL_KEY = "bga:model/transition:v0"
TRANSITION_REPR_LOGICAL_KEY = "bga:repr/transitions:v0"
TRANSITION_EVAL_LOGICAL_KEY = "bga:evaluation/transition:v0"

# Phase 5 — sequence experiment summaries
SEQUENCE_REPR_LOGICAL_KEY = "bga:repr/sequence-summaries:v0"
SEQUENCE_ENCODER_LOGICAL_KEY = "bga:model/sequence-encoder:v0"
SEQUENCE_MODEL_LOGICAL_KEY = SEQUENCE_ENCODER_LOGICAL_KEY
SEQUENCE_PREDICTOR_LOGICAL_KEY = "bga:model/sequence-predictor:v0"
SEQUENCE_EVAL_LOGICAL_KEY = "bga:evaluation/sequence:v0"

# Phase 6 — interventions and higher-order perspectives
COUNTERFACTUAL_REPR_LOGICAL_KEY = "bga:repr/counterfactuals:v0"
INTERVENTION_MODEL_LOGICAL_KEY = "bga:model/intervention:v0"
COUNTERFACTUAL_EVAL_LOGICAL_KEY = "bga:evaluation/counterfactual:v0"
HIGHER_ORDER_REPR_LOGICAL_KEY = "bga:repr/higher-order-perspectives:v0"
HIGHER_ORDER_MODEL_LOGICAL_KEY = "bga:model/higher-order-perspective:v0"
PERSPECTIVE_EVAL_LOGICAL_KEY = "bga:evaluation/perspective:v0"

# Phase 7 — representation space geometry
SITUATION_REPR_LOGICAL_KEY = "bga:repr/situations:v0"
GAME_REPR_LOGICAL_KEY = "bga:repr/games:v0"
DESIGN_SPACE_MODEL_LOGICAL_KEY = "bga:model/design-space:v0"
NORMALIZER_LOGICAL_KEY = "bga:model/design-space-normalizer:v0"
PROJECTION_LOGICAL_KEY = "bga:model/design-space-projection:v0"
NOVELTY_EVAL_LOGICAL_KEY = "bga:evaluation/novelty:v0"
CLUSTER_EVAL_LOGICAL_KEY = "bga:evaluation/clusters:v0"

__all__ = [
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
]
