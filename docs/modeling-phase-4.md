# Phase 4 — Authored-transition model

Phase 4 evaluates whether a simple **authored-transition model** predicts
encoded next states from `(z_from, z_action)` on fixture transitions.

## Model

`LinearConditionedEncoder` (ridge regression) maps context state vectors
and action-set vectors to the encoded `to_state`. A `CopyStateEncoder`
baseline copies the from-state.

Inputs come from `transition_pairs_from_situation(s)` — situation-scoped
pair bundles, not game-keyed last-wins maps.

## Evaluation

`run_transition_experiment`:

- game-safe split via `split_entities_by_game`
- representation MSE against observed next states
- optional scalar deltas from Phase 1 measurements

## Artifacts

| Key | Role |
| --- | --- |
| `bga:repr/actions:v0` | action-set encodings |
| `bga:model/action-embedder:v0` | action bag embedder |
| `bga:model/transition:v0` | fitted conditioned encoder |
| `bga:repr/transitions:v0` | predicted transition reps |
| `bga:evaluation/transition:v0` | evaluation report |

Phase 5 reuses this encoder as a prefix baseline comparison.
