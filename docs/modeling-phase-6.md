# Phase 6 — Counterfactuals and higher-order perspectives

This document describes explicit intervention queries and structural
second-order perspectives. It is not a causal analysis of real play and
not a model of what players believe about each other.

## 1. Intervention semantics

An `Intervention` is a frozen, serializable record:

```text
operation, game_id, situation_id, source_state_id,
observer / item / action identifiers, parameters, provenance
```

It is never a Python callable. The `intervention_id` is a deterministic
hash of those fields.

Application is functional:

```text
source observation
        ↓
apply_intervention
        ↓
new observation (new entity id)
```

The source record is not mutated.

## 2. Observed vs predicted vs counterfactual

| Origin | Meaning |
| --- | --- |
| `observed` | Authored fixture state, observation, or transition |
| `predicted` | Output of a model on factual inputs (for example a rollout of observed actions) |
| `counterfactual` | Constructed or predicted under an explicit intervention |

Counterfactual entity ids use a `/cf/` suffix. They are never written back
into fixture trajectories or treated as `PlaySituation` states.

## 3. Supported intervention types

v0 operations:

- `identity` — copy the observation; representation is unchanged
- `hide_information` — set one item to `hidden` and `content_known=False`
- `reveal_information` — set one item to `public` and `content_known=True`
- `change_observer` — evaluate another observer's existing view of the same state
- `substitute_action` — replace a Phase 4 action set with another **already authored** action set from the same situation

Missing observers or items raise. Missing substitute actions raise. No
action is invented. No game rules are simulated.

`reveal` does not invent payload values. Hide-then-reveal restores the
encoder view when the original item was already public and known.

## 4. Representation deltas

Observation encodings before and after an intervention are subtracted with
the platform `representation_delta` (`z_cf - z_actual`).

Scalar diagnostics computed from the observation items themselves:

- information volume
- visibility (known / n)
- hidden count

A hide must not increase visibility. A reveal must not decrease it. A
distance or delta is a representation change, not a causal effect.

## 5. Counterfactual transition queries

`counterfactual_transition` reuses the Phase 4 `ConditionedEncoder` through
the platform `InterventionQuery`:

```text
z_from, z_action_A  →  predicted z_next | factual
z_from, z_action_B  →  predicted z_next | counterfactual
```

The observed `to_state` is not an input. The result is marked
`origin=counterfactual` and keeps the intervention id and model family.

This is what the fitted transition model predicts under the substitute
action, not what the real game would do.

## 6. Bounded rollout

`counterfactual_rollout` calls the platform open-loop `rollout` after
truncating the `SequenceTable` to `max_steps` events per group.

- `max_steps` is required and must be positive
- each step is a model prediction, not an observed state
- there is no environment, no legal-move generator, and no unbounded loop

## 7. Higher-order perspective semantics

`A_about_B` at state `S` is:

```text
concat(z_A, z_B, z_A - z_B, distance, structural item-set counts)
```

The structural counts are known-to-A-only, known-to-B-only, known-to-both,
and hidden-from-both. This uses fixture observations only.

It is **not** an inference of what A believes B believes.

`A_about_B` is not defined to equal `B_about_A`. They match only when the
two views and item sets actually make the concatenated vector identical.

## 8. Explicit perspective paths

`ObserverPath` is an ordered list of observers on one state. It is
converted to the platform `PerspectivePath` and composed with
`OrderedConcatComposer` (concatenate in order, plus last−first).

- observer roles and order are preserved: `A→B→C` ≠ `C→B→A`
- missing observers raise
- cycles raise unless `allow_cycles=True`

These paths are ordered perspective compositions, not nested beliefs.

## 9. Provenance

Every counterfactual result keeps:

- source entity id
- intervention id and operation
- origin (`counterfactual` or `predicted`)
- model family where a model was used
- notes that the output is not a fixture state

Persisted logical keys:

```text
bga:model/intervention:v0
bga:model/higher-order-perspective:v0
bga:repr/counterfactuals:v0
bga:repr/higher-order-perspectives:v0
bga:evaluation/counterfactual:v0
bga:evaluation/perspective:v0
```

There is no counterfactual database and no belief-state store.

## 10. Current scientific limitations

The ten fixtures are short authored moments. Phase 6 shows that
interventions, deltas, model queries, and structural `A_about_B` can be
represented with provenance.

It does not establish causal effects, accurate counterfactual dynamics,
calibrated beliefs, strategic reasoning, or a general board-game world
model. The 200-game BGG metadata corpus is not used as transition data.
