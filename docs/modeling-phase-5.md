# Phase 5 — Sequential / trajectory modeling

This document describes the BGA trajectory layer. It is infrastructure for
short authored fixtures, not a claim about general board-game strategy.

## 1. What a BGA trajectory is

A trajectory is one **contiguous chain of authored transitions** inside a
single `PlaySituation`.

Each step is:

```text
from_state_id, action_ids (possibly empty), to_state_id
```

plus the transition/pair id, step index, and observer ids that already
exist on that transition.

Phase 1 `SequenceExample` still lists `situation.states` in authored list
order and leaves `steps` empty. Phase 5 fills a separate `SequenceExample`
with `steps` and sets `event_ids` to the chain's state ids.

`GameState.data` is never copied into the sequence. Only ids are stored.

## 2. How sequences are constructed

`sequence_from_situation` / `sequences_from_situations` walk authored
`PairExample` records:

- A unique successor (`to_state` of one step is the sole `from_state` of
  the next) extends the same chain.
- Multiple outgoing transitions become separate trajectories. They are
  never concatenated.
- Situations are never merged across game or situation boundaries.
- Missing states raise. Missing actions stay empty. Cycles are skipped
  with an explicit reason. Situations with no transitions are skipped.

Telestrations is the only current fixture with two linked steps
(`tele-s0 → tele-s1 → tele-s2`). The others are one-step trajectories.

## 3. How ordering is represented

The platform `SequenceTable` holds one row per step, grouped by
trajectory id, ordered by step index.

Two BGA encodings sit on top of that:

- **Trajectory summary** (`encode_trajectory_summaries`): first state,
  last state, mean state, last−first, mean observed action, and counts.
  This describes a fully observed walk, including the final state. It is
  not a prediction input. Reversing the walk changes first/last/delta.
- **Prefix summary** (`PrefixSequenceEncoder` + `encode_prefix_summaries`):
  for each step *i*, the input is states `0..i` and observed actions
  `0..i`. The target `to_state` of step *i* is excluded. The encoder
  concatenates first, last, mean, last−first, and step count, then BGA
  appends observed-action and observer counts.

A prefix is therefore not an unordered bag of states.

## 4. Game-safe splitting

`split_sequences_by_game` reuses `split_entities_by_game`. Every prefix
of a trajectory carries that trajectory's `game_id`. A game cannot appear
on both sides of the split, and the steps of one trajectory stay
together.

Random step-level splits are not used.

## 5. Baseline models

1. **Last state** — predict the encoded from-state of the current step.
2. **Phase 4 immediate transition** — reuse `LinearConditionedEncoder` on
   `(z_from, z_action)` of the current step only.

Both are compared with representation MSE against the encoded observed
`to_state`. Missing actions are not invented for the Phase 4 baseline;
those prefixes are skipped explicitly.

## 6. What the sequence model adds

`LinearSequencePredictor` is ridge regression from the prefix summary to
the next-state representation. It can see earlier states and actions when
a trajectory has more than one step.

On the current fixtures that extra context exists only for Telestrations.
The model is a reusable interface, not evidence of learned dynamics.

## 7. What the fixture data can and cannot establish

The ten fixtures are short authored moments (eleven transitions). They
are enough to check construction, order, leakage, game-safe splits,
baselines, and artifact persistence.

They are not enough to claim that sequence context improves next-state
prediction across board games, or that the linear prefix model captures
strategy. The 200-game BGG metadata corpus is not trajectory data and is
not used here.

Scalar diagnostics (`information_volume`, `hidden_information`,
`available_decision_count` at the target state) are reported only where
Phase 1 already computed them. Missing values stay missing.
