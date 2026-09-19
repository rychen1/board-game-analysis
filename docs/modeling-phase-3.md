# Phase 3 — First-order perspectives

Phase 3 builds platform `PerspectiveTable` objects from Phase 2 observation
embeddings.

## Core API

- `perspective_table_from_observations` — rows keyed by
  `(game_state_id, observer_id)`
- `perspective_distances` — pairwise distances within a state
- `store_perspectives` / `store_perspective_distances` — persistence

Spec: `BGA_PERSPECTIVE_SPEC` (`subject_field="game_state_id"`,
`perspective_field="observer_id"`).

## Artifacts

| Key | Content |
| --- | --- |
| `bga:repr/perspectives:v0` | perspective table |
| `bga:repr/perspective-distances:v0` | distance feature table |

## Semantics

Perspectives describe **observed views**, not beliefs. Phase 6 higher-order
pairs compose these views into directed `A_about_B` relationships.
