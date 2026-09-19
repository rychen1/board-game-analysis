# BGA modeling roadmap

Play-structure modeling lives in `src/board_game_analysis/modeling/`. It
turns authored `PlaySituation` fixtures into a **representation space** and
supports structural questions about similarity, neighborhoods, structural
kNN outlierness, and exploratory clusters.

## Phase map

| Phase | Module focus | Doc |
| --- | --- | --- |
| 1 | Examples and measurements | [modeling-phase-1.md](modeling-phase-1.md) |
| 2 | Bag encodings | [modeling-phase-2.md](modeling-phase-2.md) |
| 3 | First-order perspectives | [modeling-phase-3.md](modeling-phase-3.md) |
| 4 | Authored-transition model | [modeling-phase-4.md](modeling-phase-4.md) |
| 5 | Trajectories and prefixes | [modeling-phase-5.md](modeling-phase-5.md) |
| 6 | Interventions and higher-order | [modeling-phase-6.md](modeling-phase-6.md) |
| 7 | Representation-space geometry | [modeling-phase-7.md](modeling-phase-7.md) |

Artifact logical keys are defined once in
`board_game_analysis.modeling.logical_keys`.

## Pipeline spine

```text
PlaySituation
  → examples (Phase 1)
  → bag encodings (Phase 2)
  → perspectives (Phase 3)
  → pairs / authored-transition model (Phase 4)
  → trajectories (Phase 5)
  → interventions / higher-order (Phase 6)
  → situation & game vectors → geometry (Phase 7)
```

## What this is not

- Not a game simulator or rule engine
- Not causal inference on real play logs
- Not BGG catalog metadata modeling (see `ingestion/` and
  `analysis/descriptive.py`)
- Not a claim that clusters are genres or that structural kNN outlierness
  is commercial originality

## Audit and tests

Adversarial audit: [modeling-audit.md](modeling-audit.md).

Tests: `tests/modeling/` (examples through design space) plus
`tests/modeling/test_audit_adversarial.py`.
