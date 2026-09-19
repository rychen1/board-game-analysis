# Phase 7 — Learned play-structure / design-space analysis

Phase 7 composes the representations from Phases 1–6 into a learned
structural design space. It answers questions about **geometry of the
chosen representation**, not about game quality.

```text
structure → representation → geometry
                              ↓
                         metadata analysis
```

not:

```text
metadata + structure → representation
```

Named games in tests (Hanabi, Chess, Just One, Telestrations) are
fixtures, not architectural special cases.

## 1. Objective

Support structural questions:

- How similar are two games or two situations under this representation?
- Which entities occupy nearby regions?
- Which entities are locally unusual (structural novelty)?
- What exploratory clusters appear?
- How does variation differ within a game versus across games?

Do not read novelty as originality, quality, fun, commercial potential,
publisher interest, or blue-/green-ocean status.

## 2. Situation representation

`SituationRepresentation` is one local point in play space. It is built
from the existing Phase 1–6 objects for that fragment: examples, bag
encodings, authored transitions, Phase 5 trajectories, Phase 6
hide/reveal deltas, and Phase 6 `higher_order_pairs`.

The full encoder vectors are not concatenated. They contribute only
through centroid norm, dispersion, and higher-order pair summaries.

Each vector keeps `game_id`, `situation_id`, and `source_artifacts`.

Persisted as `bga:repr/situations:v0`.

## 3. Game aggregation

`GameRepresentation` aggregates that game's situations
deterministically:

- situations are sorted by `situation_id` before aggregation
- family values that exist on a situation are averaged
- a missing family stays missing unless at least one situation has it
- `region` is computed from the situation vectors: count, mean
  distance to the centroid (dispersion), and max pairwise distance
  (range)

A game therefore occupies a **region**. With the current fixtures that
region is a single point. Situation identity is retained.

Persisted as `bga:repr/games:v0`.

## 4. Feature schema

Version `design-space-v0`. Family order is fixed:

| Family | Present when | Contents |
| --- | --- | --- |
| `state` | ≥1 state | counts, centroid norm, dispersion of Phase 2 state encodings |
| `perspective` | ≥1 observation | volume / visibility / hidden, asymmetry, observation dispersion |
| `action` | actions, transitions, or decision spaces | type / actor / option counts, action-encoding dispersion |
| `sequence` | ≥1 Phase 5 trajectory | trajectory count and step lengths |
| `intervention` | ≥1 information item | mean hide/reveal visibility deltas |
| `higher_order` | ≥1 same-state observer pair | pair count, distances, item-set splits, directional mismatch count |
| `region` | always | `n_units`, dispersion, range (a situation is one unit) |

`family__present` precedes each family's columns. Zeros behind a
`present = 0` flag are "not computed", not imputed play.

## 5. Metadata exclusion

The canonical vector must not use title, publisher, designer, year,
rating, popularity, categories, mechanics, corpus rank, source URL, or
catalog player/time fields.

Those fields may be attached afterwards with `attach_external_metadata`
for descriptive questions only. They do not feed back into the vector.

## 6. Normalization

`FamilyStandardizer` fits per-coordinate mean and std on **training
games only**, then applies the same transform to every game and
situation vector (same schema and dim).

This is intentional cross-level geometry (H4): situation vectors use the
same coordinate system as game vectors, fit from **game-row** training
statistics. A situation's `region__n_units` is always `1`; a game's is
`n_situations`. After z-scoring, normalized situation coordinates are
comparable across the corpus under the game-fit scale — they are **not**
situation-native variance units. Use the canonical (un-normalized) view
when within-game relative scale matters.

Zero-variance training columns use `std = 1` and are listed on
`zero_variance_columns`.

Persisted as `bga:model/design-space-normalizer:v0`.

The fixture corpus is too small for statistically meaningful scale
estimates. The fit exists so later play corpora can reuse the same
leakage-safe procedure.

## 7. Projection

Projection is optional and is **not** the canonical identity.

`LeadingVarianceProjection` keeps the highest-variance training
coordinates. It is a simple linear mask, not t-SNE/UMAP and not a
learned manifold.

If used, persist `bga:model/design-space-projection:v0`. The unprojected
canonical tables remain available.

## 8. Geometry

Platform `vector_distance`, `pairwise_distances`, and `knn` are reused.

Default metric is **l2** on the normalized view. Cosine is available
for profile direction. Every neighborhood records representation level
(game vs situation), view, metric, and schema version.

There is no vector database or approximate index.

## 9. Structural kNN outlierness

`structural_novelty` at neighborhood size `k` is **structural kNN
outlierness**: the mean distance to the `k` nearest **training** games.
Test games are scored as queries against that train reference set.
`local_isolation` is the same number. `nearest_distance` is the `k = 1`
value.

`k` is configurable. Values with `k >= n_games` are skipped.
Multiple `k` may be requested.

This is not originality, quality, or corpus rank.

Persisted as `bga:evaluation/novelty:v0`.

## 10. Clustering

`DeterministicKMeans` is exploratory geometry. `n_clusters` and `seed`
are recorded. The same seed repeats. Clusters are not genres or
mechanics.

The current corpus is too small for significance. Repeatability under
one seed is the only stability diagnostic.

Persisted as `bga:evaluation/clusters:v0`.

## 11. Situation vs game geometry

These APIs stay distinct:

```text
nearest_games / game_distance / pairwise_game_distances
nearest_situations / situation_distance / pairwise_situation_distances
```

Game entity ids are `game_id`. Situation entity ids are the full
situation ids from Phase 1 (`{topology}@{digest}`). A game vector is
never silently used as a situation vector.

**Neighbor semantics differ by API:**

| API | Reference set | Use |
| --- | --- | --- |
| `novelty_table` / `novelty_score` | Train games only | Structural kNN outlierness |
| `nearest_games` / `nearest_situations` | Full corpus | Exploratory geometry |
| `cluster_games` | Train-fit, all-game predict | Exploratory labels |

`nearest_*` is not a holdout evaluation. Test entities can appear as
neighbors. Prefer `novelty_table` when the question is distance to the
training manifold.

`within_game_dispersion` and `between_game_dispersion` describe region
size versus corpus spread (full corpus).

## 12. Artifact outputs

```text
bga:repr/situations:v0
bga:repr/games:v0
bga:model/design-space:v0
bga:model/design-space-normalizer:v0
bga:model/design-space-projection:v0
bga:evaluation/novelty:v0
bga:evaluation/clusters:v0
```

Existing `LocalStore` / `put_*` helpers. No second persistence system.

## 13. Leakage controls

- catalog metadata is not a representation input
- game-safe holdout: one game never appears on both sides
- normalizer and projection `train_entity_ids` are training games only
- test games cannot change fitted means, stds, or kept coordinates
- BGG metadata corpus is not used as play-structure data

## 14. Reproducibility

A result is determined by:

```text
source PlaySituation inputs (or persisted Phase 1 example artifacts)
+ situation ids (topology + content digest)
+ schema version
+ train-only normalizer
+ optional projection
+ distance metric
+ clustering algorithm, n_clusters, seed
+ novelty k values
+ RunContext config hash
```

`DesignSpaceManifest` records the recipe (schema, train/test ids,
encoder dim, metric, clustering/novelty params, Phase 6 logical keys).
It is sufficient to understand what a result means. Exact vector replay
on a cold machine also requires the original situation inputs and code
version pin — the manifest does not embed full Phase 1–5 artifact bytes.

Aggregation sorts situations by id. Feature column order is the schema
order.

## 15. Interpretation limits

Phase 7 discovers geometry of this representation on this corpus. It
does not measure objective game-design quality, fun, commercial
potential, or originality.

Nearest neighbors are nearest under the recorded metric and view, not
psychologically similar games.

## 16. Fixture / corpus limitations

The modeling fixtures are about ten short authored moments, usually one
situation per game. Phase 7 establishes the analysis machinery.

It does not represent the board-game universe. Comparable play-structure
data—not BGG metadata—would be required before those scientific
questions could be asked.
