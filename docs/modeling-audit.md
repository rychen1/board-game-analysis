# BGA Modeling Adversarial Audit

Audit of Phases 1–7 as implemented in `src/board_game_analysis/modeling/`.
No modeling features were added. Source behavior was not “fixed” to make
tests pass. New tests assert intended invariants; confirmed defects are
`xfail(strict=True)` so they remain visible.

Evidence date: 2026-09-19. Pre-audit suite: 245 passing tests at `5e67179`.

**Post-fix status (2026-09-19):** Recommended fix order items 1–6 and
`AUDIT-HOP-COUNT` are implemented. Adversarial suite: **52 passing, 0
xfail**. Full modeling suite: **289 passing**. See [Fixes
applied](#fixes-applied) below.

## Executive Summary

Phases 1–7 form a real pipeline, not a slide diagram. The intended
spine is present:

```text
PlaySituation → examples → bag encodings → perspectives
             → pairs / transitions → sequences
             → interventions / higher-order
             → situation / game vectors → geometry
```

`domain/` does not import `ds_platform`. Catalog metadata does not enter
canonical Phase 7 vectors. Bag embedders have no-op `fit`, so encoding
the full corpus before a game-level split does not leak *learned*
parameters. `FamilyStandardizer` and `LeadingVarianceProjection` fitted
on synthetic tables with train `[0, 1]` and test `[1000, 1001]` (then
`[1e9, 1e9+1]`) keep the same mean and std.

Identity is now situation-scoped for entity ids and batch maps (fix 1).
Multi-situation corpora with local ids (`s0`, `t0`) no longer collide on
the paths covered by adversarial tests. Remaining gaps: documentation
drift, oversized public exports, and game-level normalizer semantics on
situation vectors (H4).

Scientific risks that **remain** (by design or not yet addressed):

- `run_design_space_experiment` docstring still says “held-out” while
  novelty/cluster neighbors are train-only after a train-only scale fit
  (fix 2 changed behavior; naming/docs not fully tightened).
- Phase 4/5 models consume bag encodings that discard payload values.
  Two states with the same keys and types are identical inputs.

Scientific risks **addressed** since the audit:

- Missing-family padding is no longer z-scored when `present=0` (fix 3).
- Example items and family values are frozen (fix 4).

The architecture should be tightened, not replaced. Most Phase 2–7
types are the right shape. The defects are identity scope, a few
last-wins maps, overclaiming evaluation APIs, and an oversized public
surface.

## Critical Findings

Issues that can produce incorrect results, leakage, bad identity,
corrupted provenance, silent loss, or wrong model semantics.

### C1. Identity is game-scoped; multi-situation corpora collide

**Status: FIXED (fix 1).** Entity ids and batch maps are
situation-scoped. Adversarial identity probes pass.

Original finding (pre-fix), confirmed by adversarial tests (`AUDIT-ID-*`):

| Helper / field | Formula | Collision |
| --- | --- | --- |
| `sequence_entity_id` | `{game}/seq` | every Phase 1 sequence of that game |
| `situation_id_for` | `{game}:{first_state_id}` | two situations that start at the same local state id |
| `state_entity_id` | `{game}/state/{state_id}` | overlapping local state ids (`s0` in two situations) |
| `observation_entity_id` | `{game}/obs/{observation_id}` | reused observation ids |
| `pair_entity_id` | `{game}/pair/{transition_id}` | reused transition ids |
| `trajectory_entity_id` | `{game}/seq/{first_transition_id}` | reused first transition id |
| `PrefixExample.entity_id` | pair entity id | same as pair collision |
| `DerivedMeasurement.id` | `{game}:{name}:{scope_key}` | game-scoped metrics (`legal_action_type_count`, `structural_interaction`) always collide |

`build_play_layer` concatenates state/observation/action encodings into
one `RepresentationTable`. Duplicate entity ids raise. Two situations of
game `g` that both use `s0`/`s1` cannot enter Phase 7.

This is not a fixture curiosity. The Phase 7 docs say the machinery is
for later corpora with many situations per game. That path is broken
unless every author mints globally unique local ids.

### C2. `transition_pairs_from_situations` keys situations by `game_id`

**Status: FIXED (fix 1).** Batch pair maps key by `situation_id_for`.

Original finding:

```text
by_game = {situation.game.id: situation for situation in situations}
```

The last situation of a game wins. Resolving a pair from an earlier
situation against the later one raises `KeyError` (missing state or
action). Confirmed: `AUDIT-ID-PAIRS`.

`prefixes_from_situations` has the same pattern on `situation_id_for`.
When situation ids collide, observer lookup silently uses the last
situation.

Phase 5 `sequences_from_situations` is per-situation and does **not**
have this bug. The batch pair API does.

### C3. Novelty and clustering are corpus-wide after a held-out split

**Status: FIXED (fix 2).** `novelty_table` uses train-only neighbors;
`cluster_games` fits on train and predicts on the full table.
Adversarial leakage probes pass.

Original finding:

`FamilyStandardizer` / `LeadingVarianceProjection` are train-only.
Confirmed: `test_normalizer_ignores_extreme_test_values`.

`novelty_table` and `cluster_games` then score **every** game, including
test games, as neighbors and centroid members.

Confirmed:

- `AUDIT-LEAK-NOV` — moving a held-out point from `1000` to `0.1`
  changes train-game kNN novelty.
- `AUDIT-LEAK-CLUST` — the same move changes train k-means labels.
- `AUDIT-LEAK-EXP` — on real `DesignSpace` objects, cloning a train
  game as the test game changes that train game’s novelty.
- `test_novelty_table_includes_test_game_ids` / `test_cluster_games_includes_test_ids`
  (passing) record current behavior.

`run_design_space_experiment` is named and docstringed as
“Game-held-out design-space sanity evaluation”. The persisted
`bga:evaluation/novelty:v0` and `bga:evaluation/clusters:v0` artifacts
are not held-out scores. A later reader can report “test novelty” that
used the test point as a neighbor of itself’s cohort.

This is scientific leakage, not a split-implementation bug. The split
is disjoint (`test_game_split_ids_on_design_space_are_disjoint`).

### C4. Missing-family zeros are standardized as measurements

**Status: FIXED (fix 3).** `FamilyStandardizer.transform` leaves value
columns at `0.0` when `present=0`.

Original finding:

Phase 7 docs: zeros behind `present = 0` are “not computed”, not
imputed play.

`FamilyStandardizer.transform` applies `(x - μ) / σ` to every
coordinate, including those padding zeros. A game that lacks
perspectives becomes an extreme negative outlier on
`perspective__mean_volume` after a train fit that saw real volumes.

Confirmed: `AUDIT-LEAK-MISS`.

Distances, neighbors, novelty, and clusters on the normalized view
will treat “family absent” as “family present and extremely low”.

### C5. Frozen models still expose mutable identity-bearing dicts

**Status: FIXED (fix 4).** `ObservationExample.items` uses `_FrozenDict`;
`FamilyBlock.values` is an immutable tuple with a `.value()` accessor.

Original finding, confirmed: `AUDIT-MUT-ITEMS`, `AUDIT-MUT-FAMILY`.

- `ObservationExample.items` is `tuple[dict, ...]`. Callers can mutate
  visibility, payload keys, or ids after construction. Encoder records
  and intervention sources then disagree with the original dump.
- `FamilyBlock.values` is a `dict` on a frozen model. Mutating it does
  not refresh `SituationRepresentation.vector`, so the object can
  disagree with itself.

`apply_intervention` deep-copies items; source payload isolation
**does** hold (`test_apply_intervention_does_not_mutate_source_payload`).
The hole is the example/block types, not the intervention copy.

### C6. Shared logical keys persist different payloads

**Status: FIXED (fix 5).** Phase 2 uses `bga:repr/trajectory-summaries:v0`;
Phase 5 uses `bga:repr/sequence-summaries:v0`. Probe eval uses
`bga:evaluation/obs-probes:v0`.

Original finding, confirmed: `AUDIT-ART-SEQ`.

`run_encode.SEQUENCE_REPR_LOGICAL_KEY` and
`sequence_eval.SEQUENCE_REPR_LOGICAL_KEY` are both
`bga:repr/sequences:v0`. Phase 2 trajectory summaries and Phase 5
experiment summaries occupy the same logical key.

Also (inspected, not separately tested):

- `SEQUENCE_MODEL_LOGICAL_KEY` and `SEQUENCE_ENCODER_LOGICAL_KEY` are
  both `bga:model/sequence-encoder:v0`.
- `ACTION_MODEL_LOGICAL_KEY` / `ACTION_REPR_LOGICAL_KEY` are defined in
  both `run_encode.py` and `transitions.py`.

`PROBE_EVAL_LOGICAL_KEY = "bga:eval/obs-probes:v0"` breaks the
`bga:evaluation/...` namespace (`AUDIT-ART-PROBE`).

A later `get_latest(logical_key=...)` can return the wrong generation.

## High-Priority Findings

### H1. Phase 4/5 treat bag encodings as state dynamics

`StateBagEmbedder` hashes keys, types, phase, turn, and four count
stats. Payload values are discarded. Confirmed:
`test_state_bag_discards_payload_values` — `{"token": "alpha"}` and
`{"token": "omega"}` encode identically.

That loss is an explicit Phase 2 choice. Phase 4
`LinearConditionedEncoder` and Phase 5 sequence predictors then learn
`(z_state, z_action) → z_next` in that space. They cannot distinguish
two chess positions that share key structure. Tests pass because
fixtures differ in keys/types/phase, not because the model sees play.

Using a lossy bag as if it supported transition-of-state claims is the
overclaim. The encoder itself is consistent.

### H2. `n_asymmetric` counts both directions

**Status: FIXED.** `count_asymmetric_pairs()` counts each unordered
observer pair once. Used in `run_higher_order_experiment` and Phase 7
`higher_order` family blocks. `AUDIT-HOP-COUNT` passes.

Original finding: `A→B` and `B→A` each incremented the counter when
their vectors differed, so one unordered asymmetric pair stored `2`.

### H3. Design-space model artifact cannot reproduce the result

**Status: FIXED (fix 5).** `bga:model/design-space:v0` now pickles a
`DesignSpaceManifest` with situation/game ids, encoder config,
cluster/novelty params, Phase 6 logical keys, and payload id links to
situation/game/normalizer/projection artifacts.

Original finding:

`bga:model/design-space:v0` pickled `{family, schema, metric,
train_game_ids}` only. It did not include:

- `PlayLayer` or source situation ids
- family-block values
- the fitted normalizer (separate artifact, not linked as a required
  input of a single reconstructable bundle)
- cluster seed / `n_clusters` (those live on the `GeometrySpec` hash
  and the evaluation artifact)

`situation_table` / `game_table` set `source_payload_ids` to entity ids,
not to the bytes that produced the vectors. The representation tables
are relocatable JSON; they are not independently interpretable as
“this digest of these situations under this encoder”.

Phase 7 docs §14 list a reproducibility recipe that the model blob
does not actually contain.

### H4. Game-level scale is applied to situation vectors

Documented in Phase 7 §6. Situation `region__n_units` is always `1`;
game `region__n_units` is the situation count. After a game-fit
standardizer, situation and game coordinates in the “same” column are
not in the same statistical family. `nearest_situations` on the
normalized view is not a situation-native scale.

### H5. `action_set_entity_id` is not injective

`'+'.join(action_ids)` maps `["a+b"]` and `["a", "b"]` to the same id.
Confirmed: `AUDIT-ID-ACT`. Order is also significant: `["a","b"]` ≠
`["b","a"]`. Simultaneous action sets can split or collide by encoding
accident.

### H6. Phase 7 reconstructs Phase 6 instead of composing artifacts

**Status: FIXED (fix 6).** `Phase6Context` holds higher-order pairs and
intervention summaries. Phase 7 feature families aggregate from that
context (built once via `build_phase6_context` or supplied via
`phase6_context_from_runs`). Situation `source_artifacts` cite
`bga:repr/higher-order-perspectives:v0` and
`bga:repr/counterfactuals:v0`.

Original finding: `_intervention_block` re-applied hide/reveal;
`_higher_order_block` recomputed `higher_order_pairs` without reading
Phase 6 artifact tables.

### H7. Public `__all__` is an export dump

`board_game_analysis.modeling` exports ~200 names, including every
logical key, every experiment runner, and several phase-internal
helpers. Missing from `__all__` but used: `FamilyBlock`,
`situation_id_for`, `sequence_entity_id`, `observation_entity_id`,
`InformationScalars`, `require_observers`, `counterfactual_entity_id`,
`encode_action_set`.

Callers cannot tell what is stable.

### H8. README and missing Phase 1–4 docs describe a different repo

README still says play pipelines are unimplemented, analysis is a
placeholder, and “there is no analysis engine”. `modeling/` is absent
from the architecture tree. There is no `docs/modeling-roadmap.md` and
no `docs/modeling-phase-1.md` … `phase-4.md`. Phase 5–7 docs exist and
are mostly accurate, with the overclaims noted above.

## Medium-Priority Findings

### M1. Reveal is not the inverse of hide

Documented and tested: hide then reveal forces `PUBLIC` +
`content_known=True`, even if the item was originally `PRIVATE` and
unknown. Functional, but “reveal” is “make public and known”, not
“restore prior visibility”.

Identity interventions are also labeled `origin="counterfactual"`.
Documented. Easy to misread as “this view was observed”.

### M2. `higher_order_pairs` last-wins on `(state_id, observer_id)`

A second observation for the same observer at the same state silently
replaces the first. No error.

### M3. Phase 1 and Phase 5 both produce `SequenceExample`

Phase 1: `entity_id = {game}/seq`, empty `steps`, `event_ids` = authored
state list order. Phase 5: `entity_id = {game}/seq/{transition}`, filled
`steps`. Same type, different identity rules. Confirmed by
`test_phase1_sequence_has_empty_steps_phase5_does_not`.

### M4. Duplicate helpers

`_first_item_id`, `_known_ids`, `_jaccard_distance`, `_mean_vector`,
`_as_float`, `_zero_change_table`, `_logical_keys` (tests) are copied
across modules. Behavior has already drifted (`n_asymmetric` in
`higher_order.py` vs `design_space.py` is the same directed count, but
nothing enforces that).

`ACTION_MODEL_LOGICAL_KEY` and `SEQUENCE_REPR_LOGICAL_KEY` are defined
in two modules.

### M5. `hash_vector` wraps SHA-256 bytes for `dim > 32`

`digest[index % 32]`. High dimensions repeat the same 32 bytes. Fine at
`dim=16`. Misleading if someone raises `dim` expecting more capacity.

### M6. DeterministicKMeans on identical points

`n_clusters = n` with identical vectors assigns every point to cluster
`0` (tie → lowest index). Empty centroids keep the previous vector.
Reproducible, but “n clusters” is not “n occupied clusters”.
Confirmed: `test_kmeans_identical_points_collapse_to_one_used_cluster`.

### M7. `PlayLayer` holds live `PlaySituation` objects

The layer is not a serializable artifact. `DesignSpace` is a frozen
in-memory graph that includes those domain objects. Persistence is a
lossy projection of that graph.

### M8. Intervention family uses only the first item

`_first_item_id` picks the first item with an id. Hide/reveal deltas
are not a summary of the information space; they are a probe of one
item. Acceptable if named as such. The column `n_items` counts
observations that had an item, not items.

### M9. Empty packages

`extraction/` and `simulation/` are `__init__.py` only. Not dead
modeling code; leftover architecture placeholders. README still lists
them as if they were the product.

### M10. CI `working-directory: board-game-analysis`

Correct for the nested checkout. Local `ruff check .` from the repo
root is equivalent to CI’s `src tests` only if no other Python trees
are present. Notebooks are not typechecked.

## Low-Priority / Maintenance Findings

- `PROBE_EVAL_LOGICAL_KEY` namespace (`bga:eval/` vs `bga:evaluation/`).
- `DesignSpace.feature_schema` was renamed to avoid Pydantic shadowing;
  older notes that say `schema` are stale.
- `ViewName` is exported from `design_space.py` by assignment at the
  bottom of the file, not from `__init__.py`.
- `split_transitions_by_game` / `split_sequences_by_game` are aliases
  of `split_entities_by_game`.
- Test helpers `_logical_keys` / `_run` / `_split` are copied in Phase
  6/7 tests.
- `record_source_id` uses `json.dumps(..., default=str)`. Unusual types
  become `str(obj)`, which can include memory addresses.
- Pairwise geometry is dense O(n²). Fine at ten games; see Scale.
- `GameDefinition` / `PlaySituation` / `Game` are mutable domain
  models. Modeling copies what it needs; it does not snapshot the
  situation graph.

## Confirmed Sound Areas

Only items that were actually probed.

| Area | Evidence |
| --- | --- |
| `domain/` never imports `ds_platform` | AST walk of every domain module |
| No `if game_id == "hanabi"` (etc.) in modeling | string scan of `modeling/` |
| Multi-situation Phase 7 works **if** local ids are unique | `test_distinct_state_ids_same_game_build_a_play_layer` |
| Phase 5 per-situation construction keeps boundaries when transition ids differ | `test_phase5_sequences_keep_situation_boundaries_when_transition_ids_differ` |
| Prefixes emit one row per situation when situation ids differ | `test_prefixes_from_two_situations_same_game` |
| Intervention does not mutate source payloads | deepcopy probe |
| Missing item / wrong observer rejected | `KeyError` / `ValueError` |
| `max_steps < 1` rejected | `bound_sequence_table` |
| Higher-order `A→B ≠ B→A`; `A→B→C ≠ C→B→A`; cycles rejected | Hanabi three-observer path |
| Train-only normalizer ignores extreme test values | `[0,1]` vs `[1000,1001]` / `[1e9,…]` |
| Train/test game id sets are disjoint on `DesignSpace` | synthetic three-game space |
| Bag encoding ignores dict insertion order | sorted keys |
| Empty game aggregation rejected | `represent_game([])` |
| Game vector independent of situation input order | sort-by-`situation_id` |
| Region: `n_units` matches situation count; range ≥ dispersion | two-situation game |
| Flatten of all-missing families is explicit zeros | `flatten_families` |
| k-means: `n=1` works; `n > N` raises; `n = N` on distinct 1-D points occupies all clusters | synthetic tables |
| Catalog title/rating/mechanics do not change Phase 7 vectors | existing `test_metadata_is_not_a_representation_input` |
| Game-level split helper keeps a game on one side | existing Phase 4/5/7 tests plus this audit |

## Phase-by-Phase Findings

### Phase 1 — examples and ontology-v0 measures

**What it is.** `PlaySituation` → `ExampleBundle` plus naive
`DerivedMeasurement` rows from `analysis.spec` names.

**Defects.** Phase 1 `sequence_entity_id` is game-only. Game-scoped
measurement ids ignore situation. `examples_from_situations` concatenates
without uniqueness checks. `StateExample.data` is a shallow `dict()` copy;
nested mappings remain shared with the domain object.

**Sound.** Encoder records omit title and payload values. Fixture corpus
round-trips through JSON (existing tests). Measures import the catalog
from `analysis.spec` and do not import `ds_platform`.

**Tests.** `test_examples.py` (6), `test_measures.py` (11): mostly
fixture coverage and “metric exists”. No multi-situation uniqueness
tests existed before this audit.

### Phase 2 — bag encodings

**What it is.** Frozen SHA-256 bag pooling + 4 stats. `fit` is a no-op.

**Intentional loss.** Values discarded; only keys/types/visibility/counts
remain. Two different payloads with the same keys collide. Documented
in the encoder docstring. **Not a bug** at Phase 2.

**Defects.** Collision becomes a Phase 4/5 scientific bug when those
vectors are treated as state. `hash_vector` wraps at 32 bytes.
`record_source_id(..., default=str)` can leak `repr`.

**Parallel persist path.** `run_encode.encode_*` writes representation
and model artifacts independently of later experiment runners, and
reuses sequence/action logical keys.

### Phase 3 — perspectives

**What it is.** `PerspectiveTable` from observation encodings;
pairwise distances; stored as `bga:repr/perspectives:v0`.

**Defects.** Thin. Depends on bag collisions. `aligned_distance_pairs`
assumes matching entity alignment. Only 4 dedicated tests; no empty
observer set, no duplicate observer at a state.

**Sound.** Does not invent beliefs. Distance is representation
distance.

### Phase 4 — transitions

**What it is.** `PairExample` + `ActionBagEmbedder` +
`LinearConditionedEncoder`. Game-safe split. Encodes all states/actions
then fits the linear map on train pairs only.

**Defects.** `transition_pairs_from_situations` last-wins on `game_id`
(C2). Action-set ids not injective (H5). Learned map is over bags, not
states (H1). `CopyStateEncoder` baseline is honest; the linear model’s
name is not.

**Sound.** Empty action sets are skipped, not invented. Split is
game-grouped. Bag `fit` cannot leak through the pre-split encode.

### Phase 5 — sequences

**What it is.** Contiguous authored chains; prefixes exclude the
target to-state; `PrefixSequenceEncoder` + `LinearSequencePredictor`.

**Defects.** Trajectory id ignores situation (C1). Prefix entity id is
the pair id (C1). Shared `bga:repr/sequences:v0` with Phase 2 (C6).
Doc says situations are never merged; ids can still collide.

**Sound.** `sequences_from_situations` does not concatenate across
situations. Cycles skipped with a reason. `max_steps` is a Phase 6
concern; Phase 5 prefixes are finite authored steps.

**Tests.** 26 existing tests — strongest suite after this audit’s
file. Still fixture-shaped (Telestrations as the only two-step chain).

### Phase 6 — interventions and higher-order

**What it is.** Frozen `Intervention`; observation-only apply;
`substitute_action` is a query, not an observation rewrite; rollout is
open-loop and requires `max_steps`; `A_about_B` is concat + delta +
cosine + item-set counts.

**Defects.** Mutable example items (C5). `n_asymmetric` directed
double-count (H2). Last-wins observer map (M2). Reveal ≠ inverse of
hide (M1, documented).

**Sound (tested).** Source isolation; missing targets raise;
`max_steps=0` rejected; directionality; cycle rejection; identity
operation still marked counterfactual. Rollout wraps platform
`rollout` and does not call a game engine.

**Not a simulator.** Confirmed by construction: no `PlaySituation`
mutation, no legal-action generation, `substitute_action` cannot be
applied to an observation.

### Phase 7 — design space

**What it is.** Family schema with presence flags; mean aggregation
sorted by situation id; train-only standardizer/projection; l2
geometry; exploratory k-means; structural novelty = mean kNN distance.

**Defects.** C1–C4, H3, H4, H6. Experiment API overclaims holdout.
Missing-family standardization. Model pickle is not a recipe.

**Sound (tested).** Metadata exclusion; train-only mean/std on extreme
test values; order-independent game aggregation; region `n_units`;
k-means edge cases listed above; empty game rejected; flatten padding
is zeros *before* standardization.

**Region math.** `situation_dispersion` is mean l2 distance to the
situation-vector centroid. `situation_range` is max pairwise l2.
For two points, range = 2 × dispersion. The implementation matches
the Phase 7 document.

## Cross-Phase Findings

### Actual dependency graph

```text
domain.PlaySituation
        │
        ▼
   examples.py ──────────────────────────────► artifacts.py
        │                                         │
        ├────────► measures.py ──analysis.spec    │
        │                                         │
        ├────────► pairs.py ──────► transitions.py
        │              │                  │
        ├────────► sequences.py ──► sequence_eval.py
        │              │                  │
        ▼              ▼                  ▼
   encoders/bag,state,observation,action,conditioned,sequence
        │
        ├────────► perspectives.py ──► probes.py
        │
        ├────────► interventions.py ──► counterfactuals.py
        │                │
        │                └──► higher_order.py
        │
        └────────► design_space.py ──► design_geometry.py
                      (re-calls Phase 2/4/5/6 functions)
```

There is **no import cycle**. There **are** hidden parallel paths:

| Path | Why it is parallel |
| --- | --- |
| Phase 1 `SequenceExample` vs Phase 5 `SequenceExample` | same type, different ids and `steps` |
| `run_encode` persist vs experiment persist | same logical keys, different payloads |
| `measure_situation` vs design-space families | two measurement systems |
| Phase 7 hide/reveal / hop rebuild | does not read Phase 6 artifacts |
| `analysis/descriptive.py` | metadata corpus pipeline; not on this graph |
| `ingestion/` | BGG catalog; explicitly excluded from design-space vectors |

Phase 7 is a **function-level** composition of 1–6, not an artifact-level
DAG. That is the main abstraction duplication.

### Accidental coupling

- `design_space._intervention_block` hard-codes hide-then-reveal of
  the first item. That is a Phase 6 policy smuggled into Phase 7
  features.
- `sequence_eval` imports a Phase 4 baseline encoder. Fine as a
  comparison; the evaluation report can be read as “sequence model
  vs transition model” rather than “sequence quality”.
- `measures.py` imports `analysis.spec`. Domain stays clean; modeling
  is coupled to the measurement catalog. Acceptable if catalog remains
  the name source.

### Dead / unreachable

No unreachable modeling modules. `extraction/` and `simulation/` are
empty packages. `storage/` is a placeholder. Not modeling-dead; product
dead.

## Leakage Findings

Every path that was actually tested.

| Path | Result |
| --- | --- |
| Train `[0,1]`, test `[1000,1001]` then `[1e9,…]` on `FamilyStandardizer` | **No leakage.** Means/stds unchanged. |
| Projection fit on train ids vs all ids (existing Phase 7 test) | **No leakage** into kept coordinates when using the public fit. |
| Catalog metadata mutation (existing Phase 7 test) | **No leakage** into vectors. |
| Encode-all then split, bag embedders | **No learned leakage** (`fit` is a no-op). |
| Encode-all then split, `LinearConditionedEncoder` | Train pairs only; **no split leak** if callers use `run_transition_experiment`. |
| Novelty kNN on a table that includes a movable “test” point | **Leakage.** Train scores change. |
| k-means on the same table | **Leakage.** Train labels change. |
| `build_design_space` + `novelty_table` with a cloned vs stripped test game | **Leakage.** Train novelty changes. |
| Missing-family zeros after train standardize | **Not split leakage**; **scientific distortion** of holdout geometry. |
| `run_design_space_experiment` clusters/novelty | Full corpus after train-only scale. **Evaluation leakage** if read as holdout. |
| Duplicate situations / same game different ids | Not a split leak; an identity leak into `RepresentationTable`. |
| BGG corpus into play vectors | **Blocked** by `FORBIDDEN_METADATA_FIELDS` and `external_metadata` being post-hoc. |

Untested leakage paths (coverage gaps): probe `RidgeRegressor` fit on
all observations; `measure_situations` used as labels after a bad
entity-id collision; reusing a persisted normalizer on a relocated
corpus without checking `train_entity_ids`.

## Fixture Generality Findings

The implementation has **no** fixture-name branches. It is not a Hanabi
special case in code.

It is **fixture-shaped** in identity and scale:

- one situation per game in `tests/fixtures/games/`
- local ids already globally unique (`chess-s0`, not `s0`)
- small finite action lists
- short trajectories (Telestrations is the only two-step chain)
- observer sets of size 1–3
- bag-distinguishable key structures

Synthetic probes with different player counts, missing families,
asymmetric observers, and two situations per game **work only when
every local id is unique**. Conventional local ids crash or collide.

Verdict: the *types* are general; the *identifiers and batch APIs* are
one-situation-per-game.

## Identity / Determinism Findings

| Identity | Canonical? | Notes |
| --- | --- | --- |
| Intervention id | Yes | `spec_canonical_json_bytes` of the id body |
| Bag source payload id | Mostly | `sort_keys=True`; `default=str` is the hole |
| State / obs / action entity ids | No | game-scoped, not situation-scoped |
| Situation id | No | first state id only |
| Phase 1 sequence id | No | game only |
| Phase 5 trajectory id | No | game + first transition |
| Action-set id | No | `+` join; order-sensitive |
| Measurement id | No | game + name + scope_key |
| Higher-order entity id | Yes *per state* | `{game}/hop/{state}/{A}>{B}`; last-wins if duplicate observers |
| Counterfactual entity id | Mostly | `{source}/cf/{op}/{16 hex}`; two interventions with the same op and colliding suffix space are unlikely but the suffix is truncated |
| Feature schema columns | Yes | fixed family order |
| Game aggregation | Yes | sort by `situation_id`, then `game_id` |
| Normalizer / projection | Yes | train id set + column order; dict insertion order does not matter |
| k-means seed 0 | Yes | entity-id sorted init; seed ≠ 0 uses `random.Random` |
| Design-space model blob | Incomplete | see H3 |

Timestamps, URIs, and machine info do not enter these ids. `RunContext`
timestamps affect artifact records, not payload ids, unless a caller
puts them in a spec.

## Artifact / Provenance Findings

| Key | Payload | Enough to reproduce? |
| --- | --- | --- |
| `bga:examples/play-fixtures:v0` | JSON example bundle | Yes, for the dump; situation graph itself is not in the blob |
| `bga:measurements/ontology-v0:v0` | JSONL measurements | Yes for values; method is a string |
| `bga:repr/observations:v0` etc. | `RepresentationTable` JSON | Vectors + entity ids; encoder dim/family live on a sibling model pickle |
| `bga:repr/sequences:v0` | **Two writers** | No — ambiguous |
| `bga:eval/obs-probes:v0` | evaluation | Namespace inconsistent |
| `bga:repr/perspectives:v0` | perspective table | Yes as a table; not as a belief model |
| `bga:repr/transitions:v0` | predicted transition reps | Needs the fitted encoder pickle |
| `bga:repr/counterfactuals:v0` | CF observation encodings | Intervention model is a separate pickle |
| `bga:repr/higher-order-perspectives:v0` | hop vectors | Same |
| `bga:repr/situations:v0` | situation vectors | Entity ids = situation ids; `source_payload_ids` are those ids, not upstream digests |
| `bga:repr/games:v0` | game vectors | Same |
| `bga:model/design-space:v0` | schema + metric + train ids | **No** |
| `bga:model/design-space-normalizer:v0` | full `FamilyStandardizer` | Yes for transform; not for “why these means” without the train table |
| `bga:model/design-space-projection:v0` | keep indexes | Yes for transform |
| `bga:evaluation/novelty:v0` | `EvaluationReport` | Summary metrics only; per-game scores are in-memory on `DesignSpaceRun` |
| `bga:evaluation/clusters:v0` | `EvaluationReport` | `n_clusters` + repeatability; **not** the assignment vector |

Version suffix `:v0` is a string, not a migration system. Relocating
the store keeps payload ids; logical-key latest-pointer semantics are
undefined across the duplicate keys.

## Numerical Findings

| Case | Behavior | Risk |
| --- | --- | --- |
| Missing-family zeros after z-score | Extreme coordinates | Wrong neighbors / novelty |
| Zero-variance columns | `std = 1`, listed | Honest; constant becomes 0 |
| Cosine of zero vectors (platform) | similarity 0, distance 1 | Fine if documented; do not read as “orthogonal” |
| k-means identical points | one occupied cluster | Overstated `n_clusters` |
| `n_clusters > n` | `ValueError` | Fine |
| Integer vs float inputs | coerced via `float(...)` | Fine |
| NaN/inf | not rejected in flatten or distance | Can poison kNN; fixtures are clean |
| `dim > 32` bag wrap | repeated bytes | False sense of capacity |
| Singleton situation region | dispersion 0, range 0 | Correct |
| Two-point region | range = 2 × mean-to-centroid | Correct |

No silent overflow was observed on the synthetic tables used here.
The missing-family z-score is the numerical issue that will actually
distort research results.

## Scale Findings

Current fixture scale: 10 situations, tens of states.

| Operation | Complexity | 10k situations | 100k | 1M |
| --- | --- | --- | --- | --- |
| Bag encode | O(items × dim) | fine | fine | watch memory of tables |
| `build_play_layer` concat | copies all examples | fine | watch | problem if every situation is re-encoded from scratch each experiment |
| `higher_order_pairs` | O(observers²) per state | fine | watch for high observer counts | fine if observers stay small |
| `represent_game` pairwise range | O(s²) per game | fine if s is small | watch if a game has 10k situations | architectural if full-game traces are situations |
| `pairwise_*_distances` | O(n² dim) | watch at 10k games | problem | architectural |
| `novelty_scores` / `knn` | dense pairwise | watch | problem | architectural |
| `DeterministicKMeans` | O(n k dim iter) | fine | fine | fine (not the bottleneck) |
| Phase 7 hide/reveal per observation | O(observations) Python apply | fine | watch | problem as a feature extractor |
| Pickle of embedders / normalizers | small | fine | fine | fine |
| Holding `PlayLayer` of all `PlaySituation`s | memory | watch | problem | architectural |

Do not add a vector database for ten games. Do not recompute pairwise
novelty inside a loop over games at 100k.

## Test Coverage Gaps

Existing modeling tests before this audit: 101 functions
(examples 6, measures 11, encoders 5, perspectives 4, transitions 13,
sequences 26, interventions 11, higher-order 8, design space 17).

Classification of that suite:

| Class | Rough share | Notes |
| --- | --- | --- |
| Fixture integration | high | “all ten fixtures produce X” |
| Persistence / logical keys | high | “key present, kind is DATASET” |
| Unit | medium | factories, split, flatten |
| Determinism | medium | Phase 7 stronger than 1–5 |
| Leakage | low | normalizer/projection train-only only |
| Identity uniqueness | almost none | assumed unique fixture ids |
| Numerical edge | low | zero-var, k bounds |
| Scientific semantics | low | notes exist; few assertions |
| “function runs / length ok” | material | especially encode + persist tests |

Still missing after this audit (do not treat xfails as coverage):

- `substitute_action` + rollout with invalid / missing model outputs
- long rollouts, cyclic pair graphs, repeated state ids in a chain
- `change_observer` when two observations share an observer
- probe leakage (ridge fit on all rows)
- NaN/inf in design-space columns
- `k` ties in novelty beyond platform knn’s `(distance, id)` sort
- artifact reload: write Phase 2 sequences, write Phase 5 sequences,
  `get_latest` — would fail or pick an arbitrary payload
- measurement `FeatureTable` construction over `measure_situations`
  of two situations of one game
- `record_source_id` with a nested custom object
- cosine self-distance / zero-vector through `game_distance`
- `LeadingVarianceProjection` when all train variances are 0

## API Consolidation Candidates

Do not implement this list now.

### STABLE v0 PUBLIC API

Keep and document:

- `examples_from_situation(s)`, `ExampleBundle`, example record types
- `state_entity_id`, `situation_id_for` (after identity fix)
- `StateBagEmbedder`, `ObservationBagEmbedder`, `ActionBagEmbedder`
- `perspective_table_from_observations`, `perspective_distances`
- `transition_pairs_from_situation(s)`, `PairExample`, `ActionExample`
- `sequence_from_situation`, `sequences_from_situations`,
  `prefixes_from_situations`
- `split_entities_by_game`
- `Intervention` factories, `apply_intervention`,
  `counterfactual_rollout` (with `max_steps`)
- `higher_order_perspective`, `higher_order_pairs`, `ObserverPath`,
  `compose_observer_path`
- `FeatureSchema`, `SituationRepresentation`, `GameRepresentation`,
  `build_play_layer`, `represent_situation(s)`, `represent_game(s)`
- `FamilyStandardizer`, `LeadingVarianceProjection`, `build_design_space`,
  `game_distance`, `situation_distance`, `nearest_*`, `novelty_*`,
  `cluster_games`
- logical-key constants **once**, in one module

### INTERNAL BUT USED

- `encoders.bag` (`hash_vector`, `encode_bags`)
- `_situation_index`, family-block builders
- `bound_sequence_table`
- `require_observers`
- `CopyStateEncoder`, `LastStatePredictor` (baselines)
- `RidgeRegressor`, `probe_column`

### EXPERIMENTAL

- `run_*_experiment` runners (glue, not the model)
- `DeterministicKMeans`
- `LeadingVarianceProjection`
- `attach_external_metadata`
- `intervene_higher_order`
- `LinearConditionedEncoder`, `LinearSequencePredictor`

### DUPLICATE / CONSOLIDATE

- `SEQUENCE_REPR_LOGICAL_KEY` (two modules, one string, two payloads)
- `SEQUENCE_MODEL_LOGICAL_KEY` / `SEQUENCE_ENCODER_LOGICAL_KEY`
- `ACTION_MODEL_LOGICAL_KEY` / `ACTION_REPR_LOGICAL_KEY` dual definition
- `split_transitions_by_game` / `split_sequences_by_game`
- `_mean_vector` / `_as_float` / `_jaccard_distance` / `_known_ids`
- Phase 1 vs Phase 5 `SequenceExample` construction
- Phase 7 intervention/hop rebuild vs Phase 6 artifacts

### REMOVE / DEAD

- Nothing in `modeling/` is unreachable.
- Do not delete empty `extraction/` / `simulation/` in a modeling
  patch; they are product placeholders, not this audit’s cleanup.

## Documentation Drift

| Claim | Reality |
| --- | --- |
| README: play pipelines unimplemented; analysis is a placeholder; no analysis engine | Modeling Phases 1–7 exist; corpus descriptive analysis exists |
| README architecture tree omits `modeling/` | `src/board_game_analysis/modeling/` is 24 modules |
| `docs/modeling-roadmap.md`, `modeling-phase-1.md`–`4.md` | **Do not exist** |
| Phase 5: situations never merged across boundaries | Objects are not concatenated; **ids can still collide** |
| Phase 6: intervention application is functional and source-safe | True for apply; example dicts remain mutable |
| Phase 7: zeros behind `present=0` are not imputed | True in flatten; **false after standardize** |
| Phase 7 §13 leakage controls | Accurate for scale/projection/metadata; **silent on novelty/clustering** |
| `run_design_space_experiment` “held-out … evaluation” | Split is held-out; scores are not |
| Phase 7 §14 reproducibility recipe | ~~Model pickle does not store that recipe~~ Fixed: `DesignSpaceManifest` |
| Phase 7 situation ids are `game:state` | They are `game:sorted_states|transitions` after fix 1 |
| `bga:evaluation/...` as the eval namespace | ~~Probe eval is `bga:eval/obs-probes:v0`~~ Fixed |
| Ontology README: “no mechanic taxonomy” | Still true; not a modeling defect |

Implemented and under-documented: game-normalizer-on-situations
implications (H4), first-item-only intervention probe selection,
duplicate `ACTION_*` logical keys in `run_encode` vs `transitions`.
Fixed but may need doc sync: situation-scoped identity, split sequence
keys, `count_asymmetric_pairs`, Phase 6 composition in Phase 7.

## Recommended Fix Order

1. ~~**Correctness — identity scope.**~~ **Done.**
2. ~~**Scientific validity — evaluation APIs.**~~ **Done** (train-only
   neighbors/clusters). Remaining: rename docs/artifacts if “held-out”
   language is kept misleading.
3. ~~**Scientific validity — missing families.**~~ **Done.**
4. ~~**Data integrity — mutability.**~~ **Done.**
5. ~~**Data integrity — artifacts.**~~ **Done.**
6. ~~**Architecture — compose artifacts, don’t rebuild.**~~ **Done.**
7. ~~**Test quality.** Promote the `AUDIT-*` xfails.~~ **Done** (0
   xfails remain). Remaining: multi-situation fixtures with local ids
   in the main corpus, not only adversarial tests.
8. **Maintainability.** One module for logical keys; shrink `__all__`;
   dedupe helpers.
9. **Ergonomics.** README + Phase 1–4 docs + roadmap that match the
   tree. Tighten names: “structural kNN outlierness”, “representation
   space”, “authored-transition model”.

## What NOT to Fix

These look odd and should stay unless a later spec changes the
research claim.

- **Bag encodings discarding payload values.** Phase 2 is a structural
  bag, not a state emulator. Keep the loss; stop overclaiming later
  phases.
- **Reveal = make public and known**, not restore. Documented
  intervention algebra.
- **Identity intervention origin is `counterfactual`.** Apply always
  constructs a new entity.
- **`substitute_action` cannot be applied to an observation.** Prevents
  the layer from becoming a simulator.
- **`max_steps` required on rollout.** Keep the bound.
- **Higher-order `A_about_B ≠ B_about_A`.** Direction is the point.
- **Train-only normalizer/projection.** The fit is the leakage-safe
  part. Do not “fix” it by fitting on all games.
- **Zero-variance `std = 1`.** Explicit and tested.
- **Projection as a coordinate mask, not PCA/UMAP.** Do not upgrade
  it into a manifold method to look serious.
- **k-means labeled exploratory; clusters are not genres.** Keep the
  disclaimer; fix the held-out naming instead.
- **Catalog metadata post-hoc only.** `FORBIDDEN_METADATA_FIELDS` and
  `attach_external_metadata` are the right boundary.
- **`domain/` without `ds_platform`.** Do not “simplify” by importing
  the platform into domain models.
- **`analysis/spec.py` and `analysis/descriptive.py` unchanged.** The
  metadata corpus pipeline is a different product surface.
- **O(n²) pairwise geometry at current scale.** Do not add ANN or a
  vector DB for ten points.
- **Phase 1 empty-`steps` `SequenceExample`.** Awkward, but the Phase 5
  split is intentional. After identity is fixed, consider two types
  rather than one type with two modes.
- **Frozen no-op bag `fit`.** That is what makes encode-then-split
  safe for Phase 2.

---

## Audit inventory (for the record)

### Real Phase 1–7 graph (imports / calls)

See Cross-Phase Findings. The slide graph is mostly real. Hidden
parallels: Phase 1 vs 5 sequences; `run_encode` vs experiment writers;
Phase 7 rebuild of Phase 6; analysis/ingestion off to the side.

### Public exports actually imported from `board_game_analysis.modeling`

`__all__` in `modeling/__init__.py` (lines 229–424) is the public
surface. It is larger than any documented API. Classification is in
§API Consolidation Candidates.

### Adversarial tests added

`tests/modeling/test_audit_adversarial.py` — 52 tests, all passing.

Do not delete or weaken audit probes to green the suite.

## Fixes applied

| Item | Audit ids | Summary |
| --- | --- | --- |
| 1 | C1, C2, `AUDIT-ID-*` | Situation-scoped entity ids; batch maps keyed by `situation_id` |
| 2 | C3, `AUDIT-LEAK-*` | Train-only novelty neighbors and cluster fit |
| 3 | C4, `AUDIT-LEAK-MISS` | Missing-family value columns stay 0 after normalize |
| 4 | C5, `AUDIT-MUT-*` | `_FrozenDict` items; immutable `FamilyBlock.values` |
| 5 | C6, H3, `AUDIT-ART-*` | Split sequence repr keys; probe eval namespace; `DesignSpaceManifest` |
| 6 | H6 | `Phase6Context` composes intervention/hop families |
| — | H2, `AUDIT-HOP-COUNT` | `count_asymmetric_pairs()` — one count per unordered pair |
