# BGA Modeling Pipeline — Final Adversarial Audit (Phases 1–7)

Audit date: 2026-09-19. Scope: `src/board_game_analysis/modeling/` and its
integration with `ds_platform`.

This audit follows the first adversarial audit (`docs/modeling-audit.md`) and
the fix series (items 1–9, hop-count).

## Option A fixes applied (2026-09-19)

All CRITICAL items C-F1–C-F6 from this report were implemented:

| Item | Fix |
| --- | --- |
| C-F1 | `situation_id_for` appends `@{content_digest}` after topology key |
| C-F2 | Observation labels keyed by `(situation_id, observer_id, state_id)` |
| C-F3 | Transition eval joins scoped by `pair.situation_id` |
| C-F4 | Sequence eval joins scoped by `prefix.situation_id` |
| C-F5 | Counterfactual eval maps `pair.situation_id → situation` |
| C-F6 | Hop entity ids and Phase 7 hop blocks situation-scoped |

Permanent tests: `tests/modeling/test_audit_final_adversarial.py` (5 cases).
Documentation: Phase 1 situation id format, Phase 7 H4 + kNN semantics,
manifest replay limits.

Post-fix suite: **297 passed**, ruff clean, pyright clean.

---

## Executive verdict (pre-fix)

**Not ready to freeze as v0** without addressing multi-situation identity and
join-scope defects. The previous fix series correctly hardened entity IDs,
batch maps, train-only novelty/clustering, padding semantics, immutability of
observation items, artifact logical keys, Phase 6 composition, and hop-count
aggregation. Those fixes hold on the paths the adversarial suite exercises.

However, a second pass with deliberately adversarial multi-situation fixtures
finds **remaining CRITICAL defects** that the green suite does not cover:

1. **`situation_id_for` is topology-only** — two genuinely different situations
   with the same sorted state IDs and transition IDs receive the same situation
   ID, producing duplicate entity IDs and a hard failure (or would silently
   merge if validation were relaxed).
2. **Secondary joins still key on `(game_id, …, local_state_id)`** without
   situation scope — observation labels, transition/sequence eval feature
   joins, counterfactual `by_game`, higher-order entity IDs, and Phase 7 hop
   block scoping can cross-contaminate when one game has multiple situations
   with overlapping local IDs.

Parametric leakage for learned components (Phases 4–7 normalizer/projection/novelty
fit) is **largely fixed**. Semantic misinterpretation risks remain around
full-corpus kNN neighbors and cross-level normalization (H4 — intentional).

| Metric | Value |
| --- | --- |
| Full suite | **292 passed**, 0 failed, 0 xfailed |
| Ruff | clean |
| Pyright | clean |
| Adversarial tests (`test_audit_adversarial.py`) | 44 |
| Additional manual probes (this audit) | 18 |
| **Total adversarial cases attempted** | **62** |

### Findings by severity

| Severity | Count | Must fix before v0? |
| --- | ---: | --- |
| CRITICAL | 6 | Yes |
| HIGH | 7 | 4 yes, 3 document/clarify |
| MEDIUM | 8 | No (track for v0.1) |
| LOW | 6 | No |
| INTENTIONAL | 5 | Document only |

### Direct answers

| Question | Answer |
| --- | --- |
| Is H4 a bug? | **No — intentional design choice.** Game-fit normalizer applied to both game and situation vectors is documented in Phase 7 §6 but needs clearer API-level contract. |
| Adequate multi-situation fixture coverage? | **No.** `tests/fixtures/games/` is one situation per game. Synthetic multi-situation tests require **different transition IDs**; identical-topology and label-join bleed are untested. |
| Hidden one-situation-per-game assumptions? | **Yes — several remain** in measurement joins, counterfactual eval, higher-order IDs, and hop scoping. |
| Parametric leakage remains? | **Mostly no** for fit paths. **Yes for interpretation**: full-corpus kNN neighbors, corpus-wide dispersion metrics, persisted novelty means. |
| Provenance complete? | **Partial.** Lineage within a run is traceable; cross-run exact reproduction from manifest alone is **not** possible without original `PlaySituation` inputs. |
| Determinism holds? | **Yes** on tested paths (game/situation order permutations, repeated clustering with same seed). |
| Public API coherent enough to freeze? | **Borderline.** Slim `__all__` and `logical_keys.py` help; submodule duplication and ambiguous local-ID-based joins undermine v0 claims for multi-situation corpora. |
| Ready for v0 freeze? | **No** until CRITICAL multi-situation identity/join defects are resolved or explicitly scoped out with a documented “single situation per game” v0 constraint. |

---

## 1. Multi-situation stress test

### Fixture constructed

One game (`game-1` / `g`) with two situations sharing local IDs:

```text
Situation A: states s0,s1 — transition tr-a — payload value 1
Situation B: states s0,s1 — transition tr-b — payload value 99
```

Additional variants: identical topology (same `t0`), different observation
volumes (extra `InformationItem`), reversed input order, permuted game order.

### What passes

When transition IDs differ, the pipeline keeps situations distinct through:

- Phase 1 entity IDs (`{situation_id}/state/{state_id}`, etc.)
- `build_play_layer` → 4 unique state entity IDs for 2×2 states
- Phase 7 game aggregation (`n_situations=2`, order-independent game vector)
- Phase 5 sequence/trajectory IDs (when transition IDs differ)

Confirmed by existing adversarial tests and probes:

```python
# game vector identical under situation-order permutation
represent_games(reps_fwd)[0].vector == represent_games(reps_rev)[0].vector  # True
```

### CRITICAL — identical topology → identical `situation_id`

**Finding.** `situation_id_for` hashes only `game_id`, sorted state IDs, and
sorted transition IDs. Payloads, observations, actions, and state `data` do
not participate.

```338:351:src/board_game_analysis/modeling/examples.py
def situation_id_for(situation: PlaySituation) -> str:
    return _situation_id(situation)

def _situation_id(situation: PlaySituation) -> str:
    ...
    state_ids = ",".join(sorted(game_state.id for game_state in situation.states))
    transition_ids = ",".join(sorted(transition.id for transition in situation.transitions))
    ...
    return f"{situation.game.id}:{state_ids}|{transition_ids}"
```

**Reproduction.**

```python
a = _situation(game_id="gx", transition_id="t0", payloads=({"v": 1},))
b = _situation(game_id="gx", transition_id="t0", payloads=({"v": 2},))
assert situation_id_for(a) == situation_id_for(b)  # True
build_play_layer([a, b], dim=8)  # ValidationError: entity_ids must be unique
```

**Why it matters.** Two authored moments in the same game with the same local
graph shape cannot coexist. This is a fundamental identity defect, not an edge
case — variant openings, scenario branches, and replay slices commonly reuse
local IDs.

**Affected phases.** 1–7 (situation ID is the root namespace).

**Proposed fix.** Extend `_situation_id` with a content digest (canonical hash of
observations, actions, state payloads, or an explicit author-supplied
`moment_id`). Fail fast on collision only when digests also match.

**Before v0 freeze?** **Yes.**

---

### CRITICAL — observation label join bleed

**Finding.** `observation_label_table` indexes measurements by
`(game_id, player_id, state_id)` — no `situation_id`.

```100:124:src/board_game_analysis/modeling/artifacts.py
        if measurement.scope == "observation":
            key = (measurement.game_id, measurement.player_id, measurement.state_id)
            obs_values.setdefault(key, {})[measurement.name] = value
...
        obs = obs_values.get((game_id, observer_id, state_id), {})
```

Measurements themselves are situation-scoped (`id = f"{situation_id}:{name}:{scope_key}"`),
but the join ignores that.

**Reproduction.**

```python
# Situation A: information_volume = 1.0
# Situation B: same local ids, extra item → information_volume = 2.0
labels = labels_for_observations(meas, obs, source_payload_ids=[...])
# Both rows receive vol = 2.0 (last-write-wins on shared key)
```

**Why it matters.** Phase 3 perspective tables and any downstream eval using
observation labels silently attach the wrong scalars when local state IDs overlap
within a game.

**Affected phases.** 2–3 (labels), 4–5 (transition/sequence eval features that
reuse the same join pattern).

**Proposed fix.** Key by `(situation_id, observer_id, state_id)` or join on
measurement `id` / entity alignment.

**Before v0 freeze?** **Yes.**

---

### CRITICAL — transition eval measurement joins (`_scalar_at`, `_player_count`)

**Finding.** Phase 4 transition metrics join measurements on
`(game_id, player_id, state_id)` without situation scope.

```388:404:src/board_game_analysis/modeling/transitions.py
def _scalar_at(..., game_id: str, player_id: str, state_id: str) -> float | None:
    for item in measurements:
        if (
            item.name == name
            and item.game_id == game_id
            and item.player_id == player_id
            and item.state_id == state_id
            ...
```

**Affected phases.** 4.

**Proposed fix.** Pass `situation_id` from `PairExample` into join keys.

**Before v0 freeze?** **Yes** (any multi-situation-per-game corpus).

---

### CRITICAL — sequence eval measurement joins (`_mean_at`)

**Finding.** Same pattern as transitions — averages all measurements matching
`(game_id, state_id)` across situations.

```422:438:src/board_game_analysis/modeling/sequence_eval.py
def _mean_at(..., game_id: str, state_id: str) -> float | None:
    values = [
        float(item.value)
        for item in measurements
        if item.name == name
        and item.game_id == game_id
        and item.state_id == state_id
        ...
    ]
    return sum(values) / len(values)  # mean across colliding situations
```

**Affected phases.** 5.

**Proposed fix.** Scope by `prefix.situation_id`.

**Before v0 freeze?** **Yes.**

---

### CRITICAL — counterfactual eval `by_game` last-wins

**Finding.** Phase 6 counterfactual experiment resolves the source situation by
game ID only.

```331:335:src/board_game_analysis/modeling/counterfactuals.py
    by_game = {situation.game.id: situation for situation in situations}
    ...
        situation = by_game[pair.game_id]
```

**Reproduction.** Two situations for game `g`; dict retains the last situation
only. Test-pair counterfactual queries use wrong decision spaces / alt actions.

**Affected phases.** 6.

**Proposed fix.** Key by `pair.situation_id` or `(game_id, situation_id)`.

**Before v0 freeze?** **Yes.**

---

### CRITICAL — higher-order ID and hop block scoping

**Finding A.** `higher_order_entity_id` uses `game_id`, not `situation_id`:

```134:137:src/board_game_analysis/modeling/higher_order.py
def higher_order_entity_id(game_id: str, state_id: str, focal_id: str, target_id: str) -> str:
    return f"{game_id}/hop/{state_id}/{focal_id}>{target_id}"
```

**Finding B.** `higher_order_pairs` indexes observations by
`(state_id, observer_id)` — unsafe if a batch spans situations with overlapping
local IDs.

**Finding C.** Phase 7 `_higher_order_block_from_pairs` filters pairs by local
`state_id` set only:

```676:677:src/board_game_analysis/modeling/design_space.py
    state_ids = {example.state_id for example in examples.states}
    scoped = [pair for pair in pairs if pair.state_id in state_ids]
```

When Phase 6 context is built corpus-wide, hop pairs from situation B can be
included in situation A's feature block if they share local state IDs.

**Affected phases.** 6–7.

**Proposed fix.** Situation-scope all three: entity IDs, observation maps, hop
block filter (match on pair entity prefix or explicit `situation_id` field).

**Before v0 freeze?** **Yes.**

---

### Coverage gap in existing adversarial tests

`test_overlapping_local_state_ids_do_not_crash_play_layer` requires **different
transition IDs** (`tr-a` vs `tr-b`). It does **not** cover:

- identical transition IDs / identical topology
- label or eval join bleed
- counterfactual `by_game`
- hop block cross-situation inclusion

The main fixture corpus (`tests/fixtures/games/`, ~10 games) remains **one
situation per game**, so integration tests never hit these paths.

---

## 2. Identity and persistence audit

### Sound (post-fix 1)

- Phase 1 entity IDs: `{situation_id}/state|obs|pair|actions|seq/{local}`
- Batch pair maps: `situation_id_for(situation)` in `pairs.py`
- Measurement IDs include `situation_id` prefix
- Game-level aggregation sorts by `situation_id` (order-independent)
- Artifact content identity via `ds_platform` payload digests (URI/metadata
  changes do not alter content ID when payload bytes unchanged)
- `action_set_entity_id` `+` delimiter collision fixed via `\x1f` separator

### Remaining identity risks

| Issue | Severity | Notes |
| --- | --- | --- |
| Topology-only `situation_id_for` | CRITICAL | See §1 |
| `action_set_entity_id` `\x1f` injection | MEDIUM | `["a","b"]` vs `["a\x1fb"]` collide |
| `higher_order_entity_id` game-scoped | CRITICAL | See §1 |
| `perspectives.py` uses local `state_id` as subject | HIGH | Ambiguous in multi-situation batches |
| `prefixes_from_situations` `by_id` dict | MEDIUM | Loses situation on `situation_id` collision |
| Duplicate sequence entity IDs before play layer | HIGH | Two identical-topology situations → 2 sequences, 1 unique `entity_id` |

**Adversarial inputs attempted:** reordered lists (pass), duplicate elements
(play layer rejects duplicate entity IDs), delimiter chars in action IDs (partial
— `+` safe, `\x1f` not), empty IDs (domain validation dependent), Unicode IDs
(not exhaustively tested), identical payloads in different situations with
different transition IDs (pass).

---

## 3. Split and leakage audit

### Learned components — dependency graph

| Component | Fit on | Apply to | Leakage? |
| --- | --- | --- | --- |
| Phase 2 bag encoders | n/a (no-op `fit`) | all | No parametric leakage |
| Phase 4 transition model | train pairs | test pairs | No (game split enforced) |
| Phase 5 sequence model | train sequences | test sequences | No |
| Phase 6 counterfactual model | train pairs | test pairs | No |
| Phase 6 higher-order | corpus encode | corpus | No fit; encode-only |
| Phase 7 `FamilyStandardizer` | train **game** rows | all games + situations | No parametric leakage |
| Phase 7 `LeadingVarianceProjection` | train games | all | No parametric leakage |
| Phase 7 `novelty_table` | train games as kNN reference | all games scored | **Intentional** query semantics |
| Phase 7 `cluster_games` | train games `fit` | all games `predict` | **Intentional** |
| Phase 7 `nearest_*` | none | full corpus kNN | **HIGH** — not train-filtered |

### Confirmed: test game cannot alter train fit parameters

Probed via `build_design_space` with train `(g0,g1)` test `(g2,g3)` — normalizer
means/stds depend only on train game rows. Fix 2 (train-only novelty reference)
holds.

### HIGH — semantic leakage / misinterpretation

1. **`nearest_games` / `nearest_situations`** call `_neighborhood` → `knn` on the
   **full** entity table with no train-only filter. Test games appear as
   neighbors. This differs from `novelty_table`, which uses train-only reference.

2. **`between_game_dispersion`** averages over all games including test.

3. **Persisted `_novelty_metrics`** reports corpus-wide means (train + test).

4. **Docstring drift:** `run_counterfactual_experiment` and
   `run_sequence_experiment` still say “Game-held-out sanity evaluation” while
   Phase 7 docstring was updated to “train-fit, full-corpus query”. The split
   is game-safe for **model fit**; neighbor/dispersion APIs are not holdout
   evaluations.

**Adversarial case:** one test game with many situations vs one train game with
one situation — test presence does not alter fitted means (confirmed), but does
alter nearest-neighbor rankings and dispersion (by design or oversight — must be
documented explicitly).

---

## 4. H4 investigation — game-level normalizer on situation vectors

### Questions answered

1. **What population is the normalizer estimating?** Per-coordinate mean/std over
   **training game rows** in the flattened feature schema (21+ columns including
   `region__n_units`, family present flags, etc.).

2. **Is a game-level normalizer applied to situation-level vectors?** **Yes.** The
   same `FamilyStandardizer` fit on `game_table(...)` transforms both
   `games_table` and `sits_table`.

3. **Intentional?** **Yes.** Phase 7 §6 states: “applies the same transform to
   every game and situation vector (same schema and dim).”

4. **Within-game weighting artifacts?** **Yes, interpretively.** Situation
   `region__n_units=1`; game `region__n_units=n_situations`. After z-scoring
   with game-row statistics, situation region features live on a different
   absolute scale than their within-game aggregation would imply. This is a
   **cross-level comparison space**, not situation-native geometry.

5. **Would situation-level normalizer differ?** **Fundamentally yes** — different
   means/stds, different relative weights across families.

6. **Compatible with Phase 7 goal?** **Yes**, if the goal is a single shared
   coordinate system for cross-level kNN/outlierness. **No**, if the goal is
   situation geometry invariant to how many sibling situations exist.

7. **Bug, design choice, or documentation?** **INTENTIONAL design choice**
   requiring explicit API documentation on `normalized_situations` view semantics.

**Classification: INTENTIONAL** — not a v0 blocker; document on
`SituationRepresentation`, `view_table`, and `nearest_situations`.

---

## 5. Representation semantics

### INTENTIONAL lossy encodings (do not “fix”)

| Encoding | Loss | Documented? |
| --- | --- | --- |
| State bag | payload values → key presence only | Yes (`test_state_bag_discards_payload_values`) |
| Observation bag | item payload values | Partial |
| Action bag | action identity only | Phase 2 docs |

Loss is confined to Phase 2 embedders; later phases consume embeddings
deliberately.

### Unintended indistinguishability

| Case | Severity | Notes |
| --- | --- | --- |
| Identical topology → identical situation ID | CRITICAL | Distinct moments collapse |
| Label join bleed | CRITICAL | Different observation volumes → same labels |
| Hop block cross-situation inclusion | CRITICAL | Different observer asymmetry → blended features |
| Game aggregation averages situation families | INTENTIONAL | Documented in Phase 7 §3 |

No evidence that Phase 7 accidentally re-introduces raw payload values through
backdoors. Metadata exclusion holds (`assert_schema_excludes_metadata`).

---

## 6. Provenance and lineage audit

### Traceable within a run

For a Phase 7 `GameRepresentation`:

```text
game.source_artifacts
  → situation entity_ids + situation.source_artifacts
    → Phase 1 entity ids (states, obs, pairs, sequences)
    → Phase 6 logical keys (higher_order, counterfactual)
    → hop/intervention source_entity_ids
```

`SituationRepresentation.source_artifacts` includes encoded entity IDs and Phase
6 logical key strings. `DesignSpaceManifest` records schema, train/test game
IDs, situation IDs, encoder dim, metric, clustering/novelty params.

### Gaps

| Gap | Severity |
| --- | --- |
| Manifest lacks `SplitSpec`, `config_hash`, Phase 1–5 artifact payload links | HIGH |
| `source_payload_ids` on repr tables are entity ids, not upstream content digests | HIGH |
| No reconstructor loader from manifest alone | HIGH |
| `build_phase6_context` always lists both Phase 6 keys even when families absent | LOW |
| Missing action-set entity ids in `source_artifacts` | MEDIUM |

**Verdict:** Sufficient to understand **what** a result means; **not** sufficient
to reproduce exact vectors on a cold machine without original `PlaySituation`
inputs and code version pin.

---

## 7. Determinism and order-independence

### Confirmed deterministic

- Game aggregation: sorts situations by `situation_id`
- `build_design_space` / `represent_games`: invariant under situation-order
  permutation (probed)
- `cluster_games`: identical assignments under permutation with same seed (probed)
- `DeterministicKMeans` seed=0: entity-id-ordered centroid init
- Feature column order: fixed schema
- Canonical JSON / payload IDs: deterministic where used

### Order semantics

| Order | Treatment |
| --- | --- |
| Situations within game | Semantically unordered; sorted for aggregation |
| Transitions within situation | Semantically meaningful (trajectory chains) |
| Dict/set iteration | No evidence of bare dict iteration affecting persisted outputs |
| Floating-point aggregation | Mean-based; order-independent |

### Not tested exhaustively

- Repeated normalization across processes
- Parallel artifact persistence
- Unicode ID collation edge cases

---

## 8. Immutability audit

### Frozen correctly

- `ObservationExample.items`: `_FrozenDict` (mutation raises `TypeError`)
- `FamilyBlock.values`: immutable tuple
- Pydantic `frozen=True` on modeling boundary models
- Intervention results do not alias source payloads (tested)

### Holes

| Object | Issue | Severity |
| --- | --- | --- |
| `StateExample.data` | Plain `dict`; mutable despite frozen model | HIGH |
| `_FrozenDict` | Does not freeze lists nested inside mappings | MEDIUM |
| `PlayLayer.situations` | Holds mutable `PlaySituation` domain objects by reference | MEDIUM |
| `DesignSpace.layer` | Same | MEDIUM |

**Reproduction:** `examples_from_situations([s]).states[0].data["x"] = 1` succeeds.

Cross-boundary mutation can corrupt cached encodings if callers retain references.

---

## 9. Artifact and manifest reproducibility

`DesignSpaceManifest` (persisted in model pickle) includes:

- `feature_schema`, `metric`, train/test game IDs, situation IDs
- encoder dim/family, optional projection/cluster/novelty params
- Phase 6 logical keys, output payload id references

**Missing for exact reproduction:**

- Original `PlaySituation` JSON / Phase 1 example artifacts
- `SplitSpec` and `spec_config_hash`
- Phase 2–5 encoder artifact bytes and versions
- Python/package version pin
- Explicit random seed for all stochastic paths (cluster seed present; others n/a)

**Classification:** HIGH — document as “manifest records the recipe; inputs must
be retained separately for replay.”

---

## 10. API boundary audit

### Coherent

- Slim v0 `__all__` (~90 exports) in `modeling/__init__.py`
- Canonical artifact keys in `logical_keys.py`
- Domain → modeling → ds_platform one-way (tested)
- No named-game conditionals in modeling code

### Weak for v0 freeze

- Submodule imports expose duplicate paths to the same symbols
- Public helpers (`situation_id_for`, `perspective_table_from_observations`) accept
  semantically ambiguous local IDs without situation context
- `nearest_situations` vs `novelty_table` inconsistent train-scope semantics
- “evaluation” vs “eval” naming mixed across logical keys (consistent within
  `logical_keys.py` but historical aliases in submodule docstrings)

**Verdict:** Coherent for **single-situation-per-game** v0; **not** coherent for
claimed general multi-situation corpora without join-scope fixes.

---

## 11. Error-handling audit

| Input | Behavior | Classification |
| --- | --- | --- |
| Identical-topology two situations | `ValidationError: entity_ids must be unique` | Correctly rejected (but root cause is ID collision) |
| Missing state/obs/action in domain object | Domain / builder errors | Correctly rejected (not exhaustively matrix-tested) |
| Empty sequence | Skipped with message in bundle | Correctly handled |
| Duplicate observer/state in situation | Depends on domain validation | Not fully audited |
| Invalid split (empty train/test) | `ValueError` | Correctly rejected |
| Invalid cluster count | Runtime failure in k-means | Correctly rejected |
| `novelty_table(ks=(999,))` | Silently skipped (`k >= n_train`) | **Ambiguous** — no error |
| Mismatched artifact logical key | Store/load errors | Correctly rejected (existing tests) |
| Mismatched repr dimensions | `ValueError` on transform | Correctly rejected |

**Priority finding:** label/eval join bleed is **silently wrong**, not silent pass
— worse than missing validation.

---

## 12. Scale / complexity audit

| Operation | Complexity | Reasonable until | Replacement boundary |
| --- | --- | --- | --- |
| `pairwise_*_distances` | O(n²) | ~1k entities | Approximate NN index / sampling |
| `novelty_table` | O(n²) per k | ~1k games | Same |
| `knn` (full table) | O(n²) | ~1k entities | ANN when n > 10k |
| `build_play_layer` encode | O(n) | 100k+ states | Batch encoding / caching |
| Artifact persistence | O(n) serialize | 100k artifacts | Object store sharding |
| Phase 6 higher-order pairs | O(obs²) per situation | ~100 obs/situation | Sparse pair sampling |

Current fixture (~10 games, 1 situation each) is far below thresholds. O(n²)
geometry is acceptable for v0 prototype; document ~1k entity knee.

---

## 13. Documentation / semantic audit

| Document | Status |
| --- | --- |
| `docs/modeling-phase-7.md` | Largely accurate; §6 describes H4; §16 admits one-situation fixtures |
| `docs/modeling-audit.md` | Fix status accurate; “Original finding” blocks describe pre-fix state (can confuse) |
| `run_design_space_experiment` docstring | Updated — good |
| `run_counterfactual_experiment` / sequence eval | Still “Game-held-out sanity evaluation” — **misleading** for join-scope issues |
| README modeling section | Updated in fix 9; does not warn multi-situation limits |
| “Held-out” | Accurate for model **fit**; inaccurate for `nearest_*` and dispersion |

**No doc rewrites performed in this audit** (per instructions).

---

## 14. Domain-boundary audit

**Clean.**

- `domain/` does not import `ds_platform` (`test_domain_package_never_imports_ds_platform`)
- No BGA-specific names inside `ds_platform`
- No named-game conditionals in `modeling/`
- Dependency direction: `domain → modeling → ds_platform` only

---

## 15. Classified findings

### CRITICAL (must fix before v0)

#### C-F1. Topology-only `situation_id_for`

1. **Finding:** Situation ID ignores payloads; identical local topology collides.
2. **Why it matters:** Invalidates multi-moment-per-game modeling; duplicate entity IDs.
3. **Reproduction:** Two situations, same `game_id`, same state/transition ID sets, different payloads → same `situation_id`, `build_play_layer` fails.
4. **Phases:** 1–7.
5. **Proposed fix:** Content-addressed suffix or author `moment_id` in ID derivation.
6. **Before v0?** Yes.

#### C-F2. Observation label join without situation scope

1. **Finding:** `observation_label_table` keys `(game_id, player_id, state_id)`.
2. **Why it matters:** Wrong labels silently attached in multi-situation games.
3. **Reproduction:** §1 label probe — volumes 1.0 and 2.0 measured, both labeled 2.0.
4. **Phases:** 2–3 (+ downstream eval).
5. **Proposed fix:** Situation-scoped join keys.
6. **Before v0?** Yes.

#### C-F3. Transition eval joins without situation scope

1. **Finding:** `_scalar_at` / `_player_count` ignore `situation_id`.
2. **Why it matters:** Transition metrics blend across situations.
3. **Reproduction:** Multi-situation game with shared local state IDs and different decision counts.
4. **Phases:** 4.
5. **Proposed fix:** Join on `(situation_id, player_id, state_id)`.
6. **Before v0?** Yes.

#### C-F4. Sequence eval joins without situation scope

1. **Finding:** `_mean_at` averages across all situations in a game.
2. **Why it matters:** Sequence prediction eval features wrong.
3. **Reproduction:** Same as C-F3 for prefix target states.
4. **Phases:** 5.
5. **Proposed fix:** Scope by `prefix.situation_id`.
6. **Before v0?** Yes.

#### C-F5. Counterfactual `by_game` last-wins

1. **Finding:** `{situation.game.id: situation}` drops earlier situations.
2. **Why it matters:** Test eval queries wrong situation's action space.
3. **Reproduction:** Two situations, one game; inspect dict retention.
4. **Phases:** 6.
5. **Proposed fix:** Map `pair.situation_id → situation`.
6. **Before v0?** Yes.

#### C-F6. Higher-order identity and hop scoping

1. **Finding:** Game-scoped hop IDs; local state_id filter includes foreign situation pairs.
2. **Why it matters:** Phase 7 higher_order family features cross-contaminate.
3. **Reproduction:** Build Phase 6 context on two situations sharing `s0`; inject asymmetric hop pair; observe inclusion in both situation blocks.
4. **Phases:** 6–7.
5. **Proposed fix:** Situation-scope IDs and filters.
6. **Before v0?** Yes.

---

### HIGH

#### H-F1. Full-corpus kNN in `nearest_games` / `nearest_situations`

1. **Finding:** No train-only filter unlike `novelty_table`.
2. **Why it matters:** Consumers may treat neighbors as holdout-safe.
3. **Reproduction:** Call `nearest_games` on test game; train games are neighbors but so are other test games.
4. **Phases:** 7.
5. **Proposed fix:** Add `reference_game_ids` parameter or separate `nearest_train_games`.
6. **Before v0?** Document minimally; fix preferred.

#### H-F2. Manifest cannot replay without source situations

1. **Finding:** Missing split spec, upstream artifact digests, loader.
2. **Why it matters:** Provenance gap for persisted experiments.
3. **Phases:** 7.
4. **Proposed fix:** Extend `DesignSpaceManifest`; persist Phase 1 input refs.
5. **Before v0?** Document for v0; fix for v0.1.

#### H-F3. `StateExample.data` mutability

1. **Finding:** Frozen model, mutable dict field.
2. **Why it matters:** Caller can corrupt encoder inputs after creation.
3. **Reproduction:** Mutate `.data` in place.
4. **Phases:** 1–2.
5. **Proposed fix:** `_FrozenDict` for `data` or copy-on-read in `to_encoder_record`.
6. **Before v0?** Recommended.

#### H-F4. Perspective table uses local `state_id` as subject

1. **Finding:** `subject_ids.append(example.state_id)`.
2. **Why it matters:** Perspective distances ambiguous across situations.
3. **Phases:** 3.
4. **Proposed fix:** Use situation-scoped subject key.
5. **Before v0?** Yes if multi-situation Phase 3 eval is in v0 scope.

#### H-F5. Duplicate sequence entity IDs before validation

1. **Finding:** Identical topology → two `SequenceExample` with same `entity_id`.
2. **Why it matters:** Latent collision surfaced only at encode time.
3. **Phases:** 5.
4. **Proposed fix:** Dedupe in C-F1 fix.
5. **Before v0?** Yes (follows C-F1).

#### H-F6. Corpus-wide eval metrics in persisted novelty report

1. **Finding:** `_novelty_metrics` means include test games.
2. **Why it matters:** Misleading experiment summaries.
3. **Phases:** 7.
4. **Proposed fix:** Report train-only and full-corpus separately.
5. **Before v0?** Document.

#### H-F7. “Held-out” terminology in Phase 4–6 runners

1. **Finding:** Docstrings overclaim holdout purity for feature joins.
2. **Phases:** 4–6.
5. **Proposed fix:** Rename to “game-split model eval” and note join assumptions.
6. **Before v0?** Document.

---

### MEDIUM

| ID | Finding | Phases |
| --- | --- | --- |
| M-F1 | `action_set_entity_id` `\x1f` separator injection collision | 1 |
| M-F2 | `PlayLayer` holds mutable domain objects | 7 |
| M-F3 | `_FrozenDict` does not deep-freeze lists | 1 |
| M-F4 | `attach_external_metadata` last-wins per game_id | 7 |
| M-F5 | `prefixes_from_situations` loses situations on ID collision | 5 |
| M-F6 | `novelty_table` silently skips invalid k | 7 |
| M-F7 | Missing action-set ids in `source_artifacts` | 7 |
| M-F8 | Adversarial suite count (44) < first audit report (52) | tests |

---

### LOW

| ID | Finding |
| --- | --- |
| L-F1 | Submodule duplicate import paths vs `__all__` |
| L-F2 | `build_phase6_context` injects unused logical keys |
| L-F3 | `docs/modeling-audit.md` pre-fix narrative may confuse readers |
| L-F4 | O(n²) geometry not annotated in public API docstrings |
| L-F5 | Unicode / empty ID edge cases not covered by tests |
| L-F6 | README does not state v0 single-situation-per-game constraint |

---

### INTENTIONAL

#### I-F1. H4 — cross-level normalizer (game-fit on situation vectors)

1. **Investigated:** Same `FamilyStandardizer` transforms game and situation tables.
2. **Why suspicious:** Situation `region__n_units` differs from game row semantics.
3. **Why intentional:** Phase 7 §6 — single shared coordinate system for cross-level geometry.
4. **Document:** `view_table(level="situation", view="normalized")` semantics.

#### I-F2. Bag encoders discard payload values

1. **Investigated:** Phase 2 state/obs/action bags hash structure not content.
2. **Why suspicious:** Different payloads → identical vectors.
3. **Why intentional:** Structural encoding contract; tested and documented.
4. **Document:** Already in adversarial tests; cite in Phase 2 docs.

#### I-F3. Train-fit / full-corpus-query for novelty and clustering

1. **Investigated:** Test games scored/cl assigned using train-fitted scale/clusterer.
2. **Why suspicious:** Looks like leakage if described as “holdout evaluation”.
3. **Why intentional:** Exploratory geometry query pattern (`novelty_table` docstring).
4. **Document:** Distinguish from Phase 4–6 predictive holdout.

#### I-F4. Phase 7 composes Phase 6 tables (post-fix 6)

1. **Investigated:** Intervention/hop blocks use `Phase6Context`, not independent re-encode.
2. **Why suspicious:** Could have drifted from Phase 6 semantics.
3. **Why intentional:** Single source of truth for higher-order/intervention summaries.
4. **Document:** Already in `test_design_space_composes_phase6_tables`.

#### I-F5. Missing-family padding not z-scored when `present=0` (post-fix 3)

1. **Investigated:** Zero padding behind absent family flag.
2. **Why intentional:** “Not computed” ≠ zero after standardization.
3. **Document:** Phase 7 §4.

---

## Recommended path to v0 freeze

### Option A — Full v0 (recommended)

Fix C-F1 through C-F6, add permanent adversarial tests for identical-topology
and label-join bleed, extend manifest documentation, clarify H4 and kNN semantics.

### Option B — Constrained v0

Explicitly freeze as **“single situation per game only”** in README and v0
contract; defer C-F2–C-F6 if/when that constraint is guaranteed by ingestion.
**C-F1 still blocks** even single-game multi-moment corpora.

---

## Investigation artifacts

The following probes were run locally and **not committed**:

- Multi-situation label bleed (volumes 1.0 vs 2.0 → both labeled 2.0)
- Identical topology → `situation_id` collision → `ValidationError`
- `action_set_entity_id` `\x1f` collision
- `StateExample.data` mutation
- Game vector order independence
- Cluster assignment order independence
- Duplicate sequence entity IDs (2 sequences, 1 unique id)

Recommended permanent tests (not added in this audit pass):

- `test_identical_topology_situations_get_distinct_ids_or_fail_clearly`
- `test_observation_labels_do_not_bleed_across_situations`
- `test_counterfactual_eval_resolves_situation_not_game`

---

## Summary

The modeling pipeline is **architecturally sound in shape** and **materially
improved** since the first audit. Train-only fit paths for learned components
hold. Phase 6 composition, logical keys, hop counting, and situation-scoped
entity IDs are real fixes — but **secondary joins and situation ID derivation
were not fully migrated to situation scope**.

The green suite (**292 tests**) is **insufficient evidence** for multi-situation
readiness because fixtures and adversarial tests mostly vary transition IDs while
keeping one situation per fixture game. An adversary can still construct valid
inputs where **results are wrong while tests pass**.

**Do not freeze as unrestricted v0** until CRITICAL items C-F1–C-F6 are resolved
or the v0 contract explicitly narrows scope and C-F1 is still addressed for
within-game moment identity.

---

## Post-fix verdict (Option A)

**Ready to freeze as v0** for multi-situation-per-game corpora on the paths
exercised by the adversarial suites (49 audit tests total). Remaining HIGH
items (manifest replay, `StateExample.data` mutability, full-corpus `nearest_*`
semantics) are documented and tracked for v0.1 — they do not block the core
identity/join correctness fixes.
