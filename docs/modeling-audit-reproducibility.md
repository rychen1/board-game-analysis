# BGA Modeling — Reproducibility, Artifact Integrity, Evaluation, and Cold Replay Audit

Audit date: 2026-09-19. Scope: `board-game-analysis` modeling pipeline (Phases 1–7)
and its use of `ds-platform` persistence/evaluation primitives.

**Constraint:** No production code was modified during this audit. Probes were run
in-process and via subprocess; findings are classified below.

**Baseline context:** Option A multi-situation identity/join fixes are treated as
established. This audit attacks different failure classes.

**Environment note:** The reported post–Option A baseline was **297 tests passing**.
During this audit the combined editable `ds-platform` + BGA workspace showed
**293 passing / 4 failing** (design-space and related tests fail on
`RepresentationTable` construction). That regression is itself a finding — see
§4 and §12.

---

## 1. Executive summary

The modeling stack has a **sound content-addressed artifact core** (`LocalStore`,
payload/record identity, corruption detection) and **strong in-memory determinism**
for situation IDs, bag encodings, and seed-0 k-means. Option A multi-situation
semantics verified on label-join logic (when `FeatureTable` construction succeeds).

However, the stack is **not yet a stable v0 research foundation for persisted
Phase 7 experiments and cross-machine replay**. The highest-risk gaps are:

| ID | Severity | Summary |
| --- | --- | --- |
| R-C1 | **CRITICAL** | BGA Phase 7 `RepresentationTable`s use **game/situation entity ids as `source_payload_ids`**, but ds-platform now requires **SHA-256 hex**. Phase 7 canonical tables and persistence are broken in the current combined workspace. |
| R-H1 | **HIGH** | **Game holdout splits depend on first-seen game order** in the input list; reversing situation order can change which games are train vs test (same seed). |
| R-H2 | **HIGH** | **`DesignSpaceManifest` is insufficient for cold replay** — no split spec, config hash, Phase 1 input refs, eval artifact refs, or code/version pin. |
| R-H3 | **HIGH** | **Transition/sequence scalar evaluation uses a zero-change baseline**, not model predictions — persisted scalar metrics are misleading if read as model quality. |
| R-H4 | **HIGH** | **Pickle** for fitted models/manifests — no cross-Python byte identity; class-path dependent. |
| R-H5 | **HIGH** | BGA imports **`ds_platform.modeling._spec_json`** (private) from two modules. |

**Verdict:** Safe as an **in-memory research pipeline** for Phases 1–6 with hash
encoders; **not safe to freeze** persisted Phase 7 + experiment replay until R-C1
and manifest gaps (R-H2) are addressed. Statistical leakage controls for
normalizer/cluster/novelty **hold in code** when Phase 7 runs; split order (R-H1)
and eval semantics (R-H3) can still mislead conclusions.

**Adversarial cases attempted:** 47 (28 subprocess/in-process probes, 19 code-path
traces, 49 existing audit tests reviewed).

---

## 2. Scope and methodology

### In scope

- Reproducibility and determinism (IDs, splits, geometry, metrics, bytes)
- Artifact write/read integrity and relocation
- Evaluation correctness and semantic clarity
- Train/test statistical leakage (beyond situation joins)
- Cold replay / manifest sufficiency
- Serialization and schema evolution
- Mutation and aliasing at API boundaries
- Numerical robustness contracts
- Public/private API coupling to ds-platform
- Identity vs logical-key conflation
- Fail-closed behavior
- Cross-phase contract integrity
- Adversarial corpus coverage

### Out of scope (per instructions)

- Re-litigating Option A multi-situation identity/join fixes (unless regression found)
- Architecture proposals (workflow engines, metadata DBs, feature stores, etc.)
- Production code changes

### Methods

1. Static review of persistence, evaluation, split, and Phase 7 runners
2. In-process probes (store CAS, label bleed, mutation, split order, eval edge cases)
3. Subprocess probes with `PYTHONHASHSEED ∈ {0, 1, 424242}`
4. Full-suite test run and targeted adversarial test review
5. Cross-read of ds-platform `require_source_payload_ids`, `LocalStore`, `evaluate`

---

## 3. Reproducibility findings

### INTENTIONAL — Mathematically deterministic (verified)

| Behavior | Mechanism | Evidence |
| --- | --- | --- |
| Situation IDs | Canonical JSON content digest + topology key | Stable across `PYTHONHASHSEED` 0/1/424242 and subprocess |
| Bag encodings | Sorted keys in `record_source_id` → SHA-256 | `encoders/bag.py:33–42` |
| Measurement JSONL | Sorted by `measurement.id` before serialize | `artifacts.py:37–38`; order-independent bytes |
| K-means `seed=0` | Entity-id-sorted centroid picks | `design_geometry.py:867–872` |
| K-means tie-break | Sort by `(distance, centroid_index)` | `design_geometry.py:889–901` |
| Feature schema column order | Fixed family order in `default_feature_schema()` | Tests `test_feature_ordering_is_stable` |
| Game aggregation | Situations sorted by `situation_id` before mean | `design_space.py:414` |

### Deterministic under documented assumptions

| Behavior | Assumption | Risk if violated |
| --- | --- | --- |
| Holdout game split | **First-seen unique game order** in input list | R-H1 |
| Stratified split | Label processing order `sorted(by_label, key=repr)` | Label repr changes → different splits |
| Full-corpus bag encode before split | Encoders remain no-op / deterministic | Learned encoders would leak |

### R-H1 — Split depends on input game order (HIGH)

**File/function:** `ds_platform.modeling.split.split_groups` →
`board_game_analysis.modeling.split.split_entities_by_game`

**Mechanism:** `split_groups` deduplicates games preserving **first-seen order**, then
shuffles that unique list with `Random(seed)`. Reversing the situation list reverses
first-seen game order → different train/test **game sets** (same seed).

**Minimal reproduction:**

```python
spec = SplitSpec(method="holdout", seed=0, test_size=0.34)
games = ["a", "a", "b", "b", "c", "c"]
fwd = split_groups(games, spec)[0].train_ids  # ('b', 'c')
rev = split_groups(list(reversed(games)), spec)[0].train_ids  # ('b', 'a')
assert set(fwd) == set(rev)  # fails
```

**Why existing tests miss it:** `all_situations()` uses fixed import order; tests never
permute game list order before split.

**Persisted artifacts?** Yes — train/test partition in every experiment run.

**Scientific conclusions?** Yes — different holdout games → different metrics.

**Cross-machine?** Yes — if ingestion order differs.

**Fix:** Sort unique game ids before shuffle (e.g. lexicographic) or document that
callers must pass canonical game order. **v0 blocker** for any corpus whose game
ordering is not fixed.

---

### LOW — PYTHONHASHSEED

No production path found where hash seed affects persisted outputs. Dict insertion
order is deterministic in CPython 3.13; canonical JSON uses sorted keys.

---

## 4. Artifact integrity findings

### INTENTIONAL — CAS store behavior (verified)

| Check | Result |
| --- | --- |
| `put` → `get` → `get` byte equality | Pass |
| Copy tree to new root → `get` | Pass (URI is locator) |
| Tamper payload on disk | `HashMismatchError` on read |
| `ArtifactRecord` canonical roundtrip → `record_id` | Stable |

**Files:** `ds-platform/store.py`, `ds-platform/hashing.py`

---

### R-C1 — Phase 7 `source_payload_ids` contract violation (CRITICAL)

**File/function:** `design_space.py:_table` (lines 828–843); also
`higher_order.py:higher_order_table` (uses `pair.entity_id` as sources).

**Mechanism:** ds-platform `RepresentationTable` validates
`source_payload_ids` as lowercase SHA-256 hex (`features.py:require_source_payload_ids`).
BGA sets `source_payload_ids=tuple(entity_ids)` where entity ids are game ids
(`"chess"`) or situation ids (`"hanabi:…@digest"`).

Bag encoders correctly use `record_source_id()` → SHA-256. Phase 7 canonical
tables do not.

**Minimal reproduction:**

```python
build_design_space(all_situations(), train_game_ids=..., test_game_ids=..., dim=8)
# ValidationError: source_payload_ids must be lowercase sha256 hex: 'chess'
```

**Why existing tests miss it:** Tests passed at Option A baseline; ds-platform
editable dependency now enforces validation — **cross-repo contract drift** without
a coordinated BGA update.

**Persisted artifacts?** Blocks `representation_payload_bytes` → entire Phase 7
persistence path.

**Fix:** Use row-level content hashes (encoder source ids, or hash of vector +
entity id) for canonical game/situation tables. **v0 blocker.**

---

### MEDIUM — Dual JSON encoders

| Path | Serializer | Float encoding |
| --- | --- | --- |
| Platform representations/evals | `spec_canonical_json_bytes` | Tagged `$spec_float` |
| BGA measurements/examples | `json.dumps(sort_keys=True)` | Raw JSON floats |

Identity is stable if each path keeps its serializer, but **cross-subsystem byte
identity is not interchangeable**. Mixing serializers for the same logical object
would change `payload_id`.

**Fix:** Document encoder per artifact type; align BGA dataset JSON with spec JSON
only if byte-identical cross-artifact identity is required. **v0.1.**

---

### MEDIUM — Pickle byte identity (R-H4)

Pickle dumps of `FamilyStandardizer` / `DesignSpaceManifest` are stable within one
process; **not guaranteed across Python versions, protocols, or class paths**.

**Persisted artifacts?** All `put_model_artifact(..., pickle.dumps(...))` paths.

**Fix:** Document pickle as runtime checkpoint, not long-term exact-replay format;
prefer spec JSON for manifest recipe fields already structured as Pydantic models.
**v0.1** for manifest; pickle may remain for fitted state with version pin.

---

### LOW — Logical key ambiguity

Multiple artifacts may share a `logical_key` (e.g. `bga:corpus:v0`). Keys are labels,
not identity — correct by design. Callers must use `payload_id` / `record_id`.

---

## 5. Evaluation findings

### INTENTIONAL — ds-platform `evaluate()` contract

- Drops pairs with `None` in y_true/y_pred; counts `n_missing`
- Rejects non-finite metric values (`ValidationError`)
- Perfect prediction → RMSE 0.0 (verified)
- Single-example RMSE correct (verified)

**File:** `ds-platform/modeling/evaluate.py`

---

### R-H3 — Scalar metrics score zero-change baseline, not the model (HIGH)

**File/function:** `transitions.py:run_transition_experiment` (231–239),
`sequence_eval.py:295–304`

**Mechanism:** Representation MSE evaluates the conditioned encoder on test pairs.
Scalar block uses `scalar_pred=_zero_change_table(scalar_true)` — predictions are
**always zero delta**, not model outputs.

**Why it matters:** Persisted `TransitionEvaluation.scalars` / sequence scalar
reports look like model evaluation but measure a trivial baseline vs observed deltas.

**Why tests miss it:** Tests assert experiment completes and split is game-safe;
they do not assert scalar preds derive from the model.

**Scientific conclusions?** Yes — readers can misinterpret scalar RMSE.

**Fix:** Either remove scalar block from persisted eval, rename metrics
(`baseline_zero_delta_rmse`), or wire actual model-derived scalar predictions.
**v0.1** (documentation minimum); code fix if scalars are promoted as metrics.

---

### MEDIUM — Novelty eval aggregates train + test

`_novelty_metrics` reports `mean_structural_novelty_k*` over **all scored games**
including train. Neighbor **reference** is train-only (correct); reported means are
corpus-wide (documented in Phase 7 docs post–Option A).

**Classification:** INTENTIONAL if documented; misleading if read as holdout metric.

---

### LOW — Invalid novelty k silently skipped

`novelty_table` skips `k >= len(reference)` without error. Ambiguous for callers
expecting an error.

---

### NOT A BUG — Higher-order / perspective eval

Descriptive counts only (`n_pairs`, `n_asymmetric`); no predictive metrics. No split.

---

## 6. Leakage findings

### INTENTIONAL — Train-only fit paths (code review + prior adversarial tests)

When `build_design_space` succeeds:

| Component | Fit data | Query/transform data |
| --- | --- | --- |
| `FamilyStandardizer` | Train game rows | All games + situations |
| `LeadingVarianceProjection` | Train game rows | All |
| `DeterministicKMeans.fit` | Train game rows | All (predict) |
| `novelty_table` kNN reference | Train games | All games scored |
| `LinearConditionedEncoder` | Train pairs | Test pairs |
| Bag encoders | No-op fit | All entities (deterministic) |

Extreme test game values do **not** change train normalizer means/stds when train
games unchanged (verified by prior `test_normalizer_ignores_extreme_test_values` logic;
test currently fails on R-C1 table construction).

---

### INTENTIONAL — Full-corpus encode before split

Phases 4–6 encode all entities before game split. Safe while encoders are
hash-based no-op fit. **Documented fragility** if encoders become learned.

---

### INTENTIONAL — `nearest_games` / `nearest_situations` use full corpus

Not train-filtered; differs from `novelty_table`. Documented in Phase 7 §11.

---

## 7. Cold replay findings

### Three-level assessment

| Level | Phase 7 today | Gap |
| --- | --- | --- |
| **Auditability** | Partial | Manifest + artifact records + logical keys describe recipe; eval reports slim |
| **Reproducibility** | Weak | Cannot rerun without original `PlaySituation` inputs |
| **Exact byte replay** | No | Pickle + float JSON paths + missing version pins |

### R-H2 — Manifest gaps (HIGH)

**File:** `design_geometry.py:DesignSpaceManifest` (201–222)

**Present:** schema, metric, train/test game ids, situation ids, encoder dim/family,
projection/cluster/novelty params, Phase 6 logical keys, payload id **citations**
for situation/game/normalizer/projection tables.

**Missing for cold replay:**

| Field | Why needed |
| --- | --- |
| `SplitSpec` (method, seed, test_size) | Reproduce train/test partition (R-H1 makes order explicit too) |
| `config_hash` / `GeometrySpec` bytes | Tie run to exact experiment config |
| Phase 1 example / measurement payload ids | Reconstruct inputs without author memory |
| `novelty_payload_id`, `cluster_payload_id` | Follow full eval lineage from manifest alone |
| Python + package versions | Pickle and float behavior |
| Canonical **game order** used at split time | R-H1 |

**Minimum additional metadata (no workflow engine):**

1. Embed `SplitSpec` + sorted `game_ids` order in manifest
2. Add `inputs: tuple[payload_id, ...]` for Phase 1 artifacts used
3. Store `config_hash` on manifest
4. Add `software_versions: dict[str, str]` (python, bga, ds-platform)
5. Document that exact replay still requires those input payload bytes in the store

**Fix priority:** v0 blocker for **persisted experiment claims**; documentation-only
insufficient.

---

## 8. Serialization and schema findings

### Identity-bearing vs non-identity-bearing

| Schema | Identity-bearing? | Notes |
| --- | --- | --- |
| Payload bytes | Yes | SHA-256 |
| `ArtifactRecord` | Yes (record_id) | Float-free envelope |
| `SplitSpec` / `GeometrySpec` | Config identity | Hashed into `config_hash` on RunContext |
| `DesignSpaceManifest` | Recipe, not output identity | Pickled blob gets its own payload_id |
| `EvaluationReport` | No (snapshot) | Metrics frozen via `MappingProxyType` |
| Logical keys | No | Human labels |

### MEDIUM — Record rehash vs stored bytes

ds-platform documents that `record_id(record)` hashes re-serialized model, not
stored bytes — schema default changes can shift ids. `schema_version: 0` exists;
no migration path.

### LOW — BGA ingestion JSON without `sort_keys`

`ingestion/pipeline.py` uses `json.dumps` without sorted keys for game documents —
field order follows Pydantic schema order. Stable if schema stable.

---

## 9. Mutation findings

| Object | Mutable? | Severity |
| --- | --- | --- |
| `ObservationExample.items` | Blocked (`_FrozenDict`) | OK |
| `StateExample.data` | **Yes** — plain dict | **HIGH** (R-M1) |
| `EvaluationReport.metrics` | Frozen proxy | OK |
| `TransitionEvaluation.scalars` | Outer dict frozen; values immutable | OK |
| `Intervention.parameters` | Plain dict | MEDIUM |
| `PlayLayer.situations` | Domain objects by reference | MEDIUM |
| Persisted store bytes after write | Immutable (write-once) | OK |

### R-M1 — `StateExample.data` mutable (HIGH)

**Reproduction:** `examples_from_situations([s]).states[0].data["x"] = 1` succeeds.

**Impact:** Caller alias mutation can corrupt encoder inputs after example creation.

**Fix:** Apply `_FrozenDict` to `StateExample.data` or copy-on-read in
`to_encoder_record`. **v0.1** (not blocking in-memory pipeline if callers disciplined).

---

## 10. Numerical robustness findings

| Case | Behavior | Classification |
| --- | --- | --- |
| `RepresentationTable` with NaN/Inf | Rejected at validation | Fail-closed ✓ |
| `evaluate()` with Inf label | `ValidationError` | Fail-closed ✓ |
| Zero-variance normalizer column | `std=1.0`, listed in `zero_variance_columns` | INTENTIONAL |
| Missing family `present=0` padding | Not z-scored | INTENTIONAL (Option A) |
| All-zero vectors | Allowed if finite | OK |
| Identical k-means points | Collapse to one cluster | INTENTIONAL |
| `k >= n` in novelty | Silently skipped | MEDIUM ambiguous |

---

## 11. API boundary findings

### R-H5 — Private ds-platform imports (HIGH)

```
board_game_analysis/modeling/perspectives.py:10
board_game_analysis/modeling/interventions.py:9
  from ds_platform.modeling._spec_json import spec_canonical_json_bytes
```

**Impact:** ds-platform refactor of `_spec_json` breaks BGA without semver notice.

**Fix:** Promote `spec_canonical_json_bytes` to public `ds_platform.modeling` API.
**v0.1.**

### LOW — Dual import paths

Stable v0 surface in `modeling/__init__.`__`; submodules remain importable —
documented pattern.

---

## 12. Cross-phase findings

| Boundary | Contract | Gap |
| --- | --- | --- |
| Phase 1 → 2 | Situation-scoped entity ids; bag records | OK |
| Phase 2 → 3 | `z_obs` entity alignment | OK |
| Phase 4–5 → eval | Pair/prefix `situation_id` on scalar joins | Fixed Option A |
| Phase 6 → 7 | `Phase6Context` tables composed | OK |
| Phase 7 → store | `RepresentationTable` must satisfy platform invariants | **R-C1 broken** |
| Wrong-phase artifact | No automatic schema guard beyond Pydantic types | MEDIUM — caller responsibility |

Feeding a representation table with wrong `dim` → `ValueError` on transform (fail-closed).

---

## 13. Adversarial corpus assessment

### Covered by permanent tests

| Invariant | Test location |
| --- | --- |
| Identical topology → distinct situation ids | `test_audit_final_adversarial.py` |
| Multi-situation play layer | audit final + adversarial |
| Label join situation scope | audit final (logic verified; FeatureTable ids need SHA-256 in tests) |
| Train-only normalizer | `test_audit_adversarial.py` (blocked by R-C1 in current workspace) |
| Train-only novelty reference | adversarial |
| Hop situation scope | audit final |
| Domain ↛ ds-platform | adversarial |

### Gaps — fixtures do not protect

| Missing invariant | Recommended fixture/probe |
| --- | --- |
| Split invariant to game **order** | Permute `all_situations()` order; assert stable train games |
| `source_payload_ids` SHA-256 on Phase 7 tables | Unit test on `game_table` / `situation_table` |
| Cold replay from manifest alone | Integration test with mocked store roundtrip |
| Scalar eval semantics | Assert `scalar_pred` is zero baseline or rename |
| Multi-situation in **fixture corpus** | `all_situations()` still one situation per game |
| Degenerate numerics in design space | NaN injection at representation layer |
| Cross-process pickle identity | Subprocess compare manifest bytes |

**Corpus philosophy:** Synthetic `_situation()` helper is well-designed; main fixture
corpus remains thin for multi-situation **integration** despite Option A semantics.

---

## 14. Severity table

| ID | Severity | Area | v0 blocker? |
| --- | --- | --- | --- |
| R-C1 | CRITICAL | Cross-phase `source_payload_ids` | **Yes** |
| R-H1 | HIGH | Split order dependence | **Yes** (if game order not fixed) |
| R-H2 | HIGH | Cold replay manifest | **Yes** (for persisted claims) |
| R-H3 | HIGH | Scalar eval semantics | No (document); yes if scalars are trusted |
| R-H4 | HIGH | Pickle portability | No (document + version pin) |
| R-H5 | HIGH | Private API import | No |
| R-M1 | HIGH | StateExample mutability | No |
| M1 | MEDIUM | Dual JSON encoders | No |
| M2 | MEDIUM | Record rehash drift | No |
| M3 | MEDIUM | Invalid k silent skip | No |
| M4 | MEDIUM | Cross-phase artifact guards | No |
| L1–L3 | LOW | Logical keys, import paths, novelty naming | No |

**INTENTIONAL (not bugs):** CAS integrity, train-only normalizer/projection/cluster
fit, train-only novelty reference, full-corpus nearest neighbors, hash encoder
pre-split encode, H4 cross-level normalization, zero-variance std=1, missing-family
padding semantics, k-means seed-0 determinism.

---

## 15. Recommended fixes (minimal)

1. **R-C1:** In `design_space._table`, set `source_payload_ids` to per-row SHA-256
   of canonical vector bytes or reuse encoder `record_source_id` lineage.
2. **R-H1:** Sort unique game ids before holdout shuffle in `split_groups`, or require
   sorted game input in BGA runners and test permutations.
3. **R-H2:** Extend `DesignSpaceManifest` with `split_spec`, `config_hash`,
   `input_payload_ids`, eval artifact ids, `software_versions`; document replay procedure.
4. **R-H3:** Rename or rewire scalar eval; never persist as implicit model quality.
5. **R-H4/R-H5:** Document pickle limits; promote `_spec_json` to public API.
6. **R-M1:** Freeze `StateExample.data`.
7. **Tests:** Add audit tests for R-C1 and R-H1; use valid SHA-256 placeholders in
   `FeatureTable`/`RepresentationTable` test helpers.

No metadata database or workflow engine required.

---

## 16. Explicitly intentional behavior

- Content-addressed storage: identity = bytes, URI = locator
- Logical keys are not unique and not identity
- Novelty: train-fit reference, full-corpus query scores
- Cluster: train-fit centroids, all-game label assignment
- Nearest neighbors: full corpus (not holdout)
- Bag encoders: structural hash, payload values discarded (Phase 2 contract)
- Game-level normalizer applied to situation vectors (H4)
- Measurements JSONL sorted by id for deterministic bytes

---

## 17. What is safe to freeze as v0

**Safe:**

- Phase 1 example/measurement model (in-memory)
- Situation-scoped identity semantics (Option A)
- Hash bag encoders (Phases 2–3)
- Game-safe split **invariant** (no game on both sides) when game order is fixed
- CAS store protocol usage
- Train-only statistical fits (normalizer, projection, k-means fit, novelty reference)
- Domain → modeling → ds-platform dependency direction

**Not safe to freeze yet:**

- Phase 7 persistence (`run_design_space_experiment`) until R-C1 resolved
- Cross-machine exact artifact replay (pickle + manifest gaps)
- Holdout split reproducibility across ingestion order (R-H1)
- Persisted scalar metrics as scientific evidence (R-H3)
- Claim that “297 tests green” implies Phase 7 store path works against current ds-platform

---

## 18. What should remain unfrozen

- Pickle model blob format (until versioned schema JSON alternative exists)
- Private `_spec_json` coupling
- `DesignSpaceManifest` field set (expect additive fields for replay)
- Full-corpus nearest-neighbor API semantics (may gain train-filtered variant)
- Scalar evaluation blocks in Phases 4–5

---

## Ready for stable v0 research foundation?

**Partially.**

The BGA modeling stack is a **credible in-memory research foundation** for
exploring play-structure representations with fixed fixtures and adversarial
multi-situation tests. It is **not yet** a stable foundation for **accumulating
trusted persisted conclusions** across machines and time until:

1. Phase 7 tables satisfy ds-platform representation contracts (**R-C1**)
2. Holdout splits are stable under canonical game ordering (**R-H1**)
3. Manifest records enough to reproduce **what happened** and rerun with stored inputs (**R-H2**)

Addressing R-C1 + R-H1 + minimal manifest extensions (R-H2) is the smallest path
to a defensible v0 freeze for persisted Phase 7 work — without new infrastructure.

---

## Audit probes log (investigation only)

| # | Probe | Result |
| --- | --- | --- |
| 1 | `PYTHONHASHSEED` × situation_id | Stable |
| 2 | Subprocess situation_id tail | Stable |
| 3 | Label join multi-situation | 1.0 vs 2.0 correct |
| 4 | Store put/get/relocate/tamper | Pass / HashMismatchError |
| 5 | Record canonical roundtrip id | Stable |
| 6 | Pickle same-process bytes | Stable |
| 7 | Split forward vs reverse game order | **Different train games** |
| 8 | evaluate perfect / single / Inf | Correct / rejected |
| 9 | measurements_payload order independence | Stable (sorted ids) |
| 10 | build_design_space full corpus | **ValidationError** (R-C1) |
| 11 | Manifest field presence | Missing split/config/inputs |
| 12 | StateExample.data mutation | Succeeds (R-M1) |

**Test suite at audit time:** 293 passed, 4 failed (R-C1-related and ingestion adjacency).

---

## 19. Post-fix status (2026-09-19)

Minimum v0 fixes applied. Historical findings above are unchanged.

### Fixed

| ID | Fix |
| --- | --- |
| **R-C1** | Phase 7 tables set `source_payload_ids` from encoder payload lineage via `source_payload_id` on each row and `composite_source_payload_id` for multi-source rows. Higher-order hop tables use observation encoder payload ids. |
| **R-H1** | `split_groups` sorts unique group ids lexicographically before holdout/k-fold selection. BGA game splits inherit order invariance. |
| **R-H2** | `DesignSpaceManifest` extended with `split_spec`, `config_hash`, `input_examples_payload_id`, eval payload ids, and `software_versions`. Phase 7 experiment persists Phase 1 examples. |
| **R-H3** | Transition/sequence scalar metrics use train-only ridge probes from model representation vectors (`scalar_eval.py`), not `_zero_change_table`. |

### Remain deliberately unfrozen

- **R-H4** — pickle portability across Python versions
- **R-H5** — private `_spec_json` imports
- Full-corpus `nearest_*`, train-fit/full-corpus-query novelty semantics
- Hash encoders fit before split, H4 cross-level normalization
- Zero-variance `std=1`, missing-family padding, pickle byte stability

### v0 guarantees after fix

**Auditability:** A persisted Phase 7 run's manifest plus store payloads
answer what data, split, config, upstream artifacts, evaluations, and
software versions produced the result.

**Reproducibility:** Given the same situation inputs, split spec/seed,
config, and compatible package versions, in-memory recomputation matches
within floating-point tolerance. Game holdout assignment is stable under
input order permutations.

**Not guaranteed at v0:** Universal byte-identical artifact replay across
Python/platform versions (pickle). Scalar probe quality with very small
train partitions. Full-corpus nearest-neighbor train filtering.

### Test status after fix

305 tests passing (includes 8 regression tests in
`tests/modeling/test_audit_reproducibility_fixes.py` and 2 new
`split_groups` order-invariance tests in ds-platform). Ruff and Pyright
clean on changed modeling code.

### Ready for stable v0 research foundation?

**Yes, for persisted Phase 7 work within the documented limits.** The
modeling stack is safe to freeze as a v0 research foundation when
conclusions are scoped to auditability and in-version reproducibility,
not cross-version byte-identical replay or the intentionally unfrozen
items above.
