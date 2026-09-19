# Corpus descriptive analysis v1

Project-owned specification for the first descriptive pass over the
BoardGameGeek metadata corpus. It is not the ontology measurement catalog
in [analysis.md](analysis.md), not a causal study, and not a statement
about all board games.

**Sample:** `bgg_boardgames_v1` is a 200-id convenience / coverage list.
It is not a probability sample, not representative of BGG, not a rank
snapshot, and not a best-of list. Every number below describes **this
corpus file**, not a population.

**Layer labels** used in code and findings:

| Label | Meaning |
| --- | --- |
| descriptive observation | What is on these records |
| exploratory association | Co-occurrence in this sample |
| unsupported causal interpretation | Do not write this |

## 1. Data that is actually available

Inspected: `Game` / `Mechanic` / `SourceReference`, `docs/data-dictionary.md`,
ingestion normalizer, quality helpers, `01_corpus_overview.ipynb`, the
packaged v1 id list, and the locally processed JSONL present at
specification time.

### 1.1 Fields

| Field | Analytically usable? | Notes |
| --- | --- | --- |
| `id`, `title` | identity only | required |
| `release_year` | yes, as corpus composition | `0` / omitted → null |
| `min_players`, `max_players` | yes, as a **range** | do not collapse to a single “player count” |
| `min_play_time_minutes`, `max_play_time_minutes` | yes, as a **range** | point when min = max; else keep span |
| `complexity` | yes, as BGG `averageweight` | null when `numweights` is 0; community poll, not measured difficulty |
| `mechanics[].name` | yes, multi-label source strings | a game has many; not exclusive classes |
| `mechanics[].id` | identity of the BGG link | |
| `mechanics[].category` | **no** | always null in BGG ingest |
| `mechanics[].description` | **no** | always null in BGG ingest |
| `categories` | optional context | BGG category strings, not interpretations |
| `rating`, `rating_count` | coverage / bias check only | famous-title truncation; not “quality” |
| `popularity` | **no** | never mapped |
| `designers` | not in this pass | |
| `publishers` | **no** | regional reprint lists (often 20–50+ names) |
| `sources` | provenance only | |
| Extracted interpretations | **no** | not written by ingest |
| Mechanic families / ontology | **no** | not implemented |
| Play traces / decision spaces | **no** | not in this corpus |

Missingness policy: `null` or `[]`. Do not impute.

The packaged v1 list has 200 ids. A local ingest produced **195**
board-game documents (2 skipped non-`boardgame` item types, 3 not
found). Coverage is computed on the JSONL that is actually loaded, not
assumed from the 25-game v0 smoke file. `popularity` and
`Mechanic.category` remain unused.

### 1.2 Analyses not performed

- Mechanic **families** (field is unused).
- `popularity`.
- Treating `rating` as fun, quality, or an outcome to explain.
- Causal claims (“heavier games take longer because…”).
- Population inference from the convenience sample.
- Significance tests manufactured for this pass.
- Midpoint-of-range as if it were a precise measurement (midpoints appear
  only as labeled proxies).
- Publisher “who publishes the most.”

## 2. Questions

Chosen because the populated fields can support them.

**Q1.** What player-count **profiles** appear in this corpus, and how wide
are the reported player ranges?
Layer: descriptive observation.

**Q2.** How does reported play time (min, max, span; point vs range) vary
across those player-count profiles?
Layer: exploratory association. Play time is a published range, not a
timed session.

**Q3.** How does BGG weight (`complexity`) vary with player-count profile
and reported play time?
Layer: exploratory association. Weight is a community poll.

**Q4.** Which BGG mechanic **labels** are most common, and do common
labels appear at different rates across player-count profiles?
Layer: exploratory association. Mechanics are multi-label; rare labels
are listed but not used for profile comparisons.

## 3. Definitions

### Player-count profile

Exclusive partition from the reported interval. Not “how many people
usually play.”

| Profile | Rule |
| --- | --- |
| `two_only` | min = 2 and max = 2 |
| `upto_four` | max ≤ 4, and not `two_only` |
| `five_plus` | max ≥ 5 |
| `unknown` | min or max is null |

Range width is `max_players - min_players` when both are present.

### Play time

- **Point:** min = max (BGG often copies `playingtime` to both).
- **Range:** min < max. Span = max − min.
- **Midpoint proxy:** `(min + max) / 2`, labeled as such, never as “the”
  play time.

### Mechanic prevalence

Share of games that list the label. A game with 10 mechanics contributes
to 10 counts. Profile comparison for a label uses only labels that appear
on at least five games.

### Association measure

Spearman rank correlation on paired complete cases, **without p-values**.
n must be at least 8. Reported as an exploratory rank association in this
sample.

## 4. Outputs

```text
data/derived/analysis/corpus_descriptive_v1/
  report.json
  findings.md
  run_manifest.json
  figures/*.png
```

`report.json` is stored as a `dataset` artifact (`bga:analysis/corpus-descriptive:v1`)
whose `inputs` cite the SHA-256 of the corpus JSONL bytes. No new artifact
kind.

Written interpretation of the first v1 run:
[analysis-corpus-descriptive-v1-findings.md](analysis-corpus-descriptive-v1-findings.md).

## 5. Reproduction

```bash
uv run board-game-ingest --corpus --corpus-id bgg_boardgames_v1   # if JSONL missing
uv run bga-analyze --corpus-id bgg_boardgames_v1
```

Notebook: `notebooks/02_corpus_descriptive_v1.ipynb`.

## 6. Robustness pass (second analysis)

Stress-tests the first descriptive results on the **same** corpus file.
Does not replace or overwrite the first-pass outputs.

```text
data/derived/analysis/corpus_descriptive_v1_robustness/
  report.json
  robustness_findings.md
  run_manifest.json
  figures/*.png
```

Artifact: `bga:analysis/corpus-descriptive-robustness:v1` (same `dataset`
kind; separate logical key and schema).

```bash
uv run bga-analyze --corpus-id bgg_boardgames_v1 --robustness
```

Written interpretation:
[analysis-corpus-descriptive-v1-robustness-findings.md](analysis-corpus-descriptive-v1-robustness-findings.md).

Tests use fixtures only. They never call BGG.
