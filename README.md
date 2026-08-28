# Board Game Analysis

Computational research system for analyzing board games from metadata and
rulebooks. This repository is a research and data-engineering project first.
The HTTP API exists as a future integration surface, not as the primary
product.

## Long-term vision

The intended system will:

1. Ingest thousands of games from public APIs, databases, and rulebooks.
2. Store observed metadata (title, year, player count, play time, ratings,
   designers, publishers, categories, mechanics).
3. Extract structured rules and mechanics from rulebooks.
4. Represent play formally: information spaces, decision spaces, player
   knowledge, actions, states, transitions, and turn structure.
5. Measure properties such as decision richness, information asymmetry,
   uncertainty, agency, interaction, dominance, balance, and tension.
6. Simulate games with different agent types.
7. Search, mutate, combine, and generate designs from that representation.
8. Hand successful generated designs to a separate prototyping workflow.

None of those pipelines are implemented yet. This repository currently holds
foundational schemas, package boundaries, and tooling.

## Three kinds of information

These layers must not be mixed:

1. **Observed facts** — values obtained from a source: release year, player
   count, a rulebook sentence, a published rating. Modeled as `Game`,
   `Mechanic`, and `SourceReference`.
2. **Extracted interpretations** — classifications derived from rules or
   other evidence: cooperative, hidden information, constrained
   communication. Modeled as `ExtractedInterpretation`, keyed by `game_id`.
3. **Derived measurements** — quantitative outputs of analysis: decision-
   space size, uncertainty, agency. Modeled as `DerivedMeasurement`, keyed
   by `game_id`.

`Game` does not carry interpretation flags or analysis scores. Provenance is
attached via `SourceReference` so a later extraction layer can trace a claim
back to an API record or rulebook page.

## Initial architecture

```
src/board_game_analysis/
  domain/         # Pydantic models (no FastAPI imports)
  ingestion/      # placeholder for acquisition
  extraction/     # placeholder for rule/mechanic extraction
  analysis/       # placeholder for measurements
  simulation/     # placeholder for agent play
  storage/        # placeholder for PostgreSQL / DuckDB
  api/            # FastAPI app; health check only
```

Play is represented as an id-linked graph rather than nested objects:

`Game` → `Turn` → `Player` → `InformationSpace` / `DecisionSpace` → `Action` → `StateTransition`

Identifiers are strings so BoardGameGeek ids, slugs, and generated ids all
fit. Visibility, action type, phase, and mechanic category are open strings
with `dict` payloads for game-specific structure. There is no mechanic
taxonomy and no component ontology (boards, cards, tokens) yet.

## Ontology v0

The current domain can represent a moment of play as:

`GameDefinition` (rules) → `GameState` (underlying snapshot) →
`Observation` / `InformationSpace` (per player) → `DecisionSpace` →
`Action` → `StateTransition` → next `GameState`

Design notes, vocabulary, and known limits: [docs/ontology.md](docs/ontology.md).

Manually authored situations live in `tests/fixtures/games/`. They are
test cases for the representation, not an ingestion corpus.

## Analysis specification v0

Proposed measurements, evidence levels, and research questions:
[docs/analysis.md](docs/analysis.md).

Typed metric names live in `board_game_analysis.analysis` as
`MeasurementSpec` entries. There is no analysis engine and no computed
corpus scores.

## Ingestion v0

BoardGameGeek XML API2 metadata only. Raw XML is archived, then normalized
to `Game`. Live calls need `BGG_TOKEN`. Details:
[docs/ingestion.md](docs/ingestion.md).

```bash
uv run board-game-ingest 13
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

Optional local PostgreSQL (unused until a storage layer exists):

```bash
cp .env.example .env
docker compose up -d
```

## Tests and checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
uv run pyright
```

## API

```bash
uv run uvicorn board_game_analysis.api.main:app --reload
```

`GET /health` returns `{"status": "ok"}`. There are no other endpoints.

## Data layout

- `data/raw/` — unmodified acquisitions
- `data/processed/` — cleaned or normalized tables
- `data/derived/` — analysis outputs
- `schemas/` — reserved for later JSON Schema / interchange exports
- `notebooks/` — exploratory analysis
- `scripts/` — one-off operational scripts

## Intended later architecture

Planned stack, not installed in this skeleton unless already required:

- PostgreSQL as the canonical store
- DuckDB and Polars for analytical processing
- PyMuPDF for rulebook PDFs
- httpx for API ingestion
- Playwright where browser acquisition is necessary

Future work should keep ingestion, extraction, analysis, simulation, and
storage as separate packages, and should keep domain models independent of
FastAPI. Do not put derived scores on `Game` when those pipelines are added.
