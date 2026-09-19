# Ingestion v0

Small, reproducible BoardGameGeek metadata pipeline. It is not a production
crawler, not HTML scraping, and not rulebook extraction.

The pipeline is built to ingest on the order of **~1,000** games. The frozen
id list checked into the repo is **50** well-known titles so a smoke run is
`--corpus --limit 25` or `--limit 50`. Do not scrape HTML if the XML API
rejects a request.

Live calls to BGG XML API2 require an application token (`BGG_TOKEN`).
Without a token the API returns `401 Unauthorized`. Tests never call the
network; they use checked-in XML fixtures.

## Source

- **Name:** BoardGameGeek
- **Interface:** XML API2 `GET /thing?id={id}&stats=1` (up to 20 comma-separated ids)
- **Docs:** https://boardgamegeek.com/wiki/page/BGG_XML_API2
- **Auth:** `Authorization: Bearer <token>`
  (register at https://boardgamegeek.com/applications)
- **Terms:** https://boardgamegeek.com/xmlapi/termsofuse

Do not scrape `boardgamegeek.com` HTML to bypass authentication.

## Data dictionary

Canonical fields, types, nullability, BGG mappings, and missing-data
policy: [docs/data-dictionary.md](data-dictionary.md).

Typed catalog: `board_game_analysis.ingestion.dictionary`.
JSON Schema: `schemas/game.schema.json`.

## Architecture

```text
frozen id list (comments allowed)
    → missing ids, in batches of ≤20
BggClient.fetch_games(ids)
    → per-id RawArtifact (split from the batch XML)
    → data/raw/boardgamegeek/{id}.xml
    → data/raw/boardgamegeek/{id}.meta.json
    → normalize (boardgame items only)
    → Game
    → data/processed/boardgamegeek/{id}.json
    → data/processed/boardgamegeek/corpus_v0.jsonl
    → data/derived/corpus/bgg_boardgames_v0.manifest.json
```

| Piece | Responsibility |
| --- | --- |
| Frozen corpus list | `ingestion/corpus/bgg_boardgames_v0.txt` (50 ids; replaceable with ~1000) |
| `BggClient` | HTTP only: timeout, retries, rate limit, User-Agent, Bearer token, batch `/thing` |
| `parse_thing_xml` / `parse_things_xml` | XML → source-specific `BggThing` |
| `normalize_bgg_artifact` | `BggThing` + provenance → canonical `Game` |
| `RawArchive` | Immutable-ish raw cache (resume) |
| `ingest_corpus` | limit/offset, skip non-boardgames, continue on per-id errors, manifest |

`board_game_analysis.domain` does not import this package and does not
know BGG field names.

The pipeline writes `Game` plus `SourceReference`. It does **not** write
`ExtractedInterpretation` or `DerivedMeasurement`. A BGG category such as
"Negotiation" stays a source-reported `Game.categories` string.

## Raw vs normalized

Raw XML is the research archive. Normalization never mutates it. `--force`
refetches and may replace the raw file; the default is to reuse the cache
and re-run normalization only. Existing raw+meta files skip HTTP, which is
how a 50- or 1000-id run can resume after a partial failure.

Processed JSON is `Game.model_dump(mode="json")`. The corpus JSONL is the
same payload, one game per line, in ingest order. PostgreSQL is not used.

Downloaded datasets must not be committed (see `.gitignore`).

## Provenance

Each `Game.sources` entry records:

- `source` = `boardgamegeek`
- `source_type` = `xmlapi2`
- `source_identifier` = BGG thing id
- `url` = the request URL (a batched URL when the fetch used multiple ids)
- `retrieved_at` = fetch time (provenance only; same XML + same timestamp
  → identical `Game`)

Per-field provenance is not implemented. Corpus run provenance (counts,
skips, errors, timestamps) lives on the manifest, not on `Game`.

## Caching, batching, and rate limiting

If `data/raw/boardgamegeek/{id}.xml` and `.meta.json` exist, ingest skips
HTTP for that id.

Live HTTP:

- explicit timeout (`BGG_TIMEOUT_SECONDS`, default 30)
- delay between **requests** (`BGG_RATE_LIMIT_SECONDS`, default 5)
- up to `BGG_BATCH_SIZE` ids per request (default 20, BGG maximum)
- retries on `202`, `429`, `503` with linear backoff
- one request at a time (no parallel scraping)
- `User-Agent` identifies this application

A full 1,000-id run is about 50 HTTP requests (1000 / 20) plus rate-limit
delay, not 1,000 serial calls.

## Missing-data policy

Do not invent values. BGG uses `0` as a sentinel for some fields. See the
[data dictionary](data-dictionary.md) for the full table.

| Field | Missing | Known zero |
| --- | --- | --- |
| year, player counts, play times | omitted node **or** `value="0"` → `null` | not used (0 is treated as unknown) |
| `rating_count` (`usersrated`) | omitted → `null` | `0` stays `0` |
| `rating` (`average`) | no ratings (`usersrated` 0 or omitted) → `null` | not mapped from BGG's placeholder `0.00000` |
| `complexity` (`averageweight`) | `numweights` 0 or omitted → `null` | same |
| `popularity` | not mapped from BGG (rank is inverted; ownership is a different concept) | — |
| lists (designers, …) | omitted links → `[]` | — |

Canonical ids are `bgg-{thing id}`. Mechanic ids are `bgg-{link id}` with
the source-reported name. That is a label, not a mechanic taxonomy.

If `minplaytime`/`maxplaytime` are both missing/zero but `playingtime` is
set, both min and max are filled from `playingtime`.

## How to ingest

One or more explicit ids (fail-fast):

```bash
cp .env.example .env   # set BGG_TOKEN
uv run board-game-ingest 13
# or: uv run python scripts/ingest.py 13
```

Frozen corpus (continue on per-id errors; resume from cache):

```bash
uv run board-game-ingest --corpus --limit 25   # smoke
uv run board-game-ingest --corpus --limit 50   # full packaged list
uv run board-game-ingest --corpus --ids-file path/to/ids.txt --limit 1000
```

`--force` refetches even when raw files exist.

Writes:

- `data/raw/boardgamegeek/{id}.xml` and `{id}.meta.json`
- `data/processed/boardgamegeek/{id}.json` for ingested board games
- `data/processed/boardgamegeek/corpus_v0.jsonl` (corpus runs)
- `data/derived/corpus/bgg_boardgames_v0.manifest.json` (corpus runs)

The packaged list is 50 ids. A larger frozen file (up to ~1000) can be
passed with `--ids-file` without changing the pipeline.

## Corpus behaviour

- Comments (`#`) and blank lines in id files are ignored.
- Duplicate ids keep the first occurrence. Order is part of the definition.
- `--offset` / `--limit` slice the list after uniquing.
- Item types other than `boardgame` (expansions, accessories) are skipped.
  Their raw XML may still be archived so resume does not refetch them.
  Base games often include many `boardgameexpansion` / `boardgameaccessory`
  *links*; those are related products, not the item type, and do not cause
  a skip.
- HTTP failures and missing ids are recorded on the manifest; the run
  continues. Authentication errors abort the run.
- Manifest `cached` is true when raw+meta already existed before this run.

## Known limitations

- A live 25-game smoke run succeeded. That sample is famous, well-documented
  titles; it does not exercise sparse/missing BGG fields.
- BGG `ranks` and `owned` are present on those records and are intentionally
  not mapped to `popularity`.
- Publisher lists are every regional publisher BGG knows, not a single
  imprint. Famous games commonly have 20–50+ names.
- No collection, search, hot-list, or user endpoints.
- Alternate names, descriptions, polls, ranks, and images are ignored.
- BGG "Geek Rating" (`bayesaverage`) is not mapped; user average is.
- Explicit id mode (`board-game-ingest 13 9209`) is fail-fast; `--corpus`
  continues on per-id errors.
- Not a crawler: no job queue, no multi-thousand live crawl, no HTML.

## Fixture tests

`tests/fixtures/ingestion/` holds realistic `/thing` XML, including a
two-item batch and an expansion. Pytest must not depend on live BGG.
