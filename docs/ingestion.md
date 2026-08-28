# Ingestion v0

This is a small, source-specific pipeline for BoardGameGeek metadata. It is
not a production crawler, not HTML scraping, and not rulebook extraction.

Live calls to BGG XML API2 require an application token (`BGG_TOKEN`).
Without a token the API returns `401 Unauthorized`. Tests never call the
network; they use checked-in XML fixtures.

## Source

- **Name:** BoardGameGeek
- **Interface:** XML API2 `GET /thing?id={id}&stats=1`
- **Docs:** https://boardgamegeek.com/wiki/page/BGG_XML_API2
- **Auth:** `Authorization: Bearer <token>`
  (register at https://boardgamegeek.com/applications)
- **Terms:** https://boardgamegeek.com/xmlapi/termsofuse

Do not scrape `boardgamegeek.com` HTML to bypass authentication.

## Architecture

```text
BggClient.fetch_game(id)
    → RawArtifact (XML body + request metadata)
    → data/raw/boardgamegeek/{id}.xml
    → data/raw/boardgamegeek/{id}.meta.json
    → normalize_bgg_artifact
    → Game
    → data/processed/boardgamegeek/{id}.json
```

| Piece | Responsibility |
| --- | --- |
| `BggClient` | HTTP only: timeout, retries, rate limit, User-Agent, Bearer token |
| `parse_thing_xml` | XML → source-specific `BggThing` |
| `normalize_bgg_artifact` | `BggThing` + provenance → canonical `Game` |
| `RawArchive` | Immutable-ish raw cache |

`board_game_analysis.domain` does not import this package and does not
know BGG field names.

The pipeline writes `Game` plus `SourceReference`. It does **not** write
`ExtractedInterpretation` or `DerivedMeasurement`. A BGG category such as
"Negotiation" stays a source-reported `Game.categories` string.

## Raw vs normalized

Raw XML is the research archive. Normalization never mutates it. `--force`
refetches and may replace the raw file; the default is to reuse the cache
and re-run normalization only.

Processed JSON is `Game.model_dump(mode="json")`. PostgreSQL is not used.

Downloaded datasets must not be committed (see `.gitignore`).

## Provenance

Each `Game.sources` entry records:

- `source` = `boardgamegeek`
- `source_type` = `xmlapi2`
- `source_identifier` = BGG thing id
- `url` = the request URL
- `retrieved_at` = fetch time (provenance only; same XML + same timestamp
  → identical `Game`)

Per-field provenance is not implemented.

## Caching and rate limiting

If `data/raw/boardgamegeek/{id}.xml` and `.meta.json` exist, ingest skips
HTTP. That is how normalization can be rerun offline.

Live HTTP:

- explicit timeout (`BGG_TIMEOUT_SECONDS`, default 30)
- delay between requests (`BGG_RATE_LIMIT_SECONDS`, default 5)
- retries on `202`, `429`, `503` with linear backoff
- one request at a time (no parallel scraping)
- `User-Agent` identifies this application

## Missing-data policy

Do not invent values. BGG uses `0` as a sentinel for some fields.

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

## How to ingest one game

```bash
cp .env.example .env   # set BGG_TOKEN
uv run board-game-ingest 13
# or: uv run python scripts/ingest.py 13
```

Writes:

- `data/raw/boardgamegeek/13.xml`
- `data/raw/boardgamegeek/13.meta.json`
- `data/processed/boardgamegeek/13.json`

`--force` refetches.

## Known limitations

- Token required; live ingest is unverified in environments without
  `BGG_TOKEN`.
- One thing id per HTTP request (BGG allows up to 20; v0 does not batch).
- No collection, search, hot-list, or user endpoints.
- Expansions and accessories are still mapped as `Game` if returned.
- Alternate names, descriptions, polls, ranks, and images are ignored.
- BGG "Geek Rating" (`bayesaverage`) is not mapped; user average is.
- Not a crawler: no job queue, no thousands of ids, no HTML.

## Fixture tests

`tests/fixtures/ingestion/` holds realistic `/thing` XML. Pytest must not
depend on live BGG.
