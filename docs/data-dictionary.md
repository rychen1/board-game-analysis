# Data dictionary

Canonical fields for ingested board-game metadata. This is the catalog of
**observed facts** stored on `Game`, `Mechanic`, and `SourceReference`.

It is not a taxonomy of mechanics, not an interpretation layer, and not an
analysis catalog. BGG categories such as "Negotiation" stay source strings.

The typed catalog lives in `board_game_analysis.ingestion.dictionary`. The
machine-readable interchange schema is `schemas/game.schema.json`, generated
from `Game.model_json_schema()`.

## Layers

| Layer | Models | Ingestion v0 |
| --- | --- | --- |
| Observed facts | `Game`, `Mechanic`, `SourceReference` | written |
| Extracted interpretations | `ExtractedInterpretation` | not written |
| Derived measurements | `DerivedMeasurement` | not written |

Missing values are `null` (or `[]` for lists). Do not invent defaults.
BoardGameGeek uses `0` as a sentinel for some numeric fields; those become
`null` except `rating_count`, where `0` is a known zero.

## `Game`

| Field | Type | Nullable | BGG source | Missing-data policy |
| --- | --- | --- | --- | --- |
| `id` | `str` | no | `item@id` prefixed with `bgg-` | required |
| `title` | `str` | no | primary `name/@value` | required |
| `release_year` | `int` | yes | `yearpublished` | omitted or `0` → null |
| `min_players` | `int` | yes | `minplayers` | omitted or `0` → null (never default to 1) |
| `max_players` | `int` | yes | `maxplayers` | omitted or `0` → null; reject if min > max |
| `min_play_time_minutes` | `int` | yes | `minplaytime`, else `playingtime` | omitted or `0` → null |
| `max_play_time_minutes` | `int` | yes | `maxplaytime`, else `playingtime` | omitted or `0` → null |
| `popularity` | `float` | yes | not mapped | always null in BGG v0 |
| `rating` | `float` | yes | `average` | null when `usersrated` is omitted or 0 |
| `rating_count` | `int` | yes | `usersrated` | omitted → null; `0` stays `0` |
| `complexity` | `float` | yes | `averageweight` | null when `numweights` is omitted or 0 |
| `designers` | `list[str]` | no | designer links | no links → `[]` |
| `publishers` | `list[str]` | no | publisher links | no links → `[]`. BGG lists every regional publisher; famous games often have 20–50+ names. |
| `categories` | `list[str]` | no | category links | no links → `[]`; not interpretations |
| `mechanics` | `list[Mechanic]` | no | mechanic links | no links → `[]`; not a taxonomy |
| `sources` | `list[SourceReference]` | no | request URL + thing id + fetch time | BGG ingest always writes one |

Play-time fallback: if both `minplaytime` and `maxplaytime` are missing or
zero but `playingtime` is set, min and max are filled from `playingtime`.

`rating` is the user average, not BGG Geek Rating (`bayesaverage`). Rank and
owned-count are not mapped onto `popularity`.

The only timestamp on `Game` is provenance: `sources[].retrieved_at`.

## `Mechanic`

| Field | Type | Nullable | BGG source | Missing-data policy |
| --- | --- | --- | --- | --- |
| `id` | `str` | no | link id prefixed with `bgg-` | required per named link |
| `name` | `str` | no | link `@value` | empty names dropped |
| `category` | `str` | yes | not mapped | always null in BGG v0 |
| `description` | `str` | yes | not mapped | always null in BGG v0 |

## `SourceReference`

| Field | Type | Nullable | BGG source | Missing-data policy |
| --- | --- | --- | --- | --- |
| `source` | `str` | no | literal `boardgamegeek` | required |
| `source_type` | `str` | yes | literal `xmlapi2` | always set for BGG ingest |
| `url` | `str` | yes | `/thing?id=...&stats=1` (may be batched) | always set for BGG ingest |
| `source_identifier` | `str` | yes | thing id | always set for BGG ingest |
| `retrieved_at` | `datetime` | yes | local fetch timestamp | always set for BGG ingest |
| `page_number` | `int` | yes | unused | always null for BGG v0 |

## What is not in this dictionary

- Rulebook text, descriptions, images, polls, ranks, alternate names
- Expansions and accessories (`item@type` other than `boardgame`). Related
  `boardgameexpansion` / `boardgameaccessory` links on a base game are ignored.
- Interpretations (cooperative, hidden information, …)
- Derived measurements (decision-space size, agency, …)
- Per-field provenance (record-level only)
