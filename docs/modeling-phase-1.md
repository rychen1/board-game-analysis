# Phase 1 — Examples and measurements

Phase 1 projects one authored `PlaySituation` into JSON-serializable
records that later phases consume without re-parsing domain objects.

## Example records

`examples_from_situation` / `examples_from_situations` produce an
`ExampleBundle`:

- `StateExample` — omniscient state snapshot (`data` stays a mapping)
- `ObservationExample` — one observer's view with frozen item dicts
- `PairExample` — one authored transition with action ids
- `SequenceExample` — situation states in list order; `steps` empty here

Entity ids are **situation-scoped** (`situation_id_for` disambiguates
multiple fragments of the same game). See `examples.py` helpers:
`state_entity_id`, `observation_entity_id`, `pair_entity_id`, etc.

Persisted as `bga:examples/play-fixtures:v0`.

## Measurements

`measure_situation(s)` evaluates the ontology-v0 catalog (`analysis/spec.py`)
over domain objects. Measurement ids use the situation id as scope key.

Persisted as `bga:measurements/ontology-v0:v0`.

## Boundaries

Phase 1 does not encode vectors, fit models, or merge situations across
games. Batch APIs key by `situation_id`, not `game_id` alone.
