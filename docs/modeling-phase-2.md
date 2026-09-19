# Phase 2 — Bag encodings

Phase 2 maps Phase 1 records to fixed-dimension **structural** bag vectors.
Payload values inside state/observation items are discarded by design; keys,
types, phases, and counts define the encoding.

## Embedders

| Embedder | Input record | Family |
| --- | --- | --- |
| `StateBagEmbedder` | `StateExample.to_encoder_record()` | bag hash |
| `ObservationBagEmbedder` | `ObservationExample.to_encoder_record()` | bag hash |
| `ActionBagEmbedder` | `ActionExample.to_encoder_record()` | bag hash |

`fit` is a no-op. Encoding the full fixture corpus before a game split does
not leak learned parameters.

## Trajectory summaries (Phase 2)

`encode_sequences` writes **trajectory summaries** (first/last/mean states,
action aggregates) to `bga:repr/trajectory-summaries:v0` with model
`bga:model/sequence-encoder:v0`.

This is distinct from Phase 5 sequence experiment summaries
(`bga:repr/sequence-summaries:v0`).

## Persistence helpers

`run_encode.py` wraps ds-platform `put_feature_dataset` /
`put_model_artifact` for states, observations, actions, trajectories, and
optional observation probes (`bga:evaluation/obs-probes:v0`).

Logical keys: `board_game_analysis.modeling.logical_keys`.
