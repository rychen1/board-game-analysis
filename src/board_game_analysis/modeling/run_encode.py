"""Thin wrappers around ds-platform encoding persistence."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import datetime

from ds_platform import RunContext, Store
from ds_platform.modeling.encode import EncodingResult
from ds_platform.modeling.evaluate import EvaluationReport
from ds_platform.modeling.records import (
    put_evaluation_artifact,
    put_feature_dataset,
    put_model_artifact,
    representation_payload_bytes,
)
from ds_platform.modeling.representations import RepresentationTable
from ds_platform.modeling.spec import EncodingSpec, spec_config_hash
from pydantic import BaseModel, ConfigDict

from board_game_analysis.modeling.encoders import (
    BAG_HASH_FAMILY,
    DEFAULT_DIM,
    ObservationBagEmbedder,
    StateBagEmbedder,
)
from board_game_analysis.modeling.examples import ObservationExample, StateExample

OBS_MODEL_LOGICAL_KEY = "bga:model/obs-embedder:v0"
STATE_MODEL_LOGICAL_KEY = "bga:model/state-embedder:v0"
OBS_REPR_LOGICAL_KEY = "bga:repr/observations:v0"
STATE_REPR_LOGICAL_KEY = "bga:repr/states:v0"
PROBE_EVAL_LOGICAL_KEY = "bga:eval/obs-probes:v0"


class EncodingBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    table: RepresentationTable
    result: EncodingResult


def bag_encoding_spec(*, dim: int = DEFAULT_DIM, seed: int = 0) -> EncodingSpec:
    return EncodingSpec(family=BAG_HASH_FAMILY, params={}, dim=dim, seed=seed)


def encode_observations(
    examples: Sequence[ObservationExample],
    *,
    store: Store,
    run: RunContext,
    dim: int = DEFAULT_DIM,
    created_at: datetime | None = None,
) -> EncodingBundle:
    encoder = ObservationBagEmbedder(dim=dim)
    return _encode_and_store(
        entity_ids=[example.entity_id for example in examples],
        records=[example.to_encoder_record() for example in examples],
        encoder=encoder,
        store=store,
        run=run,
        dim=dim,
        repr_key=OBS_REPR_LOGICAL_KEY,
        model_key=OBS_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )


def encode_states(
    examples: Sequence[StateExample],
    *,
    store: Store,
    run: RunContext,
    dim: int = DEFAULT_DIM,
    created_at: datetime | None = None,
) -> EncodingBundle:
    encoder = StateBagEmbedder(dim=dim)
    return _encode_and_store(
        entity_ids=[example.entity_id for example in examples],
        records=[example.to_encoder_record() for example in examples],
        encoder=encoder,
        store=store,
        run=run,
        dim=dim,
        repr_key=STATE_REPR_LOGICAL_KEY,
        model_key=STATE_MODEL_LOGICAL_KEY,
        created_at=created_at,
    )


def store_probe_evaluation(
    store: Store,
    report: EvaluationReport,
    *,
    run: RunContext,
    subject_payload_id: str,
    inputs: Sequence[str],
    logical_key: str = PROBE_EVAL_LOGICAL_KEY,
) -> tuple[str, str]:
    return put_evaluation_artifact(
        store,
        report,
        run=run,
        subject_payload_id=subject_payload_id,
        inputs=list(inputs),
        logical_key=logical_key,
    )


def _encode_and_store(
    *,
    entity_ids: list[str],
    records: list[dict[str, object]],
    encoder: ObservationBagEmbedder | StateBagEmbedder,
    store: Store,
    run: RunContext,
    dim: int,
    repr_key: str,
    model_key: str,
    created_at: datetime | None,
) -> EncodingBundle:
    spec = bag_encoding_spec(dim=dim)
    config_hash = spec_config_hash(spec)
    run_with_hash = run.model_copy(update={"config_hash": config_hash})
    encoder.fit(entity_ids, records)
    table = encoder.encode(entity_ids, records)
    representation_payload_id, _repr_record_id = put_feature_dataset(
        store,
        representation_payload_bytes(table),
        run=run_with_hash,
        inputs=[],
        media_type="application/json",
        logical_key=repr_key,
        created_at=created_at,
    )
    model_payload_id, _model_record_id = put_model_artifact(
        store,
        pickle.dumps(encoder),
        run=run_with_hash,
        inputs=[representation_payload_id],
        media_type="application/octet-stream",
        logical_key=model_key,
        created_at=created_at,
    )
    return EncodingBundle(
        table=table,
        result=EncodingResult(
            run_id=run_with_hash.run_id,
            config_hash=config_hash,
            representation_payload_id=representation_payload_id,
            model_payload_id=model_payload_id,
        ),
    )
