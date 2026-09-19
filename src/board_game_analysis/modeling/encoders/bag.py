"""Numpy-free hashed bag pooling shared by state and observation embedders."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence

from ds_platform.hashing import payload_id
from ds_platform.modeling.representations import RepresentationTable

BAG_HASH_FAMILY = "bag-hash-v0"
DEFAULT_DIM = 16
STAT_DIM = 4


def hash_vector(token: str, dim: int) -> tuple[float, ...]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return tuple((digest[index % 32] / 127.5) - 1.0 for index in range(dim))


def mean_pool(vectors: Sequence[Sequence[float]], dim: int) -> list[float]:
    if not vectors:
        return [0.0] * dim
    totals = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += float(value)
    count = float(len(vectors))
    return [total / count for total in totals]


def record_source_id(record: Mapping[str, object]) -> str:
    return payload_id(
        json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )


def concat_stats(
    pooled: Sequence[float],
    stats: Sequence[float],
    dim: int,
) -> tuple[float, ...]:
    if dim <= STAT_DIM:
        return tuple(float(value) for value in pooled[:dim])
    hash_dim = dim - STAT_DIM
    padded_stats = list(stats[:STAT_DIM])
    while len(padded_stats) < STAT_DIM:
        padded_stats.append(0.0)
    return tuple([*pooled[:hash_dim], *padded_stats])


def encode_bags(
    entity_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
    *,
    dim: int,
    token_fn: Callable[[Mapping[str, object]], Sequence[str]],
    stats_fn: Callable[[Mapping[str, object]], Sequence[float]],
) -> RepresentationTable:
    if len(entity_ids) != len(records):
        raise ValueError("entity_ids length must match records length")
    hash_dim = dim if dim <= STAT_DIM else dim - STAT_DIM
    vectors: list[tuple[float, ...]] = []
    sources: list[str] = []
    for record in records:
        tokens = token_fn(record)
        pooled = mean_pool([hash_vector(token, hash_dim) for token in tokens], hash_dim)
        vectors.append(concat_stats(pooled, stats_fn(record), dim))
        sources.append(record_source_id(record))
    return RepresentationTable(
        entity_ids=tuple(entity_ids),
        vectors=tuple(vectors),
        dim=dim,
        source_payload_ids=tuple(sources),
    )
