"""Bag-hash Embedder over observation information items."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ds_platform.modeling.representations import RepresentationTable

from board_game_analysis.domain import Visibility
from board_game_analysis.modeling.encoders.bag import (
    BAG_HASH_FAMILY,
    DEFAULT_DIM,
    encode_bags,
)


class ObservationBagEmbedder:
    """Mean-pool hashed item tokens plus bag counts. ``fit`` is a no-op."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.family = BAG_HASH_FAMILY

    def fit(
        self,
        entity_ids: Sequence[str],
        records: Sequence[Mapping[str, object]],
    ) -> None:
        del entity_ids, records

    def encode(
        self,
        entity_ids: Sequence[str],
        records: Sequence[Mapping[str, object]],
    ) -> RepresentationTable:
        return encode_bags(
            entity_ids,
            records,
            dim=self.dim,
            token_fn=_observation_tokens,
            stats_fn=_observation_stats,
        )


def _observation_tokens(record: Mapping[str, object]) -> list[str]:
    tokens: list[str] = []
    for item in _items(record):
        keys = item.get("payload_keys")
        key_text = ",".join(str(key) for key in keys) if isinstance(keys, list) else ""
        tokens.append(
            "vis={}|known={}|about={}|holder={}|keys={}".format(
                item.get("visibility"),
                1 if item.get("content_known") else 0,
                item.get("about"),
                item.get("holder_id"),
                key_text,
            )
        )
    return tokens


def _observation_stats(record: Mapping[str, object]) -> list[float]:
    items = _items(record)
    n_items = float(len(items))
    n_known = float(sum(1 for item in items if item.get("content_known")))
    n_unknown = n_items - n_known
    n_unknown_to_self = float(
        sum(1 for item in items if item.get("visibility") == Visibility.UNKNOWN_TO_SELF)
    )
    return [n_items, n_known, n_unknown, n_unknown_to_self]


def _items(record: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = record.get("items")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]
