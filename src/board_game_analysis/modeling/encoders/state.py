"""Bag-hash Embedder over GameState keys, types, phase, and turn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ds_platform.modeling.representations import RepresentationTable

from board_game_analysis.modeling.encoders.bag import (
    BAG_HASH_FAMILY,
    DEFAULT_DIM,
    encode_bags,
)


class StateBagEmbedder:
    """Hash data keys and types only. Payload values and titles are ignored."""

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
            token_fn=_state_tokens,
            stats_fn=_state_stats,
        )


def _state_tokens(record: Mapping[str, object]) -> list[str]:
    tokens = [
        f"phase={record.get('phase')}",
        f"turn={record.get('turn_number')}",
        f"active={1 if record.get('active_player_id') else 0}",
    ]
    data = record.get("data")
    if isinstance(data, Mapping):
        for key, value in sorted(data.items(), key=lambda item: str(item[0])):
            tokens.append(f"key={key}|type={type(value).__name__}")
    return tokens


def _state_stats(record: Mapping[str, object]) -> list[float]:
    data = record.get("data")
    n_keys = float(len(data)) if isinstance(data, Mapping) else 0.0
    turn = record.get("turn_number")
    turn_value = float(turn) if isinstance(turn, (int, float)) else 0.0
    has_phase = 1.0 if record.get("phase") else 0.0
    has_active = 1.0 if record.get("active_player_id") else 0.0
    return [n_keys, turn_value, has_phase, has_active]
