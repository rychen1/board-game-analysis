"""Bag-hash Embedder over structural action sets.

Tokens (deterministic, sorted):

- ``type={action_type}`` for each listed type
- ``param_key={key}`` for each parameter key (values are never hashed)
- ``n={n_actions}``

Stats (last dims): ``n_actions``, unique type count, parameter-key count, 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ds_platform.modeling.representations import RepresentationTable

from board_game_analysis.modeling.encoders.bag import (
    BAG_HASH_FAMILY,
    DEFAULT_DIM,
    encode_bags,
)


class ActionBagEmbedder:
    """Mean-pool hashed action-set tokens plus counts. ``fit`` is a no-op."""

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
            token_fn=_action_tokens,
            stats_fn=_action_stats,
        )


def _action_tokens(record: Mapping[str, object]) -> list[str]:
    types = _string_list(record.get("action_types"))
    keys = _string_list(record.get("parameter_keys"))
    n_actions = record.get("n_actions")
    n_text = str(int(n_actions)) if isinstance(n_actions, (int, float)) else "0"
    tokens = [f"type={action_type}" for action_type in types]
    tokens.extend(f"param_key={key}" for key in keys)
    tokens.append(f"n={n_text}")
    return tokens


def _action_stats(record: Mapping[str, object]) -> list[float]:
    types = _string_list(record.get("action_types"))
    keys = _string_list(record.get("parameter_keys"))
    n_actions = record.get("n_actions")
    count = float(n_actions) if isinstance(n_actions, (int, float)) else 0.0
    return [count, float(len(set(types))), float(len(keys)), 0.0]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
