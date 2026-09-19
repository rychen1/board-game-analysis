"""Linear concat ConditionedEncoder: (z_state, z_action) → z_next."""

from __future__ import annotations

from collections.abc import Sequence

from ds_platform.modeling.features import FeatureTable
from ds_platform.modeling.representations import RepresentationTable

from board_game_analysis.modeling.probes import RidgeRegressor

CONDITIONED_FAMILY = "linear-concat-v0"


class CopyStateEncoder:
    """Baseline: predicted z_next is z_from. Ignores the action."""

    family = "copy-state-v0"

    def fit(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> None:
        del contexts, conditions, pair_ids

    def encode(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> RepresentationTable:
        del conditions
        if tuple(contexts.entity_ids) != tuple(pair_ids):
            raise ValueError("contexts must be keyed by pair_ids")
        return RepresentationTable(
            entity_ids=tuple(pair_ids),
            vectors=contexts.vectors,
            dim=contexts.dim,
            source_payload_ids=contexts.source_payload_ids,
        )


class LinearConditionedEncoder:
    """Ridge map from concat(z_state, z_action) to z_next. No torch/sklearn.

    Call :meth:`set_targets` with the actual to-state table (keyed by pair
    id) before :meth:`fit`. Missing targets are not imputed.
    """

    family = CONDITIONED_FAMILY

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.weights: tuple[tuple[float, ...], ...] | None = None
        self.output_dim: int | None = None
        self._targets: RepresentationTable | None = None

    def set_targets(self, targets: RepresentationTable) -> None:
        self._targets = targets

    def fit(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> None:
        if self._targets is None:
            raise RuntimeError("set_targets must be called before fit")
        _require_pair_keys(contexts, pair_ids, "contexts")
        _require_pair_keys(conditions, pair_ids, "conditions")
        _require_pair_keys(self._targets, pair_ids, "targets")
        features = _concat_features(contexts, conditions)
        heads: list[tuple[float, ...]] = []
        for dim_index in range(self._targets.dim):
            y = [vector[dim_index] for vector in self._targets.vectors]
            model = RidgeRegressor(ridge=self.ridge)
            model.fit(features, y)
            if model.weights is None:
                raise RuntimeError("ridge fit produced no weights")
            heads.append(model.weights)
        self.weights = tuple(heads)
        self.output_dim = self._targets.dim

    def encode(
        self,
        contexts: RepresentationTable,
        conditions: RepresentationTable,
        pair_ids: Sequence[str],
    ) -> RepresentationTable:
        if self.weights is None or self.output_dim is None:
            raise RuntimeError("LinearConditionedEncoder.fit must be called first")
        _require_pair_keys(contexts, pair_ids, "contexts")
        _require_pair_keys(conditions, pair_ids, "conditions")
        features = _concat_features(contexts, conditions)
        vectors: list[tuple[float, ...]] = []
        for row in features.values:
            intercept_row = [1.0, *_numeric_row(row)]
            vectors.append(tuple(_dot(head, intercept_row) for head in self.weights))
        return RepresentationTable(
            entity_ids=tuple(pair_ids),
            vectors=tuple(vectors),
            dim=self.output_dim,
            source_payload_ids=contexts.source_payload_ids,
        )


def _concat_features(
    contexts: RepresentationTable,
    conditions: RepresentationTable,
) -> FeatureTable:
    context_cols = [f"c{index}" for index in range(contexts.dim)]
    action_cols = [f"a{index}" for index in range(conditions.dim)]
    columns = tuple([*context_cols, *action_cols])
    values = tuple(
        tuple([*context, *condition])
        for context, condition in zip(contexts.vectors, conditions.vectors, strict=True)
    )
    return FeatureTable(
        entity_ids=contexts.entity_ids,
        columns=columns,
        values=values,
        source_payload_ids=contexts.source_payload_ids,
    )


def _require_pair_keys(
    table: RepresentationTable, pair_ids: Sequence[str], name: str
) -> None:
    if tuple(table.entity_ids) != tuple(pair_ids):
        raise ValueError(f"{name} must be keyed by pair_ids")


def _numeric_row(row: Sequence[object]) -> list[float]:
    values: list[float] = []
    for cell in row:
        if isinstance(cell, bool) or cell is None:
            raise TypeError("conditioned features must be numeric")
        if isinstance(cell, (int, float)):
            values.append(float(cell))
            continue
        raise TypeError(f"expected numeric feature, got {type(cell)!r}")
    return values


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
