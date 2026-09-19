"""Tiny numpy-free linear probes over representation FeatureTables."""

from __future__ import annotations

from collections.abc import Sequence

from ds_platform.modeling.evaluate import EvaluationReport, evaluate, rmse
from ds_platform.modeling.features import FeatureTable, Scalar, extract_column
from ds_platform.modeling.spec import MetricSpec
from ds_platform.modeling.split import SplitAssignment, apply_split


class RidgeRegressor:
    """Closed-form ridge regressor. Satisfies the platform Regressor protocol."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.weights: tuple[float, ...] | None = None

    def fit(self, features: FeatureTable, y: Sequence[float]) -> None:
        if len(y) != len(features.entity_ids):
            raise ValueError("y length must match feature row count")
        design = [_with_intercept(row) for row in features.values]
        targets = [float(value) for value in y]
        self.weights = tuple(_fit_ridge(design, targets, self.ridge))

    def predict(self, features: FeatureTable) -> list[float]:
        if self.weights is None:
            raise RuntimeError("RidgeRegressor.fit must be called before predict")
        return [_dot(self.weights, _with_intercept(row)) for row in features.values]


def labeled_feature_table(
    features: FeatureTable,
    labels: Sequence[Scalar],
) -> tuple[FeatureTable, list[float]]:
    """Drop rows whose label is missing. Never impute."""
    if len(labels) != len(features.entity_ids):
        raise ValueError("labels length must match feature row count")
    kept: list[int] = []
    y: list[float] = []
    for index, label in enumerate(labels):
        if isinstance(label, (int, float)) and not isinstance(label, bool):
            kept.append(index)
            y.append(float(label))
    return (
        FeatureTable(
            entity_ids=tuple(features.entity_ids[index] for index in kept),
            columns=features.columns,
            values=tuple(features.values[index] for index in kept),
            source_payload_ids=tuple(
                features.source_payload_ids[index] for index in kept
            ),
        ),
        y,
    )


def probe_column(
    features: FeatureTable,
    labels: FeatureTable,
    column: str,
    assignment: SplitAssignment,
    *,
    ridge: float = 1e-4,
) -> tuple[EvaluationReport, float]:
    """Fit on train, score RMSE on test, and return the train-mean baseline RMSE."""
    train_features, _validation, test_features = apply_split(features, assignment)
    train_labels, _validation_labels, test_labels = apply_split(labels, assignment)
    train_x, train_y = labeled_feature_table(
        train_features, extract_column(train_labels, column)
    )
    test_x, test_y = labeled_feature_table(
        test_features, extract_column(test_labels, column)
    )
    model = RidgeRegressor(ridge=ridge)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    metrics = (MetricSpec(name="rmse"), MetricSpec(name="mae"))
    report = evaluate(test_y, predicted, metrics)
    baseline = sum(train_y) / len(train_y) if train_y else 0.0
    baseline_rmse = rmse(test_y, [baseline] * len(test_y))
    return report, baseline_rmse


def _with_intercept(row: Sequence[Scalar]) -> list[float]:
    return [1.0, *(_as_float(cell) for cell in row)]


def _as_float(value: Scalar) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected numeric feature, got {type(value)!r}")


def _fit_ridge(
    design: list[list[float]],
    y: list[float],
    ridge: float,
) -> list[float]:
    n_features = len(design[0]) if design else 0
    gram = [[0.0] * n_features for _ in range(n_features)]
    projection = [0.0] * n_features
    for row, target in zip(design, y, strict=True):
        for i in range(n_features):
            projection[i] += row[i] * target
            for j in range(n_features):
                gram[i][j] += row[i] * row[j]
    for i in range(n_features):
        gram[i][i] += ridge
    return _solve(gram, projection)


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(matrix)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda row: abs(augmented[row][i]))
        augmented[i], augmented[pivot] = augmented[pivot], augmented[i]
        diag = augmented[i][i]
        if abs(diag) < 1e-12:
            diag = 1e-12
            augmented[i][i] = diag
        scale = 1.0 / diag
        for j in range(i, n + 1):
            augmented[i][j] *= scale
        for row in range(n):
            if row == i:
                continue
            factor = augmented[row][i]
            for j in range(i, n + 1):
                augmented[row][j] -= factor * augmented[i][j]
    return [row[n] for row in augmented]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
