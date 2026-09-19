"""Train-only scalar probes from model representation vectors."""

from __future__ import annotations

from ds_platform.modeling.features import FeatureTable, Scalar
from ds_platform.modeling.representations import RepresentationTable

from board_game_analysis.modeling.probes import RidgeRegressor, labeled_feature_table


def representation_feature_table(
    table: RepresentationTable,
    *,
    source_payload_id: str | None = None,
) -> FeatureTable:
    if source_payload_id is None:
        payload_ids = table.source_payload_ids
    else:
        payload_ids = tuple(source_payload_id for _ in table.entity_ids)
    columns = tuple(f"z{index}" for index in range(table.dim))
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=columns,
        values=tuple(vector for vector in table.vectors),
        source_payload_ids=payload_ids,
    )


def zero_change_scalar_table(table: FeatureTable) -> FeatureTable:
    """Scalar baseline predicting no change from the from-state."""
    values = tuple(
        tuple(0.0 if isinstance(cell, (int, float)) else None for cell in row)
        for row in table.values
    )
    return FeatureTable(
        entity_ids=table.entity_ids,
        columns=table.columns,
        values=values,
        source_payload_ids=table.source_payload_ids,
    )


def scalar_predictions_from_representations(
    train_predicted: RepresentationTable,
    test_predicted: RepresentationTable,
    train_scalar_true: FeatureTable,
    test_scalar_true: FeatureTable,
    *,
    ridge: float = 1e-4,
) -> FeatureTable:
    """Fit ridge probes on train model vectors and predict test scalars."""
    if train_predicted.entity_ids != train_scalar_true.entity_ids:
        raise ValueError("train predicted ids must align with train scalar labels")
    if test_predicted.entity_ids != test_scalar_true.entity_ids:
        raise ValueError("test predicted ids must align with test scalar labels")
    if train_scalar_true.columns != test_scalar_true.columns:
        raise ValueError("scalar column names must match between train and test")

    train_x = representation_feature_table(train_predicted)
    test_source = (
        test_scalar_true.source_payload_ids[0]
        if test_scalar_true.source_payload_ids
        else None
    )
    test_x = representation_feature_table(
        test_predicted,
        source_payload_id=test_source,
    )

    predicted_rows: list[list[Scalar]] = [
        [None] * len(test_scalar_true.columns)
        for _ in test_scalar_true.entity_ids
    ]
    for col_index, _column in enumerate(train_scalar_true.columns):
        train_labels = [row[col_index] for row in train_scalar_true.values]
        labeled_train_x, train_y = labeled_feature_table(train_x, train_labels)
        if not train_y:
            continue
        model = RidgeRegressor(ridge=ridge)
        model.fit(labeled_train_x, train_y)
        column_preds = model.predict(test_x)
        for row_index, pred_value in enumerate(column_preds):
            predicted_rows[row_index][col_index] = pred_value

    return FeatureTable(
        entity_ids=test_scalar_true.entity_ids,
        columns=test_scalar_true.columns,
        values=tuple(tuple(row) for row in predicted_rows),
        source_payload_ids=test_scalar_true.source_payload_ids,
    )
