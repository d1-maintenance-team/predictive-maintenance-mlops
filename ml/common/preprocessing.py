"""Shared preprocessing helpers for sensor data."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(
    data: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric features and return the fitted scaler."""
    columns = list(feature_columns) if feature_columns is not None else data.columns.tolist()
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")
    values = data.loc[:, columns]
    if values.select_dtypes(exclude="number").shape[1]:
        raise TypeError("All features must be numeric")
    active_scaler = scaler or StandardScaler()
    transformed = active_scaler.fit_transform(values) if scaler is None else active_scaler.transform(values)
    return pd.DataFrame(transformed, columns=columns, index=data.index), active_scaler

