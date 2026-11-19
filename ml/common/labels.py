"""Label construction helpers for predictive maintenance."""

from __future__ import annotations

import pandas as pd


def create_failure_label(
    data: pd.DataFrame,
    failure_column: str,
    horizon: int = 1,
) -> pd.Series:
    """Label each row when a failure occurs within the following horizon."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if failure_column not in data.columns:
        raise ValueError(f"Failure column not found: {failure_column}")
    failures = data[failure_column].astype(bool)
    future_failures = pd.concat(
        [failures.shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
    return future_failures.any(axis=1).astype("int8")
