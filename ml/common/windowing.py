"""Time-series window construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_windows(
    data: pd.DataFrame | np.ndarray,
    window_size: int,
    target: pd.Series | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create overlapping sequence windows and optional end-of-window labels."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    values = data.to_numpy(dtype=np.float32) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    if len(values) < window_size:
        raise ValueError("data must contain at least window_size rows")

    windows = np.stack([values[index:index + window_size] for index in range(len(values) - window_size + 1)])
    labels = None
    if target is not None:
        target_values = np.asarray(target)
        if len(target_values) != len(values):
            raise ValueError("target must have the same number of rows as data")
        labels = target_values[window_size - 1:]
    return windows, labels

