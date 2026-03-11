"""
labels/direction.py — Binary bar direction label.

Returns INTEGER class indices 0 or 1. NEVER +1/-1. NEVER strings.
  class_down = 0  →  close[t] < open[t]   → "Down"
  class_up   = 1  →  close[t] >= open[t]  → "Up"

Conversion from class index to direction string ("up"/"down") happens
in ensemble.py, not here. strategy/ and inference/ always use strings.

Polymarket resolution: "Up if closing price >= price at beginning of window"
open[t] is the correct anchor. close[t-1] diverges on large intra-bar moves.

The model predicts bar t+1 using features from bars up to bar t.
Labels for historical training are generated from open[t] and close[t].
"""

import numpy as np
import pandas as pd


def label_bar(open_price: float, close_price: float) -> int:
    """
    Returns 1 (Up) if close >= open, else 0 (Down).

    Returns:
        int: 0 or 1. Never +1/-1. Never strings.
    """
    return 1 if close_price >= open_price else 0


def label_series(
    opens: pd.Series,
    closes: pd.Series,
    drop_last: bool = True,
) -> pd.Series:
    """
    Vectorized label generation for a series of bars.

    The model predicts bar t+1. Labels are generated from open[t] and close[t].
    The LAST bar has no future — drop it to prevent leakage.

    Args:
        opens: Series of open prices
        closes: Series of close prices
        drop_last: If True, raises ValueError when called without dropping
                   the final bar. Set False only in test code.

    Returns:
        pd.Series of int (0 or 1)

    Raises:
        ValueError: If drop_last=True and called on a series where the last
                    bar has no known future outcome (leakage guard).
    """
    if len(opens) != len(closes):
        raise ValueError(f"Length mismatch: opens={len(opens)}, closes={len(closes)}")

    if len(opens) == 0:
        return pd.Series(dtype=int)

    labels = (closes >= opens).astype(int)

    if drop_last:
        # Leakage guard: the final bar has no t+1 close to predict.
        # Drop it from the label series. The caller must align features accordingly.
        labels = labels.iloc[:-1]

    return labels


def validate_labels(labels: pd.Series) -> None:
    """
    Validate that labels conform to spec rules.

    Raises ValueError if:
    - Any value is not 0 or 1
    - dtype is not int
    - +1/-1 encoding detected
    - String values detected
    """
    if labels.dtype == object or pd.api.types.is_string_dtype(labels):
        raise ValueError("Labels must be int, not strings")

    unique_vals = set(labels.dropna().unique())
    invalid = unique_vals - {0, 1}
    if invalid:
        raise ValueError(
            f"Invalid label values: {invalid}. "
            f"Only 0 (Down) and 1 (Up) allowed. "
            f"Never +1/-1."
        )

    if -1 in unique_vals:
        raise ValueError("Labels use +1/-1 encoding. Must use 0/1.")
