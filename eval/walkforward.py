"""
eval/walkforward.py — Walk-forward evaluation with structural embargo.

Fold structure (per spec §9):
    |── Train (120d) ──|── Embargo (10d) ──|── Val (20d) ──|── Test (10d) ──|

Step 10 days. Minimum 12 folds. Embargo enforced structurally.
"""

import os
import logging
from typing import Callable, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class WalkForwardFold:
    """A single walk-forward fold with structural embargo."""
    fold_idx: int
    train_start: int  # index into DataFrame
    train_end: int
    embargo_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    metrics: dict = field(default_factory=dict)


def generate_folds(
    n_bars: int,
    bars_per_day: int = 288,  # 5m bars per day
    config: Optional[dict] = None,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward fold indices.

    Args:
        n_bars: Total number of bars in dataset
        bars_per_day: Number of bars per day (288 for 5m, 96 for 15m)
        config: Optional config override

    Returns:
        List of WalkForwardFold objects
    """
    if config is None:
        config = load_config()

    tc = config['training']
    train_bars = tc['train_window_days'] * bars_per_day
    embargo_bars = tc['embargo_days'] * bars_per_day
    val_bars = tc['val_window_days'] * bars_per_day
    test_bars = tc['test_window_days'] * bars_per_day
    step_bars = tc['fold_step_days'] * bars_per_day
    min_folds = tc['min_folds']

    fold_size = train_bars + embargo_bars + val_bars + test_bars
    folds = []

    fold_start = 0
    fold_idx = 0

    while fold_start + fold_size <= n_bars:
        train_start = fold_start
        train_end = train_start + train_bars
        embargo_end = train_end + embargo_bars
        val_start = embargo_end
        val_end = val_start + val_bars
        test_start = val_end
        test_end = test_start + test_bars

        if test_end > n_bars:
            break

        folds.append(WalkForwardFold(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            embargo_end=embargo_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        ))

        fold_start += step_bars
        fold_idx += 1

    if len(folds) < min_folds:
        logger.warning(
            "Only {} folds generated (minimum {}). Need more data.".format(
                len(folds), min_folds
            )
        )

    logger.info("Generated {} walk-forward folds".format(len(folds)))
    return folds


def run_walkforward(
    features: pd.DataFrame,
    labels: pd.Series,
    train_fn: Callable,
    predict_fn: Callable,
    p_market_history: Optional[pd.DataFrame] = None,
    bar_timestamps: Optional[pd.Series] = None,
    config: Optional[dict] = None,
    bars_per_day: int = 288,
) -> list[WalkForwardFold]:
    """
    Run walk-forward evaluation.

    Args:
        features: Feature DataFrame (aligned with labels)
        labels: Label Series (0/1)
        train_fn: Callable(X_train, y_train) -> model
        predict_fn: Callable(model, X) -> p_up_array
        p_market_history: Optional historical Polymarket prices DataFrame
        bar_timestamps: Timestamps for each bar (for p_market join)
        config: Optional config override
        bars_per_day: Bars per day

    Returns:
        List of WalkForwardFold with metrics populated
    """
    if config is None:
        config = load_config()

    from eval.metrics import compute_fold_metrics

    folds = generate_folds(len(features), bars_per_day, config)

    for fold in folds:
        logger.info("Fold {}: train[{}:{}] val[{}:{}] test[{}:{}]".format(
            fold.fold_idx,
            fold.train_start, fold.train_end,
            fold.val_start, fold.val_end,
            fold.test_start, fold.test_end,
        ))

        # Extract data — embargo gap enforced structurally
        X_train = features.iloc[fold.train_start:fold.train_end]
        y_train = labels.iloc[fold.train_start:fold.train_end]
        X_val = features.iloc[fold.val_start:fold.val_end]
        y_val = labels.iloc[fold.val_start:fold.val_end]
        X_test = features.iloc[fold.test_start:fold.test_end]
        y_test = labels.iloc[fold.test_start:fold.test_end]

        # Train
        model = train_fn(X_train, y_train)

        # Predict on val and test
        p_up_val = predict_fn(model, X_val)
        p_up_test = predict_fn(model, X_test)

        # Get p_market for val/test bars
        p_market_val = _get_p_market(
            bar_timestamps, fold.val_start, fold.val_end,
            p_market_history, config
        )
        p_market_test = _get_p_market(
            bar_timestamps, fold.test_start, fold.test_end,
            p_market_history, config
        )

        # Compute metrics
        fold.metrics = compute_fold_metrics(
            p_up=p_up_test,
            labels=y_test.values,
            p_market=p_market_test,
            config=config,
        )
        fold.metrics['val_metrics'] = compute_fold_metrics(
            p_up=p_up_val,
            labels=y_val.values,
            p_market=p_market_val,
            config=config,
        )

    return folds


def _get_p_market(
    bar_timestamps: Optional[pd.Series],
    start_idx: int,
    end_idx: int,
    p_market_history: Optional[pd.DataFrame],
    config: dict,
) -> np.ndarray:
    """
    Get p_market for a range of bars.

    Uses historical prices where available (from p_market_history.parquet).
    Falls back to assumed_market_price (0.50) when no historical price exists.
    Per spec §9: documented at every call site.
    """
    n_bars = end_idx - start_idx
    # Fallback: assumed_market_price = 0.50 (efficient-market null hypothesis)
    # This is a LOWER BOUND — live prices may diverge from 0.50
    fallback = config['backtest']['assumed_market_price']

    if bar_timestamps is None or p_market_history is None or p_market_history.empty:
        # No historical prices available — use fallback for all bars
        return np.full(n_bars, fallback)

    timestamps = bar_timestamps.iloc[start_idx:end_idx].values
    p_market = np.full(n_bars, fallback)

    if config['backtest']['use_historical_prices']:
        for i, ts in enumerate(timestamps):
            match = p_market_history[p_market_history['timestamp'] == ts]
            if len(match) > 0:
                p_market[i] = match.iloc[0]['p_market_up']

    return p_market


def summarize_folds(folds: list[WalkForwardFold]) -> dict:
    """Summarize metrics across all folds."""
    if not folds:
        return {}

    metrics_keys = [k for k in folds[0].metrics.keys() if k != 'val_metrics']
    summary = {}

    for key in metrics_keys:
        values = [f.metrics[key] for f in folds if key in f.metrics]
        if values and isinstance(values[0], (int, float)):
            summary[key + '_mean'] = float(np.mean(values))
            summary[key + '_std'] = float(np.std(values))
            summary[key + '_min'] = float(np.min(values))
            summary[key + '_max'] = float(np.max(values))

    summary['n_folds'] = len(folds)
    return summary
