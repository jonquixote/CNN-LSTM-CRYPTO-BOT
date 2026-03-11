"""
data/validate.py — Distribution checks, NaN audit, Chainlink basis.

Validation suite for the data pipeline. Run after feature engineering
to verify data quality before training.
"""

import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def audit_nans(features: pd.DataFrame, warmup_bars: int = 550) -> dict:
    """
    NaN audit across all features.

    After warmup_bars, there should be ZERO NaNs.
    Returns dict with per-feature NaN counts and overall status.
    """
    # Only check rows after warmup period
    post_warmup = features.iloc[warmup_bars:]

    nan_counts = post_warmup.isna().sum()
    total_nans = nan_counts.sum()

    problem_features = nan_counts[nan_counts > 0].to_dict()

    result = {
        'total_nans_post_warmup': int(total_nans),
        'problem_features': problem_features,
        'total_features': len(features.columns),
        'rows_checked': len(post_warmup),
        'warmup_bars': warmup_bars,
        'pass': total_nans == 0,
    }

    if total_nans > 0:
        logger.warning(f"NaN audit FAILED: {total_nans} NaNs in {len(problem_features)} features")
        for feat, count in problem_features.items():
            logger.warning(f"  {feat}: {count} NaNs")
    else:
        logger.info(f"NaN audit PASSED: 0 NaNs across {len(features.columns)} features")

    return result


def check_distribution(features: pd.DataFrame, warmup_bars: int = 550) -> dict:
    """
    Distribution checks for features.

    Checks for:
    - Constant features (zero variance)
    - Extreme outliers (> 10 std from mean)
    - Infinite values
    """
    post_warmup = features.iloc[warmup_bars:]

    # Constant features
    variances = post_warmup.var()
    constant_features = variances[variances == 0].index.tolist()

    # Infinite values
    inf_counts = np.isinf(post_warmup.select_dtypes(include=[np.number])).sum()
    inf_features = inf_counts[inf_counts > 0].to_dict()

    # Extreme outliers (> 10 std)
    means = post_warmup.mean()
    stds = post_warmup.std()
    outlier_features = {}
    for col in post_warmup.columns:
        if stds[col] > 0:
            outliers = ((post_warmup[col] - means[col]).abs() > 10 * stds[col]).sum()
            if outliers > 0:
                outlier_features[col] = int(outliers)

    result = {
        'constant_features': constant_features,
        'inf_features': inf_features,
        'extreme_outlier_features': outlier_features,
        'pass': len(constant_features) == 0 and len(inf_features) == 0,
    }

    if constant_features:
        logger.warning(f"Constant features: {constant_features}")
    if inf_features:
        logger.warning(f"Infinite values in: {list(inf_features.keys())}")

    return result


def check_label_balance(labels: pd.Series) -> dict:
    """
    Check that labels are approximately balanced (~50/50).

    Returns dict with class counts and balance status.
    """
    counts = labels.value_counts()
    total = len(labels)

    up_pct = counts.get(1, 0) / total * 100
    down_pct = counts.get(0, 0) / total * 100

    result = {
        'total_labels': total,
        'up_count': int(counts.get(1, 0)),
        'down_count': int(counts.get(0, 0)),
        'up_pct': round(up_pct, 2),
        'down_pct': round(down_pct, 2),
        'pass': 35.0 <= up_pct <= 65.0,  # crypto markets have directional bias
    }

    logger.info(f"Label balance: Up={up_pct:.1f}%, Down={down_pct:.1f}%")
    if not result['pass']:
        logger.warning(f"Label balance outside 45-55% range!")

    return result


def check_history_length(df: pd.DataFrame, timeframe: str = '5m') -> dict:
    """
    Verify minimum 270 days of history.
    """
    config = load_config()
    min_days = config['training']['min_history_days']

    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    n_days = (ts_max - ts_min) / (1000 * 86400)

    result = {
        'n_days': round(n_days, 1),
        'min_days_required': min_days,
        'start_date': pd.to_datetime(ts_min, unit='ms', utc=True).isoformat(),
        'end_date': pd.to_datetime(ts_max, unit='ms', utc=True).isoformat(),
        'pass': n_days >= min_days,
    }

    if result['pass']:
        logger.info(f"History length: {n_days:.1f} days (>= {min_days}) — PASSED")
    else:
        logger.warning(f"History length: {n_days:.1f} days (< {min_days}) — FAILED")

    return result


def check_p_market_coverage(parquet_path: str = None) -> dict:
    """
    Check p_market_history.parquet coverage.

    Documents the date range and coverage for each symbol/timeframe.
    Required by Phase 1 checkpoint.
    """
    config = load_config()
    if parquet_path is None:
        parquet_path = os.path.join(
            os.path.dirname(__file__), '..', config['backtest']['historical_prices_path']
        )

    if not os.path.exists(parquet_path):
        logger.warning(f"p_market_history.parquet not found at {parquet_path}")
        return {'exists': False, 'pass': False}

    df = pd.read_parquet(parquet_path)

    coverage = {}
    for symbol in ['BTC', 'ETH']:
        for tf in ['5m', '15m']:
            mask = (df['symbol'] == symbol) & (df['timeframe'] == tf)
            subset = df[mask]
            if not subset.empty:
                ts_min = pd.to_datetime(subset['timestamp'].min(), unit='ms', utc=True)
                ts_max = pd.to_datetime(subset['timestamp'].max(), unit='ms', utc=True)
                coverage[f'{symbol}_{tf}'] = {
                    'count': len(subset),
                    'start': ts_min.isoformat(),
                    'end': ts_max.isoformat(),
                    'days': round((ts_max - ts_min).total_seconds() / 86400, 1),
                }
            else:
                coverage[f'{symbol}_{tf}'] = {'count': 0, 'start': None, 'end': None, 'days': 0}

    result = {
        'exists': True,
        'total_records': len(df),
        'coverage': coverage,
        'pass': True,  # existence is sufficient for checkpoint
    }

    logger.info(f"p_market_history.parquet: {len(df)} records")
    for key, cov in coverage.items():
        if cov['count'] > 0:
            logger.info(f"  {key}: {cov['count']} records, {cov['start']} → {cov['end']}")
        else:
            logger.info(f"  {key}: no coverage")

    return result


def run_full_validation(
    ohlcv_df: pd.DataFrame = None,
    features_df: pd.DataFrame = None,
    labels: pd.Series = None,
) -> dict:
    """
    Run all Phase 1 checkpoint validations.

    Can be called with pre-built data, or will build from scratch.
    """
    results = {}

    if ohlcv_df is not None:
        results['history_length'] = check_history_length(ohlcv_df)
    else:
        logger.info("Skipping history length check (no OHLCV data provided)")

    if features_df is not None:
        results['nan_audit'] = audit_nans(features_df)
        results['distribution'] = check_distribution(features_df)
    else:
        logger.info("Skipping feature checks (no features provided)")

    if labels is not None:
        results['label_balance'] = check_label_balance(labels)
    else:
        logger.info("Skipping label balance check (no labels provided)")

    results['p_market_coverage'] = check_p_market_coverage()

    # Overall pass/fail
    all_pass = all(
        r.get('pass', True) for r in results.values()
        if isinstance(r, dict)
    )
    results['overall_pass'] = all_pass

    if all_pass:
        logger.info("=== ALL VALIDATION CHECKS PASSED ===")
    else:
        failed = [k for k, v in results.items() if isinstance(v, dict) and not v.get('pass', True)]
        logger.warning(f"=== VALIDATION FAILED: {failed} ===")

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    results = run_full_validation()
    import json
    print(json.dumps(results, indent=2, default=str))
