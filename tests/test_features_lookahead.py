"""
tests/test_features_lookahead.py — No lookahead leakage in features.

Core test: compute features on full dataset vs truncated dataset.
Features for bar t must be identical in both cases.
If they differ, a future bar is influencing a historical feature → leakage.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.features import (
    build_features,
    build_price_volume_features,
    build_microstructure_features,
    build_volatility_regime_features,
    build_technical_features,
    get_warmup_bars,
)


def _generate_synthetic_ohlcv(n_bars: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic but realistic OHLCV data for testing."""
    np.random.seed(seed)

    # Random walk with realistic BTC-like properties
    log_returns = np.random.normal(0, 0.002, n_bars)
    close = 50000 * np.exp(np.cumsum(log_returns))

    # Generate OHLCV from closes
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]

    # High/low spread around close
    spreads = np.abs(np.random.normal(0, 50, n_bars))
    high = np.maximum(open_prices, close) + spreads
    low = np.minimum(open_prices, close) - spreads

    volume = np.abs(np.random.normal(100, 30, n_bars)) * 1e6

    timestamps = np.arange(n_bars) * 300_000  # 5m bars in ms

    return pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })


class TestNoLookahead:
    """
    Test that features at bar t are the same whether computed on
    the full dataset or on a dataset truncated at bar t.

    This is the fundamental anti-leakage test.
    """

    @pytest.fixture
    def synthetic_data(self):
        return _generate_synthetic_ohlcv(n_bars=2000)

    def test_full_vs_truncated_features(self, synthetic_data):
        """
        Compare features computed on full data vs truncated data.
        For a bar in the middle, features must be identical.
        """
        df_full = synthetic_data
        warmup = get_warmup_bars()

        # Pick test bars after warmup
        test_bars = [warmup + 50, warmup + 100, warmup + 200, len(df_full) - 100]

        for bar_idx in test_bars:
            # Full dataset features
            features_full = build_features(df_full)

            # Truncated dataset (only bars up to bar_idx + 1)
            df_truncated = df_full.iloc[:bar_idx + 1].copy().reset_index(drop=True)
            features_truncated = build_features(df_truncated)

            # Compare features at the last bar of truncated set
            # features_full at bar_idx should match features_truncated at its last row
            full_row = features_full.iloc[bar_idx]
            trunc_row = features_truncated.iloc[-1]

            for col in features_full.columns:
                val_full = full_row[col]
                val_trunc = trunc_row[col]

                # Both NaN is OK (warmup period)
                if pd.isna(val_full) and pd.isna(val_trunc):
                    continue

                # One NaN and one not → leakage
                if pd.isna(val_full) != pd.isna(val_trunc):
                    pytest.fail(
                        f"Lookahead DETECTED in '{col}' at bar {bar_idx}: "
                        f"full={val_full}, truncated={val_trunc}"
                    )

                # Values must match within floating point tolerance
                if not np.isclose(val_full, val_trunc, rtol=1e-10, atol=1e-15):
                    pytest.fail(
                        f"Lookahead DETECTED in '{col}' at bar {bar_idx}: "
                        f"full={val_full}, truncated={val_trunc}, "
                        f"diff={abs(val_full - val_trunc)}"
                    )

    def test_price_volume_no_lookahead(self, synthetic_data):
        """Specifically test price/volume derivative features."""
        df = synthetic_data
        bar_idx = 800

        features_full = build_price_volume_features(df)
        features_trunc = build_price_volume_features(df.iloc[:bar_idx + 1].reset_index(drop=True))

        for col in features_full.columns:
            val_full = features_full.iloc[bar_idx][col]
            val_trunc = features_trunc.iloc[-1][col]
            if pd.isna(val_full) and pd.isna(val_trunc):
                continue
            if not np.isclose(val_full, val_trunc, rtol=1e-10, atol=1e-15, equal_nan=True):
                pytest.fail(f"Lookahead in {col}: full={val_full}, trunc={val_trunc}")

    def test_microstructure_no_lookahead(self, synthetic_data):
        """Specifically test microstructure features."""
        df = synthetic_data
        bar_idx = 800

        features_full = build_microstructure_features(df)
        features_trunc = build_microstructure_features(df.iloc[:bar_idx + 1].reset_index(drop=True))

        for col in features_full.columns:
            val_full = features_full.iloc[bar_idx][col]
            val_trunc = features_trunc.iloc[-1][col]
            if pd.isna(val_full) and pd.isna(val_trunc):
                continue
            if not np.isclose(val_full, val_trunc, rtol=1e-10, atol=1e-15, equal_nan=True):
                pytest.fail(f"Lookahead in {col}: full={val_full}, trunc={val_trunc}")

    def test_volatility_regime_no_lookahead(self, synthetic_data):
        """Specifically test volatility/regime features."""
        df = synthetic_data
        bar_idx = 800

        features_full = build_volatility_regime_features(df)
        features_trunc = build_volatility_regime_features(df.iloc[:bar_idx + 1].reset_index(drop=True))

        for col in features_full.columns:
            val_full = features_full.iloc[bar_idx][col]
            val_trunc = features_trunc.iloc[-1][col]
            if pd.isna(val_full) and pd.isna(val_trunc):
                continue
            if not np.isclose(val_full, val_trunc, rtol=1e-10, atol=1e-15, equal_nan=True):
                pytest.fail(f"Lookahead in {col}: full={val_full}, trunc={val_trunc}")

    def test_technical_no_lookahead(self, synthetic_data):
        """Specifically test technical indicator features."""
        df = synthetic_data
        bar_idx = 800

        features_full = build_technical_features(df)
        features_trunc = build_technical_features(df.iloc[:bar_idx + 1].reset_index(drop=True))

        for col in features_full.columns:
            val_full = features_full.iloc[bar_idx][col]
            val_trunc = features_trunc.iloc[-1][col]
            if pd.isna(val_full) and pd.isna(val_trunc):
                continue
            if not np.isclose(val_full, val_trunc, rtol=1e-10, atol=1e-15, equal_nan=True):
                pytest.fail(f"Lookahead in {col}: full={val_full}, trunc={val_trunc}")


class TestFeatureProperties:
    """Test general feature properties."""

    @pytest.fixture
    def synthetic_data(self):
        return _generate_synthetic_ohlcv(n_bars=2000)

    def test_no_inf_values(self, synthetic_data):
        """Features should have no infinite values."""
        features = build_features(synthetic_data)
        inf_mask = np.isinf(features.select_dtypes(include=[np.number]))
        inf_counts = inf_mask.sum()
        problem_cols = inf_counts[inf_counts > 0]
        assert len(problem_cols) == 0, f"Infinite values found in: {problem_cols.to_dict()}"

    def test_no_nan_after_warmup(self, synthetic_data):
        """After warmup period, no NaN values should exist."""
        features = build_features(synthetic_data)
        warmup = get_warmup_bars()
        post_warmup = features.iloc[warmup:]
        nan_counts = post_warmup.isna().sum()
        problem_cols = nan_counts[nan_counts > 0]
        assert len(problem_cols) == 0, f"NaN values after warmup in: {problem_cols.to_dict()}"

    def test_feature_count_range(self, synthetic_data):
        """Should have ~33-40 features per spec."""
        features = build_features(synthetic_data)
        n_features = len(features.columns)
        assert 30 <= n_features <= 45, f"Expected 30-45 features, got {n_features}"

    def test_same_length_as_input(self, synthetic_data):
        """Feature DataFrame must have same length as input."""
        features = build_features(synthetic_data)
        assert len(features) == len(synthetic_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
