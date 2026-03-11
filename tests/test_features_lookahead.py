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
        """Should have ~55 features after expansion."""
        features = build_features(synthetic_data)
        n_features = len(features.columns)
        assert 50 <= n_features <= 70, f"Expected 50-70 features, got {n_features}"

    def test_same_length_as_input(self, synthetic_data):
        """Feature DataFrame must have same length as input."""
        features = build_features(synthetic_data)
        assert len(features) == len(synthetic_data)


class TestNewFeatureSmoke:
    """Smoke tests for prompt2 expanded features."""

    @pytest.fixture
    def synthetic_data(self):
        df = _generate_synthetic_ohlcv(n_bars=2000)
        # Add supplementary columns that new features expect
        df['funding_rate'] = np.random.normal(0.0001, 0.0005, len(df))
        df['open_interest'] = np.abs(np.random.normal(50000, 5000, len(df))) * 1e6
        df['sol_close'] = 150 * np.exp(np.cumsum(np.random.normal(0, 0.003, len(df))))
        return df

    def test_all_new_feature_names_present(self, synthetic_data):
        """All new feature names from prompt2 must be present."""
        features = build_features(synthetic_data)
        expected = [
            # Group 1
            'funding_rate', 'funding_rate_zscore', 'oi_delta',
            'oi_delta_zscore', 'signed_oi_delta',
            # Group 2
            'upper_wick_pct', 'lower_wick_pct', 'wick_imbalance',
            # Group 3
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            # Group 4
            'prev_bar_direction', 'streak_count',
            # Group 5
            'signed_volume', 'signed_volume_sum_5', 'signed_volume_sum_15',
            # Group 6
            'return_skewness_30', 'return_percentile_100',
            # Group 7
            'dist_from_ma50', 'dist_from_ma200',
            # Group 8
            'sol_ret_lag1', 'sol_ret_lag3', 'sol_ret_lag6', 'btc_sol_corr_50',
        ]
        missing = [f for f in expected if f not in features.columns]
        assert len(missing) == 0, f"Missing features: {missing}"

    def test_no_nan_after_warmup(self, synthetic_data):
        """No NaN values in new features after warmup."""
        features = build_features(synthetic_data)
        warmup = get_warmup_bars()
        post_warmup = features.iloc[warmup:]
        nan_counts = post_warmup.isna().sum()
        problem_cols = nan_counts[nan_counts > 0]
        assert len(problem_cols) == 0, f"NaN after warmup: {problem_cols.to_dict()}"

    def test_upper_wick_pct_range(self, synthetic_data):
        """upper_wick_pct must be in [0, 1]."""
        features = build_features(synthetic_data)
        warmup = get_warmup_bars()
        col = features['upper_wick_pct'].iloc[warmup:]
        assert col.min() >= -1e-10, f"upper_wick_pct min={col.min()}"
        assert col.max() <= 1 + 1e-10, f"upper_wick_pct max={col.max()}"

    def test_lower_wick_pct_range(self, synthetic_data):
        """lower_wick_pct must be in [0, 1]."""
        features = build_features(synthetic_data)
        warmup = get_warmup_bars()
        col = features['lower_wick_pct'].iloc[warmup:]
        assert col.min() >= -1e-10, f"lower_wick_pct min={col.min()}"
        assert col.max() <= 1 + 1e-10, f"lower_wick_pct max={col.max()}"

    def test_temporal_features_range(self, synthetic_data):
        """Temporal sin/cos features must be in [-1, 1]."""
        features = build_features(synthetic_data)
        for col_name in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
            col = features[col_name]
            assert col.min() >= -1 - 1e-10, f"{col_name} min={col.min()}"
            assert col.max() <= 1 + 1e-10, f"{col_name} max={col.max()}"

    def test_return_percentile_range(self, synthetic_data):
        """return_percentile_100 must be in [0, 1]."""
        features = build_features(synthetic_data)
        warmup = get_warmup_bars()
        col = features['return_percentile_100'].iloc[warmup:]
        assert col.min() >= -1e-10, f"return_percentile_100 min={col.min()}"
        assert col.max() <= 1 + 1e-10, f"return_percentile_100 max={col.max()}"

    def test_funding_rate_zscore_no_inf(self, synthetic_data):
        """funding_rate_zscore must have no inf values."""
        features = build_features(synthetic_data)
        assert not np.isinf(features['funding_rate_zscore']).any(), \
            "funding_rate_zscore contains inf values"

    def test_no_inf_in_any_new_feature(self, synthetic_data):
        """No infinite values in any new feature."""
        features = build_features(synthetic_data)
        inf_mask = np.isinf(features.select_dtypes(include=[np.number]))
        inf_counts = inf_mask.sum()
        problem_cols = inf_counts[inf_counts > 0]
        assert len(problem_cols) == 0, f"Inf values in: {problem_cols.to_dict()}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

