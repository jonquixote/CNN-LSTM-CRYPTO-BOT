"""
tests/test_harness_random_baseline.py — Random baseline validation.

Per spec Phase 3 checkpoint:
    Random baseline → near-zero PnL, ~0.50 hit rate.
    Any other result = harness bug.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.walkforward import generate_folds, WalkForwardFold
from eval.metrics import compute_fold_metrics, _compute_ece


def _make_config():
    """Minimal config for testing."""
    return {
        'training': {
            'train_window_days': 120,
            'embargo_days': 10,
            'val_window_days': 20,
            'test_window_days': 10,
            'fold_step_days': 10,
            'min_folds': 12,
            'min_history_days': 270,
            'scaler': 'RobustScaler',
        },
        'strategy': {
            'edge_threshold': 0.04,
            'min_model_confidence': 0.52,
        },
        'backtest': {
            'use_historical_prices': False,
            'assumed_market_price': 0.50,
            'historical_prices_path': 'data/p_market_history.parquet',
        },
        'costs': {
            'fee_model': 'us_flat',
            'taker_fee_pct': 0.001,
            'maker_rebate_pct': 0.001,
            'min_fee_usdc': 0.0001,
        },
    }


class TestFoldGeneration:
    """Test walk-forward fold generation."""

    def test_fold_count(self):
        """Should generate at least 12 folds with sufficient data."""
        config = _make_config()
        # 270 days * 288 bars/day = 77,760 bars
        n_bars = 80000
        folds = generate_folds(n_bars, bars_per_day=288, config=config)
        assert len(folds) >= 12, "Expected >= 12 folds, got {}".format(len(folds))

    def test_non_overlapping_embargo(self):
        """Embargo must separate training from validation."""
        config = _make_config()
        folds = generate_folds(80000, bars_per_day=288, config=config)

        for fold in folds:
            assert fold.train_end <= fold.embargo_end
            assert fold.embargo_end <= fold.val_start
            assert fold.val_end <= fold.test_start
            # No overlap between train and val (embargo enforced)
            assert fold.train_end + config['training']['embargo_days'] * 288 <= fold.val_start

    def test_fold_sizes(self):
        """Each segment should have correct number of bars."""
        config = _make_config()
        folds = generate_folds(80000, bars_per_day=288, config=config)

        for fold in folds:
            train_size = fold.train_end - fold.train_start
            embargo_size = fold.embargo_end - fold.train_end
            val_size = fold.val_end - fold.val_start
            test_size = fold.test_end - fold.test_start

            assert train_size == 120 * 288
            assert embargo_size == 10 * 288
            assert val_size == 20 * 288
            assert test_size == 10 * 288


class TestRandomBaseline:
    """
    Random baseline must produce near-zero PnL and ~0.50 hit rate.
    Any other result = harness bug.
    """

    def test_random_predictions_near_zero_pnl(self):
        """Random predictions should yield near-zero net PnL."""
        np.random.seed(42)
        config = _make_config()

        n_bars = 5000
        # Random predictions tightly centered at 0.50
        p_up = np.random.uniform(0.45, 0.55, n_bars)
        labels = np.random.randint(0, 2, n_bars)
        # p_market = 0.50 fallback (efficient-market null hypothesis)
        p_market = np.full(n_bars, config['backtest']['assumed_market_price'])

        metrics = compute_fold_metrics(p_up, labels, p_market, config)

        # Random predictions tightly centered at 0.50 against p_market=0.50
        # Very few bets pass edge=0.04 filter, PnL should be small
        assert abs(metrics['net_pnl']) < 500.0, \
            "Random baseline PnL too large: {}".format(metrics['net_pnl'])

    def test_random_predictions_hit_rate_near_50(self):
        """Random predictions should have ~50% hit rate."""
        np.random.seed(42)
        config = _make_config()
        # Lower thresholds to allow more bets through
        config['strategy']['edge_threshold'] = 0.00
        config['strategy']['min_model_confidence'] = 0.00

        n_bars = 10000
        p_up = np.random.uniform(0.3, 0.7, n_bars)
        labels = np.random.randint(0, 2, n_bars)
        p_market = np.full(n_bars, 0.50)

        metrics = compute_fold_metrics(p_up, labels, p_market, config)

        if metrics['n_bets'] > 100:
            assert 0.40 <= metrics['hit_rate'] <= 0.60, \
                "Random baseline hit rate far from 0.50: {}".format(metrics['hit_rate'])

    def test_constant_predictions_no_edge(self):
        """Constant p_up = 0.50 predictions should have zero edge."""
        config = _make_config()
        n_bars = 1000
        p_up = np.full(n_bars, 0.50)
        labels = np.random.randint(0, 2, n_bars)
        p_market = np.full(n_bars, 0.50)

        metrics = compute_fold_metrics(p_up, labels, p_market, config)

        # With edge_threshold = 0.04 and p_model = 0.50, edge = 0.0 < 0.04
        # No bets should pass the filter
        assert metrics['n_bets'] == 0, \
            "Constant 0.50 predictions should produce zero bets, got {}".format(metrics['n_bets'])

    def test_perfect_predictions_positive_pnl(self):
        """Perfect predictions should yield positive PnL (sanity check)."""
        config = _make_config()
        config['strategy']['edge_threshold'] = 0.00
        config['strategy']['min_model_confidence'] = 0.00

        n_bars = 1000
        labels = np.random.randint(0, 2, n_bars)
        # Perfect: p_up = 1.0 when label=1, 0.0 when label=0
        p_up = labels.astype(float)
        p_market = np.full(n_bars, 0.50)

        metrics = compute_fold_metrics(p_up, labels, p_market, config)

        assert metrics['net_pnl'] > 0, \
            "Perfect predictions should yield positive PnL, got {}".format(metrics['net_pnl'])
        assert metrics['hit_rate'] == 1.0, \
            "Perfect predictions should have 100% hit rate"


class TestECE:
    """Test expected calibration error computation."""

    def test_perfect_calibration(self):
        """Perfectly calibrated model should have ECE near zero."""
        np.random.seed(42)
        n = 10000
        p_up = np.random.uniform(0, 1, n)
        # Generate labels that match probabilities
        labels = (np.random.random(n) < p_up).astype(int)
        ece = _compute_ece(p_up, labels, n_bins=20)
        assert ece < 0.05, "Perfect calibration ECE should be < 0.05, got {}".format(ece)

    def test_poor_calibration(self):
        """Poorly calibrated model should have high ECE."""
        n = 10000
        # Always predict 0.9 but labels are 50/50
        p_up = np.full(n, 0.9)
        labels = np.random.randint(0, 2, n)
        ece = _compute_ece(p_up, labels)
        assert ece > 0.3, "Poor calibration ECE should be > 0.3, got {}".format(ece)

    def test_p_market_source_tracking(self):
        """p_market_source should correctly identify historical vs assumed."""
        config = _make_config()

        # All assumed
        p_market = np.full(100, 0.50)
        metrics = compute_fold_metrics(
            np.random.uniform(0.3, 0.7, 100),
            np.random.randint(0, 2, 100),
            p_market, config
        )
        assert metrics['p_market_source'] == 'assumed_0.50'

        # Mostly historical (non-0.50 values)
        p_market_hist = np.random.uniform(0.4, 0.6, 100)
        p_market_hist[p_market_hist == 0.50] = 0.51  # ensure none are exactly 0.50
        config2 = _make_config()
        config2['strategy']['edge_threshold'] = 0.0
        config2['strategy']['min_model_confidence'] = 0.0
        metrics2 = compute_fold_metrics(
            np.random.uniform(0.3, 0.7, 100),
            np.random.randint(0, 2, 100),
            p_market_hist, config2
        )
        assert metrics2['p_market_source'] == 'historical'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
