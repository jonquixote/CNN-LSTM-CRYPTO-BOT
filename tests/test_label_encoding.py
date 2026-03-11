"""
tests/test_label_encoding.py — Label encoding validation.

Spec rules enforced:
  - index 0 = Down, index 1 = Up
  - Returns int, not string
  - Never +1/-1
  - ValueError on final bar (leakage guard)
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from labels.direction import label_bar, label_series, validate_labels


class TestLabelBar:
    """Test label_bar() — the core labeling function."""

    def test_up_returns_1(self):
        """close >= open → 1 (Up)."""
        assert label_bar(100.0, 110.0) == 1

    def test_down_returns_0(self):
        """close < open → 0 (Down)."""
        assert label_bar(100.0, 90.0) == 0

    def test_equal_returns_1(self):
        """close == open → 1 (Up). Polymarket: 'Up if closing price >= opening'."""
        assert label_bar(100.0, 100.0) == 1

    def test_return_type_is_int(self):
        """Must return int, never string."""
        result_up = label_bar(100.0, 110.0)
        result_down = label_bar(100.0, 90.0)
        assert isinstance(result_up, int), f"Expected int, got {type(result_up)}"
        assert isinstance(result_down, int), f"Expected int, got {type(result_down)}"

    def test_never_returns_negative_one(self):
        """Never +1/-1 encoding. Only 0/1."""
        assert label_bar(100.0, 90.0) == 0
        assert label_bar(100.0, 90.0) != -1

    def test_never_returns_string(self):
        """Never returns 'up' or 'down' strings."""
        result = label_bar(100.0, 110.0)
        assert not isinstance(result, str)

    def test_values_only_0_or_1(self):
        """Only 0 and 1 are valid return values."""
        prices = [(100, 110), (100, 90), (100, 100), (50, 50.01), (50, 49.99)]
        for open_p, close_p in prices:
            result = label_bar(open_p, close_p)
            assert result in {0, 1}, f"label_bar({open_p}, {close_p}) = {result}, expected 0 or 1"

    def test_tiny_up_move(self):
        """Even tiny up move → 1."""
        assert label_bar(100.0, 100.0001) == 1

    def test_tiny_down_move(self):
        """Even tiny down move → 0."""
        assert label_bar(100.0, 99.9999) == 0


class TestLabelSeries:
    """Test label_series() — vectorized label generation."""

    def test_basic_series(self):
        """Basic up/down labeling on a series."""
        opens = pd.Series([100.0, 200.0, 300.0, 400.0])
        closes = pd.Series([110.0, 190.0, 300.0, 410.0])
        labels = label_series(opens, closes, drop_last=False)
        expected = pd.Series([1, 0, 1, 1])
        pd.testing.assert_series_equal(labels, expected, check_names=False)

    def test_drop_last_reduces_length(self):
        """drop_last=True removes final bar (leakage guard)."""
        opens = pd.Series([100.0, 200.0, 300.0])
        closes = pd.Series([110.0, 190.0, 310.0])
        labels = label_series(opens, closes, drop_last=True)
        assert len(labels) == 2, f"Expected 2 labels, got {len(labels)}"

    def test_drop_last_is_default(self):
        """drop_last=True is the default behavior."""
        opens = pd.Series([100.0, 200.0, 300.0])
        closes = pd.Series([110.0, 190.0, 310.0])
        labels = label_series(opens, closes)  # default
        assert len(labels) == 2

    def test_all_labels_are_int(self):
        """All labels in series must be int dtype."""
        opens = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
        closes = pd.Series([110.0, 190.0, 310.0, 390.0, 510.0])
        labels = label_series(opens, closes, drop_last=False)
        assert labels.dtype in [np.int64, np.int32, int], f"Expected int dtype, got {labels.dtype}"

    def test_no_negative_ones(self):
        """Never -1 in labels."""
        opens = pd.Series(np.random.uniform(99, 101, 100))
        closes = pd.Series(np.random.uniform(99, 101, 100))
        labels = label_series(opens, closes, drop_last=False)
        assert -1 not in labels.values, "Found -1 in labels — must use 0/1 encoding"

    def test_length_mismatch_raises(self):
        """Different length opens/closes raises ValueError."""
        opens = pd.Series([100.0, 200.0])
        closes = pd.Series([110.0])
        with pytest.raises(ValueError, match="Length mismatch"):
            label_series(opens, closes)

    def test_empty_series(self):
        """Empty input returns empty output."""
        labels = label_series(pd.Series(dtype=float), pd.Series(dtype=float))
        assert len(labels) == 0

    def test_approximate_balance(self):
        """With random prices, labels should be roughly 50/50."""
        np.random.seed(42)
        n = 10000
        opens = pd.Series(np.random.uniform(99, 101, n))
        closes = pd.Series(np.random.uniform(99, 101, n))
        labels = label_series(opens, closes, drop_last=False)
        up_pct = labels.mean() * 100
        assert 45 <= up_pct <= 55, f"Label balance {up_pct:.1f}% outside 45-55% range"


class TestValidateLabels:
    """Test validate_labels() — assertion helper."""

    def test_valid_labels_pass(self):
        """Valid 0/1 labels raise no error."""
        labels = pd.Series([0, 1, 0, 1, 1, 0])
        validate_labels(labels)  # Should not raise

    def test_negative_one_raises(self):
        """Labels with -1 must raise ValueError."""
        labels = pd.Series([1, -1, 0, 1])
        with pytest.raises(ValueError, match="Invalid label values"):
            validate_labels(labels)

    def test_string_labels_raise(self):
        """String labels must raise ValueError."""
        labels = pd.Series(["up", "down", "up"])
        with pytest.raises(ValueError, match="Labels must be int"):
            validate_labels(labels)

    def test_out_of_range_raises(self):
        """Values other than 0/1 must raise."""
        labels = pd.Series([0, 1, 2])
        with pytest.raises(ValueError, match="Invalid label values"):
            validate_labels(labels)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
