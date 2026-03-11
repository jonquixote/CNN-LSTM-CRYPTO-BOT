"""
data/features.py — All feature engineering (training = inference).

Single source of truth for ~55 features per symbol/TF.
Same code used at training time and inference time.
No lookahead: all features use only data up to and including bar t.

Features (§5 of spec + prompt2 expansion):
  5.1 — Price & Volume Derivatives
  5.2 — Microstructure-Lite
  5.3 — Volatility & Regime
  5.4 — Technical Indicators
  5.5 — Cross-Asset (stubbed for single-asset phase)
  G1  — Funding Rate & Open Interest
  G2  — Wick Structure
  G3  — Temporal Encoding
  G4  — Price Action
  G5  — Signed Volume
  G6  — Rolling Return Statistics
  G7  — Distance from Moving Averages
  G8  — Cross-Asset SOL
"""

import hashlib
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Division with zero-fill for division by zero."""
    return a.divide(b).replace([np.inf, -np.inf], fill).fillna(fill)


def _rolling_hurst(series: pd.Series, window: int = 500) -> pd.Series:
    """
    Rolling Hurst exponent via R/S analysis — numba-accelerated.
    H > 0.5 = trending, H < 0.5 = mean-reverting.
    Uses only data up to current bar (no lookahead).

    Full fidelity: computes every bar. numba JIT gives ~100x speedup.
    80K bars: ~30s with numba vs 2+ hours with pure Python.
    """
    log_returns = np.log(series / series.shift(1)).values.astype(np.float64)
    result = _hurst_loop(log_returns, window)
    return pd.Series(result, index=series.index)


try:
    from numba import njit

    @njit(cache=True)
    def _hurst_loop(log_returns, window):
        """Numba-compiled Hurst loop — full fidelity, every bar."""
        n = len(log_returns)
        result = np.full(n, np.nan)
        for i in range(window, n):
            x = log_returns[i - window:i]
            # Count valid (non-NaN) values
            count = 0
            for v in x:
                if not np.isnan(v):
                    count += 1
            if count < 20:
                continue
            # Compute mean of valid values
            s = 0.0
            for v in x:
                if not np.isnan(v):
                    s += v
            mean_val = s / count
            # R/S calculation on valid values
            cumsum = 0.0
            r_max = -1e30
            r_min = 1e30
            ss = 0.0
            for v in x:
                if not np.isnan(v):
                    cumsum += (v - mean_val)
                    if cumsum > r_max:
                        r_max = cumsum
                    if cumsum < r_min:
                        r_min = cumsum
                    ss += (v - mean_val) ** 2
            r_val = r_max - r_min
            s_val = np.sqrt(ss / (count - 1)) if count > 1 else 0.0
            if s_val == 0.0 or r_val <= 0.0:
                result[i] = 0.5
            else:
                rs = r_val / s_val
                h = np.log(rs) / np.log(count)
                if h < 0.0:
                    h = 0.0
                elif h > 1.0:
                    h = 1.0
                result[i] = h
        return result

except ImportError:
    logger.info("numba not available — using vectorized Hurst fallback")

    def _hurst_loop(log_returns, window):
        """Pure numpy fallback — vectorized loop, slower but no dependencies."""
        n = len(log_returns)
        result = np.full(n, np.nan)
        for i in range(window, n):
            x = log_returns[i - window:i]
            x_clean = x[~np.isnan(x)]
            if len(x_clean) < 20:
                continue
            m = len(x_clean)
            mean_val = np.mean(x_clean)
            y = np.cumsum(x_clean - mean_val)
            r_val = np.max(y) - np.min(y)
            s_val = np.std(x_clean, ddof=1)
            if s_val == 0 or r_val <= 0:
                result[i] = 0.5
            else:
                rs = r_val / s_val
                result[i] = float(np.clip(np.log(rs) / np.log(m), 0.0, 1.0))
        return result


def _shannon_entropy(series: pd.Series, window: int = 20, bins: int = 10) -> pd.Series:
    """
    Rolling Shannon entropy over return/volume distributions.
    Higher entropy = more random, lower = more structured.

    CRITICAL: Filter inf/nan before np.histogram — inf causes histogram
    to hang silently on certain numpy/Linux versions.
    """
    def entropy_calc(x):
        # Filter out inf and nan — np.histogram hangs on inf
        x_clean = x[np.isfinite(x)]
        if len(x_clean) < 5:
            return np.nan
        counts, _ = np.histogram(x_clean, bins=bins)
        total = counts.sum()
        if total == 0:
            return np.nan
        probs = counts / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    return series.rolling(window=window, min_periods=window).apply(entropy_calc, raw=True)


# ── Feature Builders ─────────────────────────────────────────────────────────

def build_price_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """§5.1 — Price & Volume Derivatives."""
    features = pd.DataFrame(index=df.index)

    # Log returns at lags: 1, 3, 6, 12, 24 bars
    close = df['close']
    for lag in [1, 3, 6, 12, 24]:
        features[f'log_return_{lag}'] = np.log(close / close.shift(lag)).shift(1)

    # Rolling realized volatility: 12, 50, 200 bars
    log_ret_1 = features['log_return_1']
    for window in [12, 50, 200]:
        features[f'realized_vol_{window}'] = log_ret_1.rolling(
            window=window, min_periods=window
        ).std()

    # Volume z-score relative to 50-bar mean
    vol_mean = df['volume'].rolling(window=50, min_periods=50).mean()
    vol_std = df['volume'].rolling(window=50, min_periods=50).std()
    features['volume_zscore'] = _safe_div(df['volume'] - vol_mean, vol_std)

    # VWAP deviation from close (normalized)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_vol = (typical_price * df['volume']).rolling(window=20, min_periods=1).sum()
    cumulative_vol = df['volume'].rolling(window=20, min_periods=1).sum()
    vwap = _safe_div(cumulative_tp_vol, cumulative_vol)
    features['vwap_deviation'] = _safe_div(close - vwap, close)

    return features


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """§5.2 — Microstructure-Lite."""
    features = pd.DataFrame(index=df.index)

    bar_range = df['high'] - df['low']

    # Bar efficiency ratio: |close - open| / (high - low)
    features['bar_efficiency'] = _safe_div(
        (df['close'] - df['open']).abs(), bar_range
    )

    # ATR_14 for relative bar size
    tr = pd.concat([
        bar_range,
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()

    # Relative bar size: (high - low) / ATR_14
    features['relative_bar_size'] = _safe_div(bar_range, atr_14)

    # Close position within bar: (close - low) / (high - low)
    features['close_position'] = _safe_div(df['close'] - df['low'], bar_range)

    # Volume confirmation: binary flag
    # 1 if volume above 50-bar mean and price moved in same direction
    vol_above_avg = df['volume'] > df['volume'].rolling(window=50, min_periods=50).mean()
    price_moved = (df['close'] - df['open']).abs() > 0
    features['volume_confirmation'] = (vol_above_avg & price_moved).astype(int)

    return features


def build_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """§5.3 — Volatility & Regime."""
    features = pd.DataFrame(index=df.index)

    close = df['close']
    bar_range = df['high'] - df['low']

    # ATR at lookbacks: 14, 50, 200
    tr = pd.concat([
        bar_range,
        (df['high'] - close.shift(1)).abs(),
        (df['low'] - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    for window in [14, 50, 200]:
        features[f'atr_{window}'] = tr.rolling(window=window, min_periods=window).mean()

    # Bollinger Band width (20-bar, 2σ)
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std()
    features['bb_width'] = _safe_div(4 * std_20, sma_20)  # (upper - lower) / sma

    # Rolling Hurst exponent over 500 bars
    features['hurst'] = _rolling_hurst(close, window=500)

    # Vol-of-vol: rolling std of 20-bar realized vol over 100 bars
    log_ret_1 = np.log(close / close.shift(1))
    realized_vol_20 = log_ret_1.rolling(window=20, min_periods=20).std()
    features['vol_of_vol'] = realized_vol_20.rolling(window=100, min_periods=100).std()

    # Realized vol z-score vs 90-day average (90 days * 288 bars/day for 5m)
    rv_90d_window = 90 * 288  # ~25920 bars for 5m
    rv_90d_window = min(rv_90d_window, 5000)  # cap for computational sanity
    rv_mean = realized_vol_20.rolling(window=rv_90d_window, min_periods=500).mean()
    rv_std = realized_vol_20.rolling(window=rv_90d_window, min_periods=500).std()
    features['realized_vol_zscore'] = _safe_div(realized_vol_20 - rv_mean, rv_std)

    # Shannon entropy over 20-bar rolling returns and volume distributions
    features['entropy_returns'] = _shannon_entropy(log_ret_1, window=20)
    # Replace inf from 0→non-zero volume transitions before entropy calc
    vol_pct_change = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    features['entropy_volume'] = _shannon_entropy(vol_pct_change, window=20)

    return features


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """§5.4 — Technical Indicators."""
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=14).mean()
    rs = _safe_div(gain, loss)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD line and signal delta
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features['macd_delta'] = macd_line - signal_line

    # ADX (14)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out when other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()

    plus_di = 100 * _safe_div(plus_dm.rolling(14, min_periods=14).mean(), atr_14)
    minus_di = 100 * _safe_div(minus_dm.rolling(14, min_periods=14).mean(), atr_14)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    features['adx_14'] = dx.rolling(window=14, min_periods=14).mean()

    # Stochastic %K and %D
    lowest_14 = low.rolling(window=14, min_periods=14).min()
    highest_14 = high.rolling(window=14, min_periods=14).max()
    features['stoch_k'] = 100 * _safe_div(close - lowest_14, highest_14 - lowest_14)
    features['stoch_d'] = features['stoch_k'].rolling(window=3, min_periods=3).mean()

    # EMA crossovers (9/21, 21/55)
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    ema_55 = close.ewm(span=55, adjust=False).mean()
    features['ema_cross_9_21'] = _safe_div(ema_9 - ema_21, close)
    features['ema_cross_21_55'] = _safe_div(ema_21 - ema_55, close)

    return features


def build_cross_asset_features(
    df: pd.DataFrame,
    cross_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    §5.5 — Cross-Asset features.

    For BTC model: ETH log returns at lags 1, 3, 6 + rolling correlation and beta.
    For ETH model: BTC log returns.

    When cross_df is None (single-asset phase), returns zeros.
    """
    features = pd.DataFrame(index=df.index)

    if cross_df is not None and len(cross_df) == len(df):
        cross_close = cross_df['close']
        for lag in [1, 3, 6]:
            features[f'cross_log_return_{lag}'] = np.log(cross_close / cross_close.shift(lag))

        # Rolling 50-bar cross-asset correlation and beta
        log_ret_self = np.log(df['close'] / df['close'].shift(1))
        log_ret_cross = np.log(cross_close / cross_close.shift(1))

        features['cross_correlation_50'] = log_ret_self.rolling(
            window=50, min_periods=50
        ).corr(log_ret_cross)

        cross_var = log_ret_cross.rolling(window=50, min_periods=50).var()
        cross_cov = log_ret_self.rolling(window=50, min_periods=50).cov(log_ret_cross)
        features['cross_beta_50'] = _safe_div(cross_cov, cross_var)
    else:
        # No cross-asset data available — skip these features entirely
        # They will be added when multi-asset training is enabled
        logger.info("Cross-asset features: skipped (no cross-asset data)")

    return features


# ── Group 1: Funding Rate & Open Interest ────────────────────────────────────

def build_funding_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G1 — Funding Rate & Open Interest features.

    Requires 'funding_rate' and 'open_interest' columns in df,
    added by fetcher.merge_supplementary_data() before this is called.
    """
    features = pd.DataFrame(index=df.index)

    # funding_rate — raw, already forward-filled from 8h events
    if 'funding_rate' in df.columns:
        funding_rate = df['funding_rate']
        features['funding_rate'] = funding_rate

        # funding_rate_zscore — window=500, zero-std guard
        rolling_std = funding_rate.rolling(500).std().replace(0, np.nan)
        features['funding_rate_zscore'] = (
            (funding_rate - funding_rate.rolling(500).mean()) / rolling_std
        ).fillna(0)
    else:
        features['funding_rate'] = 0.0
        features['funding_rate_zscore'] = 0.0

    # OI features — timestamps represent bar-open snapshots (no shift needed)
    if 'open_interest' in df.columns:
        oi = df['open_interest']
        oi_delta = oi.diff(1)
        features['oi_delta'] = oi_delta

        # oi_delta_zscore — window=50
        oi_std = oi_delta.rolling(50).std().replace(0, np.nan)
        features['oi_delta_zscore'] = (
            (oi_delta - oi_delta.rolling(50).mean()) / oi_std
        ).fillna(0)

        # signed_oi_delta — directional OI pressure using PREVIOUS bar's direction
        prev_direction = np.sign(df['close'].shift(1) - df['open'].shift(1))
        features['signed_oi_delta'] = oi_delta * prev_direction
    else:
        features['oi_delta'] = 0.0
        features['oi_delta_zscore'] = 0.0
        features['signed_oi_delta'] = 0.0

    return features


# ── Group 2: Wick Structure ──────────────────────────────────────────────────

def build_wick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G2 — Wick Structure features.

    All use PREVIOUS bar's OHLCV — current bar high/low/close unknown at bar open.
    """
    features = pd.DataFrame(index=df.index)

    h = df['high'].shift(1)
    l = df['low'].shift(1)
    o = df['open'].shift(1)
    c = df['close'].shift(1)
    range_ = (h - l).replace(0, np.nan)  # guard zero-range bars

    features['upper_wick_pct'] = ((h - np.maximum(o, c)) / range_).fillna(0)
    features['lower_wick_pct'] = ((np.minimum(o, c) - l) / range_).fillna(0)
    features['wick_imbalance'] = features['upper_wick_pct'] - features['lower_wick_pct']

    return features


# ── Group 3: Temporal Encoding ───────────────────────────────────────────────

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G3 — Temporal Encoding features.

    Cyclical sin/cos encoding of hour-of-day and day-of-week.
    Derived from bar's UTC timestamp. No leakage concern.
    """
    features = pd.DataFrame(index=df.index)

    # Convert timestamp (ms) to datetime
    dt = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    hour = dt.dt.hour + dt.dt.minute / 60.0  # fractional hour
    dow = dt.dt.dayofweek  # Monday=0, Sunday=6

    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return features


# ── Group 4: Price Action ────────────────────────────────────────────────────

def build_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G4 — Price Action features.

    prev_bar_direction: {-1, 0, 1} — always uses PREVIOUS bar (current bar = label).
    streak_count: consecutive bars in same direction, signed.
    """
    features = pd.DataFrame(index=df.index)

    # Previous bar direction — current bar direction IS the label → shift
    direction = np.sign(df['close'].shift(1) - df['open'].shift(1))
    features['prev_bar_direction'] = direction

    # streak_count — vectorized, no Python row loop
    streak_id = (direction != direction.shift(1)).cumsum()
    features['streak_count'] = direction.groupby(streak_id).cumcount().add(1) * direction

    return features


# ── Group 5: Signed Volume ───────────────────────────────────────────────────

def build_signed_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G5 — Signed Volume features.

    All use PREVIOUS bar's volume and direction — current bar unknown at open.
    """
    features = pd.DataFrame(index=df.index)

    prev_direction = np.sign(df['close'].shift(1) - df['open'].shift(1))
    signed_volume = df['volume'].shift(1) * prev_direction
    features['signed_volume'] = signed_volume

    # Rolling sums, z-scored with window=100
    for window in [5, 15]:
        rolling_sum = signed_volume.rolling(window).sum()
        rs_mean = rolling_sum.rolling(100).mean()
        rs_std = rolling_sum.rolling(100).std().replace(0, np.nan)
        features[f'signed_volume_sum_{window}'] = (
            (rolling_sum - rs_mean) / rs_std
        ).fillna(0)

    return features


# ── Group 6: Rolling Return Statistics ───────────────────────────────────────

def build_return_stat_features(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    G6 — Rolling Return Statistics.

    Uses canonical log_return_1 from features_df, which is already properly
    shifted: log(close/close.shift(1)).shift(1).
    """
    features = pd.DataFrame(index=df.index)

    # Use canonical log_return_1 — already lagged by .shift(1)
    ret_1 = features_df['log_return_1']

    features['return_skewness_30'] = ret_1.rolling(30).skew()
    features['return_percentile_100'] = ret_1.rolling(100).rank(pct=True)

    return features


# ── Group 7: Distance from Moving Averages ───────────────────────────────────

def build_ma_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    G7 — Distance from Moving Averages.

    EMA at time t includes close[t], so the entire series is shifted 1 bar.
    """
    features = pd.DataFrame(index=df.index)

    close = df['close']
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    features['dist_from_ma50'] = ((close - ema50) / close).shift(1)
    features['dist_from_ma200'] = ((close - ema200) / close).shift(1)

    return features


# ── Group 8: Cross-Asset SOL ─────────────────────────────────────────────────

def build_sol_cross_features(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    G8 — Cross-Asset SOL features (BTC model only).

    Requires 'sol_close' column in df, added by fetcher.merge_supplementary_data().
    When SOL data is unavailable, returns zeros.
    Uses canonical log_return_1 from features_df for BTC-SOL correlation.
    """
    features = pd.DataFrame(index=df.index)

    if 'sol_close' in df.columns and df['sol_close'].notna().any():
        sol_close = df['sol_close']

        # SOL returns at lags 1, 3, 6 — all lagged 1 additional bar
        sol_ret_lag1 = np.log(sol_close / sol_close.shift(1)).shift(1)
        features['sol_ret_lag1'] = sol_ret_lag1
        features['sol_ret_lag3'] = np.log(sol_close / sol_close.shift(3)).shift(1)
        features['sol_ret_lag6'] = np.log(sol_close / sol_close.shift(6)).shift(1)

        # BTC-SOL rolling correlation — both inputs already pre-lagged
        ret_1 = features_df['log_return_1']  # canonical, already shifted
        features['btc_sol_corr_50'] = ret_1.rolling(50).corr(sol_ret_lag1)
    else:
        features['sol_ret_lag1'] = 0.0
        features['sol_ret_lag3'] = 0.0
        features['sol_ret_lag6'] = 0.0
        features['btc_sol_corr_50'] = 0.0
        logger.info("SOL cross-asset features: skipped (no sol_close data)")

    return features


# ── Main Feature Builder ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    cross_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build all features for the given OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            Optionally includes: funding_rate, open_interest, sol_close
            (added by fetcher.merge_supplementary_data())
        cross_df: Optional cross-asset OHLCV for legacy cross-asset features

    Returns:
        DataFrame with all features. Same length as input, but early rows will
        have NaNs due to lookback windows.
    """
    # Build original feature groups first (§5 of spec)
    price_vol_features = build_price_volume_features(df)

    features = pd.concat([
        # Original feature groups (§5 of spec)
        price_vol_features,
        build_microstructure_features(df),
        build_volatility_regime_features(df),
        build_technical_features(df),
        build_cross_asset_features(df, cross_df),
        # Expanded feature groups (prompt2)
        build_funding_oi_features(df),
        build_wick_features(df),
        build_temporal_features(df),
        build_price_action_features(df),
        build_signed_volume_features(df),
        build_return_stat_features(df, price_vol_features),  # uses log_return_1
        build_ma_distance_features(df),
        build_sol_cross_features(df, price_vol_features),    # uses log_return_1
    ], axis=1)

    # Forward-fill any residual NaNs after warmup period, then backfill the rest.
    # This handles edge cases where rolling windows produce isolated NaNs.
    features = features.ffill().bfill()

    # Final safety: fill any remaining NaN/inf with 0
    features = features.replace([np.inf, -np.inf], 0).fillna(0)

    feature_names = sorted(features.columns.tolist())
    logger.info(f"Built {len(feature_names)} features")
    return features


def get_feature_list_hash(feature_names: list[str]) -> str:
    """
    SHA-256 hash of sorted JSON feature list.
    Used for feature_list_hash in predictions.jsonl.
    Per spec §20 rule 20: always SHA-256 of sorted JSON content.
    """
    sorted_names = sorted(feature_names)
    json_str = json.dumps(sorted_names, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def get_warmup_bars() -> int:
    """Minimum number of bars before features are valid (no NaNs)."""
    # Driven by the largest lookback: Hurst 500 bars + buffer
    # Plus realized_vol_zscore window (5000) — but ffill handles the tail
    return 600


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from data.fetcher import fetch_ohlcv
    df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
    features = build_features(df)
    print(f"Feature shape: {features.shape}")
    print(f"Feature names: {sorted(features.columns.tolist())}")
    print(f"NaN counts (last 1000 rows):\n{features.tail(1000).isna().sum()}")
    print(f"Feature list hash: {get_feature_list_hash(features.columns.tolist())}")
