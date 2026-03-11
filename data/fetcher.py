"""
data/fetcher.py — Binance OHLCV fetch + Chainlink basis validation stub.

Fetches BTC/USDT (and optionally ETH/USDT) 5m candles from Binance.
Validates minimum history requirements (270+ days gap-free).
Chainlink basis validation stub for future implementation.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ── Binance OHLCV ────────────────────────────────────────────────────────────

def get_binance_client() -> ccxt.binance:
    """
    Create Binance client for public OHLCV data.
    API keys are NOT used for OHLCV fetches — they trigger geo-restricted
    authenticated endpoints on US-based VPS. OHLCV is public data.
    """
    # Try international Binance first (public endpoints only)
    try:
        client = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        # Quick test — fetch 1 candle to verify access
        client.fetch_ohlcv('BTC/USDT', '5m', limit=1)
        return client
    except Exception as e:
        logger.warning(f"International Binance failed: {e}")

    # Fallback to Binance US
    try:
        client = ccxt.binanceus({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        client.fetch_ohlcv('BTC/USDT', '5m', limit=1)
        logger.info("Using Binance US fallback")
        return client
    except Exception as e:
        logger.warning(f"Binance US also failed: {e}")

    # Last resort: international Binance with explicit public-only config
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'fetchCurrencies': False,
        },
    })


def fetch_ohlcv(
    symbol: str = 'BTC/USDT',
    timeframe: str = '5m',
    since_days: int = 280,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance.

    Args:
        symbol: Trading pair (e.g. 'BTC/USDT')
        timeframe: Candle timeframe (e.g. '5m')
        since_days: Number of days of history to fetch
        max_retries: Max retries on rate limit / network errors

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        Index is NOT set — timestamp is a column with Unix ms values.
    """
    client = get_binance_client()
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000)

    all_candles = []
    current_since = since_ms

    logger.info(f"Fetching {symbol} {timeframe} OHLCV from {since_days} days ago...")

    while True:
        for attempt in range(max_retries):
            try:
                candles = client.fetch_ohlcv(
                    symbol, timeframe, since=current_since, limit=1000
                )
                break
            except (ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
                logger.warning(f"Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Failed to fetch OHLCV after {max_retries} retries")

        if not candles:
            break

        all_candles.extend(candles)
        current_since = candles[-1][0] + 1  # next ms after last candle

        # Stop if we've reached current time
        if candles[-1][0] >= int(datetime.now(timezone.utc).timestamp() * 1000) - 300_000:
            break

        time.sleep(0.1)  # respect rate limits

    if not all_candles:
        raise RuntimeError(f"No OHLCV data returned for {symbol} {timeframe}")

    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    # Remove duplicates (overlapping fetches)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
    return df


def validate_history(df: pd.DataFrame, timeframe: str = '5m', min_days: int = 270) -> dict:
    """
    Validate OHLCV data quality: minimum history, gaps, data integrity.

    Returns:
        dict with keys: valid, n_bars, n_days, n_gaps, gap_details
    """
    config = load_config()
    min_days = config['training']['min_history_days']

    # Calculate expected bar interval in ms
    tf_minutes = {'5m': 5, '15m': 15, '1h': 60}
    interval_ms = tf_minutes.get(timeframe, 5) * 60 * 1000

    # Check history length
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    n_days = (ts_max - ts_min) / (1000 * 86400)

    # Check gaps
    diffs = df['timestamp'].diff().dropna()
    gaps = diffs[diffs > interval_ms]
    gap_details = []
    for idx in gaps.index:
        gap_start = df.loc[idx - 1, 'timestamp']
        gap_end = df.loc[idx, 'timestamp']
        gap_minutes = (gap_end - gap_start) / 60000
        gap_details.append({
            'start': datetime.fromtimestamp(gap_start / 1000, tz=timezone.utc).isoformat(),
            'end': datetime.fromtimestamp(gap_end / 1000, tz=timezone.utc).isoformat(),
            'gap_minutes': gap_minutes,
        })

    result = {
        'valid': n_days >= min_days and len(gaps) == 0,
        'n_bars': len(df),
        'n_days': round(n_days, 1),
        'n_gaps': len(gaps),
        'gap_details': gap_details,
    }

    if not result['valid']:
        if n_days < min_days:
            logger.warning(f"Insufficient history: {n_days:.1f} days (need {min_days})")
        if len(gaps) > 0:
            logger.warning(f"Found {len(gaps)} gaps in data")

    return result


# ── Chainlink Basis Validation ────────────────────────────────────────────────
# Training uses Binance OHLCV. Polymarket settles on Chainlink BTC/USD CX-Price
# Data Stream. Chainlink can lag during fast markets due to aggregation.
# This basis risk is unhedgeable — monitor daily.

def validate_chainlink_basis(
    df: pd.DataFrame,
    lookback_days: int = 7,
    warn_bps: int = 15,
) -> dict:
    """
    Stub: Validate Chainlink vs Binance price basis.

    In production, this compares Binance close prices against Chainlink
    BTC/USD feed prices at corresponding timestamps. The basis difference
    is unhedgeable but must be monitored.

    Args:
        df: Binance OHLCV DataFrame
        lookback_days: Number of recent days to check
        warn_bps: Basis warning threshold in basis points

    Returns:
        dict with: mean_basis_bps, max_basis_bps, alert (bool), note
    """
    config = load_config()
    warn_bps = config['chainlink']['basis_warn_bps']
    lookback_days = config['chainlink']['basis_lookback_days']

    # STUB: Chainlink on-chain price feed integration not yet implemented.
    # Returns zero basis as placeholder. Real implementation will:
    # 1. Fetch Chainlink BTC/USD prices from Polygon chain
    # 2. Align timestamps with Binance closes
    # 3. Compute abs(binance_close - chainlink_price) / binance_close * 10000 (bps)
    logger.info("Chainlink basis validation: STUB — returning zero basis")

    return {
        'mean_basis_bps': 0.0,
        'max_basis_bps': 0.0,
        'alert': False,
        'note': 'STUB: Chainlink feed not yet integrated. Basis assumed zero.',
    }


# ── Convenience ───────────────────────────────────────────────────────────────

def fetch_and_validate(
    symbol: str = 'BTC/USDT',
    timeframe: str = '5m',
    since_days: int = 280,
) -> tuple[pd.DataFrame, dict]:
    """Fetch OHLCV and run validation. Returns (df, validation_result)."""
    df = fetch_ohlcv(symbol, timeframe, since_days)
    validation = validate_history(df, timeframe)
    return df, validation


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df, val = fetch_and_validate()
    print(f"Bars: {val['n_bars']}, Days: {val['n_days']}, Gaps: {val['n_gaps']}, Valid: {val['valid']}")
    basis = validate_chainlink_basis(df)
    print(f"Chainlink basis: {basis}")
