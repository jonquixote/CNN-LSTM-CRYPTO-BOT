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


# ── Funding Rate ──────────────────────────────────────────────────────────────

def fetch_funding_rate(
    symbol: str = 'BTCUSDT',
    since_days: int = 280,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch funding rate from Binance Futures.

    Funding events occur every 8 hours. Returns a DataFrame with
    columns: timestamp, funding_rate. Forward-fill to bar frequency
    is done in merge_supplementary_data().
    """
    import requests

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=since_days)).timestamp() * 1000)

    all_records = []
    current_start = start_ms

    logger.info(f"Fetching funding rate for {symbol}...")

    while current_start < end_ms:
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    'https://fapi.binance.com/fapi/v1/fundingRate',
                    params={
                        'symbol': symbol,
                        'startTime': current_start,
                        'endTime': end_ms,
                        'limit': 1000,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                logger.warning(f"Funding rate retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
        else:
            logger.warning("Failed to fetch funding rate after retries")
            return pd.DataFrame(columns=['timestamp', 'funding_rate'])

        if not data:
            break

        for record in data:
            all_records.append({
                'timestamp': record['fundingTime'],
                'funding_rate': float(record['fundingRate']),
            })

        current_start = data[-1]['fundingTime'] + 1
        time.sleep(0.2)

    if not all_records:
        logger.warning("No funding rate data returned")
        return pd.DataFrame(columns=['timestamp', 'funding_rate'])

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Fetched {len(df)} funding rate records")
    return df


def fetch_open_interest(
    symbol: str = 'BTCUSDT',
    period: str = '5m',
    since_days: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch open interest history from Binance Futures.

    Available at 5m resolution. Binance limits to ~500 records per request
    and max 30 days history for this endpoint.

    NOTE: Binance openInterestHist endpoint maximum history is 30 days at 5m resolution.
    If re-running features from scratch on data older than 30 days ago, OI will have gaps
    and oi_delta / oi_delta_zscore / signed_oi_delta will be NaN for those bars.
    The NaN handling in features.py fills these with 0 — this is a documented assumption.

    Returns DataFrame with columns: timestamp, open_interest
    """
    import requests

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=min(since_days, 30))).timestamp() * 1000)

    all_records = []
    current_start = start_ms

    logger.info(f"Fetching open interest for {symbol} ({period})...")

    while current_start < end_ms:
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    'https://fapi.binance.com/futures/data/openInterestHist',
                    params={
                        'symbol': symbol,
                        'period': period,
                        'startTime': current_start,
                        'endTime': end_ms,
                        'limit': 500,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                logger.warning(f"Open interest retry {attempt+1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
        else:
            logger.warning("Failed to fetch open interest after retries")
            return pd.DataFrame(columns=['timestamp', 'open_interest'])

        if not data:
            break

        for record in data:
            all_records.append({
                'timestamp': record['timestamp'],
                'open_interest': float(record['sumOpenInterest']),
            })

        current_start = data[-1]['timestamp'] + 1
        time.sleep(0.5)  # This endpoint is more rate-limited

    if not all_records:
        logger.warning("No open interest data returned")
        return pd.DataFrame(columns=['timestamp', 'open_interest'])

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Fetched {len(df)} open interest records")
    return df


def fetch_sol_ohlcv(
    timeframe: str = '5m',
    since_days: int = 280,
) -> pd.DataFrame:
    """Fetch SOL/USDT OHLCV for cross-asset features."""
    return fetch_ohlcv('SOL/USDT', timeframe, since_days)


def merge_supplementary_data(
    df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    oi_df: Optional[pd.DataFrame] = None,
    sol_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge funding rate, open interest, and SOL data onto the main OHLCV DataFrame.

    All supplementary data is aligned to the main bar index via merge_asof
    (forward-fill for funding rate) or left join + ffill (OI, SOL).
    Does not drop any bars from the main DataFrame.

    Args:
        df: Main OHLCV DataFrame with 'timestamp' column
        funding_df: Funding rate DataFrame (timestamp, funding_rate)
        oi_df: Open interest DataFrame (timestamp, open_interest)
        sol_df: SOL OHLCV DataFrame (timestamp, open, high, low, close, volume)

    Returns:
        Main DataFrame with supplementary columns added
    """
    result = df.copy()

    # Funding rate — forward-fill from 8h events onto 5m bars
    if funding_df is not None and len(funding_df) > 0:
        funding_df = funding_df.sort_values('timestamp')
        result = pd.merge_asof(
            result.sort_values('timestamp'),
            funding_df[['timestamp', 'funding_rate']],
            on='timestamp',
            direction='backward',
        )
        result['funding_rate'] = result['funding_rate'].ffill().fillna(0)
        logger.info(f"Merged funding rate ({len(funding_df)} events)")
    else:
        result['funding_rate'] = 0.0
        logger.info("No funding rate data — filled with 0")

    # Open interest — join on timestamp, ffill gaps
    if oi_df is not None and len(oi_df) > 0:
        oi_df = oi_df.sort_values('timestamp')
        result = pd.merge_asof(
            result.sort_values('timestamp'),
            oi_df[['timestamp', 'open_interest']],
            on='timestamp',
            direction='backward',
        )
        result['open_interest'] = result['open_interest'].ffill().fillna(0)
        logger.info(f"Merged open interest ({len(oi_df)} records)")
    else:
        result['open_interest'] = 0.0
        logger.info("No open interest data — filled with 0")

    # SOL close — merge onto BTC bar index, ffill gaps
    if sol_df is not None and len(sol_df) > 0:
        sol_df = sol_df.sort_values('timestamp')
        sol_renamed = sol_df[['timestamp', 'close']].rename(columns={'close': 'sol_close'})
        result = pd.merge_asof(
            result.sort_values('timestamp'),
            sol_renamed,
            on='timestamp',
            direction='backward',
        )
        result['sol_close'] = result['sol_close'].ffill()
        logger.info(f"Merged SOL close ({len(sol_df)} bars)")
    else:
        result['sol_close'] = np.nan
        logger.info("No SOL data — sol_close is NaN")

    # Restore original order
    result = result.sort_values('timestamp').reset_index(drop=True)
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
