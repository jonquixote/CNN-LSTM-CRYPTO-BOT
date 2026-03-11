"""
data/polymarket_historical.py — Jon-Becker dataset loader → p_market_history.parquet.

Parses the prediction-market-analysis dataset from ~/cnn_lstm_v1/data/jon_becker/
to extract historical Polymarket BTC/ETH Up/Down market prices.
Aligns to Binance 5m bar timestamps and outputs p_market_history.parquet.
"""

import os
import logging
import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def discover_jon_becker_files(data_dir: str) -> list[str]:
    """
    Discover all relevant data files in the Jon-Becker dataset.
    The dataset structure may vary — look for CSV/JSON/parquet files
    containing market price data.
    """
    patterns = ['**/*.csv', '**/*.json', '**/*.parquet', '**/*.jsonl']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    logger.info(f"Discovered {len(files)} data files in Jon-Becker dataset")
    return sorted(files)


def parse_jon_becker_dataset(data_dir: str) -> pd.DataFrame:
    """
    Parse the Jon-Becker prediction-market-analysis dataset.

    Expected: CSV/JSON files with Polymarket market data including:
    - Timestamps or market window identifiers
    - Market prices for Up/Down outcomes
    - Symbol (BTC/ETH) and timeframe (5m/15m)

    Returns DataFrame with columns:
        timestamp (Unix ms), symbol, timeframe, p_market_up, p_market_down, source
    """
    files = discover_jon_becker_files(data_dir)

    if not files:
        logger.warning(f"No data files found in {data_dir}")
        return pd.DataFrame(columns=[
            'timestamp', 'symbol', 'timeframe', 'p_market_up', 'p_market_down', 'source'
        ])

    all_records = []

    for filepath in files:
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.parquet'):
                df = pd.read_parquet(filepath)
            elif filepath.endswith('.json') or filepath.endswith('.jsonl'):
                df = pd.read_json(filepath, lines=filepath.endswith('.jsonl'))
            else:
                continue

            # Try to identify relevant columns
            # The Jon-Becker dataset may use various column names
            # We look for price/probability columns and timestamp columns
            records = _extract_market_prices(df, filepath)
            if records:
                all_records.extend(records)
                logger.info(f"Extracted {len(records)} records from {os.path.basename(filepath)}")

        except Exception as e:
            logger.warning(f"Failed to parse {filepath}: {e}")
            continue

    if not all_records:
        logger.warning("No market price records extracted from Jon-Becker dataset")
        return pd.DataFrame(columns=[
            'timestamp', 'symbol', 'timeframe', 'p_market_up', 'p_market_down', 'source'
        ])

    result = pd.DataFrame(all_records)
    result = result.drop_duplicates(subset=['timestamp', 'symbol', 'timeframe'])
    result = result.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Total extracted market prices: {len(result)}")
    return result


def _extract_market_prices(df: pd.DataFrame, filepath: str) -> list[dict]:
    """
    Extract market price records from a DataFrame.
    Handles various column naming conventions from the Jon-Becker dataset.
    """
    records = []

    # Normalize column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Look for timestamp column
    ts_col = None
    for candidate in ['timestamp', 'ts', 'time', 'datetime', 'date', 'created_at', 'start_time']:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        return records

    # Look for price columns
    price_col = None
    for candidate in ['price', 'market_price', 'yes_price', 'probability', 'prob', 'outcome_price']:
        if candidate in df.columns:
            price_col = candidate
            break

    # Look for symbol/market identification
    symbol_col = None
    for candidate in ['symbol', 'market', 'asset', 'token', 'market_slug', 'question', 'title']:
        if candidate in df.columns:
            symbol_col = candidate
            break

    if price_col is None:
        return records

    for _, row in df.iterrows():
        try:
            # Parse timestamp
            ts_val = row[ts_col]
            if isinstance(ts_val, str):
                dt = pd.to_datetime(ts_val, utc=True)
                ts_ms = int(dt.timestamp() * 1000)
            elif isinstance(ts_val, (int, float)):
                # Could be Unix seconds or ms
                if ts_val > 1e12:
                    ts_ms = int(ts_val)
                else:
                    ts_ms = int(ts_val * 1000)
            else:
                continue

            # Parse price
            p_up = float(row[price_col])
            if p_up < 0 or p_up > 1:
                continue
            p_down = 1.0 - p_up

            # Detect symbol and timeframe from market name or filepath
            symbol, timeframe = _detect_symbol_tf(
                row.get(symbol_col, '') if symbol_col else '',
                filepath
            )

            records.append({
                'timestamp': ts_ms,
                'symbol': symbol,
                'timeframe': timeframe,
                'p_market_up': round(p_up, 6),
                'p_market_down': round(p_down, 6),
                'source': 'jon_becker',
            })
        except (ValueError, TypeError, KeyError):
            continue

    return records


def _detect_symbol_tf(market_str: str, filepath: str) -> tuple[str, str]:
    """Detect symbol and timeframe from market identifier or filepath."""
    combined = f"{market_str} {filepath}".upper()

    symbol = 'BTC'  # default
    if 'ETH' in combined:
        symbol = 'ETH'
    elif 'BTC' in combined or 'BITCOIN' in combined:
        symbol = 'BTC'

    timeframe = '5m'  # default
    if '15M' in combined or '15MIN' in combined:
        timeframe = '15m'
    elif '5M' in combined or '5MIN' in combined:
        timeframe = '5m'

    return symbol, timeframe


def align_to_bars(
    market_prices: pd.DataFrame,
    bar_timestamps_ms: pd.Series,
    symbol: str = 'BTC',
    timeframe: str = '5m',
) -> pd.DataFrame:
    """
    Align historical Polymarket prices to Binance bar timestamps.

    For each bar timestamp, find the closest market price within the bar window.
    Uses merge_asof for efficient timestamp alignment.
    """
    # Filter to symbol/timeframe
    mask = (market_prices['symbol'] == symbol) & (market_prices['timeframe'] == timeframe)
    filtered = market_prices[mask].copy()

    if filtered.empty:
        logger.warning(f"No historical prices for {symbol} {timeframe}")
        return pd.DataFrame(columns=['timestamp', 'p_market_up', 'p_market_down'])

    # Create bar DataFrame
    bars = pd.DataFrame({'timestamp': bar_timestamps_ms})
    bars = bars.sort_values('timestamp').reset_index(drop=True)
    filtered = filtered.sort_values('timestamp').reset_index(drop=True)

    # merge_asof: find closest historical price at or before each bar
    tf_minutes = {'5m': 5, '15m': 15}
    tolerance_ms = tf_minutes.get(timeframe, 5) * 60 * 1000  # within one bar

    aligned = pd.merge_asof(
        bars,
        filtered[['timestamp', 'p_market_up', 'p_market_down']],
        on='timestamp',
        direction='nearest',
        tolerance=tolerance_ms,
    )

    coverage = aligned['p_market_up'].notna().sum()
    total = len(aligned)
    logger.info(
        f"Aligned {coverage}/{total} bars with historical prices "
        f"({coverage/total*100:.1f}% coverage) for {symbol} {timeframe}"
    )

    return aligned


def build_parquet(
    data_dir: str = None,
    output_path: str = None,
) -> str:
    """
    Build p_market_history.parquet from the Jon-Becker dataset.

    Args:
        data_dir: Path to Jon-Becker dataset. Defaults to ~/cnn_lstm_v1/data/jon_becker/
        output_path: Output parquet path. Defaults from config.

    Returns:
        Path to written parquet file, and coverage summary string.
    """
    config = load_config()

    if data_dir is None:
        data_dir = os.path.expanduser('~/cnn_lstm_v1/data/jon_becker/')
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', config['backtest']['historical_prices_path']
        )

    logger.info(f"Parsing Jon-Becker dataset from {data_dir}")
    market_prices = parse_jon_becker_dataset(data_dir)

    if market_prices.empty:
        logger.warning("No market prices found — writing empty parquet")
        market_prices.to_parquet(output_path, index=False)
        return output_path

    # Write raw historical prices
    market_prices.to_parquet(output_path, index=False)
    logger.info(f"Written {len(market_prices)} records to {output_path}")

    # Document coverage
    for symbol in ['BTC', 'ETH']:
        for tf in ['5m', '15m']:
            mask = (market_prices['symbol'] == symbol) & (market_prices['timeframe'] == tf)
            subset = market_prices[mask]
            if not subset.empty:
                ts_min = pd.to_datetime(subset['timestamp'].min(), unit='ms', utc=True)
                ts_max = pd.to_datetime(subset['timestamp'].max(), unit='ms', utc=True)
                logger.info(f"{symbol} {tf}: {len(subset)} prices from {ts_min} to {ts_max}")
            else:
                logger.info(f"{symbol} {tf}: no coverage")

    return output_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = build_parquet()
    print(f"Written to: {path}")
