"""
Script to inspect Jon-Becker dataset and build p_market_history.parquet.
Run on VPS: python build_historical_prices.py
"""
import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MARKETS_DIR = '/root/cnn_lstm_v1/data/jon_becker/data/polymarket/markets/'
TRADES_DIR = '/root/cnn_lstm_v1/data/jon_becker/data/polymarket/trades/'
OUTPUT_PATH = '/root/cnn_lstm_v1/data/p_market_history.parquet'


def load_all_markets():
    """Load all market parquet files."""
    files = sorted([f for f in os.listdir(MARKETS_DIR)
                    if f.endswith('.parquet') and not f.startswith('._')])
    logger.info("Loading {} market files...".format(len(files)))
    dfs = [pd.read_parquet(os.path.join(MARKETS_DIR, f)) for f in files]
    markets = pd.concat(dfs, ignore_index=True)
    logger.info("Total markets: {}".format(len(markets)))
    return markets


def find_updown_markets(markets):
    """Find BTC/ETH 5m/15m up/down markets."""
    results = {}

    for symbol, patterns in [
        ('BTC', ['BTC', 'Bitcoin']),
        ('ETH', ['ETH', 'Ethereum']),
    ]:
        sym_mask = markets['question'].str.contains(
            '|'.join(patterns), case=False, na=False
        )
        updown_mask = markets['question'].str.contains(
            'up or down|Up or Down|go up|go down', case=False, na=False
        )

        # Also check for 5-minute and 15-minute patterns
        for tf_label, tf_patterns in [
            ('5m', ['5 minute', '5-minute', '5min', '5 min']),
            ('15m', ['15 minute', '15-minute', '15min', '15 min']),
        ]:
            tf_mask = markets['question'].str.contains(
                '|'.join(tf_patterns), case=False, na=False
            )
            matched = markets[sym_mask & (updown_mask | tf_mask)]
            key = "{}_{}".format(symbol, tf_label)
            results[key] = matched
            logger.info("{}: {} markets found".format(key, len(matched)))

            if len(matched) > 0:
                # Show sample
                sample = matched.head(3)
                for _, row in sample.iterrows():
                    logger.info("  Q: {}".format(row['question'][:120]))
                    logger.info("  Prices: {}".format(row.get('outcome_prices', 'N/A')))

    return results


def extract_market_prices(markets_df):
    """
    Extract p_market_up from outcome_prices for each market.
    Returns DataFrame with: timestamp, p_market_up, p_market_down
    """
    records = []

    for _, row in markets_df.iterrows():
        try:
            # Parse outcome prices
            prices = row.get('outcome_prices')
            if prices is None:
                continue

            # outcome_prices may be a JSON string like "[\"0.51\", \"0.49\"]"
            if isinstance(prices, str):
                prices = json.loads(prices)

            if isinstance(prices, list) and len(prices) >= 2:
                p_up = float(prices[0])  # Usually first outcome is Up/Yes
                p_down = float(prices[1])
            else:
                continue

            # Parse timestamp from created_at or end_date
            ts_str = row.get('end_date') or row.get('created_at')
            if ts_str is None:
                continue

            if isinstance(ts_str, str):
                dt = pd.to_datetime(ts_str, utc=True)
            else:
                dt = pd.to_datetime(ts_str, utc=True)

            ts_ms = int(dt.timestamp() * 1000)

            # Detect symbol and timeframe from question
            question = str(row.get('question', ''))
            symbol = 'BTC'
            if 'ETH' in question.upper() or 'ETHEREUM' in question.upper():
                symbol = 'ETH'

            timeframe = '5m'
            if '15' in question:
                timeframe = '15m'

            records.append({
                'timestamp': ts_ms,
                'symbol': symbol,
                'timeframe': timeframe,
                'p_market_up': round(p_up, 6),
                'p_market_down': round(p_down, 6),
                'market_id': row.get('id', ''),
                'source': 'jon_becker',
            })

        except Exception as e:
            continue

    return pd.DataFrame(records)


def main():
    # Load markets
    markets = load_all_markets()

    # Find relevant markets
    updown = find_updown_markets(markets)

    # Combine all relevant markets
    all_relevant = pd.concat(updown.values(), ignore_index=True)
    all_relevant = all_relevant.drop_duplicates(subset='id')
    logger.info("Total unique relevant markets: {}".format(len(all_relevant)))

    # If we found specific 5m/15m markets, use those
    # Otherwise fall back to broader BTC/ETH up/down search
    if len(all_relevant) < 100:
        logger.info("Few specific markets found, widening search...")
        btc_mask = markets['question'].str.contains('BTC|Bitcoin', case=False, na=False)
        eth_mask = markets['question'].str.contains('ETH|Ethereum', case=False, na=False)
        updown_broad = markets['question'].str.contains(
            'up|down|higher|lower|above|below|price',
            case=False, na=False
        )
        all_relevant = markets[(btc_mask | eth_mask) & updown_broad]
        all_relevant = all_relevant.drop_duplicates(subset='id')
        logger.info("Broadened to {} markets".format(len(all_relevant)))

    # Extract prices
    logger.info("Extracting market prices...")
    prices = extract_market_prices(all_relevant)
    logger.info("Extracted {} price records".format(len(prices)))

    if len(prices) > 0:
        prices = prices.sort_values('timestamp').reset_index(drop=True)

        # Coverage summary
        for symbol in ['BTC', 'ETH']:
            for tf in ['5m', '15m']:
                mask = (prices['symbol'] == symbol) & (prices['timeframe'] == tf)
                subset = prices[mask]
                if len(subset) > 0:
                    ts_min = pd.to_datetime(subset['timestamp'].min(), unit='ms', utc=True)
                    ts_max = pd.to_datetime(subset['timestamp'].max(), unit='ms', utc=True)
                    logger.info("{} {}: {} records, {} to {}".format(
                        symbol, tf, len(subset),
                        ts_min.strftime('%Y-%m-%d'),
                        ts_max.strftime('%Y-%m-%d')
                    ))
                else:
                    logger.info("{} {}: no coverage".format(symbol, tf))

    # Write output
    prices.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Written to: {}".format(OUTPUT_PATH))
    logger.info("Total records: {}".format(len(prices)))


if __name__ == '__main__':
    main()
