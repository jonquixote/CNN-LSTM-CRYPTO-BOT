"""
upload_to_volume.py — Upload dataset and config to Modal Volume.

Fetches BTC 5m OHLCV + supplementary data (funding rate, open interest, SOL),
builds all 55 features (including expanded G1-G8 groups), and uploads to the
'cnn-lstm-data' Modal Volume. Must be run from VPS with Modal token set.

Usage:
    cd ~/cnn_lstm_v1 && source venv/bin/activate
    python upload_to_volume.py
"""

import os
import sys
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    import modal
    import numpy as np
    import pandas as pd

    # ── Step 1: Fetch BTC OHLCV data ─────────────────────────────────────
    logger.info("Fetching BTC/USDT 5m OHLCV data...")
    from data.fetcher import (
        fetch_ohlcv,
        fetch_funding_rate,
        fetch_open_interest,
        fetch_sol_ohlcv,
        merge_supplementary_data,
    )
    df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
    logger.info(f"Fetched {len(df)} BTC bars")

    # ── Step 2: Fetch supplementary data ─────────────────────────────────
    logger.info("Fetching supplementary data (funding rate, OI, SOL)...")

    funding_df = fetch_funding_rate(symbol='BTCUSDT', since_days=280)
    logger.info(f"  Funding rate: {len(funding_df)} records")

    oi_df = fetch_open_interest(symbol='BTCUSDT', period='5m', since_days=30)
    logger.info(f"  Open interest: {len(oi_df)} records")

    sol_df = fetch_sol_ohlcv(timeframe='5m', since_days=280)
    logger.info(f"  SOL OHLCV: {len(sol_df)} bars")

    # ── Step 3: Merge supplementary data onto BTC bars ───────────────────
    logger.info("Merging supplementary data...")
    df = merge_supplementary_data(df, funding_df, oi_df, sol_df)

    # Verify supplementary columns exist
    for col in ['funding_rate', 'open_interest', 'sol_close']:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' after merge — aborting")
        non_null = df[col].notna().sum()
        logger.info(f"  {col}: {non_null}/{len(df)} non-null")

    # Save merged OHLCV to temp parquet
    ohlcv_path = os.path.join(tempfile.gettempdir(), 'btc_5m_ohlcv.parquet')
    df.to_parquet(ohlcv_path)
    logger.info(f"Saved merged OHLCV ({len(df.columns)} cols) to {ohlcv_path}")

    # ── Step 4: Build features ───────────────────────────────────────────
    logger.info("Building features...")
    from data.features import build_features
    features = build_features(df)
    features_path = os.path.join(tempfile.gettempdir(), 'btc_5m_features.parquet')
    features.to_parquet(features_path)
    logger.info(f"Built {features.shape[1]} features, {len(features)} rows")

    # Sanity check: verify key features are present and non-zero
    spot_checks = ['funding_rate', 'hour_sin', 'upper_wick_pct', 'log_return_1',
                   'sol_ret_lag1', 'oi_delta', 'signed_volume', 'dist_from_ma50']
    present = [c for c in spot_checks if c in features.columns]
    missing = [c for c in spot_checks if c not in features.columns]
    if missing:
        raise RuntimeError(f"Missing expected features: {missing}")
    logger.info(f"Spot-check: all {len(present)} key features present ✓")
    logger.info(f"Total features: {features.shape[1]}")

    # ── Step 5: Upload to Modal Volume ───────────────────────────────────
    logger.info("Creating/opening Modal Volume 'cnn-lstm-data'...")
    vol = modal.Volume.from_name("cnn-lstm-data", create_if_missing=True)

    logger.info("Uploading files to Volume...")
    with vol.batch_upload(force=True) as batch:
        batch.put_file("config.yaml", "config.yaml")
        batch.put_file(ohlcv_path, "btc_5m_ohlcv.parquet")
        batch.put_file(features_path, "btc_5m_features.parquet")

        # Upload feature_list.json if it exists
        if os.path.exists("feature_list.json"):
            batch.put_file("feature_list.json", "feature_list.json")
            logger.info("  Uploaded feature_list.json")

    logger.info("Upload complete!")
    logger.info("Volume contents:")
    logger.info("  /data/config.yaml")
    logger.info("  /data/btc_5m_ohlcv.parquet")
    logger.info(f"  /data/btc_5m_features.parquet ({features.shape[1]} features)")

    # Cleanup temp files
    os.unlink(ohlcv_path)
    os.unlink(features_path)
    logger.info("Cleaned up temp files")


if __name__ == '__main__':
    main()
