"""
upload_to_volume.py — One-time upload of dataset and config to Modal Volume.

Fetches BTC 5m OHLCV data, builds features, and uploads everything to the
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

    # ── Step 1: Fetch OHLCV data ─────────────────────────────────────────
    logger.info("Fetching BTC/USDT 5m OHLCV data...")
    from data.fetcher import fetch_ohlcv
    df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
    logger.info(f"Fetched {len(df)} bars")

    # Save OHLCV to temp parquet
    ohlcv_path = os.path.join(tempfile.gettempdir(), 'btc_5m_ohlcv.parquet')
    df.to_parquet(ohlcv_path)
    logger.info(f"Saved OHLCV to {ohlcv_path}")

    # ── Step 2: Build features ───────────────────────────────────────────
    logger.info("Building features...")
    from data.features import build_features
    features = build_features(df)
    features_path = os.path.join(tempfile.gettempdir(), 'btc_5m_features.parquet')
    features.to_parquet(features_path)
    logger.info(f"Saved features ({features.shape[1]} cols) to {features_path}")

    # ── Step 3: Upload to Modal Volume ───────────────────────────────────
    logger.info("Creating/opening Modal Volume 'cnn-lstm-data'...")
    vol = modal.Volume.from_name("cnn-lstm-data", create_if_missing=True)

    logger.info("Uploading files to Volume...")
    with vol.batch_upload() as batch:
        # Remote paths are relative to Volume root
        # Volume mounted at /data → "config.yaml" → /data/config.yaml
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
    logger.info("  /data/btc_5m_features.parquet")

    # Cleanup temp files
    os.unlink(ohlcv_path)
    os.unlink(features_path)
    logger.info("Cleaned up temp files")


if __name__ == '__main__':
    main()
