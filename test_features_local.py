import numpy as np
import pandas as pd
import time
import logging
from data.features import build_features

logging.basicConfig(level=logging.INFO)

def test_local_features():
    print("Creating synthetic dataset (80,000 bars)...")
    n = 80000
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='5min'),
        'open': np.random.uniform(50000, 60000, n),
        'high': np.random.uniform(60000, 61000, n),
        'low': np.random.uniform(49000, 50000, n),
        'close': np.random.uniform(50000, 60000, n),
        'volume': np.random.uniform(0, 100, n)
    })
    
    # Introduce the "inf" trigger: 0 to non-zero volume transitions
    df.loc[100:110, 'volume'] = 0.0
    df.loc[111, 'volume'] = 10.0
    
    print("Starting feature build (Hurst + Entropy)...")
    start_time = time.time()
    
    try:
        features = build_features(df)
        elapsed = time.time() - start_time
        print(f"SUCCESS: Feature build completed in {elapsed:.2f} seconds.")
        print(f"Feature shape: {features.shape}")
        
        # Check for NaNs and Infs in tricky features
        print("\nSanity Check (post-warmup):")
        warmup = 600
        post = features.iloc[warmup:]
        print(f"Residual NaNs: {post.isna().sum().sum()}")
        print(f"Residual Infs: {np.isinf(post.values).sum()}")
        
        if 'hurst' in features.columns:
            print(f"Hurst range: {features['hurst'].min():.2f} - {features['hurst'].max():.2f}")
        if 'entropy_volume' in features.columns:
            print(f"Entropy Volume range: {features['entropy_volume'].min():.2f} - {features['entropy_volume'].max():.2f}")

    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_local_features()
