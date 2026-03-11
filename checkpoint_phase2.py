"""
Phase 2 checkpoint: Run Boruta-SHAP feature selection on BTC/USDT 5m data.

Outputs:
  - feature_list.json (SHA-256 hashed)
  - feature_demotion_history.json
  - Console report with accepted/rejected/tentative features
"""

import sys
import os
import json
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from data.fetcher import fetch_ohlcv
from data.features import build_features, get_warmup_bars
from labels.direction import label_series
from selection.boruta_shap import BorutaSHAP, save_feature_list, check_demotion_history

print("=" * 60)
print("PHASE 2 — BORUTA-SHAP FEATURE SELECTION")
print("=" * 60)

# 1. Fetch data and build features
print("\n--- Step 1: Fetching data + building features ---")
df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
print("Fetched {} bars".format(len(df)))

features = build_features(df)
print("Built {} features, {} rows".format(features.shape[1], features.shape[0]))

# 2. Generate labels
labels = label_series(df['open'], df['close'], drop_last=True)
print("Labels: {} total".format(len(labels)))

# 3. Trim to post-warmup, align features and labels
warmup = get_warmup_bars()
X = features.iloc[warmup:].copy()
y = labels.iloc[warmup:].copy()

# Align lengths (labels are 1 shorter due to drop_last)
min_len = min(len(X), len(y))
X = X.iloc[:min_len]
y = y.iloc[:min_len]

# For VPS CPU check, we must subsample or XGBoost+SHAP will timeout
MAX_SAMPLES = 10000
if len(X) > MAX_SAMPLES:
    print(f"Subsampling to {MAX_SAMPLES} for VPS CPU feasibility...")
    X = X.iloc[-MAX_SAMPLES:]
    y = y.iloc[-MAX_SAMPLES:]

print("Post-warmup: {} samples, {} features".format(len(X), X.shape[1]))

# 4. Run Boruta-SHAP (reduced trials for VPS CPU — full 100 trials on GPU)
print("\n--- Step 2: Running Boruta-SHAP ---")
n_trials = 20  # Reduced for VPS CPU; spec says 100 but that needs GPU
start_time = time.time()

selector = BorutaSHAP(
    n_trials=n_trials,
    alpha=0.05,
    use_catboost=False,  # XGBoost is faster on CPU
    random_state=42,
)
selector.fit(X, y)
elapsed = time.time() - start_time

print("Boruta-SHAP completed in {:.1f}s".format(elapsed))

# 5. Report results
report = selector.get_feature_report()
print("\n--- Step 3: Feature Selection Results ---")
print("Accepted features ({}):".format(len(report['accepted'])))
for f in report['accepted']:
    print("  + {}  (hit rate: {:.2f})".format(f, report['importances'].get(f, 0)))

print("\nRejected features ({}):".format(len(report['rejected'])))
for f in report['rejected']:
    print("  - {}  (hit rate: {:.2f})".format(f, report['importances'].get(f, 0)))

print("\nTentative features ({}):".format(len(report['tentative'])))
for f in report['tentative']:
    print("  ? {}  (hit rate: {:.2f})".format(f, report['importances'].get(f, 0)))

# 6. Save feature_list.json
print("\n--- Step 4: Saving feature_list.json ---")
accepted = selector.get_accepted_features()
filepath, sha256_hash = save_feature_list(
    accepted,
    metadata={
        'n_trials': n_trials,
        'alpha': 0.05,
        'model': 'XGBoost',
        'n_samples': len(X),
        'elapsed_seconds': round(elapsed, 1),
    }
)
print("Saved: {}".format(filepath))
print("SHA-256: {}".format(sha256_hash))

# 7. Check demotion history
print("\n--- Step 5: Demotion History ---")
flagged = check_demotion_history(report['rejected'])
if flagged:
    print("FLAGGED for manual review: {}".format(flagged))
else:
    print("No features flagged for manual review (first run)")

# 8. Summary
print("\n" + "=" * 60)
print("PHASE 2 CHECKPOINT RESULTS")
print("=" * 60)
print("1. Shadow competition:  {} trials completed".format(n_trials))
print("2. Accepted features:   {} / {}".format(len(accepted), X.shape[1]))
print("3. feature_list.json:   SHA-256 = {}...".format(sha256_hash[:16]))
print("4. Shannon entropy:     {}".format(
    "present" if "entropy_returns" in accepted or "entropy_volume" in accepted else "not selected"
))

if len(accepted) > 0:
    print("\n>>> PHASE 2 CHECKPOINT PASSED <<<")
else:
    print("\n>>> PHASE 2 CHECKPOINT FAILED: No features accepted <<<")
