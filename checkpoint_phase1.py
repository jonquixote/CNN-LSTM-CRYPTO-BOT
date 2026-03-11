"""
Phase 1 checkpoint validation script.
Runs all checkpoint criteria from the spec.
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from data.fetcher import fetch_ohlcv, validate_history, validate_chainlink_basis
from data.features import build_features, get_warmup_bars, get_feature_list_hash
from labels.direction import label_series, validate_labels
from data.validate import (
    audit_nans, check_distribution, check_label_balance,
    check_history_length, check_p_market_coverage
)

print("=" * 60)
print("PHASE 1 CHECKPOINT VALIDATION")
print("=" * 60)

# 1. Fetch OHLCV data
print("\n--- Step 1: Fetching BTC/USDT 5m OHLCV ---")
df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
print("Fetched {} bars".format(len(df)))

# 2. Validate history
print("\n--- Step 2: History Validation ---")
val = validate_history(df, '5m')
print("Days: {}, Gaps: {}, Valid: {}".format(val['n_days'], val['n_gaps'], val['valid']))

# 3. Build features
print("\n--- Step 3: Feature Engineering ---")
features = build_features(df)
print("Features: {} columns, {} rows".format(features.shape[1], features.shape[0]))

# 4. Generate labels
print("\n--- Step 4: Label Generation ---")
labels = label_series(df['open'], df['close'], drop_last=True)
validate_labels(labels)
print("Labels generated: {} (validated OK)".format(len(labels)))

# 5. Checkpoint validation
print("\n--- Step 5: Checkpoint Checks ---")
warmup = get_warmup_bars()

nan_result = audit_nans(features, warmup)
dist_result = check_distribution(features, warmup)
label_result = check_label_balance(labels)
history_result = check_history_length(df)
pm_result = check_p_market_coverage()
basis_result = validate_chainlink_basis(df)

print("\n" + "=" * 60)
print("PHASE 1 CHECKPOINT RESULTS")
print("=" * 60)

pass_str = "PASS"
fail_str = "FAIL"

nan_status = pass_str if nan_result['pass'] else fail_str
dist_status = pass_str if dist_result['pass'] else fail_str
label_status = pass_str if label_result['pass'] else fail_str
history_status = pass_str if history_result['pass'] else fail_str

print("1. NaN audit:       {} ({} NaNs post-warmup)".format(nan_status, nan_result['total_nans_post_warmup']))
print("2. Distribution:    {}".format(dist_status))
if dist_result.get('constant_features'):
    print("   Constant feats:  {}".format(dist_result['constant_features']))
print("3. Label balance:   {} (Up={}%, Down={}%)".format(label_status, label_result['up_pct'], label_result['down_pct']))
print("4. History length:  {} ({} days)".format(history_status, history_result['n_days']))
print("5. p_market parquet: exists={}".format(pm_result['exists']))
print("6. Feature count:   {}".format(features.shape[1]))
print("7. Feature hash:    {}...".format(get_feature_list_hash(features.columns.tolist())[:16]))
print("8. Chainlink basis: {}".format(basis_result['note']))

all_pass = all([
    nan_result['pass'],
    dist_result['pass'],
    label_result['pass'],
    history_result['pass'],
])

print()
if all_pass:
    print(">>> ALL CHECKPOINT CRITERIA MET <<<")
else:
    failed = []
    if not nan_result['pass']: failed.append('NaN')
    if not dist_result['pass']: failed.append('Distribution')
    if not label_result['pass']: failed.append('Labels')
    if not history_result['pass']: failed.append('History')
    print(">>> CHECKPOINT FAILED: {} <<<".format(failed))
