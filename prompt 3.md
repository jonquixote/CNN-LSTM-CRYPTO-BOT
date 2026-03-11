# Pre-Boruta-SHAP Fixes — Required Before Next Modal Run

Do not submit any job to Modal until all items in this prompt are resolved. Items are ordered
by severity. The Boruta-SHAP run with the expanded 55-feature pool is blocked on item 1.

-----

## Step 1 — Fix Leaky Existing Features (🔴 Blocker)

### Background

The gate check correctly identified that `log_return_1 = log(close / close.shift(1))` lacks
an outer `.shift(1)` — making it leaky. The previous “no modifications to existing features”
rule was intended to prevent accidentally breaking working features, not to protect confirmed
leakage. The rule has one explicit exception:

> **If an existing feature is confirmed leaky by the gate check, fix it in `features.py`
> before the next Boruta-SHAP run. Log the change explicitly.**

`log_return_1` and `log_return_3` are both in `FEATURE_COLUMNS` and both in the previously
confirmed Boruta-SHAP set. Their apparent importance to XGBoost is likely partly or entirely
driven by leakage — they have a direct correlation with the label (`sign(close - open)`) by
construction. Letting them enter the next Boruta run will cause them to appear artificially
important and potentially displace legitimate features.

### Fix

In `features.py`, update both features to include the outer `.shift(1)`:

```python
# BEFORE (leaky):
log_return_1 = np.log(close / close.shift(1))
log_return_3 = np.log(close / close.shift(3))

# AFTER (correct):
log_return_1 = np.log(close / close.shift(1)).shift(1)
log_return_3 = np.log(close / close.shift(3)).shift(1)
```

**Check all other existing log return features** (`log_return_6`, `log_return_12`,
`log_return_24` or equivalent) for the same pattern. Fix any that are missing the outer
`.shift(1)`.

**Update the new derived features** (`return_skewness_30`, `return_percentile_100`,
`btc_sol_corr_50`) to use canonical `log_return_1` directly. The internal `ret_1` was
only needed to work around the leaky canonical version — now that `log_return_1` is
correctly shifted, retain a single canonical series and remove the internal computation.

### Restoration log

Write an entry to `selection/boruta_shap_results.json` (or a separate `fix_log.json`) noting:

```json
{
  "fix_ts": "2026-...",
  "changes": [
    {
      "feature": "log_return_1",
      "issue": "missing outer .shift(1) — leaky, correlated with label by construction",
      "before": "log(close / close.shift(1))",
      "after":  "log(close / close.shift(1)).shift(1)"
    },
    {
      "feature": "log_return_3",
      "issue": "same as log_return_1",
      "before": "log(close / close.shift(3))",
      "after":  "log(close / close.shift(3)).shift(1)"
    }
  ],
  "boruta_impact": "log_return_1 and log_return_3 were in the previously confirmed 6-feature set. Their importance scores are likely inflated by leakage. The next Boruta run is expected to produce materially different results."
}
```

### Tests

Re-run the full leakage test suite after the fix:

```bash
pytest tests/test_features_lookahead.py -v
```

All 17 tests must still pass. If the leakage tests don’t cover `log_return_1` and
`log_return_3` explicitly, add them now — these features should have been caught earlier.

**Do not run Boruta-SHAP until the leakage tests pass with the fixed features.**

-----

## Step 2 — Fix `get_ssh_key_path()` for Base64-Encoded Secret (🔴 Blocker)

The Modal Secret was re-created as `SSH_PRIVATE_KEY_B64` (base64-encoded) to preserve
OpenSSH PEM newlines. The `get_ssh_key_path()` helper in `modal_gpu.py` must be updated to
decode before writing — if it still reads `SSH_PRIVATE_KEY` and writes the raw string, it
writes the base64 text to the `.pem` file and every rsync call fails with a key format error.

```python
import os, tempfile, base64

def get_ssh_key_path() -> str:
    """Decode base64 SSH key from Modal Secret and write to temp file for rsync."""
    key_b64 = os.environ["SSH_PRIVATE_KEY_B64"]
    key = base64.b64decode(key_b64).decode("utf-8")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
        f.write(key)
        key_path = f.name
    os.chmod(key_path, 0o600)
    return key_path
```

Verify the decoded key starts with `-----BEGIN OPENSSH PRIVATE KEY-----` before the first
rsync call, or test with a dry-run rsync (`-n` flag) before the Boruta-SHAP job.

-----

## Step 3 — Confirm Modal Volume Persistence with `vol.commit()` (🟡 Important)

In Modal versions below ~0.60, writes to a mounted Volume inside a container are **not
automatically persisted** — `vol.commit()` must be called explicitly before the function
exits or the writes are silently discarded. If the Volume is designated as the primary backup
and `vol.commit()` is not being called, the primary backup doesn’t exist.

**First: confirm the actual artifact write path inside the container.** The original
infrastructure spec wrote artifacts to `/tmp/` and rsynced to VPS — the Volume at `/data/`
was read-only input only. The implementing agent designated the Volume as primary backup,
but the mechanism matters:

- If artifacts are being written directly to `/data/artifacts/` or copied there from `/tmp/`,
  `vol.commit()` is the correct fix.
- If artifacts are only written to `/tmp/` and the “Volume backup” was a misunderstanding,
  then `vol.commit()` is irrelevant — the Volume write needs to be added from scratch first.

Inspect `modal_gpu.py` and confirm which path artifact files are written to before
implementing the `vol.commit()` fix below.

Check the Modal version in use:

```bash
python -c "import modal; print(modal.__version__)"
```

- If version ≥ 0.60 and auto-commit on function exit is confirmed in the Modal changelog
  for that version: document the version and proceed.
- If version < 0.60, or if uncertain: add explicit `vol.commit()` calls after every write
  to `/data/` in each GPU function. `vol.commit()` is a no-op if auto-commit is already
  active — safe to add regardless.

This is especially important given that rsync is currently non-fatal (best-effort). Without
a confirmed Volume commit, a job can exit clean with no artifacts written anywhere:
Volume write discarded silently → rsync fails silently → job exits 0 → artifacts lost.

If Volume write fails (any reason), it must be fatal — add an explicit check:

```python
# After writing artifacts to /data/:
try:
    vol.commit()
except Exception as e:
    raise RuntimeError(f"Volume commit failed — artifacts not persisted: {e}")
```

-----

## Step 4 — Restore rsync to Fatal on Failure (🟡 Important)

rsync was downgraded to best-effort (non-fatal) with the Volume designated as primary backup.
This creates a silent failure risk: if both Volume and rsync fail, the job exits clean with
no artifacts. The `check=True` + `rsync_with_retry` pattern was specified precisely to avoid
this.

Restore rsync to fatal after confirming Step 3 (Volume persistence). The correct failure
hierarchy is:

1. Write artifacts to `/data/` (Volume) — **fatal if it fails**
1. `vol.commit()` — **fatal if it fails**
1. rsync to VPS — **fatal if it fails** (3 retries via `rsync_with_retry`)

If the VPS is unreachable and rsync fails after 3 retries, the job should fail visibly.
Silent artifact loss is worse than a visible job failure — a failed job can be re-run,
lost artifacts cannot.

-----

## Step 5 — Check for Redundant SOL OHLCV Fetch (🟢 Minor)

`fetch_sol_ohlcv()` was added to `fetcher.py` for the cross-asset features. If SOL OHLCV
was already being fetched for other trading slots (SOL/USDT is a live trading pair in the
system), `fetcher.py` may now fetch SOL twice. Check for duplicate fetches and consolidate
if present — fetch SOL once and reuse the result for both the trading pipeline and the
cross-asset feature computation.

-----

## Step 6 — Document OI 30-Day History Cap in `fetcher.py` (🟢 Minor)

The Binance `openInterestHist` endpoint returns a maximum of 30 days of 5m OI history.
Add a comment to `fetch_open_interest()` noting this constraint:

```python
# NOTE: Binance openInterestHist endpoint maximum history is 30 days at 5m resolution.
# If re-running features from scratch on data older than 30 days ago, OI will have gaps
# and oi_delta / oi_delta_zscore / signed_oi_delta will be NaN for those bars.
# The NaN handling in features.py fills these with 0 — document this assumption.
```

-----

## Execution Order

1. Fix `log_return_1`, `log_return_3`, check all other log return features → re-run leakage tests (all 17 must pass)
1. Fix `get_ssh_key_path()` for base64 decode
1. Confirm/add `vol.commit()` — check Modal version
1. Restore rsync to fatal
1. Check for redundant SOL fetch
1. Add OI cap comment
1. **Rebuild features parquet locally** — re-run the feature build pipeline (`python data/build_features.py` or equivalent) to regenerate `data/btc_5m_features.parquet` with the fixed features and all 55 columns including supplementary data (funding rate, OI, SOL). Confirm all 55 columns are present before proceeding. `upload_to_volume.py` uploads whatever is on disk — skipping this step means Boruta-SHAP runs on the old parquet with leaky `log_return_1` and `log_return_3` values regardless of the `features.py` fix.
1. Re-run `upload_to_volume.py` with expanded feature set
1. Submit Boruta-SHAP job on B200

-----

## What You Are Not Changing

- Any feature that is already correctly implemented (`.shift(1)` confirmed)
- The 55-feature `FEATURE_COLUMNS` list (except renaming any incorrectly shifted features)
- Model architecture, training loop, eval harness, strategy layer, inference
- Walk-forward fold structure or data split logic
- The Optuna SQLite persistence pattern (already correct)
