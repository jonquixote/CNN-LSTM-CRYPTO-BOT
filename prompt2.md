# Feature Expansion: Add New Candidate Features to `data/features.py`

## Context

The CNN-LSTM system currently has ~37 features in `data/features.py`. We are expanding the
candidate pool to ~67 features before the next Boruta-SHAP run. Boruta-SHAP will confirm which
features survive — do not pre-filter. Your job is to add all features listed below correctly,
without breaking any existing features or the feature pipeline.

**Single source of truth rule:** All feature logic lives in `data/features.py`. Training and
inference both call this file. Do not add feature logic anywhere else.

**Leakage rule:** Every feature that references `close`, `high`, `low`, `volume`, or `open`
of the current bar — whether in a rolling window, a point-in-time formula, or a directional
sign computation — must apply `.shift(1)` to those inputs or to the entire computed series
before use. `sign(close - open)` is the label. Using it unshifted in any feature is direct
target leakage. Confirm leakage-free before marking any feature done.

-----

## New Data Requirements — Fetch Before Features

Two feature groups require data not available in the standard OHLCV fetch. Add both fetches
to `data/fetcher.py` and merge onto the OHLCV DataFrame on timestamp before `features.py`
is called.

### Funding Rate

```python
# Binance Futures endpoint — one rate per 8 hours, forward-filled to bar frequency
# GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000
# Fields: fundingTime (ms epoch), fundingRate (float)
# Merge: forward-fill onto 5m bar index — funding rate is constant between funding events
```

### Open Interest

```python
# Binance Futures endpoint — available at 5m resolution
# GET https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=500
# Fields: timestamp (ms epoch), sumOpenInterest (float)
# Merge: join on bar timestamp, ffill any gaps
```

Both fetches must be added alongside the existing OHLCV fetch in `fetcher.py`. They must
be aligned to the same bar index as the OHLCV data before `features.py` is called. If either
endpoint is unavailable for a bar, forward-fill. Do not drop bars.

-----

## Features to Add

Implement every feature below in `data/features.py`. Group them in the file under clearly
labeled sections matching the groups below.

### Group 1 — Funding Rate & Open Interest

**OI timestamp convention:** Binance `sumOpenInterest` timestamps represent the snapshot at the START of each 5m bar (bar open). This means OI at bar t is available at bar t open without a shift — `oi_delta = open_interest.diff(1)` is not leaky. Verify this against the Binance API documentation before implementation.

**`funding_rate_zscore` formula — zero-std guard required:**

```python
rolling_std = funding_rate.rolling(500).std().replace(0, np.nan)
funding_rate_zscore = ((funding_rate - funding_rate.rolling(500).mean()) / rolling_std).fillna(0)
# replace(0, np.nan) prevents division producing inf; fillna(0) catches the resulting NaN.
# Add smoke test: assert not np.isinf(df["funding_rate_zscore"]).any()
```

|Feature              |Formula                                          |Notes                                                                                           |
|---------------------|-------------------------------------------------|------------------------------------------------------------------------------------------------|
|`funding_rate`       |Raw funding rate at bar time                     |Forward-filled from 8h funding events                                                           |
|`funding_rate_zscore`|See code block above                             |window=500 (~41 hours, ~5 funding events). Zero-std produces `inf` (not NaN) — guard explicitly.|
|`oi_delta`           |`open_interest.diff(1)`                          |Bar-over-bar OI change                                                                          |
|`oi_delta_zscore`    |Rolling z-score of `oi_delta`, window=50         |OI changes every bar so variance accumulates quickly — window=50 is correct here                |
|`signed_oi_delta`    |`oi_delta * sign(close.shift(1) - open.shift(1))`|Directional OI pressure — use shifted close/open, not current bar                               |

### Group 2 — Wick Structure

All wick features use the **previous bar’s** OHLCV — current bar high/low/close are unknown at
bar open. Shift all four OHLCV inputs before computing:

```python
h = high.shift(1)
l = low.shift(1)
o = open_.shift(1)
c = close.shift(1)
range_ = (h - l).replace(0, np.nan)  # guard zero-range bars
upper_wick_pct = ((h - np.maximum(o, c)) / range_).fillna(0)
lower_wick_pct = ((np.minimum(o, c) - l) / range_).fillna(0)
wick_imbalance  = upper_wick_pct - lower_wick_pct
```

|Feature         |Formula                          |Notes                     |
|----------------|---------------------------------|--------------------------|
|`upper_wick_pct`|See code block above             |0 if zero-range bar       |
|`lower_wick_pct`|See code block above             |0 if zero-range bar       |
|`wick_imbalance`|`upper_wick_pct - lower_wick_pct`|Positive = upper rejection|

### Group 3 — Temporal Encoding

|Feature   |Formula                  |Notes                        |
|----------|-------------------------|-----------------------------|
|`hour_sin`|`sin(2π * hour / 24)`    |`hour` = UTC hour of bar open|
|`hour_cos`|`cos(2π * hour / 24)`    |                             |
|`dow_sin` |`sin(2π * dayofweek / 7)`|Monday=0, Sunday=6           |
|`dow_cos` |`cos(2π * dayofweek / 7)`|                             |

Derive from the bar’s UTC timestamp index. Do not use local time.

### Group 4 — Price Action

|Feature             |Formula                                   |Notes                                                                                       |
|--------------------|------------------------------------------|--------------------------------------------------------------------------------------------|
|`prev_bar_direction`|`sign(close.shift(1) - open.shift(1))`    |{-1, 0, 1} — current bar direction is the label; always use previous bar                    |
|`streak_count`      |Consecutive bars in same direction, signed|+3 = 3 consecutive up bars; -2 = 2 consecutive down bars; resets on direction change or doji|

Check whether `ret_1` and `ret_5` already exist in `features.py` (the plan includes log return
lags). If they do, skip. If they are named differently (e.g. `log_ret_lag1`), add aliases or
confirm they are equivalent. Do not add duplicates.

**Critical gate before building derived features:** Confirm `ret_1` is computed as
`log(close / close.shift(1)).shift(1)` — i.e., it is already lagged one bar. If `ret_1` is
`log(close / close.shift(1))` without the outer `.shift(1)`, it is leaky. `return_skewness_30`,
`return_percentile_100`, and `btc_sol_corr_50` are all derived from `ret_1` and inherit
any leakage in it. Do not proceed with those features until this is confirmed.

**`streak_count` implementation note:** Use previous bar’s direction — current bar direction
is the label. Vectorized approach (no Python row loop):

```python
direction = np.sign(close.shift(1) - open.shift(1))  # shift — current bar is the label
streak_id = (direction != direction.shift(1)).cumsum()
streak_count = direction.groupby(streak_id).cumcount().add(1) * direction
```

### Group 5 — Signed Volume

|Feature               |Formula                                                 |Notes                                                                  |
|----------------------|--------------------------------------------------------|-----------------------------------------------------------------------|
|`signed_volume`       |`volume.shift(1) * sign(close.shift(1) - open.shift(1))`|Use previous bar — current volume and direction are unknown at bar open|
|`signed_volume_sum_5` |Rolling 5-bar sum of `signed_volume`, z-scored          |`(rolling_sum - mean) / std`, window=100 for z-score                   |
|`signed_volume_sum_15`|Rolling 15-bar sum of `signed_volume`, z-scored         |Same z-score window                                                    |

### Group 6 — Rolling Return Statistics

|Feature                |Formula                                                            |Notes                                                                                                                             |
|-----------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
|`return_skewness_30`   |Rolling 30-bar skewness of `ret_1` (or equivalent 1-bar log return)|Use `pandas.Series.rolling(30).skew()`                                                                                            |
|`return_percentile_100`|Percentile rank of `ret_1` within its 100-bar rolling window       |`ret_1.rolling(100).rank(pct=True)` — result in [0, 1]. `ret_1` is already lagged per the gate check — do not use unshifted close.|

### Group 7 — Distance from Moving Averages

|Feature          |Formula  |Notes                |
|-----------------|---------|---------------------|
|`dist_from_ma50` |See below|EMA span=50, shifted |
|`dist_from_ma200`|See below|EMA span=200, shifted|

EMA at time t includes `close[t]`, so both the EMA and the divisor are leaky if unshifted.
Compute on unshifted close, then shift the entire series one step:

```python
ema50  = close.ewm(span=50, adjust=False).mean()
ema200 = close.ewm(span=200, adjust=False).mean()
dist_from_ma50  = ((close - ema50)  / close).shift(1)
dist_from_ma200 = ((close - ema200) / close).shift(1)
```

Signed: positive = above MA, negative = below.

### Group 8 — Cross-Asset (BTC model only)

|Feature          |Formula                                       |Notes                                                                                                                 |
|-----------------|----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
|`sol_ret_lag1`   |`log(sol_close / sol_close.shift(1)).shift(1)`|1-bar return, lagged 1                                                                                                |
|`sol_ret_lag3`   |`log(sol_close / sol_close.shift(3)).shift(1)`|3-bar cumulative return, lagged 1                                                                                     |
|`sol_ret_lag6`   |`log(sol_close / sol_close.shift(6)).shift(1)`|6-bar cumulative return, lagged 1                                                                                     |
|`btc_sol_corr_50`|`ret_1.rolling(50).corr(sol_ret_lag1)`        |Both inputs are pre-lagged — no additional shift needed. `sol_ret_lag1` is the SOL series defined above in this group.|

SOL OHLCV is already fetched for other slots. Confirm `fetcher.py` exposes it and merge
onto the BTC bar index before `features.py` is called. If SOL bars don’t align perfectly
with BTC bars (timestamp gaps), forward-fill SOL returns onto the BTC index — do not drop
BTC bars.

-----

## Implementation Requirements

### NaN handling

Every new feature must produce no NaN values in the output DataFrame for bars past the
maximum warmup window. Warmup period for the new features is determined by the longest window used —
The longest warmup among new features is `funding_rate_zscore` at 500 bars, which equals
the existing Hurst window. **No update to the global warmup constant is needed.** Fill any remaining
NaNs with 0 before returning.

### RobustScaler

The existing features are scaled by `RobustScaler` in the training pipeline. The new
features will be scaled the same way — no changes needed to the scaler. Do not add
per-feature scaling inside `features.py`.

### Feature name registry

If `features.py` maintains an explicit list of feature names (e.g. `FEATURE_COLUMNS` or
equivalent), add every new feature name to that list. Training and inference both read
from this list to select columns — missing entries mean the feature is computed but never
used. Note: the feature is named `prev_bar_direction`, not `current_bar_direction`.

### No changes to

- `labels/direction.py`
- `models/architecture.py`
- `eval/walkforward.py`
- Any strategy or inference files

-----

## Verification

After implementing, run the existing leakage test:

```bash
pytest tests/test_features_lookahead.py -v
```

Then add a smoke test confirming:

1. The output DataFrame has no NaN values past the warmup window
1. All new feature names are present in the output columns
1. `upper_wick_pct` and `lower_wick_pct` are both in [0, 1] for all bars
1. `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` are all in [-1, 1]
1. `return_percentile_100` is in [0, 1]
1. `funding_rate_zscore` contains no `inf` values — `assert not np.isinf(df["funding_rate_zscore"]).any()`

Do not run Boruta-SHAP or Optuna until these tests pass.

-----

## Optuna Trial Count Note

The prompt that governs the Optuna search specifies 100 trials. Based on observed runtime
on B200, **run 50 trials first.** At trial 50, check the Optuna dashboard or
`optuna_results.json` for the best Val Sharpe trajectory:

- If the best score is still improving meaningfully at trial 50 (i.e. the top-5 trial
  scores are spread across recent trials, not clustered in the first 20), extend to 100
  trials using a second Modal run. The resume requires a **persistent SQLite storage
  backend** and an explicit VPS pull at the start of run 2 — a new container starts with
  an empty filesystem and `create_study(..., load_if_exists=True)` against a non-existent
  file just creates a fresh empty study, silently discarding the first 50 trials.

**Run 1 pattern** — configure in `run_optuna_search()`:

```python
import os
os.makedirs("/tmp/tuning", exist_ok=True)
storage = "sqlite:////tmp/tuning/optuna_study.db"
study = optuna.create_study(
    study_name="btc_5m_cnn_lstm",
    storage=storage,
    load_if_exists=True,   # parameter to create_study, not to optimize
    direction="maximize"
)
study.optimize(objective, n_trials=50)
```

Add `optuna_study.db` to the Step 6 rsync block — without this the DB is destroyed on
container exit and resume is impossible:

```python
rsync_with_retry([
    "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
    "/tmp/tuning/optuna_study.db",
    remote_jobs + "/tuning/"
])
```

**Run 2 pattern** (only if extending) — pull the DB from VPS before `create_study()`,
then run 50 more trials:

```python
# Pull existing DB from VPS — must happen before create_study(), not after
os.makedirs("/tmp/tuning", exist_ok=True)
remote_jobs = config.infrastructure.jobs_remote_path.rstrip("/")  # from loaded config
key_path = get_ssh_key_path()
try:
    rsync_with_retry([
        "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        remote_jobs + "/tuning/optuna_study.db",
        "/tmp/tuning/optuna_study.db"
    ])
finally:
    os.unlink(key_path)

# Now create_study finds the existing 50 trials and continues from trial 51
storage = "sqlite:////tmp/tuning/optuna_study.db"
study = optuna.create_study(
    study_name="btc_5m_cnn_lstm",
    storage=storage,
    load_if_exists=True,
    direction="maximize"
)
study.optimize(objective, n_trials=50)  # adds trials 51–100
# Rsync DB + results back to VPS as in Run 1
```

- If the curve has clearly plateaued by trial 40–50, stop at 50 and proceed with the
  winner.

**Do not cut to 30 trials regardless of runtime.** TPE needs ~20 trials for exploration
before exploitation begins — 30 trials gives almost no exploitation budget. 50 is the
floor.

-----

## What You Are Not Changing

- Existing feature logic — do not modify any feature already in `features.py`
- Label generation — `labels/direction.py` is untouched
- Model architecture, training loop, eval harness, strategy layer, inference
- Any walk-forward fold structure or data split logic
