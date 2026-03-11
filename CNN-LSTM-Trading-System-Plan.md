# CNN-LSTM Crypto Trading System

## Architecture, Build Plan & Engineering Specification — v9

### Target Venue: Polymarket BTC/ETH Up/Down 5m & 15m Markets

---

## 1. Vision & Goals

Build a production-grade, self-retraining crypto direction prediction engine using a multi-scale CNN-BiLSTM-Attention architecture, deployed against Polymarket's BTC and ETH Up/Down binary markets on 5m and 15m timeframes.

**The model's task:** a 2-class classification problem. Given the last N fully closed bars, predict whether the closing price of the **next** bar will be higher (Up, class index 1) or lower (Down, class index 0) than the opening price of that next bar. This aligns exactly with Polymarket's resolution rule: *"Up if closing price ≥ price at the beginning of the window."* There is no neutral class — "no bet" is a strategy-layer decision made by the edge filter, not a model output.

During walk-forward evaluation, historical Polymarket market prices are loaded from the Jon-Becker dataset where available. For bars with no historical price, the conservative fallback `p_market = 0.50` is used — the efficient-market null hypothesis. A model that cannot beat a market priced at 50/50 has no edge. This assumption is documented in `config.yaml` and `eval/metrics.py` wherever it is used.

**The execution model:** signals are only acted on when the model's calibrated probability meaningfully exceeds the live Polymarket market price for that direction. Edge over market price — not raw directional confidence — is the primary execution criterion.

**Design principles:**

- No magic numbers — everything lives in `config.yaml`
- No lookahead leakage — enforced structurally, not by convention
- No accuracy theater — primary metric is net PnL after Polymarket fees
- No monolithic code — each layer is independently testable with corresponding tests
- No bet without a fresh market price — stale prices older than 30 seconds suppress the trade
- No bet in the final 60 seconds of a market window
- No optimistic accounting — fees, slippage, and Chainlink basis risk all modeled explicitly
- All timestamps are UTC. The system never uses local time
- Signal direction is always a string (`"up"` or `"down"`) at the strategy and execution layer; class indices (0/1) exist only inside `direction.py`, `architecture.py`, and `test_label_encoding.py`
- Build BTC 5m end-to-end before expanding to other symbols or timeframes

**Scope note:** The full system (BTC + ETH × 5m + 15m) is 4–6 months of careful solo engineering. Build BTC 5m through all phases first.

---

## 2. Infrastructure

### GPU — Lightning.ai T4 (~80 hrs/month free)

Training only. Never runs inference. After each retrain, artifacts are pushed to the VPS via rsync.

| Job | Frequency | Estimated GPU Hours |
|-----|-----------|-------------------|
| Initial hyperparameter search (Optuna + Hyperband) | Once per symbol/TF | ~6 hrs |
| Weekly full retrain (all active slots × 5 seeds) | Weekly | ~2–3 hrs |
| Walk-forward evaluation + PnL audit | After each retrain | ~1 hr |
| Boruta-SHAP feature audit | After each retrain | ~0.5 hr |
| Probability calibration | After each retrain | ~0.25 hr |
| Quarterly lightweight re-search | Quarterly | ~2 hrs |
| **Total per month** | | **~15–20 hrs** |

Remaining ~60 hrs/month available for experiments and ablations.

### CPU — Inference Server

Always-on Linux VPS (Hetzner CX22, ~$5–6/month). Runs 24/7 independently of Lightning.ai sessions.

- ONNX Runtime inference
- Feature builder and bar buffer
- Polymarket market price poller and order execution
- Position manager and JSONL logger
- Health checks and alerting

### Artifact Transfer

After hot-swap guard approves new artifacts, `weekly_retrain.py` pushes the versioned artifact directory to the VPS via rsync. SSH keypair must be provisioned before first push (see Phase 8 Step 0).

```yaml
infrastructure:
  gpu_platform: lightning_ai
  inference_host: vps
  artifact_sync: rsync
  artifact_remote_path: "user@vps-ip:/home/user/cnn_lstm_v1/inference/artifacts/"
```

---

## 3. Project Structure

```
cnn_lstm_v1/
├── config.yaml
├── requirements.txt
│
├── data/
│   ├── fetcher.py                     # Binance OHLCV fetch + Chainlink basis validation
│   ├── features.py                    # All feature engineering (training = inference)
│   ├── polymarket_historical.py       # Jon-Becker dataset loader → p_market_history.parquet
│   └── validate.py                    # Distribution checks, NaN audit, Chainlink basis
│
├── labels/
│   └── direction.py                   # Returns 0 (Down) or 1 (Up). Never +1/-1, never strings.
│
├── models/
│   ├── architecture.py                # CNN-BiLSTM-Attention, 2-class output
│   │                                  # softmax ordering + BiLSTM backward pass documented
│   ├── train.py
│   └── ensemble.py
│
├── calibration/
│   └── isotonic.py
│
├── selection/
│   └── boruta_shap.py
│
├── eval/
│   ├── walkforward.py
│   └── metrics.py                     # p_market fallback=0.50 documented; uses parquet where available
│
├── tuning/
│   └── optuna_search.py
│
├── strategy/
│   ├── edge.py
│   ├── filters.py                     # coherence_gap_behavior respected; coherence_status logged
│   ├── sizer.py                       # Binary Kelly + vol scalar + negative Kelly guard
│   └── calendar.py
│
├── inference/
│   ├── export_onnx.py
│   ├── bar_buffer.py
│   ├── polymarket_client.py           # negRisk, GTD, tick size (price+size), tick_size_change, heartbeat recovery
│   ├── position_manager.py            # CONFIRMED/FAILED/RETRYING+timeout, per-account fill tracking
│   └── live.py
│
├── logging/
│   ├── prediction_logger.py
│   └── trade_logger.py
│
├── utils/
│   └── timing.py
│
├── tests/
│   ├── test_label_encoding.py         # index 0=Down, 1=Up; returns int not string
│   ├── test_features_lookahead.py
│   ├── test_harness_random_baseline.py
│   ├── test_atomic_deployment.py
│   ├── test_position_manager.py       # CONFIRMED, FAILED refund, RETRYING timeout, per-account divergence
│   ├── test_gap_handling.py
│   ├── test_polymarket_client.py      # negRisk, GTD, tick size (price+size rounding), heartbeat recovery, tick_size_change
│   ├── test_edge_computation.py
│   ├── test_timing_alignment.py      # boundary, one-before, one-after UTC inputs
│   └── test_tick_size_precision.py   # price AND size rounded correctly for all tick size tiers
│
└── jobs/
    ├── weekly_retrain.py
    ├── quarterly_retune.py
    └── healthcheck.py
```

---

## 4. `config.yaml`

```yaml
# ── Active slots ─────────────────────────────────────────────────────────────
active_slots:
  - symbol: BTC
    timeframe: 5m

all_slots:
  - { symbol: BTC,  timeframe: 5m  }
  - { symbol: ETH,  timeframe: 5m  }
  - { symbol: BTC,  timeframe: 15m }
  - { symbol: ETH,  timeframe: 15m }

# ── Labels ────────────────────────────────────────────────────────────────────
labels:
  type: binary_direction
  # Class encoding — must match softmax output ordering in architecture.py.
  # direction.py returns integer indices 0 or 1. Never +1/-1. Never strings.
  # Strategy/execution layer uses strings "up"/"down". Conversion happens in ensemble.py.
  class_down: 0    # close[t] < open[t]  → "Down"
  class_up: 1      # close[t] >= open[t] → "Up"

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  sequence_length: 750
  conv_filters: 256
  lstm_hidden_dim: 512
  lstm_layers: 2
  attention_heads: 8
  use_global_attention: true
  dropout: 0.3
  batch_size: 1024
  max_epochs: 100
  early_stopping_patience: 10
  learning_rate: 0.0005
  ensemble_seeds: [0, 1, 2, 3, 4]
  output_classes: 2

# ── Training ──────────────────────────────────────────────────────────────────
training:
  train_window_days: 120
  embargo_days: 10
  val_window_days: 20
  test_window_days: 10
  fold_step_days: 10
  min_folds: 12
  min_history_days: 270              # 12 folds × 10-day step + 160-day initial window
  scaler: RobustScaler

# ── Backtest assumptions ──────────────────────────────────────────────────────
backtest:
  # Historical Polymarket market prices from Jon-Becker dataset.
  # Coverage: BTC/ETH 5m/15m from ~late 2024. Verify before setting use_historical_prices: true.
  # For bars with no historical price, assumed_market_price is used as fallback.
  use_historical_prices: true
  historical_prices_path: data/p_market_history.parquet
  assumed_market_price: 0.50         # efficient-market null hypothesis fallback
  # assumed_market_price = 0.50 means: edge = p_model - 0.50
  # Backtest PnL using fallback is a LOWER BOUND — live prices may diverge from 0.50

# ── Strategy ──────────────────────────────────────────────────────────────────
strategy:
  edge_threshold: 0.04               # min p_model - p_market to place bet
  # US flat 0.10% fee → edge_threshold >= 0.02 is safe; 0.04 recommended for margin
  # Non-US variable 1.56% peak → edge_threshold >= 0.06 required
  min_model_confidence: 0.52
  hurst_trend_threshold: 0.52
  hurst_mean_reversion_threshold: 0.48
  hurst_high_confidence_bonus: 0.58
  threshold_hurst_adjustment: 0.03
  seed_disagreement_max: 0.08
  vol_zscore_threshold: 2.5
  entropy_threshold: 0.95

  # Multi-asset coherence
  # Only active when ALL coherence_assets are in active_slots.
  # "Signals Up/Down" = p_calibrated > coherence_confidence_threshold AND edge > 0.
  # A weak lean is not sufficient.
  coherence_assets: ["BTC", "ETH"]
  coherence_required: true
  coherence_confidence_threshold: 0.55
  # What to do when a coherence asset has a gap (no signal available):
  #   bypass       → treat coherence as passed for this bar; log coherence_status: "bypassed_gap"
  #   suppress_all → block all bets this bar; log coherence_status: "failed_gap"
  coherence_gap_behavior: bypass

  min_market_liquidity_usd: 500
  max_bid_ask_spread: 0.04
  blackout_bars: 2

  target_vol:
    BTC: 0.0022
    ETH: 0.0028
  kelly_fraction_multiplier: 0.25
  kelly_max_fraction: 0.15
  kelly_recalibration_days: 10
  daily_loss_limit_pct: 0.03
  weekly_loss_limit_pct: 0.07
  max_concurrent_positions: 4

# ── Execution ─────────────────────────────────────────────────────────────────
execution:
  order_type: GTD
  max_entry_seconds: 240
  poll_interval_seconds: 5
  heartbeat_interval_seconds: 5      # chain heartbeat_id every call; 400 is recoverable
  stale_price_max_seconds: 30
  tick_size_cache_ttl_seconds: 3600  # periodic refresh; also reset on tick_size_change event
  retrying_max_minutes: 15           # after this, treat RETRYING trade as FAILED, refund stake
  allow_mid_window_exit: false       # paper trading phase: all bets held to resolution

# ── Fees ──────────────────────────────────────────────────────────────────────
costs:
  fee_model: us_flat
  taker_fee_pct: 0.001               # 0.10% of contract premium
  maker_rebate_pct: 0.001
  min_fee_usdc: 0.0001               # Polymarket stated minimum; 0.0010 was incorrect
  # Global variable fee (non-US reference only):
  # fee_formula: "C * p * 0.25 * (p * (1-p))^2"
  # max_fee_pct: 0.0156 at p=0.50 → raise edge_threshold to 0.06 minimum

# ── Accounts ──────────────────────────────────────────────────────────────────
# Each account applies Kelly sizing to its own portfolio_value independently.
# position_fraction multiplies the final Kelly stake. 1.0 = full Kelly. 0.1 = 10%.
accounts:
  - id: account_a
    key_env: POLY_KEY_A
    position_fraction: 1.0
    starting_capital: 100.0
    logs_dir: logs/account_a/
    state_dir: inference/state/account_a/
  - id: account_b
    key_env: POLY_KEY_B
    position_fraction: 0.1
    starting_capital: 10.0
    logs_dir: logs/account_b/
    state_dir: inference/state/account_b/

# ── Inference ─────────────────────────────────────────────────────────────────
inference:
  artifacts_dir: inference/artifacts/
  staging_dir: inference/staging/
  bar_buffer_extra: 10
  gap_tolerance_minutes: 0
  maintenance_windows: []

# ── Chainlink basis validation ────────────────────────────────────────────────
chainlink:
  basis_warn_bps: 15
  basis_check_interval_hours: 24
  basis_lookback_days: 7

# ── Health monitoring ─────────────────────────────────────────────────────────
health:
  canary_accuracy_floor: 0.50
  canary_lookback_days: 3
  feature_kl_alert_threshold: 0.10
  calibration_drift_alert_threshold: 0.05
  live_pnl_alert_floor: -0.05        # alert if rolling 7-day net PnL < -5% of capital
  alert_method: telegram             # telegram | slack | ntfy | log_only
  alert_webhook_url: ""              # required unless log_only

# ── Paper trading ─────────────────────────────────────────────────────────────
paper_trading:
  allow_mid_window_exit: false       # all bets held to resolution in paper phase

# ── Paper trading gate ────────────────────────────────────────────────────────
paper_trading_gate:
  min_sharpe_60d: 1.0
  max_drawdown_60d: 0.15
  required_months: 3
  min_drawdown_recovery_pct: 0.05

# ── Jobs ──────────────────────────────────────────────────────────────────────
jobs:
  retrain_day: Monday
  retrain_hour_utc: 2
  quarterly_retune_months: [1, 4, 7, 10]
  calendar_source: nasdaq_api
  calendar_api_url: https://api.nasdaq.com/api/calendar/economicevents
  calendar_manual_path: strategy/events_manual.json
  high_impact_event_filter: [FOMC, CPI, NFP, PPI, FOMC_MINUTES]

# ── Infrastructure ────────────────────────────────────────────────────────────
infrastructure:
  gpu_platform: lightning_ai
  inference_host: vps
  artifact_sync: rsync
  artifact_remote_path: "user@vps-ip:/home/user/cnn_lstm_v1/inference/artifacts/"
```

---

## 5. Features

All features computed in `data/features.py` — single source of truth for training and inference.

### 5.1 Price & Volume Derivatives

- Log returns at lags: 1, 3, 6, 12, 24 bars
- Rolling realized volatility: 12, 50, 200 bars
- Volume z-score relative to 50-bar mean
- VWAP deviation from close (normalized)

### 5.2 Microstructure-Lite

- Bar efficiency ratio: `|close - open| / (high - low)`
- Relative bar size: `(high - low) / ATR_14`
- Close position within bar: `(close - low) / (high - low)`
- Volume confirmation: binary flag

### 5.3 Volatility & Regime

- ATR at lookbacks: 14, 50, 200
- Bollinger Band width (20-bar, 2σ)
- Rolling Hurst exponent over 500 bars
- Vol-of-vol: rolling std of 20-bar realized vol over 100 bars
- Realized vol z-score vs 90-day average
- Shannon entropy over 20-bar rolling returns and volume distributions

### 5.4 Technical Indicators

- RSI (14), MACD line and signal delta, ADX (14), Stochastic %K and %D, EMA crossovers (9/21, 21/55)

### 5.5 Cross-Asset

- ETH log return at lags 1, 3, 6 bars (for BTC model); BTC for ETH model
- Rolling 50-bar cross-asset correlation and beta

**Total: ~33–40 features per symbol/TF.** `RobustScaler` fit only on training data per fold.

### 5.6 Feature Selection: Boruta-SHAP

SHAP values from XGBoost/CatBoost replace Gini impurity. Run per fold on GPU. Features demoted across 3 consecutive retrains trigger a manual review flag. `feature_list.json` versioned with SHA-256 hash of sorted JSON content.

---

## 6. Labels: Binary Bar Direction

```python
# labels/direction.py
#
# Returns INTEGER class indices 0 or 1. NEVER +1/-1. NEVER strings.
# class_down = 0  →  close[t] < open[t]   → "Down"
# class_up   = 1  →  close[t] >= open[t]  → "Up"
#
# Conversion from class index to direction string ("up"/"down") happens
# in ensemble.py, not here. strategy/ and inference/ always use strings.
#
# Polymarket resolution: "Up if closing price >= price at beginning of window"
# open[t] is the correct anchor. close[t-1] diverges on large intra-bar moves.
#
# The model predicts bar t+1 using features from bars up to bar t.
# Labels for historical training are generated from open[t] and close[t].

def label_bar(open_price: float, close_price: float) -> int:
    """Returns 1 (Up) if close >= open, else 0 (Down)."""
    return 1 if close_price >= open_price else 0
```

**Leakage guard:** `direction.py` raises `ValueError` if called on the final bar of a series (no `t+1` close exists for prediction).

**Minimum history:** 270 days of clean gap-free Binance OHLCV.

---

## 7. Model Architecture

### 7.1 Softmax Output Class Ordering

**Fixed — enforced throughout. Class indices 0 and 1, not +1/-1:**

| Index | Class | Condition | Direction string | Polymarket outcome |
|-------|-------|-----------|------------------|--------------------|
| 0 | `class_down` | `close < open` | `"down"` | "Down" resolves |
| 1 | `class_up` | `close >= open` | `"up"` | "Up" resolves |

Enforced by assertion in `test_label_encoding.py`. An encoding mismatch silently inverts all signals.

### 7.2 Architecture

```
Input: (batch, sequence_length, n_features)
          │
          ├─── Conv1D(filters, kernel=3,  padding='causal') ── BatchNorm ── GELU
          ├─── Conv1D(filters, kernel=7,  padding='causal') ── BatchNorm ── GELU
          └─── Conv1D(filters, kernel=15, padding='causal') ── BatchNorm ── GELU
                          │
                   Concatenate → (batch, seq_len, filters*3)
                          │
                  Dropout
                          │
           BiLSTM(hidden_dim, layers, dropout)
                          │
        [Optional] Multi-head Self-Attention  ← use_global_attention from config
        O(n²) in seq_len — profile T4 VRAM before committing
                          │
          ┌───────────────┴───────────────┐
   Global Avg Pool                  Last Timestep
          │                              │
          └──────── Concatenate ─────────┘
                          │
                Dense(256, GELU) → Dropout → Dense(128, GELU) → Dense(2, Softmax)
                                                                   ↑
                                                    index 0=Down, index 1=Up
```

Causal padding mandatory on all Conv1D layers.

### 7.3 BiLSTM Backward Pass — Not Leakage

The backward LSTM sees later timesteps within the input window — all fully closed historical bars at inference time. Genuine leakage would come from bars after the sequence end, prevented by causal padding and the `direction.py` guard. Document as a comment block in `architecture.py`.

### 7.4 ONNX Export

```python
torch.onnx.export(
    model, dummy_input, path,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=17
)
# Fixed-batch export causes silent inference failures at batch=1
```

### 7.5 Ensemble

```python
p_up              = mean(seed_probs[:, config.labels.class_up])
p_down            = 1 - p_up
seed_disagreement = std(seed_probs[:, config.labels.class_up])

# Direction string conversion (only place this happens):
direction = "up" if p_up >= p_down else "down"
```

### 7.6 Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, weight decay 1e-4 |
| LR schedule | OneCycleLR with warmup |
| Loss | Focal loss (γ=2) |
| Batch size | 1024 |
| Max epochs | 100, patience 10 |
| Stride | 50 bars |

---

## 8. Probability Calibration

Isotonic regression maps raw softmax `p_up` to empirically observed bar-up rates. Fit on `(raw_p_up, outcome)` pairs from validation folds. Re-fit after every weekly retrain. All downstream logic uses calibrated probabilities. Both raw and calibrated logged for drift monitoring.

---

## 9. Walk-Forward Evaluation

### Fold structure

```
|── Train (120d) ──|── Embargo (10d) ──|── Val (20d) ──|── Test (10d) ──|
```

Step 10 days. Minimum 12 folds. Embargo enforced structurally.

### Historical Polymarket price handling

`eval/metrics.py` loads `p_market_history.parquet` (from `data/polymarket_historical.py`) and joins on Unix timestamp. For each bar:

```python
p_market = historical_prices.get(bar_ts, config.backtest.assumed_market_price)
edge = p_model_calibrated - p_market
```

When `use_historical_prices: false` or no price exists for a bar, `assumed_market_price = 0.50` is used. This is documented with a comment at every call site. **BTC/ETH 5m/15m markets launched ~late 2024 — verify historical coverage before enabling.**

### Metrics per fold

| Metric | Notes |
|--------|-------|
| Hit rate (filtered) | On bars where simulated bet was placed |
| Net PnL after fees | Polymarket US 0.10% flat; `p_market` per above |
| Simulated Sharpe | Annualized daily |
| Max drawdown | Per fold equity curve |
| Turnover | Bets per day |
| Filter pass rate | % of bars signaled |
| Per-filter rejection rate | Per individual filter |
| Edge distribution | `p_model - p_market` at simulated bet time |
| Negative Kelly suppression rate | Post-edge trades killed by Kelly guard |
| ECE | Expected calibration error |
| p_market_source | `"historical"` or `"assumed_0.50"` — logged per fold for auditing |

---

## 10. Hyperparameter Search

### Initial (100 trials, Optuna + Hyperband, T4)

| Parameter | Range |
|-----------|-------|
| Conv filters | 128, 256, 512 |
| LSTM hidden dim | 256, 512 |
| LSTM layers | 1, 2, 3 |
| Attention heads | 4, 8 |
| `use_global_attention` | true, false |
| Dropout | 0.1 – 0.5 |
| Learning rate | 1e-4 – 1e-3 log |
| Sequence length | 500, 750, 1000 |

**Objective:** mean Val Sharpe across 6 folds using simulated Polymarket PnL. Prune Val Sharpe < 0 after fold 3. Profile T4 VRAM for (seq_len=1000, attention=true, filters=512) before launching.

### Quarterly re-search (30 trials)

Sequence length, LSTM layers, attention heads, `use_global_attention`.

---

## 11. Strategy Layer

### 11.1 Edge computation

```python
# strategy/edge.py
def compute_edge(p_model: float, p_market: float) -> float:
    return p_model - p_market

def should_bet(p_model: float, p_market: float) -> bool:
    edge = compute_edge(p_model, p_market)
    return (
        edge >= config.strategy.edge_threshold
        and p_model >= config.strategy.min_model_confidence
    )
```

### 11.2 Execution timing and order placement

```python
# utils/timing.py
def next_market_open_ts() -> int:
    now = int(time.time())
    return ((now // 300) + 1) * 300
```

**GTD order construction:**

```python
def place_bet(signal, p_market: float, market_close_ts: int):
    tick_size = get_tick_size(signal.token_id)   # fetched, never assumed
    # Round price to tick precision
    price_rounded = round(round(p_market / float(tick_size)) * float(tick_size), 10)
    # Round size to 2 decimal places (Polymarket precision requirement)
    shares = round(signal.stake_usdc / price_rounded, 2)

    order = {
        "tokenID":    signal.token_id,
        "price":      price_rounded,
        "size":       shares,
        "side":       "BUY",
        "order_type": "GTD",
        "expiration": market_close_ts - 61,  # hard stop at exchange level
        "negRisk":    False                  # mandatory; always False for binary markets
    }
    return client.create_order(order)
```

**Stale price check (two-point validation):**

```python
def execute_with_timing(signal, market_close_ts: int):
    if time.time() - signal.market_price_fetched_at > config.execution.stale_price_max_seconds:
        log("Signal-time market price stale — skipping")
        return None

    window_remaining = market_close_ts - time.time()
    if window_remaining < (300 - config.execution.max_entry_seconds):
        log("Entry window expired — skipping")
        return None

    while time.time() < (market_close_ts - 60):
        p_market = fetch_current_price(signal.direction)
        if time.time() - p_market.fetched_at > config.execution.stale_price_max_seconds:
            log("Fresh fetch stale — suppressing")
            return None
        if should_bet(signal.p_model, p_market.value):
            return place_bet(signal, p_market.value, market_close_ts)
        time.sleep(config.execution.poll_interval_seconds)

    log("Edge never confirmed — no bet")
    return None
```

**Heartbeat loop with recovery:**

```python
def run_heartbeat_loop():
    heartbeat_id = ""
    while orders_open():
        try:
            resp = client.post_heartbeat(heartbeat_id)
            heartbeat_id = resp["heartbeat_id"]
        except HTTP400Error as e:
            # 400 is recoverable — response body contains correct ID
            heartbeat_id = e.response["heartbeat_id"]
            log.warning(f"Heartbeat ID corrected — retrying with {heartbeat_id}")
            resp = client.post_heartbeat(heartbeat_id)
            heartbeat_id = resp["heartbeat_id"]
        time.sleep(config.execution.heartbeat_interval_seconds)
```

**Tick size change handling:**

```python
def on_tick_size_change(event):
    # Reinitialize client to clear cached tick size
    self.client = build_new_client()
    log.warning(f"Tick size changed for {event.token_id} — client reinitialized")
```

Subscribe to `tick_size_change` WebSocket events on startup. Do not rely on TTL refresh alone.

### 11.3 Filters

**Hard filters** (binary reject):

- **Edge:** `p_model - p_market < edge_threshold`
- **Confidence floor:** `p_model < min_model_confidence`
- **Seed agreement:** `seed_disagreement >= seed_disagreement_max`
- **Market availability:** no active Polymarket market for this symbol/TF
- **Liquidity:** `market_liquidity_usd < min_market_liquidity_usd`
- **Spread:** `bid_ask_spread > max_bid_ask_spread`
- **Stale price:** signal-time market price older than `stale_price_max_seconds`
- **Entry window:** past `max_entry_seconds` into window
- **Hurst regime:** trend only when `H > hurst_trend_threshold`; mean-reversion only when `H < hurst_mean_reversion_threshold`
- **Multi-asset coherence:** see guard below
- **Negative Kelly:** `kelly_fraction_raw < 0`
- **Macro blackout:** within `blackout_bars` of high-impact event

**Soft filters** (threshold adjustments):

- **Hurst conditional:** adjusted thresholds at strong trend confirmation
- **Shannon entropy:** threshold raised at warning level; hard reject at ceiling
- **Correlation cap:** reduced max concurrent bets when inter-asset correlation elevated

**Coherence filter guard:**

```python
def passes_coherence(signals: dict) -> tuple[bool, str]:
    if not config.strategy.coherence_required:
        return True, "disabled"
    required = config.strategy.coherence_assets
    active = [s["symbol"] for s in config.active_slots]
    if not all(asset in active for asset in required):
        return True, "bypassed: not all assets in active_slots"
    for asset in required:
        sig = signals.get(asset)
        if sig is None:
            if config.strategy.coherence_gap_behavior == "bypass":
                return True, f"bypassed: {asset} gap"
            else:
                return False, f"failed: {asset} gap"
        if sig.p_calibrated_up <= config.strategy.coherence_confidence_threshold:
            return False, f"failed: {asset} p_calibrated below threshold"
        if sig.edge <= 0:
            return False, f"failed: {asset} edge not positive"
    directions = [signals[a].direction for a in required]
    if len(set(directions)) != 1:
        return False, "failed: direction mismatch"
    return True, "passed"
# Returns (bool, status_string) — status_string is written to coherence_status in predictions.jsonl
```

### 11.4 Position sizing

```python
# strategy/sizer.py
p = p_up if signal.direction == "up" else p_down    # calibrated; string comparison
q = 1 - p
b = (1 - p_market) / p_market                       # implied odds

kelly_fraction_raw = (p * b - q) / b

if kelly_fraction_raw < 0:
    suppress_bet()

vol_scalar      = target_vol / realized_vol_14
capped_fraction = min(kelly_fraction_raw * kelly_fraction_multiplier, kelly_max_fraction)
final_fraction  = capped_fraction * vol_scalar
stake_usd       = portfolio_value * final_fraction * account.position_fraction
```

`position_fraction` is a per-account multiplier applied after all Kelly capping and vol scaling.

### 11.5 Economic calendar

Default: Nasdaq API. Fallback: `events_manual.json`. Interface: `calendar.is_blackout(ts, bars) -> bool`.

---

## 12. Cost Model

### Polymarket US — Flat Fee (ACTIVE)

```
fee = stake_usd * 0.001    (0.10%)

# Win: shares resolve at $1.00
pnl_gross = (stake_usd / market_price_at_entry) * (1.0 - market_price_at_entry)

# Loss: shares resolve at $0.00
pnl_gross = -stake_usd

pnl_net = pnl_gross - fee
min_fee = 0.0001 USDC
```

### Global Variable Fee (Reference — Non-US)

```
fee = C × p × 0.25 × (p × (1 - p))²
Maximum: ~1.56% at p = 0.50 → raise edge_threshold to 0.06 minimum
```

### No Funding

Polymarket binary markets have no funding mechanism.

### Chainlink Basis Risk

Training uses Binance OHLCV. Polymarket settles on Chainlink BTC/USD CX-Price Data Stream. Chainlink can lag during fast markets due to aggregation. Unhedgeable — documented in `data/fetcher.py`, monitored daily via `validate_chainlink_basis()`.

---

## 13. Position Manager

### Trade status handling

```python
for trade in client.get_trades(market=market_id):
    if trade["status"] == "CONFIRMED":
        outcome = "win" if resolution_matches_direction(trade) else "loss"
        pnl_gross = compute_pnl_gross(trade, outcome)
        fee = trade["stake_usd"] * config.costs.taker_fee_pct
        pnl_net = pnl_gross - fee
        portfolio_value += pnl_net
        write_trade_record(trade, outcome, pnl_gross, fee, pnl_net)
        remove_from_open_bets(trade)

    elif trade["status"] == "FAILED":
        log.warning(f"Trade FAILED permanently: {trade['id']}")
        portfolio_value += trade["stake_usd"]   # refund reserved stake
        remove_from_open_bets(trade)

    elif trade["status"] == "RETRYING":
        elapsed = (now - trade["placed_ts"]).minutes
        if elapsed > config.execution.retrying_max_minutes:
            log.warning(f"Trade RETRYING timeout: {trade['id']} — treating as FAILED")
            portfolio_value += trade["stake_usd"]
            remove_from_open_bets(trade)
        # else: hold, check next cycle
```

**Per-account fill tracking:** each account submits its own GTD order. Fill status is tracked independently per account. A fill for account_a is never assumed to imply a fill for account_b. Divergence is logged with `fill_status_account_a` and `fill_status_account_b` fields in `paper_trades.jsonl`.

**Mid-window exits:** all bets held to resolution. `config.paper_trading.allow_mid_window_exit: false` enforced in position manager; any mid-window sell call raises an error during paper phase.

**Important:** `USDC wallet balance ≠ portfolio_value`. Wallet balance is on-chain USDC. Portfolio value is the internal equity curve. Never read wallet balance as a proxy for performance.

**Hot-swap interface:** `has_open_bets() -> bool`.

**State files (per account):**

```
inference/state/{account_id}/
├── open_bets.json
└── portfolio.json   # { "value": 104.32, "last_updated": "2026-03-07T14:40:03Z" }
```

---

## 14. JSONL Logging

Append-only. All timestamps UTC. Log rotation and archiving addressed in v10 live execution plan.

### `predictions.jsonl`

```json
{
  "ts": "2026-03-07T14:35:00Z",
  "symbol": "BTC",
  "timeframe": "5m",
  "bar_open": 87290.0,
  "bar_close": 87340.5,
  "last_bar_direction": 1,
  "last_bar_direction_note": "class index of bar t (just closed) — NOT the bar being predicted (t+1)",
  "features_hash": "a3f9b2...",
  "feature_list_hash": "c72d1a...",
  "feature_list_hash_algorithm": "sha256_sorted_json",
  "model_version": "v1.4_20260307",
  "model_val_sharpe": 0.84,
  "model_val_sharpe_note": "backtest val Sharpe at deploy time — not live rolling Sharpe",
  "p_up_raw": 0.71,
  "p_up_calibrated": 0.68,
  "p_down_calibrated": 0.32,
  "seed_probs_up": [0.66, 0.71, 0.68, 0.69, 0.67],
  "seed_disagreement": 0.018,
  "market_price_up": 0.51,
  "market_price_fetched_at": "2026-03-07T14:35:02Z",
  "edge": 0.17,
  "regime": {
    "hurst": 0.59,
    "vol_zscore": 0.8,
    "realized_vol_14": 0.0023,
    "shannon_entropy": 0.42
  },
  "coherence_status": "bypassed: ETH not in active_slots",
  "signal": "up",
  "filtered_out": false,
  "filter_reason": null,
  "edge_threshold_used": 0.04,
  "blackout_active": false,
  "gap_detected": false,
  "market_available": true
}
```

`coherence_status` values: `"passed"` | `"bypassed: not all assets in active_slots"` | `"bypassed: ETH gap"` | `"failed: direction mismatch"` | `"failed: ETH p_calibrated below threshold"` | `"failed: ETH edge not positive"` | `"failed: ETH gap"` (when `coherence_gap_behavior: suppress_all`)

### `paper_trades.jsonl`

```json
{
  "trade_id": "BTC-5m-20260307-143500",
  "account_id": "account_a",
  "symbol": "BTC",
  "timeframe": "5m",
  "direction": "up",
  "model_version_at_entry": "v1.4_20260307",
  "model_val_sharpe_at_entry": 0.84,
  "market_id": "btc-updown-5m-1741355700",
  "market_open_ts": "2026-03-07T14:35:00Z",
  "market_close_ts": "2026-03-07T14:40:00Z",
  "bet_placed_ts": "2026-03-07T14:35:08Z",
  "order_type": "GTD",
  "order_expiration": 1741355879,
  "tick_size_used": "0.01",
  "neg_risk": false,
  "p_up_calibrated_at_entry": 0.68,
  "market_price_at_entry": 0.51,
  "edge_at_entry": 0.17,
  "seed_disagreement_at_entry": 0.018,
  "kelly_fraction_raw": 0.31,
  "kelly_fraction_capped": 0.08,
  "vol_scalar": 0.97,
  "kelly_fraction_final": 0.078,
  "position_fraction": 1.0,
  "stake_usd": 7.80,
  "shares_purchased": 15.29,
  "fill_status_account_a": "filled",
  "fill_status_account_b": "filled",
  "portfolio_value_at_entry": 104.32,
  "regime_at_entry": {
    "hurst": 0.59,
    "vol_zscore": 0.8,
    "shannon_entropy": 0.42
  },
  "resolution_ts": "2026-03-07T14:40:03Z",
  "resolution_open_price": 87340.5,
  "resolution_close_price": 87980.0,
  "trade_status": "CONFIRMED",
  "outcome": "win",
  "pnl_gross": 7.50,
  "fee": 0.0078,
  "pnl_net": 7.49,
  "portfolio_value_after_resolution": 111.81
}
```

---

## 15. Inference Pipeline (CPU — VPS)

```
Bar t closes (Unix ts divisible by 300)
      │
utils/timing.py: confirm alignment to Polymarket window boundary
      │
position_manager: check all open bets
  ├── CONFIRMED → PnL, portfolio update, write trade record
  ├── FAILED    → refund stake, portfolio update, remove
  └── RETRYING  → check elapsed; if > retrying_max_minutes → treat as FAILED
      │
calendar.is_blackout? → log blackout_active=true, skip
      │
bar_buffer.append(bar t) → gap check → dirty? log gap_detected=true, skip
      │
features.py: build feature vector
      │
RobustScaler transform (frozen artifact)
      │
Sequence assembly: last sequence_length bars
      │
ONNX Runtime × 5 seeds (parallel)
      │
Ensemble average → p_up_raw → direction string conversion
      │
Isotonic calibration → p_up_calibrated
      │
polymarket_client: fetch live p_market + staleness check
      │
strategy/edge.py: edge = p_up_calibrated - p_market
      │
strategy/filters.py: all hard and soft gates (coherence_status logged)
      │
strategy/sizer.py: Kelly sizing
      │
prediction_logger: write predictions.jsonl (always, including skips)
      │
If signal passes all filters:
  For each account in parallel:
    polymarket_client.execute_with_timing():
      Stale-price pre-check
      Poll market price every 5s within entry window
      Place GTD order (negRisk=false, price+size tick-rounded) when edge confirmed
      Run heartbeat loop with 400-recovery until order settles
      Log fill price, slippage, fill_status per account
    position_manager: add to open_bets for this account
```

### Atomic artifact deployment

```python
shutil.copytree(staging_dir, live_artifacts_dir + "_new")
os.rename(live_artifacts_dir + "_new", live_artifacts_dir)  # atomic on Linux
```

---

## 16. Weekly Retrain & Hot-Swap Guard

### Job sequence (Lightning.ai GPU)

1. Fetch new Binance OHLCV data
2. Run `validate_chainlink_basis()` — alert if elevated
3. Refresh economic calendar
4. Run Boruta-SHAP; validate feature list compatibility
5. Retrain 5 seeds × all active slots
6. Isotonic calibration
7. Walk-forward evaluation → `eval_summary.json`
8. ONNX export → staging directory
9. Hot-swap guard
10. If approved: push artifacts to VPS via rsync; atomic deployment on VPS

### Hot-swap guard

```
new_val_sharpe      = eval_summary.json → mean val fold Sharpe
deployed_val_sharpe = artifacts/deployed_model_val_sharpe.json

if new_val_sharpe > deployed_val_sharpe:
    validate feature list compatibility → block if incompatible
    position_manager.has_open_bets() → defer if any bet open
    push artifacts to VPS → atomic deployment
    update deployed_model_val_sharpe.json
else:
    retain current artifacts
    alert via configured alert_method
```

---

## 17. Paper Trading Success Gate

All conditions met simultaneously and sustained for 3 consecutive months.

| Condition | Threshold |
|-----------|-----------|
| Rolling 60-day Sharpe | > 1.0 |
| Max drawdown any 60-day window | < 15% |
| Sustained duration | 3 consecutive months |
| Drawdown recovery | Recovered from at least one drawdown > 5% |
| Net PnL positive | `pnl_net` after Polymarket fees |
| Mean Chainlink basis | < 15 bps over paper trading period |

Meeting the gate enables a manual go/no-go decision. It does not automatically trigger live execution.

**v10 prerequisites:** live execution requires USDC.e on Polygon, API wallet with CTF Exchange contract allowances set, gas funding, and log rotation infrastructure. None are in scope for v9 paper trading.

---

## 18. Build Order

**BTC 5m only until paper trading gate is met.**

### Phase 1 — Data Foundation

1. `data/fetcher.py` — Binance OHLCV; Chainlink basis stub
2. `data/features.py` — all features, lookahead-tested
3. `data/polymarket_historical.py` — load Jon-Becker dataset, align to Binance bars, output `p_market_history.parquet`
4. `data/validate.py` — NaN audit, distribution checks, `validate_chainlink_basis()`
5. `labels/direction.py` — returns 0/1, not +1/-1, not strings
6. `tests/test_label_encoding.py` — assert return type is int; assert 0=Down, 1=Up; assert no +1/-1
7. `tests/test_features_lookahead.py`
8. **Checkpoint:** zero NaN features, ~50/50 label balance, leakage tests pass, 270+ days of clean history confirmed, `p_market_history.parquet` coverage verified and documented

### Phase 2 — Feature Selection

9. `selection/boruta_shap.py`
2. **Checkpoint:** shadow competition working; Shannon entropy result noted; `feature_list.json` hashed with SHA-256

### Phase 3 — Evaluation Harness

11. `eval/walkforward.py`, `eval/metrics.py`
2. Document `p_market` source logic (historical parquet → 0.50 fallback) prominently
3. `tests/test_harness_random_baseline.py`
4. **Checkpoint:** random baseline → near-zero PnL, ~0.50 hit rate; `p_market_source` field logged per fold; any other result = harness bug

### Phase 4 — Baseline Model

15. `models/architecture.py` — 2-class, class ordering documented, BiLSTM backward pass documented
2. `models/train.py`, `models/ensemble.py` — direction string conversion in `ensemble.py`
3. `calibration/isotonic.py`
4. **Checkpoint:** loss decreases; ECE lower post-calibration; direction strings flow cleanly to strategy layer

### Phase 5 — Hyperparameter Search (GPU)

19. `tuning/optuna_search.py`
2. Profile T4 VRAM before launching; fix winning config in `config.yaml`
3. **Checkpoint:** consistent Val Sharpe > 0; `use_global_attention` result noted

### Phase 6 — Strategy Layer

22. `strategy/edge.py`
2. `strategy/calendar.py`
3. `strategy/filters.py` — all gates; coherence guard returns `(bool, status_string)`; gap behavior respected
4. `strategy/sizer.py` — binary Kelly + `position_fraction`
5. `tests/test_edge_computation.py`
6. **Checkpoint:** simulated bet pass rate 10–35% of bars; coherence_status `"bypassed"` fires correctly when single-slot; negative Kelly suppression rate logged; all filter rejection rates logged

### Phase 7 — Wallet, Polymarket Client & Inference

**Step 0 — One-time setup (before any code):**

```
- Fund VPS wallet with USDC.e on Polygon
- Call setApprovalForAll on CTF Exchange contract
- Call setAllowance on USDC.e for Exchange address
- Run client.create_api_key() → store POLY_API_KEY, POLY_SECRET, POLY_PASSPHRASE in VPS .env
- Verify via client.get_api_keys() and client.get_balance()
- Confirm USDC wallet balance ≠ portfolio_value (they are different things)
```

1. `utils/timing.py`; `tests/test_timing_alignment.py` — exact boundary, one-second-before, one-second-after UTC inputs
2. `inference/polymarket_client.py` — market data, GTD orders with `negRisk=false`, tick size fetch (price AND size), `tick_size_change` WebSocket handler, heartbeat with 400-recovery
3. `inference/bar_buffer.py`
4. `inference/export_onnx.py` — dynamic batch axis
5. `inference/live.py`
6. `inference/position_manager.py` — CONFIRMED/FAILED/RETRYING+timeout, per-account fill tracking, `allow_mid_window_exit` enforcement
7. `logging/prediction_logger.py`, `logging/trade_logger.py`
8. `tests/test_polymarket_client.py` — `negRisk=false` present; GTD rejected without `negRisk`; price rounded to tick; size rounded to 2 dp; heartbeat 400 recovery; tick_size_change reinitializes client
9. `tests/test_tick_size_precision.py` — price AND size correctly rounded for all three tick size tiers (0.01, 0.001)
10. `tests/test_position_manager.py` — CONFIRMED win/loss PnL; FAILED refund; RETRYING timeout triggers refund; account_a fill does not imply account_b fill
11. `tests/test_gap_handling.py`
12. **Checkpoint:** historical replay with mocked Polymarket prices; `negRisk` present in every order; tick rounding correct for price and size; heartbeat 400 recovery verified; RETRYING timeout fires at configured threshold; per-account fill divergence logged correctly; `allow_mid_window_exit` raises on attempt

### Phase 8 — GPU Jobs, VPS Automation & Deployment

**Step 0 — SSH keypair provisioning:**

```
- Generate SSH keypair for artifact transfer
- Add public key to VPS ~/.ssh/authorized_keys
- Store private key as Lightning.ai environment secret
- Test rsync connectivity before building weekly_retrain.py
```

1. `jobs/weekly_retrain.py` — full pipeline + rsync artifact push + atomic VPS deployment
2. `jobs/quarterly_retune.py`
3. `jobs/healthcheck.py` — canary, feature drift, calibration drift, Chainlink basis, live PnL; alert fires to configured `alert_method`
4. `tests/test_atomic_deployment.py`
5. VPS: systemd service for `live.py`; cron for `healthcheck.py`; configure `alert_webhook_url`
6. **Checkpoint:** end-to-end retrain simulation; hot-swap rejects degraded model; deployment defers with open bet; atomic swap survives mid-write interruption; artifacts arrive on VPS; alert fires to webhook

### Phase 9 — Paper Trading (BTC 5m)

46. Run paper trading against live Polymarket markets for both accounts
2. Monitor weekly: edge distribution, coherence_status breakdown, filter pass rates, Chainlink basis, Sharpe, drawdown, per-account portfolio values
3. Do not expand to additional slots until success gate is met

### Phase 10 — Expansion

49. ETH 5m → add to `active_slots`; this **activates the coherence filter on BTC 5m** — expect BTC bet frequency to change; monitor 2 weeks before proceeding
2. BTC 15m → independent model, separate Polymarket market
3. ETH 15m → last; validate risk limits across all four concurrent slots carefully

---

## 19. Health Monitoring

| Monitor | Trigger | Action |
|---------|---------|--------|
| **Canary** | Logistic regression on 5 stable features; accuracy < `canary_accuracy_floor` for `canary_lookback_days` | Pause inference; alert |
| **Feature drift** | Nightly KL divergence vs training distribution > `feature_kl_alert_threshold` | Alert; flag for retrain review |
| **Calibration drift** | Rolling gap between raw and calibrated `p_up` > `calibration_drift_alert_threshold` | Alert; reflag calibration |
| **Chainlink basis** | Daily `validate_chainlink_basis()`; mean absolute basis > `basis_warn_bps` | Alert; investigate feed |
| **Live PnL** | Rolling 7-day net PnL < `live_pnl_alert_floor` | Alert; manual review |

All alerts routed to `config.health.alert_method`. If `log_only`, alerts write to a dedicated `alerts.log` file.

---

## 20. Key Rules

These rules take precedence over everything. Check this list when something feels wrong before touching code.

1. **Softmax index 0 = Down, index 1 = Up.** Fixed. Never changed. Enforced by `test_label_encoding.py`. An inversion here silently bets backwards on every trade.

2. **`direction.py` returns 0 or 1 only.** Never +1/-1. Never strings. The one place integer class indices exist. Conversion to strings `"up"`/`"down"` happens only in `ensemble.py`.

3. **All strategy and execution code uses direction strings `"up"`/`"down"`.** Never class indices below the ensemble layer.

4. **`negRisk: False` on every order.** Always. Omitting it causes silent order rejection.

5. **Heartbeat 400 is recoverable.** The response body contains the corrected ID. A 400 is not a crash — it is a re-sync. The loop has a recovery path.

6. **Price AND size are tick-rounded.** Price to tick size. Size to 2 decimal places. Both. Unrounded size causes rejection.

7. **`tick_size_change` events reinitialize the client.** TTL refresh alone is not sufficient. Subscribe to WebSocket events on startup.

8. **`p_market = 0.50` is a backtest fallback, not a constant.** During backtest, use historical prices where available. Log `p_market_source` per fold.

9. **`USDC wallet balance ≠ portfolio_value`.** Never read wallet balance as equity. Portfolio value is tracked internally in `portfolio.json`.

10. **RETRYING trades have a timeout.** After `retrying_max_minutes`, treat as FAILED and refund stake. Never hold indefinitely.

11. **Coherence filter bypasses gracefully when ETH not in `active_slots`.** Logs `coherence_status: "bypassed: not all assets in active_slots"`. Does not block BTC bets.

12. **Adding ETH to `active_slots` activates the coherence filter on BTC.** BTC bet frequency will change. Monitor for 2 weeks before adding further slots.

13. **Per-account fills are independent.** A fill for account_a does not imply a fill for account_b. Log both separately.

14. **SSH keypair must be provisioned before first rsync.** Private key in Lightning.ai secrets. Public key in VPS `authorized_keys`. No key = no artifact delivery.

15. **`create_api_key()` must be called once before any authenticated order.** POLY_API_KEY, POLY_SECRET, and POLY_PASSPHRASE are all required. Wallet address alone is not sufficient.

16. **Never expand to a new slot until the paper trading gate is fully met.** All six gate conditions simultaneously sustained for 3 consecutive months.

17. **`allow_mid_window_exit: false` during paper trading.** All bets held to resolution. Any mid-window sell path raises an error.

18. **Alert destination must be configured before Phase 8.** `log_only` is valid but must be explicit. Silent alerts defeat health monitoring entirely.

19. **Jon-Becker coverage must be verified before `use_historical_prices: true`.** BTC/ETH 5m/15m markets launched late 2024. Document verified coverage range in Phase 1 checkpoint.

20. **`feature_list_hash` uses SHA-256 of sorted JSON content.** Always. Consistent across all environments.
