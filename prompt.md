# Infrastructure Upgrade: T4/CPU → B200 + Full Boruta-SHAP + Optuna Search

## Context

Two degradations were made to get the system running on constrained hardware. Both must be fixed before anything else runs:

1. **Training config degraded for T4:** `batch=256, seq=500` to fit within 16GB VRAM
1. **Boruta-SHAP degraded for CPU:** reduced estimator count, fewer shadow iterations, possibly reduced tree depth or sample size to make it tractable without a GPU

Both compromises are now gone. We are upgrading to **Modal.com B200 (180GB usable VRAM / 192GB HBM3e physical, $6.25/hr)**. Modal does not charge while waiting for GPU allocation.

**Before starting: commit all current state to git.** This is your rollback point if any step needs to be reverted.

**Also before starting: register the Modal Secret from your local machine** (cannot run inside a container):

```bash
modal secret create vps-ssh-key SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)"
```

**VPS firewall prerequisite:** Modal containers egress from dynamic AWS/GCP IPs. If your VPS has `ufw` or `iptables` rules restricting SSH ingress to a specific source IP, every rsync call will fail with `CalledProcessError` and the container will die before writing any artifact. Run this once from your VPS before submitting any GPU job:

```bash
sudo ufw allow 22
```

**Execution order for this session — complete each step fully before starting the next:**

1. Restore full training config
1. Restore full Boruta-SHAP config
1. Provision Modal Volume for dataset and config
1. Add job duration guards to all Modal GPU functions
1. Run full Boruta-SHAP on B200 (BTC 5m)
1. Run full Optuna search on B200 (BTC 5m)

Step 4 (timeout guards) must be completed before any GPU job is submitted. Do not submit to Modal until guards are in place.

-----

## Step 1 — Restore Full Training Config

Undo all T4 compromises in `config.yaml` and `train.py`.

```yaml
# config.yaml — restore these values
model:
  batch_size: 1024
  sequence_length: 750           # starting value — Optuna will search 500/750/1000
  norm_type: layer_norm          # correct for causal Conv at variable seq lengths
  # Remove physical_batch_size entirely — no longer a valid config field
  # Remove accumulation_steps entirely — no longer a valid config field

infrastructure:
  gpu_platform: modal
  gpu_type: B200                 # confirmed Modal identifier — uppercase, see Step 4
  artifact_remote_path: "user@vps-ip:/home/user/cnn_lstm_v1/inference/artifacts/"
  jobs_remote_path: "user@vps-ip:/home/user/cnn_lstm_v1/jobs/"  # must end with /; rstrip("/") in log_gpu_timing guards against omission
  config_remote_path: "user@vps-ip:/home/user/cnn_lstm_v1/config.yaml"  # explicit path — avoids fragile relative navigation
```

In `train.py`: remove the gradient accumulation loop entirely. Replace with a standard single forward-backward pass:

```python
# REMOVE this entire block:
accumulation_steps = config.model.batch_size // config.model.physical_batch_size
optimizer.zero_grad()
for i, (x, y) in enumerate(dataloader):
    loss = focal_loss(model(x), y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# REPLACE with standard loop:
for x, y in dataloader:
    optimizer.zero_grad()
    loss = focal_loss(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

Keep `LayerNorm` after every `Conv1D` — this is the correct architectural choice for causal convolutions at variable sequence lengths. The justification is training stability, not VRAM.

Flash Attention 2 stays — faster on B200, mathematically identical outputs.

Also revert the `OneCycleLR` scheduler instantiation. The accumulation implementation previously fixed `steps_per_epoch` to use logical steps (`len(dataloader) // accumulation_steps`). With accumulation removed, `steps_per_epoch` must revert to the physical batch count:

```python
# CORRECT — accumulation removed, steps_per_epoch is just len(dataloader)
scheduler = OneCycleLR(
    optimizer,
    max_lr=config.model.learning_rate,
    steps_per_epoch=len(dataloader),   # physical batch count — not divided by accumulation_steps
    epochs=config.model.max_epochs
)
```

If the agent that implemented accumulation applied the logical-steps fix, it will not automatically revert. Check explicitly that `steps_per_epoch` is `len(dataloader)` and not `len(dataloader) // accumulation_steps` or any variant of that expression.

-----

## Step 2 — Restore Full Boruta-SHAP Config

Identify every parameter in `selection/boruta_shap.py` and its config that was reduced for CPU execution and restore it to the full production specification.

The full production Boruta-SHAP spec is:

- **Base estimator:** XGBoost or CatBoost, GPU-accelerated
  - Use the following version check at the top of `boruta_shap.py` — do not specify both params:

```python
import xgboost as xgb
from packaging import version

xgb_gpu_param = (
    {"device": "cuda"}
    if version.parse(xgb.__version__) >= version.parse("2.0.0")
    else {"tree_method": "gpu_hist"}
)
# Pass **xgb_gpu_param to XGBClassifier(...)
```

- **Shadow iterations:** `n_shadow_iterations = 50`. Default to 50 — B200 makes the difference between 50 and 100 negligible for runtime, and 50 is the conservative, auditable default. Log the value explicitly in the restoration log so it is on record and intentional, not incidental.
- **SHAP computation:** computed across all training instances, not a subsample
- **Feature confirmation:** binomial test across all iterations, not early-stopped
- **Runs per fold:** once per walk-forward fold — this does not change
- **Output:** versioned `feature_list.json` written to the staging artifacts directory (same location the hot-swap guard reads from — not to `selection/`)

For every parameter that was degraded, restore it and log what the degraded value was and what it has been restored to:

```json
{
  "restored_at": "2026-03-10T...",
  "changes": [
    {"param": "n_estimators", "degraded_value": 50, "restored_value": 200},
    {"param": "n_shadow_iterations", "degraded_value": 10, "restored_value": 50},
    {"param": "shap_sample_size", "degraded_value": 1000, "restored_value": "all"}
  ]
}
```

If you are uncertain whether a parameter was degraded or was always its current value, restore it to the plan's specification and log it as a precautionary restoration.

Optuna trials use the fixed confirmed feature list from `feature_list.json` in staging. Do not run Boruta-SHAP inside individual Optuna trials.

-----

## Step 3 — Provision Modal Volume for Dataset and Config

Every Modal GPU container starts with an empty filesystem. The OHLCV dataset, pre-built features, and `config.yaml` are not automatically available. This must be resolved before any GPU job runs.

### Decision: Modal Volume

Use a **Modal Volume** to share the dataset and config across all containers. This is the correct strategy for data that is read by many containers but written rarely.

```python
# Create once from local machine:
import modal
vol = modal.Volume.from_name("cnn-lstm-data", create_if_missing=True)

# Mount in every GPU function:
@app.function(
    gpu="B200",
    volumes={"/data": vol},
    ...
)
def run_boruta_shap():
    # Dataset available at /data/btc_5m_ohlcv.parquet
    # Config available at /data/config.yaml
    ...
```

### One-time upload (run from local machine before any GPU job)

```python
# upload_to_volume.py — run once
import modal

vol = modal.Volume.from_name("cnn-lstm-data", create_if_missing=True)
with vol.batch_upload() as batch:
    # Remote paths are relative to the Volume root.
    # With the Volume mounted at /data in the container, "config.yaml" → /data/config.yaml.
    # Do NOT include /data/ prefix here — it would create /data/data/config.yaml.
    batch.put_file("config.yaml",                  "config.yaml")
    batch.put_file("data/btc_5m_ohlcv.parquet",    "btc_5m_ohlcv.parquet")
    batch.put_file("data/btc_5m_features.parquet", "btc_5m_features.parquet")
```

Add `modal_volume_name` to `config.yaml`:

```yaml
infrastructure:
  modal_volume_name: "cnn-lstm-data"
  modal_volume_mount: "/data"
```

### Config loading inside containers

Every container function must load `config.yaml` explicitly from the Volume mount. Do not rely on a module-level import that references a local path — that path does not exist in the empty container filesystem and will raise `FileNotFoundError` or silently read a stale config.

Add this block immediately after the CUDA assert in every GPU function body (use your existing config loader if one exists):

```python
import yaml
from types import SimpleNamespace

with open("/data/config.yaml") as f:
    _cfg = yaml.safe_load(f)

def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})

config = _ns(_cfg)
# config.infrastructure.jobs_remote_path, config.infrastructure.artifact_remote_path, etc.
# are now available for the rest of the function body.
```

### All local file paths inside containers use `/tmp/` for writes

Modal containers have empty filesystems — relative paths like `"jobs/timing_log.jsonl"` will raise `FileNotFoundError` because the `jobs/` directory does not exist. All files written inside containers must use `/tmp/` with explicit `os.makedirs`:

```python
# Wrong — relative path, directory won't exist:
local_path = "jobs/timing_log.jsonl"

# Correct — absolute path, directory created explicitly:
import os
os.makedirs("/tmp/jobs", exist_ok=True)
local_path = "/tmp/jobs/timing_log.jsonl"
```

Apply this convention to every locally-written file across all GPU functions:

- Timing log: `/tmp/jobs/timing_log.jsonl`
- Staging artifacts: `/tmp/staging_artifacts/`
- Boruta-SHAP results: `/tmp/selection/boruta_shap_results.json`
- Optuna results: `/tmp/tuning/optuna_results.json`
- Updated config: `/tmp/config.yaml`

-----

## Step 4 — Add Job Duration Guards to All Modal GPU Functions

Every `@app.function` that uses a GPU must have an explicit `timeout` parameter before any job is submitted. Modal enforces this at the container level — it is a hard kill, not a soft signal. A hung job on a B200 at $6.25/hr is expensive.

### Modal B200 GPU string

Modal's confirmed identifier is `"B200"` (uppercase). Do not use lowercase `"b200"` — it causes silent fallback to a different GPU or submission failure, and will NOT be caught by the VRAM preflight check in Step 6.

Also available: `"B200+"`, which schedules on either B200 or B300. B300 requires CUDA ≥ 13.0 — do not use `"B200+"` unless you have confirmed your CUDA version. Stick with `"B200"` for now.

Modal exposes **180GB usable VRAM** to B200 containers (physical is 192GB HBM3e). The preflight check in Step 6 should reason against 180GB, not 192GB.

### Timeout table

| Job                             | Hard timeout  | Estimated actual runtime | Note                                       |
|---------------------------------|---------------|--------------------------|--------------------------------------------|
| Boruta-SHAP                     | **30 minutes**| ~5 min (single slot)     | Scales with slot count at expansion        |
| Optuna search (100 trials)      | **60 minutes**| ~25–30 min               |                                            |
| Weekly retrain (5 seeds)        | **30 minutes**| ~10–12 min (single slot) | Must be revisited before adding a 3rd slot |
| Quarterly re-search (30 trials) | **30 minutes**| ~8 min                   |                                            |

**The retrain and Boruta-SHAP timeouts apply to single-slot only.** Both scale linearly with active slot count. Revisit before expanding beyond 2 slots.

### SSH key helper

The Modal Secret registered in the preamble (`vps-ssh-key`) is mounted in every GPU function. Write the key to a temp file inside the container using this helper:

```python
import os, tempfile

def get_ssh_key_path() -> str:
    """Write SSH private key from Modal Secret to a temp file for rsync."""
    key = os.environ["SSH_PRIVATE_KEY"]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as f:
        f.write(key)
        key_path = f.name
    os.chmod(key_path, 0o600)
    return key_path
```

Pass `key_path` to rsync via `-e "ssh -i {key_path} -o StrictHostKeyChecking=no"`.

### rsync retry helper

All rsync calls use `check=True`. Wrap every rsync in this helper to handle transient VPS hiccups without silently dropping artifacts. On final failure the exception is still raised — never swallow it, as silent artifact loss is worse than a visible job failure.

```python
import subprocess, time as _time

def rsync_with_retry(cmd: list, retries: int = 3, delay: float = 5.0) -> None:
    """Run an rsync command with retries. Raises CalledProcessError on final failure."""
    for attempt in range(retries):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries - 1:
                raise
            _time.sleep(delay)
```

### GPU wall-clock logging

Start the clock from the moment the GPU function body begins executing — not from job submission, not from Modal container startup.

```python
import time, json, logging

log = logging.getLogger(__name__)

def log_gpu_timing(job_name: str, gpu_start_ts: float,
                   timeout_seconds: int, estimated_seconds: int,
                   remote_jobs_path: str):
    # remote_jobs_path: pass config.infrastructure.jobs_remote_path explicitly
    # Making it a parameter avoids implicit module-level config access.
    gpu_elapsed = time.time() - gpu_start_ts
    record = {
        "job":               job_name,
        "gpu":               "B200",
        "started_at":        gpu_start_ts,
        "elapsed_seconds":   gpu_elapsed,
        "estimated_seconds": estimated_seconds,
        "timeout_seconds":   timeout_seconds,
        "headroom_seconds":  timeout_seconds - gpu_elapsed,
        "headroom_pct":      round((timeout_seconds - gpu_elapsed) / timeout_seconds * 100, 1),
        "cost_usd":          round(gpu_elapsed / 3600 * 6.25, 4)  # B200 rate $6.25/hr
    }

    import os
    os.makedirs("/tmp/jobs", exist_ok=True)
    local_path = "/tmp/jobs/timing_log.jsonl"
    with open(local_path, "a") as f:
        json.dump(record, f)
        f.write("\n")

    # Sync to VPS immediately — Modal containers are ephemeral.
    remote_path = remote_jobs_path.rstrip("/") + "/timing_log.jsonl"
    key_path = get_ssh_key_path()
    try:
        rsync_with_retry([
            "rsync", "-az",
            "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
            local_path, remote_path
        ])
    finally:
        os.unlink(key_path)  # always clean up — even if rsync fails

    log.info(
        f"GPU job complete | job={job_name} | "
        f"elapsed={gpu_elapsed:.1f}s | headroom={timeout_seconds - gpu_elapsed:.1f}s"
    )
```

Call `log_gpu_timing()` at the end of every GPU function body, before the function returns. The rsync must complete before the container exits — do not run it in a background thread.

### Implementation

`gpu_start_ts` must be the **literal first line** of each function body — before any imports, setup, or data loading. Any code above it inflates the elapsed time measurement. Config is loaded explicitly from the Volume immediately after the CUDA assert (see Step 3):

```python
@app.function(
    gpu="B200",      # confirmed Modal identifier — uppercase
    timeout=1800,    # 30 minutes hard kill
    volumes={"/data": modal.Volume.from_name("cnn-lstm-data")},
    secrets=[modal.Secret.from_name("vps-ssh-key")]
)
def run_boruta_shap():
    gpu_start_ts = time.time()  # ← first line, before anything else

    # CUDA sanity check — B200 requires CUDA 12.4+
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cuda_ver = tuple(int(x) for x in torch.version.cuda.split("."))
    assert cuda_ver >= (12, 4), f"CUDA 12.4+ required for B200, got {torch.version.cuda}"

    # Load config from Volume — explicit in-body load required; no local path exists in container
    import yaml
    from types import SimpleNamespace
    def _ns(d):
        return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    with open("/data/config.yaml") as f:
        config = _ns(yaml.safe_load(f))

    import xgboost as xgb
    # ... rest of function ...
    log_gpu_timing("boruta_shap", gpu_start_ts, 1800, 300,
                   config.infrastructure.jobs_remote_path)


@app.function(
    gpu="B200",
    timeout=3600,    # 60 minutes hard kill
    volumes={"/data": modal.Volume.from_name("cnn-lstm-data")},
    secrets=[modal.Secret.from_name("vps-ssh-key")]
)
def run_optuna_search():
    gpu_start_ts = time.time()  # ← first line

    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cuda_ver = tuple(int(x) for x in torch.version.cuda.split("."))
    assert cuda_ver >= (12, 4), f"CUDA 12.4+ required for B200, got {torch.version.cuda}"

    import yaml
    from types import SimpleNamespace
    def _ns(d):
        return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    with open("/data/config.yaml") as f:
        config = _ns(yaml.safe_load(f))

    # ... gate check, then rest of function ...
    log_gpu_timing("optuna_search", gpu_start_ts, 3600, 1800,
                   config.infrastructure.jobs_remote_path)


@app.function(
    gpu="B200",
    timeout=1800,    # 30 minutes — single slot only, revisit at 3+ slots
    volumes={"/data": modal.Volume.from_name("cnn-lstm-data")},
    secrets=[modal.Secret.from_name("vps-ssh-key")]
)
def run_weekly_retrain():
    gpu_start_ts = time.time()  # ← first line

    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cuda_ver = tuple(int(x) for x in torch.version.cuda.split("."))
    assert cuda_ver >= (12, 4), f"CUDA 12.4+ required for B200, got {torch.version.cuda}"

    import yaml
    from types import SimpleNamespace
    def _ns(d):
        return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    with open("/data/config.yaml") as f:
        config = _ns(yaml.safe_load(f))

    # ... rest of function ...
    log_gpu_timing("weekly_retrain", gpu_start_ts, 1800, 660,
                   config.infrastructure.jobs_remote_path)
```

-----

## Step 5 — Run Full Boruta-SHAP on B200

With restored config and timeout guards in place, run Boruta-SHAP on BTC 5m as specified in Section 5.6 of the plan.

### Inputs required

- Full BTC 5m OHLCV dataset — loaded from `/data/btc_5m_ohlcv.parquet` via Modal Volume mounted at `/data`
- Pre-built features — loaded from `/data/btc_5m_features.parquet` via Modal Volume mounted at `/data`

(Both files were uploaded to the Volume in Step 3. The Volume is mounted at `/data` in the container — treat as read-only for dataset files.)

### What it does

- Fits XGBoost/CatBoost (GPU-accelerated) on real + shadow features
- Computes SHAP values across all training instances
- Runs 50 shadow iterations (`n_shadow_iterations = 50`) with binomial confirmation test
- Outputs versioned `feature_list.json` to the staging artifacts directory

### On completion

1. Write `feature_list.json` to the staging artifacts directory with content hash — this is where the hot-swap guard and Step 6 (Optuna) will read it from
1. Write full results to `/tmp/selection/boruta_shap_results.json` including:
   - Confirmed features list
   - Rejected features list
   - Iteration stats
   - Shannon entropy result — log as `"confirmed"` or `"rejected"` (not both, not ambiguous)
   - Restoration log from Step 2 (what was degraded and what was restored)
   - `feature_list_hash` — computed as `hashlib.sha256(json.dumps(sorted(confirmed_features), separators=(',', ':')).encode()).hexdigest()`. Hash only the confirmed feature list, not the whole file. `sorted()` and `separators=(',', ':')` are both required: `sorted()` for key-order determinism, `separators` to suppress the default whitespace in `json.dumps` which is consistent within a Python version but not guaranteed across environments.
1. Rsync both artifacts to VPS — `log_gpu_timing()` only syncs the timing log, not these files. Without this step, Step 6's gate check has nothing to pull and crashes:

```python
key_path = get_ssh_key_path()
try:
    # feature_list.json → artifacts dir (where hot-swap guard and Step 6 read from)
    rsync_with_retry([
        "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        "/tmp/staging_artifacts/feature_list.json",
        config.infrastructure.artifact_remote_path.rstrip("/") + "/"
    ])
    # boruta_shap_results.json → jobs/selection/ for audit trail
    remote_selection = config.infrastructure.jobs_remote_path.rstrip("/") + "/selection/"
    rsync_with_retry([
        "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        "/tmp/selection/boruta_shap_results.json",
        remote_selection
    ])
finally:
    os.unlink(key_path)
```

1. Call `log_gpu_timing()` — rsyncs timing log to VPS
2. **Do not proceed to Step 6 until `feature_list.json` exists in the staging artifacts directory and its content hash is written.** Optuna trials must train on the confirmed feature list only.

### Expected output structure

```json
{
  "symbol": "BTC",
  "timeframe": "5m",
  "run_ts": "2026-03-10T...",
  "gpu": "B200",
  "n_features_input": 38,
  "n_features_confirmed": 28,
  "n_features_rejected": 10,
  "shannon_entropy_result": "confirmed",
  "feature_list_hash": "...",
  "confirmed_features": ["..."],
  "rejected_features": ["..."],
  "restoration_log": { "...": "..." },
  "timing": { "elapsed_seconds": 0, "timeout_seconds": 1800 }
}
```

-----

## Step 6 — Run Full Optuna Search on B200

With Boruta-SHAP complete and `feature_list.json` written to staging, run the Optuna hyperparameter search as specified in Section 10 of the plan.

### Gate check — verify `feature_list.json` before doing anything else

Add this as the first substantive block inside `run_optuna_search()`, immediately after `gpu_start_ts = time.time()`, the CUDA assert, and the config load:

```python
import hashlib, json, os
from pathlib import Path

# Pull feature_list.json from VPS into container — each Modal container starts empty.
# run_boruta_shap() wrote it to the VPS; this container has no local copy.
key_path = get_ssh_key_path()
local_staging = Path("/tmp/staging_artifacts")
local_staging.mkdir(parents=True, exist_ok=True)
try:
    rsync_with_retry([
        "rsync", "-az",
        "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        f"{config.infrastructure.artifact_remote_path.rstrip('/')}/feature_list.json",
        str(local_staging / "feature_list.json")
    ])
finally:
    os.unlink(key_path)

feature_list_path = local_staging / "feature_list.json"
assert feature_list_path.exists(), (
    "feature_list.json not found in staging — run Boruta-SHAP (Step 5) first"
)
with open(feature_list_path) as f:
    feature_data = json.load(f)

# Hash only the confirmed feature list — a file cannot contain its own SHA-256 hash.
# sorted() + separators are required for determinism across Python versions/environments.
expected_hash = feature_data["feature_list_hash"]
actual_hash = hashlib.sha256(
    json.dumps(sorted(feature_data["confirmed_features"]), separators=(',', ':')).encode()
).hexdigest()
assert actual_hash == expected_hash, (
    f"feature_list.json hash mismatch: {actual_hash} != {expected_hash} — file may be corrupted"
)
```

This prevents a silent run on a missing or corrupted feature list. The assert fires before any GPU compute is used.

### Pre-flight VRAM check

Before launching 100 trials, run one full trial (all 6 folds, early stopping enabled) at the most expensive config to confirm it fits in B200 VRAM:

- `seq=1000, attention=true, filters=512, batch=1024`

If this trial completes all 6 folds without OOM error, proceed to the full search. This takes 2–3 minutes. A wrong GPU string (see Step 4) will not be caught here — the preflight only catches VRAM issues.

### Search space

```python
sequence_length:       [500, 750, 1000]
conv_filters:          [128, 256, 512]
lstm_hidden_dim:       [256, 512]
lstm_layers:           [1, 2, 3]
attention_heads:       [4, 8]
use_global_attention:  [true, false]
dropout:               0.1 – 0.5       # continuous uniform
learning_rate:         1e-4 – 1e-3     # log scale
```

### Search configuration

- **100 trials**, `MedianPruner` (replaces HyperbandPruner — see note below)
- **Folds per trial:** 6 (not 12 — the 12-fold minimum applies to the full production walk-forward retrain, not to Optuna trials)
- **Objective:** mean Val Sharpe across 6 folds; simulated Polymarket PnL after fees (`p_market=0.50`) as tiebreaker
- **Pruning:** `MedianPruner` handles Optuna-level pruning; additionally apply a hard Sharpe gate after fold 3:

```python
for fold_idx in range(6):
    val_sharpe = train_fold(fold_idx)
    trial.report(val_sharpe, step=fold_idx)

    # Hard Sharpe gate fires after fold 3 (index 2) only
    if fold_idx == 2 and val_sharpe < 0:
        trial.set_user_attr("pruned_by", "sharpe_gate")
        raise optuna.TrialPruned()

    # MedianPruner checks after every fold — needs intermediate values
    # from multiple folds to compare against the median of completed trials.
    # Checking only at fold 3 gives it no opportunity to fire on folds 4/5.
    if trial.should_prune():
        trial.set_user_attr("pruned_by", "median_pruner")
        raise optuna.TrialPruned()
```

**Why MedianPruner instead of HyperbandPruner:** Hyperband uses its own successive-halving schedule and fires at different fold thresholds than the manual Sharpe check. Running both produces ambiguous logs and unpredictable pruning order. MedianPruner + hard Sharpe gate is simpler, predictable, and auditable.

- **Batch size:** `batch_size=1024`, no accumulation
- **Feature set:** confirmed features from `feature_list.json` in staging only — not the full feature set
- **Symbol/TF:** BTC 5m only

### On completion

1. Write full trial results to `/tmp/tuning/optuna_results.json` — all 100 trials, not just the winner
1. Write winning hyperparameters into `/tmp/config.yaml` under the `model:` block
1. Log explicitly in `optuna_results.json`:
   - Winning `sequence_length`
   - Whether `use_global_attention` was `true` or `false`
   - Winning `learning_rate` — **this value is valid for `batch_size=1024` and must not be rescaled**
   - Shannon entropy: was it in the confirmed feature set that Optuna trained on?
1. Rsync both artifacts to VPS before the container exits — they will be permanently lost otherwise:

```python
key_path = get_ssh_key_path()
try:
    remote_jobs = config.infrastructure.jobs_remote_path.rstrip("/")
    # optuna_results.json → jobs/tuning/
    rsync_with_retry([
        "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        "/tmp/tuning/optuna_results.json",
        remote_jobs + "/tuning/"
    ])
    # Updated config.yaml → VPS project root via dedicated config_remote_path key
    rsync_with_retry([
        "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
        "/tmp/config.yaml",
        config.infrastructure.config_remote_path
    ])
finally:
    os.unlink(key_path)
```

1. Call `log_gpu_timing()` — rsyncs timing log to VPS
2. **Re-run `upload_to_volume.py` from your local machine** after confirming the VPS has the updated `config.yaml`. The Volume copy at `/data/config.yaml` is still the pre-Optuna version — every future container (`run_weekly_retrain`, etc.) reads from the Volume and will use stale hyperparameters until this is done.
3. **Verify before the next weekly job:** SSH to VPS and confirm `feature_list.json`, `optuna_results.json`, and `timing_log.jsonl` exist with expected timestamps. Spot-check `config.yaml` for the winning `sequence_length` and `learning_rate`. Run a single CPU dry-pass of `train.py` to confirm no import or shape errors from the new sequence length before the next scheduled B200 job.

-----

## What You Are Not Changing

- `labels/direction.py` — `close[t] >= open[t]`, returns `0` or `1`, never `+1`/`-1`
- Walk-forward fold structure — 120d train / 10d embargo / 20d val / 10d test; **12-fold minimum applies to production retrain only, not Optuna trials**
- `p_market=0.50` backtest assumption — documented in `config.yaml` and `metrics.py`
- Strategy layer — edge computation, Kelly sizing, all filters
- Polymarket client — GTD orders, tick size fetch, heartbeat ID chaining
- Position manager — CONFIRMED/FAILED/RETRYING handling
- Every Key Rule in Section 20 of the plan

**The only things changing in this session:**

- GPU hardware: Modal B200 (180GB usable / 192GB HBM3e physical, $6.25/hr)
- `batch_size` restored to 1024, `sequence_length` restored to 750 as starting value
- Gradient accumulation loop removed from `train.py`
- `physical_batch_size` and `accumulation_steps` removed from config entirely
- `OneCycleLR` `steps_per_epoch` reverted to `len(dataloader)` — not divided by accumulation steps
- Boruta-SHAP restored to full production parameters (GPU-accelerated, full sample, 50 shadow iterations)
- Boruta-SHAP does NOT run inside Optuna trials — Optuna uses the fixed `feature_list.json` from staging
- Timeout guards added to all Modal GPU functions
- Modal Secret `vps-ssh-key` registered; mounted in all GPU functions for rsync persistence
- GPU timing log with VPS rsync persistence added to all GPU jobs
- Modal Volume provisioned for OHLCV dataset, features, and config — one-time upload before first job
- All container file writes use `/tmp/` with explicit `os.makedirs`
- Explicit in-body config load from `/data/config.yaml` in every GPU function
- rsync wrapped in `rsync_with_retry` (3 attempts, 5s delay) across all GPU functions
- Artifact rsync added to Steps 5 and 6 completion blocks (separate from timing log rsync)
- CUDA 12.4+ assert added to every GPU function body
- `cost_usd` field added to timing log records
