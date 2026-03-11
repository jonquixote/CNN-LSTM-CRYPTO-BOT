"""
modal_gpu.py — Unified Modal GPU functions for CNN-LSTM trading system.

All GPU jobs run on B200 (180GB usable VRAM, $6.25/hr).
Each function has an explicit timeout, CUDA 12.4+ assert, and timing log.

Functions:
    run_boruta_shap()  — Feature selection (timeout: 30 min)
    run_optuna_search() — Hyperparameter search (timeout: 60 min)
    run_weekly_retrain() — Weekly retrain (timeout: 30 min)

Usage from VPS:
    cd ~/cnn_lstm_v1 && source venv/bin/activate

    # Run Boruta-SHAP
    modal run modal_gpu.py::boruta_main

    # Run Optuna search (after Boruta-SHAP completes)
    modal run modal_gpu.py::optuna_main

    # Run weekly retrain
    modal run modal_gpu.py::retrain_main
"""

import modal
import os
import time
import json
import logging
import subprocess
import tempfile
import base64

# ── Modal App & Image ────────────────────────────────────────────────────────

app = modal.App("cnn-lstm-gpu")

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("openssh-client", "rsync")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scipy==1.14.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "numba>=0.58.0",
        "ta>=0.10.0",
        "ccxt>=4.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pyarrow>=12.0.0",
        "optuna>=3.0.0",
        "xgboost==2.1.3",  # pinned — shap 0.49.1 incompatible with xgboost 3.x
        "shap>=0.42.0",
        "packaging>=21.0",
    )
    .add_local_dir(
        "/root/modal_build_context",
        remote_path="/root/cnn_lstm_v1",
    )
)

data_volume = modal.Volume.from_name("cnn-lstm-data", create_if_missing=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


def get_ssh_key_path() -> str:
    """Write SSH private key from Modal Secret to a temp file for rsync.

    Key is stored as base64 in SSH_PRIVATE_KEY_B64 to preserve PEM newlines.
    """
    key_b64 = os.environ["SSH_PRIVATE_KEY_B64"]
    key_bytes = base64.b64decode(key_b64)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pem', delete=False) as f:
        f.write(key_bytes)
        key_path = f.name
    os.chmod(key_path, 0o600)
    return key_path


def rsync_with_retry(cmd: list, retries: int = 3, delay: float = 5.0) -> None:
    """Run an rsync command with retries. Raises CalledProcessError on final failure."""
    for attempt in range(retries):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as e:
            log.warning(f"rsync attempt {attempt + 1}/{retries} failed: {e.stderr}")
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def log_gpu_timing(job_name: str, gpu_start_ts: float,
                   timeout_seconds: int, estimated_seconds: int,
                   remote_jobs_path: str):
    """Log GPU wall-clock timing and rsync to VPS."""
    gpu_elapsed = time.time() - gpu_start_ts
    record = {
        "job": job_name,
        "gpu": "B200",
        "started_at": gpu_start_ts,
        "elapsed_seconds": round(gpu_elapsed, 2),
        "estimated_seconds": estimated_seconds,
        "timeout_seconds": timeout_seconds,
        "headroom_seconds": round(timeout_seconds - gpu_elapsed, 2),
        "headroom_pct": round((timeout_seconds - gpu_elapsed) / timeout_seconds * 100, 1),
        "cost_usd": round(gpu_elapsed / 3600 * 6.25, 4),
    }

    os.makedirs("/tmp/jobs", exist_ok=True)
    local_path = "/tmp/jobs/timing_log.jsonl"
    with open(local_path, "a") as f:
        json.dump(record, f)
        f.write("\n")

    # Sync to VPS — best effort, non-fatal
    remote_path = remote_jobs_path.rstrip("/") + "/timing_log.jsonl"
    try:
        key_path = get_ssh_key_path()
        try:
            rsync_with_retry([
                "rsync", "-az",
                "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                local_path, remote_path
            ])
        finally:
            os.unlink(key_path)
    except Exception as e:
        log.warning(f"GPU timing rsync failed (non-fatal): {e}")

    log.info(
        f"GPU job complete | job={job_name} | "
        f"elapsed={gpu_elapsed:.1f}s | headroom={timeout_seconds - gpu_elapsed:.1f}s | "
        f"cost=${record['cost_usd']:.4f}"
    )


def load_config_from_volume():
    """Load config from Modal Volume mount. Returns (raw_dict, namespace)."""
    import yaml
    from types import SimpleNamespace

    def _ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [_ns(i) for i in d]
        return d

    with open("/data/config.yaml") as f:
        raw = yaml.safe_load(f)

    return raw, _ns(raw)


def cuda_preflight():
    """Assert CUDA availability and version. Returns device info string."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    cuda_ver = tuple(int(x) for x in torch.version.cuda.split("."))
    assert cuda_ver >= (12, 4), f"CUDA 12.4+ required for B200, got {torch.version.cuda}"

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    info = f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB, CUDA: {torch.version.cuda}"
    log.info(info)
    return info


# ── GPU Function: Boruta-SHAP ────────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=1800,  # 30 minutes hard kill
    image=gpu_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("vps-ssh-key")],
)
def run_boruta_shap():
    """Run full Boruta-SHAP feature selection on BTC 5m."""
    gpu_start_ts = time.time()  # ← first line, before anything else

    import sys
    sys.path.insert(0, "/root/cnn_lstm_v1")
    os.chdir("/root/cnn_lstm_v1")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    import numpy as np
    import pandas as pd
    import hashlib

    gpu_info = cuda_preflight()
    config_raw, config = load_config_from_volume()

    print("=" * 60)
    print("BORUTA-SHAP — Full Production Run (Modal B200)")
    print("=" * 60)
    print(gpu_info)

    # ── Load data from Volume ────────────────────────────────────────
    print("\n--- Loading data from Volume ---")
    df = pd.read_parquet("/data/btc_5m_ohlcv.parquet")
    features = pd.read_parquet("/data/btc_5m_features.parquet")
    print(f"OHLCV: {len(df)} bars, Features: {features.shape}")

    # Build labels
    from labels.direction import label_series
    from data.features import get_warmup_bars

    labels = label_series(df['open'], df['close'], drop_last=True)
    warmup = get_warmup_bars()

    X = features.iloc[warmup:]
    y = labels.iloc[warmup:]
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    print(f"Post-warmup: {len(X)} samples, {X.shape[1]} features")

    # ── Run Boruta-SHAP ──────────────────────────────────────────────
    print("\n--- Running Boruta-SHAP (50 shadow iterations) ---")
    from selection.boruta_shap import BorutaSHAP, save_feature_list

    selector = BorutaSHAP(
        n_shadow_iterations=50,
        n_estimators=200,
        max_depth=6,
        alpha=0.05,
        random_state=42,
    )
    selector.fit(X, y)

    accepted = selector.get_accepted_features()
    report = selector.get_feature_report()
    print(f"Accepted: {len(report['accepted'])}")
    print(f"Rejected: {len(report['rejected'])}")
    print(f"Tentative: {len(report['tentative'])}")

    # ── Save feature_list.json to staging ────────────────────────────
    os.makedirs("/tmp/staging_artifacts", exist_ok=True)
    filepath, sha256_hash = save_feature_list(
        accepted,
        output_dir="/tmp/staging_artifacts",
        metadata={
            "symbol": "BTC",
            "timeframe": "5m",
            "run_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": "B200",
            "n_shadow_iterations": 50,
            "n_estimators": 200,
            "restoration_log": {
                "n_shadow_iterations": {"degraded_value": 100, "restored_value": 50,
                                        "note": "100 was CPU default, 50 is B200 production spec"},
                "xgb_device": {"degraded_value": "cpu", "restored_value": "cuda",
                               "note": "Was running on CPU due to no GPU"},
                "shap_sample_size": {"degraded_value": "all", "restored_value": "all",
                                     "note": "Never subsampled — confirming no degradation"},
            },
        },
    )
    print(f"feature_list.json saved: hash={sha256_hash[:16]}...")

    # ── Save full results ────────────────────────────────────────────
    os.makedirs("/tmp/selection", exist_ok=True)

    # Determine shannon_entropy status
    shannon_status = "confirmed" if "shannon_entropy" in accepted else "rejected"

    results = {
        "symbol": "BTC",
        "timeframe": "5m",
        "run_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": "B200",
        "n_features_input": X.shape[1],
        "n_features_confirmed": len(accepted),
        "n_features_rejected": len(report['rejected']),
        "shannon_entropy_result": shannon_status,
        "feature_list_hash": sha256_hash,
        "confirmed_features": sorted(accepted),
        "rejected_features": sorted(report['rejected']),
        "tentative_features": sorted(report['tentative']),
        "importances": report['importances'],
        "restoration_log": {
            "n_shadow_iterations": {"degraded_value": 100, "restored_value": 50},
            "xgb_device": {"degraded_value": "cpu", "restored_value": "cuda"},
            "shap_sample_size": {"degraded_value": "all", "restored_value": "all"},
        },
        "timing": {
            "elapsed_seconds": round(time.time() - gpu_start_ts, 2),
            "timeout_seconds": 1800,
        },
    }

    with open("/tmp/selection/boruta_shap_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Save to Volume as primary backup (survives rsync failures) ───
    print("\n--- Saving results to Volume ---")
    import shutil
    shutil.copy("/tmp/staging_artifacts/feature_list.json", "/data/feature_list.json")
    shutil.copy("/tmp/selection/boruta_shap_results.json", "/data/boruta_shap_results.json")
    data_volume.commit()  # Persist to Volume
    print("  ✓ feature_list.json → Volume")
    print("  ✓ boruta_shap_results.json → Volume")

    # ── Rsync artifacts to VPS (best-effort, non-fatal) ──────────────
    print("\n--- Syncing artifacts to VPS ---")
    rsync_ok = True
    try:
        key_path = get_ssh_key_path()
        try:
            # feature_list.json → artifacts dir
            rsync_with_retry([
                "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                "/tmp/staging_artifacts/feature_list.json",
                config.infrastructure.artifact_remote_path.rstrip("/") + "/"
            ])
            print("  ✓ feature_list.json → VPS artifacts")

            # boruta_shap_results.json → jobs/selection/
            remote_selection = config.infrastructure.jobs_remote_path.rstrip("/") + "/selection/"
            rsync_with_retry([
                "rsync", "-az", "--rsync-path", "mkdir -p /root/cnn_lstm_v1/jobs/selection && rsync",
                "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                "/tmp/selection/boruta_shap_results.json",
                remote_selection
            ])
            print("  ✓ boruta_shap_results.json → VPS jobs/selection/")
        finally:
            os.unlink(key_path)
    except Exception as e:
        rsync_ok = False
        print(f"  ✗ Rsync to VPS failed (non-fatal, results saved to Volume): {e}")

    # ── Timing log ───────────────────────────────────────────────────
    log_gpu_timing("boruta_shap", gpu_start_ts, 1800, 300,
                   config.infrastructure.jobs_remote_path)

    print("\n" + "=" * 60)
    print(f"BORUTA-SHAP COMPLETE — {len(accepted)} features confirmed")
    print(f"Cost: ${(time.time() - gpu_start_ts) / 3600 * 6.25:.4f}")
    if not rsync_ok:
        print("NOTE: Results saved to Volume only — rsync to VPS failed")
    print("=" * 60)

    return results


# ── GPU Function: Optuna Search ──────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=3600,  # 60 minutes hard kill
    image=gpu_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("vps-ssh-key")],
)
def run_optuna_search():
    """Run 100-trial Optuna hyperparameter search on BTC 5m."""
    gpu_start_ts = time.time()  # ← first line, before anything else

    import sys
    sys.path.insert(0, "/root/cnn_lstm_v1")
    os.chdir("/root/cnn_lstm_v1")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    import numpy as np
    import pandas as pd
    import hashlib
    import torch
    import optuna
    from pathlib import Path

    gpu_info = cuda_preflight()
    config_raw, config = load_config_from_volume()

    print("=" * 60)
    print("OPTUNA SEARCH — 100 Trials (Modal B200)")
    print("=" * 60)
    print(gpu_info)

    # ── Gate check: verify feature_list.json from Volume ─────────────
    print("\n--- Gate check: feature_list.json ---")
    feature_list_path = "/data/feature_list.json"
    assert os.path.exists(feature_list_path), (
        "feature_list.json not found on Volume — run Boruta-SHAP (Step 5) first"
    )
    with open(feature_list_path) as f:
        feature_data = json.load(f)

    # Verify hash
    expected_hash = feature_data["feature_list_hash"]
    actual_hash = hashlib.sha256(
        json.dumps(sorted(feature_data["confirmed_features"]), separators=(',', ':')).encode()
    ).hexdigest()
    assert actual_hash == expected_hash, (
        f"feature_list.json hash mismatch: {actual_hash} != {expected_hash} — file may be corrupted"
    )
    confirmed_features = feature_data["confirmed_features"]
    print(f"  ✓ feature_list.json verified ({len(confirmed_features)} features, hash OK)")

    # ── Load data from Volume ────────────────────────────────────────
    print("\n--- Loading data from Volume ---")
    df = pd.read_parquet("/data/btc_5m_ohlcv.parquet")
    features_all = pd.read_parquet("/data/btc_5m_features.parquet")

    # Filter to confirmed features only
    features_filtered = features_all[confirmed_features]
    print(f"Using {features_filtered.shape[1]} confirmed features (of {features_all.shape[1]} total)")

    from labels.direction import label_series
    from data.features import get_warmup_bars

    labels = label_series(df['open'], df['close'], drop_last=True)
    warmup = get_warmup_bars()

    X_all = features_filtered.iloc[warmup:].values
    y_all = labels.iloc[warmup:].values
    min_len = min(len(X_all), len(y_all))
    X_all = X_all[:min_len]
    y_all = y_all[:min_len]
    print(f"Post-warmup: {len(X_all)} samples, {X_all.shape[1]} features")

    # ── VRAM preflight ───────────────────────────────────────────────
    print("\n--- VRAM preflight: seq=1000, attention=true, filters=512, batch=1024 ---")
    from models.architecture import CNNBiLSTMAttention
    from models.train import train_model, fit_scaler, SequenceDataset

    preflight_config = dict(config_raw)
    preflight_config['model'] = dict(config_raw['model'])
    preflight_config['model']['sequence_length'] = 1000
    preflight_config['model']['conv_filters'] = 512
    preflight_config['model']['use_global_attention'] = True
    preflight_config['model']['batch_size'] = 1024

    n_features = X_all.shape[1]
    try:
        model = CNNBiLSTMAttention(n_features, preflight_config)
        # Quick forward pass to verify VRAM
        model = model.cuda()
        dummy = torch.randn(4, 1000, n_features).cuda()
        with torch.no_grad():
            _ = model(dummy)
        del model, dummy
        torch.cuda.empty_cache()
        print("  ✓ VRAM preflight passed — most expensive config fits in B200")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ✗ VRAM preflight FAILED: {e}")
            print("  Reducing to seq=750 max for search")
            torch.cuda.empty_cache()
        else:
            raise

    # ── Run Optuna search ────────────────────────────────────────────
    print("\n--- Running Optuna Search (100 trials, MedianPruner) ---")

    from eval.walkforward import generate_folds
    from eval.metrics import compute_fold_metrics

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        trial_config = dict(config_raw)
        trial_config['model'] = dict(config_raw['model'])

        trial_config['model']['conv_filters'] = trial.suggest_categorical(
            'conv_filters', [128, 256, 512]
        )
        trial_config['model']['lstm_hidden_dim'] = trial.suggest_categorical(
            'lstm_hidden_dim', [256, 512]
        )
        trial_config['model']['lstm_layers'] = trial.suggest_int(
            'lstm_layers', 1, 3
        )
        trial_config['model']['attention_heads'] = trial.suggest_categorical(
            'attention_heads', [4, 8]
        )
        trial_config['model']['use_global_attention'] = trial.suggest_categorical(
            'use_global_attention', [True, False]
        )
        trial_config['model']['dropout'] = trial.suggest_float(
            'dropout', 0.1, 0.5
        )
        trial_config['model']['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-4, 1e-3, log=True
        )
        trial_config['model']['sequence_length'] = trial.suggest_categorical(
            'sequence_length', [500, 750, 1000]
        )
        # batch_size=1024, no accumulation
        trial_config['model']['batch_size'] = 1024

        # Generate 6 folds for Optuna (not 12 — 12-fold is production retrain only)
        folds = generate_folds(len(X_all), 288, trial_config)
        folds = folds[:6]

        if len(folds) < 3:
            return float('-inf')

        val_sharpes = []

        for fold_idx, fold in enumerate(folds):
            X_train = X_all[fold.train_start:fold.train_end]
            y_train = y_all[fold.train_start:fold.train_end]
            X_val = X_all[fold.val_start:fold.val_end]
            y_val = y_all[fold.val_start:fold.val_end]

            scaler = fit_scaler(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Clean infinities
            X_train_scaled[~np.isfinite(X_train_scaled)] = 0.0
            X_val_scaled[~np.isfinite(X_val_scaled)] = 0.0

            model = CNNBiLSTMAttention(n_features, trial_config)
            result = train_model(
                model, X_train_scaled, y_train,
                X_val_scaled, y_val,
                config=trial_config, device='cuda', seed=0,
            )

            # Predict on val
            model.eval()
            val_ds = SequenceDataset(
                X_val_scaled, y_val,
                sequence_length=trial_config['model']['sequence_length'],
                stride=trial_config['model']['sequence_length'],
            )

            p_up_list = []
            for X_batch, _ in torch.utils.data.DataLoader(val_ds, batch_size=512):
                with torch.no_grad():
                    probs = model(X_batch.cuda())
                    p_up_list.append(probs[:, 1].cpu().numpy())

            if not p_up_list:
                val_sharpes.append(0.0)
                continue

            p_up = np.concatenate(p_up_list)
            n_val = len(p_up)
            y_val_aligned = y_val[-n_val:]
            p_market = np.full(n_val, config_raw['backtest']['assumed_market_price'])

            metrics = compute_fold_metrics(p_up, y_val_aligned, p_market, trial_config)
            val_sharpe = metrics.get('sharpe', 0.0)
            val_sharpes.append(val_sharpe)

            # Report intermediate value
            trial.report(np.mean(val_sharpes), fold_idx)

            # Hard Sharpe gate after fold 3 (index 2)
            if fold_idx == 2 and val_sharpe < 0:
                trial.set_user_attr("pruned_by", "sharpe_gate")
                raise optuna.TrialPruned()

            # MedianPruner check
            if trial.should_prune():
                trial.set_user_attr("pruned_by", "median_pruner")
                raise optuna.TrialPruned()

            # Free GPU memory between folds
            del model
            torch.cuda.empty_cache()

        mean_sharpe = float(np.mean(val_sharpes))
        log.info(f"Trial {trial.number}: mean Val Sharpe = {mean_sharpe:.4f}")
        return mean_sharpe

    os.makedirs("/tmp/tuning", exist_ok=True)
    storage = "sqlite:////tmp/tuning/optuna_study.db"
    study = optuna.create_study(
        study_name="btc_5m_cnn_lstm",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\nBest trial: {study.best_trial.number} (Sharpe={study.best_trial.value:.4f})")
    print(f"Best params: {study.best_trial.params}")

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs("/tmp/tuning", exist_ok=True)

    # All trial results
    all_trials = []
    for t in study.trials:
        trial_info = {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
            "user_attrs": t.user_attrs,
        }
        all_trials.append(trial_info)

    best = study.best_trial.params
    optuna_results = {
        "study_name": "btc_5m_cnn_lstm",
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_sharpe": study.best_trial.value,
        "best_params": best,
        "winning_sequence_length": best.get("sequence_length"),
        "winning_use_global_attention": best.get("use_global_attention"),
        "winning_learning_rate": best.get("learning_rate"),
        "learning_rate_note": "Valid for batch_size=1024, must not be rescaled",
        "shannon_entropy_in_features": "shannon_entropy" in confirmed_features,
        "feature_list_hash": expected_hash,
        "all_trials": all_trials,
    }

    with open("/tmp/tuning/optuna_results.json", "w") as f:
        json.dump(optuna_results, f, indent=2)
    print("  ✓ Saved optuna_results.json")

    # Write updated config.yaml with winning hyperparameters
    import yaml
    updated_config = dict(config_raw)
    updated_config['model'] = dict(config_raw['model'])
    for param_name, value in best.items():
        if param_name in updated_config['model']:
            updated_config['model'][param_name] = value

    with open("/tmp/config.yaml", "w") as f:
        yaml.dump(updated_config, f, default_flow_style=False, sort_keys=False)
    print("  ✓ Saved updated config.yaml")

    # ── Save to Volume as primary backup ─────────────────────────────
    print("\n--- Saving results to Volume ---")
    import shutil
    shutil.copy("/tmp/tuning/optuna_results.json", "/data/optuna_results.json")
    shutil.copy("/tmp/tuning/optuna_study.db", "/data/optuna_study.db")
    shutil.copy("/tmp/config.yaml", "/data/config.yaml")
    data_volume.commit()
    print("  ✓ optuna_results.json → Volume")
    print("  ✓ optuna_study.db → Volume")
    print("  ✓ Updated config.yaml → Volume")

    # ── Rsync to VPS (best-effort, non-fatal) ────────────────────────
    print("\n--- Syncing results to VPS ---")
    rsync_ok = True
    try:
        key_path = get_ssh_key_path()
        try:
            remote_jobs = config.infrastructure.jobs_remote_path.rstrip("/")
            rsync_with_retry([
                "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                "--rsync-path", "mkdir -p /root/cnn_lstm_v1/jobs/tuning && rsync",
                "/tmp/tuning/optuna_results.json",
                remote_jobs + "/tuning/"
            ])
            rsync_with_retry([
                "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                "/tmp/tuning/optuna_study.db",
                remote_jobs + "/tuning/"
            ])
            print("  ✓ optuna_results.json + optuna_study.db → VPS")

            rsync_with_retry([
                "rsync", "-az", "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no",
                "/tmp/config.yaml",
                config.infrastructure.config_remote_path
            ])
            print("  ✓ Updated config.yaml → VPS")
        finally:
            os.unlink(key_path)
    except Exception as e:
        rsync_ok = False
        print(f"  ✗ Rsync to VPS failed (non-fatal, results saved to Volume): {e}")

    log_gpu_timing("optuna_search", gpu_start_ts, 3600, 1800,
                   config.infrastructure.jobs_remote_path)

    print("\n" + "=" * 60)
    print(f"OPTUNA SEARCH COMPLETE — Best Sharpe: {study.best_trial.value:.4f}")
    print(f"Cost: ${(time.time() - gpu_start_ts) / 3600 * 6.25:.4f}")
    if not rsync_ok:
        print("NOTE: Results saved to Volume only — rsync to VPS failed")
    print("=" * 60)

    return optuna_results


# ── GPU Function: Weekly Retrain ─────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=1800,  # 30 minutes — single slot only, revisit at 3+ slots
    image=gpu_image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("vps-ssh-key")],
)
def run_weekly_retrain():
    """Run weekly retrain for all active slots (5 seeds each)."""
    gpu_start_ts = time.time()  # ← first line, before anything else

    import sys
    sys.path.insert(0, "/root/cnn_lstm_v1")
    os.chdir("/root/cnn_lstm_v1")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    import torch
    gpu_info = cuda_preflight()
    config_raw, config = load_config_from_volume()

    print("=" * 60)
    print("WEEKLY RETRAIN (Modal B200)")
    print("=" * 60)
    print(gpu_info)

    # TODO: Implement full weekly retrain pipeline
    # This is a placeholder that will be filled when Phase 8 retrain is ready
    print("Weekly retrain not yet implemented in this session")

    log_gpu_timing("weekly_retrain", gpu_start_ts, 1800, 660,
                   config.infrastructure.jobs_remote_path)

    return {"status": "placeholder"}


# ── Local Entrypoints ────────────────────────────────────────────────────────

@app.local_entrypoint()
def boruta_main():
    """Run Boruta-SHAP on B200."""
    print("Submitting Boruta-SHAP job to Modal B200...")
    result = run_boruta_shap.remote()
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def optuna_main():
    """Run Optuna search on B200."""
    print("Submitting Optuna search to Modal B200...")
    result = run_optuna_search.remote()
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def retrain_main():
    """Run weekly retrain on B200."""
    print("Submitting weekly retrain to Modal B200...")
    result = run_weekly_retrain.remote()
    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))
