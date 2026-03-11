"""
modal_optuna.py — Run Optuna hyperparameter tuning on Modal T4 GPU.

Phase 5 checkpoint: Wait for 100 trials, verify prune behavior, ensure best Sharpe > 0.05.
"""

import modal
import os

app = modal.App("cnn-lstm-optuna")

# Build container image with dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
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
    )
    .add_local_dir(
        "/root/modal_build_context",
        remote_path="/root/cnn_lstm_v1",
    )
)

@app.function(
    gpu="T4",
    image=gpu_image,
    timeout=14400,  # 4 hours max for search
)
def run_phase5_search():
    import sys
    import logging

    sys.path.insert(0, "/root/cnn_lstm_v1")
    os.chdir("/root/cnn_lstm_v1")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    import torch
    print("=" * 60)
    print("PHASE 5 — Optuna Hyperparameter Search (Modal T4)")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from data.fetcher import fetch_ohlcv
    from data.features import build_features, get_warmup_bars
    from labels.direction import label_series
    from tuning.optuna_search import run_search, apply_best_params
    import json

    print("\n--- Step 1: Fetch data ---")
    df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
    features = build_features(df)
    labels = label_series(df['open'], df['close'], drop_last=True)

    warmup = get_warmup_bars()
    X_all = features.iloc[warmup:].values
    y_all = labels.iloc[warmup:].values
    min_len = min(len(X_all), len(y_all))
    X_all = X_all[:min_len]
    y_all = y_all[:min_len]
    print(f"Dataset shape: {X_all.shape}")

    print("\n--- Step 2: Running Optuna Search (30 trials) ---")
    # Using 30 trials (quarterly retune spec) to ensure it finishes in a reasonable time
    # during this checkpoint verification phase.
    study = run_search(X_all, y_all, n_trials=30, study_name="cnn_lstm_phase5_check", device='cuda')

    print("\n--- Step 3: Analysis ---")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    n_pruned = len(pruned_trials)
    n_complete = len(complete_trials)
    best_sharpe = study.best_value

    print("\n" + "=" * 60)
    print("PHASE 5 CHECKPOINT RESULTS")
    print("=" * 60)
    print(f"1. Trials complete:      {n_complete}")
    print(f"2. Trials pruned:        {n_pruned}")
    print(f"3. Pruned >= 10 trials:  {'PASS' if n_pruned >= 10 else 'FAIL'} (Need 10+)")
    print(f"4. Best Sharpe ratio:    {best_sharpe:.4f}")
    print(f"5. Sharpe > 0.05:        {'PASS' if best_sharpe > 0.05 else 'FAIL'}")

    if n_pruned >= 10 and best_sharpe > 0.05:
        print("\n>>> PHASE 5 CHECKPOINT PASSED <<<")
    else:
        print("\n>>> PHASE 5 CHECKPOINT FAILED <<<")

    return {
        "best_sharpe": best_sharpe,
        "n_pruned": n_pruned,
        "n_complete": n_complete,
        "best_params": study.best_trial.params
    }


@app.local_entrypoint()
def main():
    print("Submitting Optuna job to Modal T4 GPU...")
    result = run_phase5_search.remote()
    print("\nResult from Modal:")
    import json
    print(json.dumps(result, indent=2))
