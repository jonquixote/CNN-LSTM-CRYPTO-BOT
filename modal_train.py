"""
modal_train.py — Run CNN-BiLSTM-Attention training on Modal T4 GPU.

Phase 4 checkpoint: Train on T4, verify loss decrease + ECE improvement.

Usage from VPS:
    cd ~/cnn_lstm_v1 && source venv/bin/activate
    modal run modal_train.py
"""

import modal

# ── Modal App & Image ───────────────────────────────────────────────────────

app = modal.App("cnn-lstm-train")

# Build a container image with dependencies + project code baked in
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
    )
    .add_local_dir(
        "/root/modal_build_context",
        remote_path="/root/cnn_lstm_v1",
    )
)


@app.function(
    gpu="T4",
    image=gpu_image,
    timeout=3600,  # 1 hour max
)
def train_phase4():
    """Run full Phase 4 training pipeline on T4 GPU."""
    import sys
    import os
    import logging

    sys.path.insert(0, "/root/cnn_lstm_v1")
    os.chdir("/root/cnn_lstm_v1")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    import numpy as np
    import torch
    print("=" * 60)
    print("PHASE 4 — CNN-BiLSTM-Attention Training (Modal T4)")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Fetch data and build features ────────────────────────────
    print("\n--- Step 1: Data & Features ---")
    from data.fetcher import fetch_ohlcv
    from data.features import build_features, get_warmup_bars
    from labels.direction import label_series

    df = fetch_ohlcv('BTC/USDT', '5m', since_days=280)
    print(f"Fetched {len(df)} bars")

    features = build_features(df)
    labels = label_series(df['open'], df['close'], drop_last=True)

    warmup = get_warmup_bars()
    X_all = features.iloc[warmup:].values
    y_all = labels.iloc[warmup:].values
    min_len = min(len(X_all), len(y_all))
    X_all = X_all[:min_len]
    y_all = y_all[:min_len]
    print(f"Post-warmup: {len(X_all)} samples, {X_all.shape[1]} features")

    # ── Step 2: Walk-forward split (use first fold for checkpoint) ────────
    print("\n--- Step 2: Train/Val/Test Split ---")
    bars_per_day = 288
    train_days = 120
    embargo_days = 10
    val_days = 20
    test_days = 10

    train_end = train_days * bars_per_day
    embargo_end = train_end + embargo_days * bars_per_day
    val_end = embargo_end + val_days * bars_per_day
    test_end = val_end + test_days * bars_per_day

    if test_end > len(X_all):
        test_end = len(X_all)
        val_end = test_end - test_days * bars_per_day
        embargo_end = val_end - embargo_days * bars_per_day
        train_end = embargo_end - embargo_days * bars_per_day

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]
    X_val = X_all[embargo_end:val_end]
    y_val = y_all[embargo_end:val_end]
    X_test = X_all[val_end:test_end]
    y_test = y_all[val_end:test_end]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Step 3: Scale features ───────────────────────────────────────────
    print("\n--- Step 3: RobustScaler ---")
    from models.train import fit_scaler
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    for arr in [X_train_scaled, X_val_scaled, X_test_scaled]:
        arr[~np.isfinite(arr)] = 0.0

    print("Scaling complete (fit on train only)")

    # ── Step 4: Train model ──────────────────────────────────────────────
    print("\n--- Step 4: Training CNN-BiLSTM-Attention ---")
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    from models.architecture import CNNBiLSTMAttention
    from models.train import train_model

    mc = config['model']
    n_features = X_train_scaled.shape[1]

    # Train with seed 0 for checkpoint validation
    print(f"\n  Training seed 0 (n_features={n_features})...")
    model = CNNBiLSTMAttention(
        n_features=n_features,
        config=config,
    )

    result = train_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
        config=config,
        device='auto',
        seed=0,
    )

    print(f"  epochs={result['epochs_trained']}, "
          f"best_val_loss={result['best_val_loss']:.4f}, "
          f"best_val_acc={result['best_val_accuracy']:.4f}")

    # ── Step 5: Verify loss decrease ─────────────────────────────────────
    print("\n--- Step 5: Loss Decrease Verification ---")
    h = result['history']
    if len(h['train_loss']) >= 2:
        first_loss = h['train_loss'][0]
        last_loss = h['train_loss'][-1]
        best_loss = min(h['train_loss'])
        improved = best_loss < first_loss
        print(f"  first_loss={first_loss:.4f} → best_loss={best_loss:.4f} "
              f"({'IMPROVED' if improved else 'NO IMPROVEMENT'})")
    else:
        improved = False
        print("  Only 1 epoch trained")

    # ── Step 6: ECE check ────────────────────────────────────────────────
    print("\n--- Step 6: ECE Pre-Calibration ---")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    from models.train import SequenceDataset
    from torch.utils.data import DataLoader
    test_ds = SequenceDataset(
        X_test_scaled, y_test,
        sequence_length=mc['sequence_length'],
        stride=mc['sequence_length'],
    )

    ece = None
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=mc['batch_size'], shuffle=False)
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                probs = torch.softmax(model(X_batch), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y_batch.numpy())

        probs = np.concatenate(all_probs)
        test_labels = np.concatenate(all_labels)
        p_up = probs[:, 1]

        from eval.metrics import _compute_ece
        ece = _compute_ece(p_up, test_labels)
        print(f"  ECE (pre-calibration): {ece:.4f}")
    else:
        print("  Insufficient test data for ECE")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 CHECKPOINT RESULTS")
    print("=" * 60)

    print(f"1. Loss decreased:      {'PASS' if improved else 'FAIL'}")
    print(f"2. Best val accuracy:   {result['best_val_accuracy']:.4f}")
    print(f"3. Epochs trained:      {result['epochs_trained']}")
    if ece is not None:
        print(f"4. ECE pre-calibration: {ece:.4f}")
    print(f"5. GPU used:            {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if improved:
        print("\n>>> PHASE 4 CHECKPOINT PASSED <<<")
    else:
        print("\n>>> PHASE 4 CHECKPOINT FAILED: Loss did not decrease <<<")

    return {
        "loss_decreased": improved,
        "best_val_loss": result['best_val_loss'],
        "best_val_accuracy": result['best_val_accuracy'],
        "epochs_trained": result['epochs_trained'],
        "ece_pre_calibration": ece,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint — triggers the remote GPU function."""
    print("Submitting training job to Modal T4 GPU...")
    result = train_phase4.remote()
    print("\nResult from Modal:")
    print(result)
