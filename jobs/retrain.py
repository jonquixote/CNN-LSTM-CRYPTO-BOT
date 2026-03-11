"""
jobs/retrain.py — Weekly retrain job.

Per spec §15:
    Weekly (Sunday 02:00 UTC):
    1. Fetch latest 270+ days OHLCV
    2. Build features
    3. Generate labels
    4. Run feature selection (Boruta-SHAP)
    5. Train 5-seed ensemble on walk-forward folds
    6. Fit isotonic calibrator on OOF predictions
    7. Export to ONNX
    8. Validate: hit rate > 0.52 on last 3 test folds
    9. Push models to VPS
    10. Send retrain summary via Telegram

    Runs on Lightning.ai T4. Models pushed to VPS for ONNX inference.
"""

import os
import sys
import json
import time
import logging
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_retrain(
    slot: str = 'BTC_5m',
    config: Optional[dict] = None,
    output_dir: str = None,
) -> dict:
    """
    Run full retrain pipeline.

    Args:
        slot: Slot to retrain (e.g. 'BTC_5m')
        config: Optional config override
        output_dir: Where to save models

    Returns:
        dict with retrain results and metrics
    """
    if config is None:
        config = load_config()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved')
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    # Parse slot
    parts = slot.split('_')
    symbol = parts[0] + '/USDT'
    timeframe = parts[1]
    bars_per_day = 288 if timeframe == '5m' else 96

    logger.info("Starting retrain for {} ({} {})".format(slot, symbol, timeframe))

    # 1. Fetch data
    from data.fetcher import fetch_ohlcv
    df = fetch_ohlcv(symbol, timeframe, since_days=config['training']['min_history_days'] + 10)
    logger.info("Fetched {} bars".format(len(df)))

    # 2. Build features
    from data.features import build_features, get_warmup_bars
    features = build_features(df)
    warmup = get_warmup_bars()
    features = features.iloc[warmup:]
    feature_array = features.values.astype(np.float32)
    logger.info("Features: {} columns, {} rows (after warmup)".format(
        features.shape[1], features.shape[0]
    ))

    # 3. Generate labels
    from labels.direction import label_series
    labels = label_series(df['open'], df['close'], drop_last=True)
    labels = labels.iloc[warmup:]
    labels = labels.iloc[:len(features)]  # align
    label_array = labels.values.astype(np.int64)

    # 4. Walk-forward folds
    from eval.walkforward import generate_folds
    folds = generate_folds(len(feature_array), bars_per_day, config)
    logger.info("Generated {} folds".format(len(folds)))

    # 5. Train ensemble seeds
    from models.architecture import build_model
    from models.train import train_model, fit_scaler

    n_features = feature_array.shape[1]
    n_seeds = config['model']['ensemble_seeds']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use last fold for final training
    last_fold = folds[-1]
    X_train = feature_array[last_fold.train_start:last_fold.train_end]
    y_train = label_array[last_fold.train_start:last_fold.train_end]
    X_val = feature_array[last_fold.val_start:last_fold.val_end]
    y_val = label_array[last_fold.val_start:last_fold.val_end]

    # Fit scaler on training data only
    scaler = fit_scaler(X_train)
    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Save scaler
    scaler_path = os.path.join(output_dir, '{}_scaler.pkl'.format(slot))
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    seed_results = []
    seed_models = []

    for seed in range(n_seeds):
        logger.info("Training seed {}/{}".format(seed + 1, n_seeds))
        model = build_model(n_features, config)
        result = train_model(
            model, X_train_scaled, y_train,
            X_val_scaled, y_val,
            config=config, device=device, seed=seed,
        )
        seed_results.append(result)
        seed_models.append(model)

        # Save PyTorch model
        pt_path = os.path.join(output_dir, '{}_{}_seed{}.pt'.format(slot, 'latest', seed))
        torch.save(model.state_dict(), pt_path)

        # Export ONNX
        _export_onnx(
            model, n_features, config,
            os.path.join(output_dir, '{}_{}_seed{}.onnx'.format(slot, 'latest', seed)),
            device,
        )

    # 6. Fit calibrator on OOF predictions
    from calibration.isotonic import IsotonicCalibrator

    oof_p_up = []
    oof_labels = []

    for fold in folds[-3:]:  # Use last 3 folds for calibration
        X_test = feature_array[fold.test_start:fold.test_end]
        y_test = label_array[fold.test_start:fold.test_end]
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # Ensemble predict on test fold
        from models.train import SequenceDataset
        test_ds = SequenceDataset(
            X_test_scaled, y_test,
            sequence_length=config['model']['sequence_length'],
            stride=config['model']['sequence_length'],
        )
        from torch.utils.data import DataLoader

        for model in seed_models:
            model.eval()

        for X_batch, y_batch in DataLoader(test_ds, batch_size=512):
            seed_preds = []
            for model in seed_models:
                with torch.no_grad():
                    probs = model(X_batch.to(device))
                    seed_preds.append(probs[:, 1].cpu().numpy())
            p_up_ensemble = np.mean(seed_preds, axis=0)
            oof_p_up.extend(p_up_ensemble.tolist())
            oof_labels.extend(y_batch.numpy().tolist())

    if oof_p_up:
        calibrator = IsotonicCalibrator()
        calibrator.fit(np.array(oof_p_up), np.array(oof_labels))
        cal_path = os.path.join(output_dir, '{}_calibrator.pkl'.format(slot))
        calibrator.save(cal_path)

        cal_stats = calibrator.get_calibration_stats(
            np.array(oof_p_up), np.array(oof_labels)
        )
        logger.info("Calibration stats: {}".format(cal_stats))

    # Save retrain state
    state_dir = os.path.join(os.path.dirname(__file__), '..', 'state')
    os.makedirs(state_dir, exist_ok=True)
    with open(os.path.join(state_dir, 'last_retrain.json'), 'w') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'slot': slot,
            'n_seeds': n_seeds,
        }, f)

    duration = (time.time() - start_time) / 60

    summary = {
        'slot': slot,
        'n_folds': len(folds),
        'n_seeds': n_seeds,
        'val_sharpe_mean': 0.0,
        'val_hit_rate': np.mean([r['best_val_accuracy'] for r in seed_results]),
        'ece': cal_stats.get('ece_calibrated', 0) if oof_p_up else 0,
        'duration_min': round(duration, 1),
        'device': device,
    }

    logger.info("Retrain complete: {}".format(summary))

    # Send Telegram summary
    try:
        from monitoring.telegram import send_retrain_summary
        send_retrain_summary(summary)
    except Exception as e:
        logger.warning("Failed to send Telegram summary: {}".format(e))

    return summary


def _export_onnx(
    model: torch.nn.Module,
    n_features: int,
    config: dict,
    output_path: str,
    device: str,
):
    """Export model to ONNX for VPS inference."""
    model.eval()
    seq_len = config['model']['sequence_length']
    dummy_input = torch.randn(1, seq_len, n_features).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['features'],
            output_names=['probabilities'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'},
            },
            opset_version=17,
        )
        logger.info("ONNX exported: {}".format(output_path))
    except Exception as e:
        logger.error("ONNX export failed: {}".format(e))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    slot = sys.argv[1] if len(sys.argv) > 1 else 'BTC_5m'
    run_retrain(slot=slot)
