"""
tuning/optuna_search.py — Hyperparameter search with Optuna + MedianPruner.

Per spec §10:
    Initial: 100 trials, objective = mean Val Sharpe across 6 folds
    Prune Val Sharpe < 0 after fold 3
    Quarterly re-search: 30 trials (seq_len, LSTM layers, attention heads)

    Search ranges:
        Conv filters:            128, 256, 512
        LSTM hidden dim:         256, 512
        LSTM layers:             1, 2, 3
        Attention heads:         4, 8
        use_global_attention:    true, false
        Dropout:                 0.1 – 0.5
        Learning rate:           1e-4 – 1e-3 log
        Sequence length:         500, 750, 1000

    Profile T4 VRAM for (seq_len=1000, attention=true, filters=512) before launching.
"""

import os
import logging
from typing import Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_objective(
    features: np.ndarray,
    labels: np.ndarray,
    bar_timestamps: Optional[np.ndarray] = None,
    p_market_history=None,
    n_folds: int = 6,
    bars_per_day: int = 288,
    device: str = 'auto',
):
    """
    Create Optuna objective function.

    Objective: mean Val Sharpe across folds using simulated Polymarket PnL.
    Prune if Val Sharpe < 0 after fold 3.
    """
    import torch
    from models.architecture import build_model, CNNBiLSTMAttention
    from models.train import train_model, fit_scaler, SequenceDataset
    from eval.walkforward import generate_folds
    from eval.metrics import compute_fold_metrics

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        config = load_config()

        config['model']['conv_filters'] = trial.suggest_categorical(
            'conv_filters', [128, 256, 512]
        )
        config['model']['lstm_hidden_dim'] = trial.suggest_categorical(
            'lstm_hidden_dim', [256, 512]
        )
        config['model']['lstm_layers'] = trial.suggest_int(
            'lstm_layers', 1, 3
        )
        config['model']['attention_heads'] = trial.suggest_categorical(
            'attention_heads', [4, 8]
        )
        config['model']['use_global_attention'] = trial.suggest_categorical(
            'use_global_attention', [True, False]
        )
        config['model']['dropout'] = trial.suggest_float(
            'dropout', 0.1, 0.5
        )
        config['model']['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-4, 1e-3, log=True
        )
        config['model']['sequence_length'] = trial.suggest_categorical(
            'sequence_length', [500, 750, 1000]
        )

        # VRAM safety check
        if (config['model']['sequence_length'] == 1000 and
            config['model']['use_global_attention'] and
            config['model']['conv_filters'] == 512):
            logger.warning("High VRAM config — monitoring memory")

        # Generate folds
        folds = generate_folds(len(features), bars_per_day, config)
        folds = folds[:n_folds]

        if len(folds) < 3:
            return float('-inf')

        n_features = features.shape[1]
        val_sharpes = []

        for fold_idx, fold in enumerate(folds):
            # Extract data
            X_train = features[fold.train_start:fold.train_end]
            y_train = labels[fold.train_start:fold.train_end]
            X_val = features[fold.val_start:fold.val_end]
            y_val = labels[fold.val_start:fold.val_end]

            # Scale
            scaler = fit_scaler(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Build and train model (single seed for speed)
            model = build_model(n_features, config)
            result = train_model(
                model, X_train_scaled, y_train,
                X_val_scaled, y_val,
                config=config, device=device, seed=0,
            )

            # Predict on val
            model.eval()
            import torch as th
            val_dataset = SequenceDataset(
                X_val_scaled, y_val,
                sequence_length=config['model']['sequence_length'],
                stride=config['model']['sequence_length'],
            )

            p_up_list = []
            for X_batch, _ in th.utils.data.DataLoader(val_dataset, batch_size=512):
                with th.no_grad():
                    probs = model(X_batch.to(device))
                    p_up_list.append(probs[:, 1].cpu().numpy())

            if not p_up_list:
                val_sharpes.append(0.0)
                continue

            p_up = np.concatenate(p_up_list)
            n_val = len(p_up)
            y_val_aligned = y_val[-n_val:]

            # p_market fallback = 0.50 (efficient-market null hypothesis)
            p_market = np.full(n_val, config['backtest']['assumed_market_price'])

            metrics = compute_fold_metrics(p_up, y_val_aligned, p_market, config)
            val_sharpes.append(metrics.get('sharpe', 0.0))

            # Pruning: report intermediate and check gates
            trial.report(np.mean(val_sharpes), fold_idx)

            # Hard Sharpe gate fires after fold 3 (index 2) only
            if fold_idx == 2 and np.mean(val_sharpes) < 0:
                trial.set_user_attr("pruned_by", "sharpe_gate")
                raise optuna.exceptions.TrialPruned()

            # MedianPruner check after every fold
            if trial.should_prune():
                trial.set_user_attr("pruned_by", "median_pruner")
                raise optuna.exceptions.TrialPruned()

        mean_sharpe = float(np.mean(val_sharpes))
        logger.info("Trial {}: mean Val Sharpe = {:.4f}".format(
            trial.number, mean_sharpe
        ))
        return mean_sharpe

    return objective


def run_search(
    features: np.ndarray,
    labels: np.ndarray,
    n_trials: int = 100,
    study_name: str = 'cnn_lstm_btc_5m',
    device: str = 'auto',
    **kwargs,
) -> optuna.Study:
    """
    Run hyperparameter search.

    Args:
        features: Full feature array
        labels: Full label array
        n_trials: Number of trials (100 initial, 30 quarterly)
        study_name: Optuna study name
        device: Device to train on

    Returns:
        Optuna Study object with results
    """
    objective = create_objective(features, labels, device=device, **kwargs)

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=MedianPruner(),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Best trial: {} (Sharpe={:.4f})".format(
        study.best_trial.number, study.best_trial.value
    ))
    logger.info("Best params: {}".format(study.best_trial.params))

    return study


def apply_best_params(study: optuna.Study, config_path: str = None) -> dict:
    """Apply best hyperparameters to config.yaml."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    config = load_config()
    best = study.best_trial.params

    param_map = {
        'conv_filters': ('model', 'conv_filters'),
        'lstm_hidden_dim': ('model', 'lstm_hidden_dim'),
        'lstm_layers': ('model', 'lstm_layers'),
        'attention_heads': ('model', 'attention_heads'),
        'use_global_attention': ('model', 'use_global_attention'),
        'dropout': ('model', 'dropout'),
        'learning_rate': ('model', 'learning_rate'),
        'sequence_length': ('model', 'sequence_length'),
    }

    for param_name, (section, key) in param_map.items():
        if param_name in best:
            config[section][key] = best[param_name]

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Applied best params to config.yaml")
    return config
