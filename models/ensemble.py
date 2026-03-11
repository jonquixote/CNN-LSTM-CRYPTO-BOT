"""
models/ensemble.py — 5-seed ensemble averaging.

Direction string conversion happens HERE and ONLY here (spec §7.5, §20 rule 2).
Below this layer: integer class indices 0/1.
Above this layer: direction strings "up"/"down".

Per spec §7.5:
    p_up              = mean(seed_probs[:, config.labels.class_up])
    p_down            = 1 - p_up
    seed_disagreement = std(seed_probs[:, config.labels.class_up])
    direction = "up" if p_up >= p_down else "down"
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class EnsembleResult:
    """Result of ensemble prediction."""
    p_up: float               # mean P(Up) across seeds
    p_down: float             # 1 - p_up
    direction: str            # "up" or "down" — ONLY place this conversion happens
    seed_disagreement: float  # std of P(Up) across seeds
    seed_probs_up: list       # per-seed P(Up) values
    p_up_raw: float           # before calibration


def ensemble_predict(
    seed_probs: np.ndarray,
    config: Optional[dict] = None,
) -> EnsembleResult:
    """
    Average predictions across ensemble seeds.

    This is the ONLY place where integer class indices are converted to
    direction strings. All code above this layer uses "up"/"down" strings.
    All code below uses integer indices 0/1.

    Args:
        seed_probs: (n_seeds, 2) array of [p_down, p_up] per seed
            Column 0 = class_down (index 0)
            Column 1 = class_up   (index 1)

    Returns:
        EnsembleResult with direction string, probabilities, disagreement
    """
    if config is None:
        config = load_config()

    class_up = config['labels']['class_up']    # 1
    class_down = config['labels']['class_down']  # 0

    # Mean P(Up) across seeds
    seed_p_up = seed_probs[:, class_up]
    p_up = float(np.mean(seed_p_up))
    p_down = 1.0 - p_up

    # Seed disagreement
    seed_disagreement = float(np.std(seed_p_up))

    # Direction string conversion — ONLY PLACE THIS HAPPENS
    # Per spec §7.5: direction = "up" if p_up >= p_down else "down"
    direction = "up" if p_up >= p_down else "down"

    return EnsembleResult(
        p_up=p_up,
        p_down=p_down,
        direction=direction,
        seed_disagreement=seed_disagreement,
        seed_probs_up=[float(p) for p in seed_p_up],
        p_up_raw=p_up,  # will be overwritten by calibration
    )


def run_ensemble_inference(
    models: list,
    X: np.ndarray,
    device: str = 'cpu',
) -> EnsembleResult:
    """
    Run inference across all seed models and ensemble.

    Args:
        models: List of trained PyTorch models (one per seed)
        X: Input features (1, sequence_length, n_features)
        device: Device to run on

    Returns:
        EnsembleResult
    """
    import torch

    seed_probs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(device)
            if x_tensor.dim() == 2:
                x_tensor = x_tensor.unsqueeze(0)
            probs = model(x_tensor)  # (1, 2) → [p_down, p_up]
            seed_probs.append(probs.cpu().numpy()[0])

    seed_probs = np.array(seed_probs)  # (n_seeds, 2)
    return ensemble_predict(seed_probs)


def run_onnx_ensemble_inference(
    sessions: list,
    X: np.ndarray,
) -> EnsembleResult:
    """
    Run ONNX inference across all seed sessions and ensemble.

    Args:
        sessions: List of ONNX Runtime sessions (one per seed)
        X: Input features (1, sequence_length, n_features)

    Returns:
        EnsembleResult
    """
    seed_probs = []

    for session in sessions:
        input_name = session.get_inputs()[0].name
        if X.ndim == 2:
            X_input = X[np.newaxis, ...]  # add batch dim
        else:
            X_input = X

        result = session.run(None, {input_name: X_input.astype(np.float32)})
        seed_probs.append(result[0][0])  # (2,) → [p_down, p_up]

    seed_probs = np.array(seed_probs)  # (n_seeds, 2)
    return ensemble_predict(seed_probs)
