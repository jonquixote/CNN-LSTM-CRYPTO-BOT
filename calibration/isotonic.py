"""
calibration/isotonic.py — Isotonic regression probability calibration.

Per spec §8:
    Isotonic regression maps raw softmax p_up to empirically observed
    bar-up rates. Fit on (raw_p_up, outcome) pairs from validation folds.
    Re-fit after every weekly retrain.
    All downstream logic uses calibrated probabilities.
    Both raw and calibrated logged for drift monitoring.
"""

import os
import logging
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for model probabilities.

    Maps raw softmax P(Up) to empirically observed bar-up rates.
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0.01,  # clip to avoid exact 0/1
            y_max=0.99,
            out_of_bounds='clip',
        )
        self.is_fitted = False

    def fit(self, raw_p_up: np.ndarray, outcomes: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit calibrator on validation data.

        Args:
            raw_p_up: Raw softmax P(Up) values from model
            outcomes: Ground truth labels (0=Down, 1=Up)

        Returns:
            self
        """
        assert len(raw_p_up) == len(outcomes)
        assert set(np.unique(outcomes)).issubset({0, 1})

        self.calibrator.fit(raw_p_up, outcomes)
        self.is_fitted = True

        logger.info(
            "Isotonic calibrator fitted on {} samples".format(len(raw_p_up))
        )

        return self

    def calibrate(self, raw_p_up: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.

        Args:
            raw_p_up: Raw softmax P(Up) values

        Returns:
            Calibrated P(Up) values
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        calibrated = self.calibrator.predict(raw_p_up)
        return np.clip(calibrated, 0.01, 0.99)

    def calibrate_single(self, raw_p_up: float) -> float:
        """Calibrate a single probability value."""
        result = self.calibrate(np.array([raw_p_up]))
        return float(result[0])

    def save(self, filepath: str) -> None:
        """Save calibrator to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.calibrator, f)
        logger.info("Calibrator saved to {}".format(filepath))

    def load(self, filepath: str) -> 'IsotonicCalibrator':
        """Load calibrator from disk."""
        with open(filepath, 'rb') as f:
            self.calibrator = pickle.load(f)
        self.is_fitted = True
        logger.info("Calibrator loaded from {}".format(filepath))
        return self

    def get_calibration_stats(
        self, raw_p_up: np.ndarray, outcomes: np.ndarray
    ) -> dict:
        """
        Compute calibration statistics for monitoring.

        Both raw and calibrated are logged for drift monitoring (per spec §8).
        """
        calibrated = self.calibrate(raw_p_up)

        # ECE before and after calibration
        ece_raw = _compute_ece(raw_p_up, outcomes)
        ece_calibrated = _compute_ece(calibrated, outcomes)

        # Mean drift
        drift = float(np.mean(np.abs(calibrated - raw_p_up)))

        return {
            'ece_raw': float(ece_raw),
            'ece_calibrated': float(ece_calibrated),
            'ece_improvement': float(ece_raw - ece_calibrated),
            'mean_drift': drift,
            'n_samples': len(raw_p_up),
        }


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = labels[mask].mean()
        bin_weight = mask.sum() / len(probs)
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece
