"""
selection/boruta_shap.py — Boruta-SHAP feature selection.

SHAP values from XGBoost/CatBoost replace Gini impurity.
Run per fold on GPU. Features demoted across 3 consecutive retrains
trigger a manual review flag.
feature_list.json versioned with SHA-256 hash of sorted JSON content.

Per spec §5.6.
"""

import os
import json
import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BorutaSHAP:
    """
    Boruta-style feature selection using SHAP values.

    Instead of original Boruta's Gini importance, uses SHAP values from
    XGBoost/CatBoost for more accurate feature importance estimates.
    """

    def __init__(
        self,
        n_trials: int = 100,
        alpha: float = 0.05,
        use_catboost: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            n_trials: Number of Boruta iterations
            alpha: Statistical significance level
            use_catboost: Use CatBoost instead of XGBoost
            random_state: Random seed for reproducibility
        """
        self.n_trials = n_trials
        self.alpha = alpha
        self.use_catboost = use_catboost
        self.random_state = random_state
        self.accepted_features: list[str] = []
        self.rejected_features: list[str] = []
        self.tentative_features: list[str] = []
        self.feature_importances: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BorutaSHAP':
        """
        Run Boruta-SHAP feature selection.

        Args:
            X: Feature DataFrame (post-warmup, no NaNs)
            y: Label Series (0/1 binary)

        Returns:
            self
        """
        import shap

        if self.use_catboost:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                verbose=0,
                random_seed=self.random_state,
                task_type='GPU' if _gpu_available_catboost() else 'CPU',
            )
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state,
                tree_method='hist',
                device='cuda' if _gpu_available_xgb() else 'cpu',
                verbosity=0,
            )

        n_features = X.shape[1]
        feature_names = X.columns.tolist()

        # Track hits: how many times each feature beats the best shadow
        hits = np.zeros(n_features, dtype=int)

        logger.info(
            "Starting Boruta-SHAP: {} features, {} trials".format(n_features, self.n_trials)
        )

        for trial in range(self.n_trials):
            # Create shadow features by shuffling each column
            X_shadow = X.apply(np.random.permutation).copy()
            X_shadow.columns = ['shadow_' + c for c in feature_names]

            # Combine real + shadow
            X_combined = pd.concat([X, X_shadow], axis=1)

            # Fit model
            model.fit(X_combined, y)

            # Compute SHAP values
            if self.use_catboost:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_combined)

            # For binary classification, shap_values may be a list [class_0, class_1]
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[1])  # class_up importance
            else:
                shap_vals = np.abs(shap_values)

            # Mean absolute SHAP per feature
            mean_shap = shap_vals.mean(axis=0)
            real_shap = mean_shap[:n_features]
            shadow_shap = mean_shap[n_features:]

            # Best shadow threshold
            shadow_max = shadow_shap.max()

            # Count hits
            hits += (real_shap > shadow_max).astype(int)

            if (trial + 1) % 20 == 0:
                logger.info("  Trial {}/{}".format(trial + 1, self.n_trials))

        # Statistical test: binomial test at alpha significance
        from scipy import stats

        for i, feat_name in enumerate(feature_names):
            # Under null hypothesis, P(hit) = 0.5
            p_value = stats.binomtest(hits[i], self.n_trials, 0.5, alternative='greater').pvalue
            self.feature_importances[feat_name] = float(hits[i] / self.n_trials)

            if p_value < self.alpha:
                self.accepted_features.append(feat_name)
            elif p_value > (1 - self.alpha):
                self.rejected_features.append(feat_name)
            else:
                self.tentative_features.append(feat_name)

        logger.info(
            "Boruta-SHAP complete: {} accepted, {} rejected, {} tentative".format(
                len(self.accepted_features),
                len(self.rejected_features),
                len(self.tentative_features),
            )
        )

        return self

    def get_accepted_features(self) -> list[str]:
        """Return list of accepted features (include tentative by default)."""
        return sorted(self.accepted_features + self.tentative_features)

    def get_feature_report(self) -> dict:
        """Return full feature selection report."""
        return {
            'accepted': sorted(self.accepted_features),
            'rejected': sorted(self.rejected_features),
            'tentative': sorted(self.tentative_features),
            'importances': self.feature_importances,
            'n_trials': self.n_trials,
            'alpha': self.alpha,
        }


def save_feature_list(
    feature_names: list[str],
    output_dir: str = None,
    metadata: Optional[dict] = None,
) -> tuple[str, str]:
    """
    Save feature_list.json with SHA-256 hash of sorted JSON content.

    Per spec §5.6: feature_list.json versioned with SHA-256 hash.
    Per spec §20 rule 20: always SHA-256 of sorted JSON content.

    Args:
        feature_names: List of accepted feature names
        output_dir: Directory to save to (defaults to project root)
        metadata: Optional metadata to include

    Returns:
        (filepath, sha256_hash)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..')

    sorted_names = sorted(feature_names)
    content = {
        'features': sorted_names,
        'count': len(sorted_names),
    }
    if metadata:
        content['metadata'] = metadata

    # Compute hash of sorted feature list (just the names, not metadata)
    json_str = json.dumps(sorted_names, sort_keys=True)
    sha256_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    content['feature_list_hash'] = sha256_hash
    content['feature_list_hash_algorithm'] = 'sha256_sorted_json'

    filepath = os.path.join(output_dir, 'feature_list.json')
    with open(filepath, 'w') as f:
        json.dump(content, f, indent=2)

    logger.info("Saved feature_list.json: {} features, hash={}...".format(
        len(sorted_names), sha256_hash[:16]
    ))

    return filepath, sha256_hash


def load_feature_list(filepath: str = None) -> tuple[list[str], str]:
    """
    Load feature_list.json and verify its hash.

    Returns:
        (feature_names, sha256_hash)

    Raises:
        ValueError: If hash verification fails
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'feature_list.json')

    with open(filepath, 'r') as f:
        content = json.load(f)

    features = content['features']
    stored_hash = content['feature_list_hash']

    # Verify hash
    json_str = json.dumps(sorted(features), sort_keys=True)
    computed_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    if computed_hash != stored_hash:
        raise ValueError(
            "Feature list hash mismatch! "
            "Stored: {}, Computed: {}".format(stored_hash[:16], computed_hash[:16])
        )

    return features, stored_hash


def check_demotion_history(
    current_rejected: list[str],
    history_file: str = None,
    consecutive_threshold: int = 3,
) -> list[str]:
    """
    Check for features demoted across consecutive retrains.
    Features demoted across 3 consecutive retrains trigger a manual review flag.

    Args:
        current_rejected: Features rejected in current run
        history_file: Path to demotion history JSON
        consecutive_threshold: Number of consecutive demotions to flag

    Returns:
        List of features flagged for manual review
    """
    if history_file is None:
        history_file = os.path.join(
            os.path.dirname(__file__), '..', 'feature_demotion_history.json'
        )

    # Load history
    history = {}
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)

    # Update history
    flagged = []
    for feat in current_rejected:
        count = history.get(feat, 0) + 1
        history[feat] = count
        if count >= consecutive_threshold:
            flagged.append(feat)
            logger.warning(
                "Feature '{}' demoted {} consecutive times — manual review flagged".format(
                    feat, count
                )
            )

    # Reset count for features NOT rejected this time
    for feat in list(history.keys()):
        if feat not in current_rejected:
            history[feat] = 0

    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    return flagged


def _gpu_available_xgb() -> bool:
    """Check if GPU is available for XGBoost."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_available_catboost() -> bool:
    """Check if GPU is available for CatBoost."""
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(task_type='GPU', iterations=1, verbose=0)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Boruta-SHAP module loaded. Use BorutaSHAP class for feature selection.")
    print("Run during training pipeline, not standalone.")
