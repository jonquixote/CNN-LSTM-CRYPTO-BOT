"""
strategy/filters.py — Pre-trade filter cascade.

Per spec §11.1, filters stop a trade before the order is placed.
All filter logic uses direction STRINGS ("up"/"down") — we are above ensemble.py.

Filter cascade order:
    1. Edge filter: p_model - p_market >= 0.04
    2. Kelly filter: kelly_fraction > 0
    3. Entropy filter: pred_entropy < 0.64
    4. Minimum confidence: max(p_up, p_down) >= 0.52
    5. Seed disagreement: disagreement < 0.08
    6. Regime entropy: 6h rolling entropy < 0.95

Each filter returns (pass: bool, reason: str, value: float).
Every rejection is logged per-filter for rate monitoring.
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
class FilterResult:
    """Result of a single filter check."""
    name: str
    passed: bool
    reason: str
    value: float


@dataclass
class CascadeResult:
    """Result of the entire filter cascade."""
    passed: bool
    filter_results: list  # List[FilterResult]
    rejection_reason: str  # first failing filter

    @property
    def rejection_filter(self) -> str:
        for fr in self.filter_results:
            if not fr.passed:
                return fr.name
        return ''


def run_filter_cascade(
    p_up: float,
    p_down: float,
    p_market: float,
    direction: str,
    seed_disagreement: float = 0.0,
    regime_entropy: float = 0.0,
    config: Optional[dict] = None,
) -> CascadeResult:
    """
    Run the full pre-trade filter cascade.

    All args use direction STRINGS — this is above ensemble.py.

    Args:
        p_up: Calibrated P(Up)
        p_down: Calibrated P(Down) = 1 - p_up
        p_market: Current market price
            - From p_market_history.parquet where available
            - Live: from order book or API
            - Fallback: assumed_market_price = 0.50
        direction: "up" or "down" (from ensemble.py)
        seed_disagreement: Std of P(Up) across ensemble seeds
        regime_entropy: 6h rolling Shannon entropy
        config: Optional config override

    Returns:
        CascadeResult
    """
    if config is None:
        config = load_config()

    sc = config['strategy']
    results = []

    p_model = p_up if direction == "up" else p_down

    # 1. Edge filter: p_model - p_market >= threshold
    edge = p_model - p_market
    edge_pass = edge >= sc['edge_threshold']
    results.append(FilterResult(
        name='edge',
        passed=edge_pass,
        reason='' if edge_pass else 'edge {:.4f} < {}'.format(edge, sc['edge_threshold']),
        value=edge,
    ))

    # 2. Kelly filter: kelly_fraction > 0
    b = (1.0 - p_market) / max(p_market, 1e-10)
    q = 1 - p_model
    kelly_raw = (p_model * b - q) / max(b, 1e-10)
    kelly_pass = kelly_raw > 0
    results.append(FilterResult(
        name='kelly',
        passed=kelly_pass,
        reason='' if kelly_pass else 'negative kelly {:.4f}'.format(kelly_raw),
        value=kelly_raw,
    ))

    # 3. Entropy filter: prediction entropy < threshold
    # Binary entropy: -p*log2(p) - (1-p)*log2(1-p)
    eps = 1e-10
    pred_entropy = -(p_up * np.log2(p_up + eps) + p_down * np.log2(p_down + eps))
    entropy_pass = pred_entropy < sc.get('entropy_threshold', 0.64)
    results.append(FilterResult(
        name='entropy',
        passed=entropy_pass,
        reason='' if entropy_pass else 'entropy {:.4f} >= {}'.format(
            pred_entropy, sc.get('entropy_threshold', 0.64)
        ),
        value=pred_entropy,
    ))

    # 4. Minimum confidence
    min_conf = sc['min_model_confidence']
    conf_pass = p_model >= min_conf
    results.append(FilterResult(
        name='confidence',
        passed=conf_pass,
        reason='' if conf_pass else 'confidence {:.4f} < {}'.format(p_model, min_conf),
        value=p_model,
    ))

    # 5. Seed disagreement
    max_disagree = sc.get('seed_disagreement_max', 0.08)
    disagree_pass = seed_disagreement < max_disagree
    results.append(FilterResult(
        name='seed_disagreement',
        passed=disagree_pass,
        reason='' if disagree_pass else 'disagreement {:.4f} >= {}'.format(
            seed_disagreement, max_disagree
        ),
        value=seed_disagreement,
    ))

    # 6. Regime entropy filter
    max_regime = sc.get('max_regime_entropy', 0.95)
    regime_pass = regime_entropy < max_regime
    results.append(FilterResult(
        name='regime_entropy',
        passed=regime_pass,
        reason='' if regime_pass else 'regime_entropy {:.4f} >= {}'.format(
            regime_entropy, max_regime
        ),
        value=regime_entropy,
    ))

    # Cascade: all must pass
    all_passed = all(r.passed for r in results)
    rejection = ''
    if not all_passed:
        rejection = next(r.reason for r in results if not r.passed)

    return CascadeResult(
        passed=all_passed,
        filter_results=results,
        rejection_reason=rejection,
    )
