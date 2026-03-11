"""
strategy/sizing.py — Kelly-fraction position sizing.

Per spec §11.2:
    kelly_bet = (p * b - q) / b
    Where:
        p = calibrated model confidence for chosen direction
        q = 1 - p
        b = (1 - p_market) / p_market  (binary payoff)

    Then:
        capped_kelly = min(kelly_bet, max_position_size)
        stake         = bankroll * capped_kelly

    Negative Kelly → skip (filter catches this).
    kelly_bet capped at max_position_size (0.05 = 5% of bankroll).
    Minimum order: $1 (Polymarket minimum).
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    kelly_raw: float
    kelly_capped: float
    stake_usdc: float
    bankroll: float
    skip: bool
    skip_reason: str


def compute_kelly_stake(
    p_model: float,
    p_market: float,
    bankroll: float,
    config: Optional[dict] = None,
) -> SizingResult:
    """
    Compute Kelly-fraction position size.

    Args:
        p_model: Calibrated probability for chosen direction
        p_market: Current market price (entry cost)
            - From order book or API in live trading
            - p_market_history.parquet in backtesting
            - Fallback: assumed_market_price = 0.50
        bankroll: Current bankroll in USDC
        config: Optional config override

    Returns:
        SizingResult with stake amount
    """
    if config is None:
        config = load_config()

    sc = config['strategy']

    # Binary payoff odds: b = (1 - p_market) / p_market
    # If p_market = 0.50, b = 1.0 (even odds)
    b = (1.0 - p_market) / max(p_market, 1e-10)
    q = 1.0 - p_model

    # Kelly formula
    kelly_raw = (p_model * b - q) / max(b, 1e-10)

    # Skip on negative Kelly (no edge)
    if kelly_raw <= 0:
        return SizingResult(
            kelly_raw=kelly_raw,
            kelly_capped=0.0,
            stake_usdc=0.0,
            bankroll=bankroll,
            skip=True,
            skip_reason='negative_kelly',
        )

    # Cap at max position size (kelly_max_fraction from config)
    max_pos = sc.get('kelly_max_fraction', 0.15)
    kelly_capped = min(kelly_raw, max_pos)

    # Compute stake
    stake = bankroll * kelly_capped

    # Enforce minimum order
    min_order = sc.get('min_order_usdc', 1.0)
    if stake < min_order:
        return SizingResult(
            kelly_raw=kelly_raw,
            kelly_capped=kelly_capped,
            stake_usdc=0.0,
            bankroll=bankroll,
            skip=True,
            skip_reason='below_min_order',
        )

    return SizingResult(
        kelly_raw=kelly_raw,
        kelly_capped=kelly_capped,
        stake_usdc=round(stake, 2),
        bankroll=bankroll,
        skip=False,
        skip_reason='',
    )
