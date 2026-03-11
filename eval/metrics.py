"""
eval/metrics.py — Evaluation metrics for walk-forward testing.

p_market fallback = 0.50 documented; uses parquet where available.
Per spec §9: historical Polymarket prices from Jon-Becker dataset.

Metrics per fold:
  - Hit rate (filtered)
  - Net PnL after fees
  - Simulated Sharpe
  - Max drawdown
  - Turnover (bets per day)
  - Filter pass rate
  - Per-filter rejection rate
  - Edge distribution
  - Negative Kelly suppression rate
  - ECE (expected calibration error)
  - p_market_source ("historical" or "assumed_0.50")
"""

import os
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


def compute_fold_metrics(
    p_up: np.ndarray,
    labels: np.ndarray,
    p_market: np.ndarray,
    config: Optional[dict] = None,
) -> dict:
    """
    Compute all metrics for a single fold.

    Args:
        p_up: Model predicted P(Up) for each bar (calibrated)
        labels: Ground truth labels (0=Down, 1=Up)
        p_market: Market price for each bar
            - From p_market_history.parquet where available
            - Fallback: assumed_market_price = 0.50 (documented here per spec)
        config: Optional config override

    Returns:
        dict of metrics
    """
    if config is None:
        config = load_config()

    sc = config['strategy']
    cc = config['costs']

    n_bars = len(p_up)
    if n_bars == 0:
        return {'n_bars': 0, 'hit_rate': 0.0}

    # Determine direction and edge for each bar
    p_down = 1.0 - p_up
    direction = np.where(p_up >= p_down, 1, 0)  # 1=up, 0=down
    p_model = np.where(direction == 1, p_up, p_down)

    # Edge computation: p_model - p_market
    # p_market = historical prices where available, 0.50 fallback otherwise
    edge = p_model - p_market

    # Determine which bars pass the edge filter
    bet_mask = (
        (edge >= sc['edge_threshold']) &
        (p_model >= sc['min_model_confidence'])
    )

    n_bets = bet_mask.sum()
    filter_pass_rate = n_bets / n_bars if n_bars > 0 else 0.0

    # p_market source tracking
    # assumed_market_price = 0.50 per config — this is the efficient-market null hypothesis
    assumed_price = config['backtest']['assumed_market_price']
    n_historical = np.sum(p_market != assumed_price)
    n_assumed = np.sum(p_market == assumed_price)
    p_market_source = 'historical' if n_historical > n_assumed else 'assumed_0.50'

    if n_bets == 0:
        return {
            'n_bars': int(n_bars),
            'n_bets': 0,
            'hit_rate': 0.0,
            'net_pnl': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'turnover_per_day': 0.0,
            'filter_pass_rate': 0.0,
            'ece': _compute_ece(p_up, labels),
            'p_market_source': p_market_source,
        }

    # Hit rate on filtered bars
    bet_labels = labels[bet_mask]
    bet_direction = direction[bet_mask]
    bet_correct = (bet_direction == bet_labels)
    hit_rate = bet_correct.mean()

    # PnL computation — Polymarket binary market
    # Per spec §12: US flat fee model
    fee_pct = cc['taker_fee_pct']
    min_fee = cc['min_fee_usdc']

    stake = 10.0  # normalized stake for PnL computation
    pnl_per_bet = []

    bet_p_market = p_market[bet_mask]
    for i in range(n_bets):
        entry_price = bet_p_market[i]
        fee = max(stake * fee_pct, min_fee)

        if bet_correct[i]:
            # Win: shares resolve at $1.00
            shares = stake / entry_price
            pnl_gross = shares * (1.0 - entry_price)
        else:
            # Loss: shares resolve at $0.00
            pnl_gross = -stake

        pnl_net = pnl_gross - fee
        pnl_per_bet.append(pnl_net)

    pnl_array = np.array(pnl_per_bet)
    net_pnl = pnl_array.sum()

    # Sharpe ratio (annualized from daily)
    # Assume ~288 bars per day for 5m
    bars_per_day = 288
    n_days = n_bars / bars_per_day
    daily_pnl = []
    bets_per_day_count = []
    bet_indices = np.where(bet_mask)[0]

    for day in range(int(n_days)):
        day_start = day * bars_per_day
        day_end = (day + 1) * bars_per_day
        day_bets = (bet_indices >= day_start) & (bet_indices < day_end)
        day_pnl = pnl_array[day_bets].sum() if day_bets.any() else 0.0
        daily_pnl.append(day_pnl)
        bets_per_day_count.append(day_bets.sum())

    daily_pnl = np.array(daily_pnl)
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    # Max drawdown
    equity_curve = np.cumsum(pnl_array)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-10)
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0

    # Turnover
    turnover_per_day = n_bets / max(n_days, 1)

    # Edge distribution
    bet_edges = edge[bet_mask]
    edge_mean = float(bet_edges.mean())
    edge_std = float(bet_edges.std()) if len(bet_edges) > 1 else 0.0

    # ECE
    ece = _compute_ece(p_up, labels)

    # Negative Kelly suppression rate
    # Simulated: % of edge-passing bets that would have negative Kelly
    kelly_suppressed = 0
    for i in range(n_bets):
        p = p_up[bet_mask][i] if bet_direction[i] == 1 else p_down[bet_mask][i]
        q = 1 - p
        pm = bet_p_market[i]
        b = (1 - pm) / max(pm, 1e-10)
        kelly_raw = (p * b - q) / max(b, 1e-10)
        if kelly_raw < 0:
            kelly_suppressed += 1

    kelly_suppression_rate = kelly_suppressed / max(n_bets, 1)

    return {
        'n_bars': int(n_bars),
        'n_bets': int(n_bets),
        'hit_rate': float(hit_rate),
        'net_pnl': float(net_pnl),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'turnover_per_day': float(turnover_per_day),
        'filter_pass_rate': float(filter_pass_rate),
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'kelly_suppression_rate': float(kelly_suppression_rate),
        'ece': float(ece),
        'p_market_source': p_market_source,
    }


def _compute_ece(p_up: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.

    Measures how well calibrated the model probabilities are.
    Lower is better. 0 = perfectly calibrated.
    """
    if len(p_up) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (p_up >= bin_boundaries[i]) & (p_up < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_confidence = p_up[mask].mean()
        bin_accuracy = labels[mask].mean()
        bin_weight = mask.sum() / len(p_up)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def load_p_market_history(config: Optional[dict] = None) -> pd.DataFrame:
    """
    Load p_market_history.parquet.

    Per spec §9: p_market = historical prices from Jon-Becker dataset.
    For bars with no historical price, assumed_market_price (0.50) is used.
    This fallback is the efficient-market null hypothesis.
    """
    if config is None:
        config = load_config()

    parquet_path = os.path.join(
        os.path.dirname(__file__), '..',
        config['backtest']['historical_prices_path']
    )

    if not os.path.exists(parquet_path):
        logger.warning(
            "p_market_history.parquet not found at {}. "
            "Using assumed_market_price = {} fallback for all bars.".format(
                parquet_path, config['backtest']['assumed_market_price']
            )
        )
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded {} historical market prices".format(len(df)))
    return df
