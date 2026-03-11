"""
strategy/execution.py — Trade execution logic for Polymarket.

Per spec §12:
    - Places limit orders at 1 tick inside best bid/ask
    - Wait 30s for fill
    - Cancel & replace if not filled
    - Final timeout at T-30s before market close
    - All fills logged to trade_log.json

Direction strings only ("up"/"down") — above ensemble.py.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict

import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class TradeOrder:
    """A trade order to be placed."""
    slot: str              # e.g. "BTC_5m"
    direction: str         # "up" or "down"
    entry_price: float     # limit price
    stake_usdc: float      # position size
    p_model: float         # calibrated model probability
    p_market: float        # market price at decision time
    edge: float            # p_model - p_market
    kelly_fraction: float  # capped Kelly
    seed_disagreement: float
    timestamp: str         # ISO format UTC


@dataclass
class TradeResult:
    """Result of trade execution."""
    order: TradeOrder
    filled: bool
    fill_price: float
    fill_time: str
    outcome: Optional[int]   # 0=loss, 1=win (set after settlement)
    pnl: Optional[float]     # set after settlement
    status: str              # 'filled', 'timeout', 'cancelled', 'error'


class TradeExecutor:
    """Manages trade execution against Polymarket API."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.exec_config = config['execution']
        self.trade_log_path = os.path.join(
            os.path.dirname(__file__), '..', 'trade_log.json'
        )

    def execute_trade(self, order: TradeOrder) -> TradeResult:
        """
        Execute a trade on Polymarket.

        Strategy:
            1. Get current best bid/ask
            2. Place limit order at 1 tick inside
            3. Wait fill_wait_seconds (30s)
            4. If not filled, cancel & replace
            5. Final timeout at market_close - 30s

        For paper trading (Phase 7): simulates fills at p_market.

        Args:
            order: TradeOrder to execute

        Returns:
            TradeResult
        """
        # Paper trading mode
        if self.config.get('paper_trading', {}).get('enabled', True):
            return self._paper_trade(order)

        # Live trading — to be implemented in Phase 8
        return self._live_trade(order)

    def _paper_trade(self, order: TradeOrder) -> TradeResult:
        """Simulate trade fill for paper trading."""
        result = TradeResult(
            order=order,
            filled=True,
            fill_price=order.entry_price,
            fill_time=datetime.now(timezone.utc).isoformat(),
            outcome=None,
            pnl=None,
            status='filled',
        )

        self._log_trade(result)
        return result

    def _live_trade(self, order: TradeOrder) -> TradeResult:
        """
        Execute live trade via Polymarket CLOB API.
        TODO: Implement in Phase 8.
        """
        raise NotImplementedError("Live trading not yet implemented. Use paper trading.")

    def settle_trade(
        self,
        result: TradeResult,
        actual_label: int,
    ) -> TradeResult:
        """
        Settle a trade after the bar closes.

        Args:
            result: TradeResult from execute_trade
            actual_label: 0=Down, 1=Up (ground truth)

        Returns:
            Updated TradeResult with outcome and PnL
        """
        # Determine if trade won
        direction_label = 1 if result.order.direction == "up" else 0
        won = (direction_label == actual_label)

        result.outcome = 1 if won else 0

        # PnL calculation
        stake = result.order.stake_usdc
        entry = result.fill_price
        fee_pct = self.config['costs']['taker_fee_pct']
        fee = max(stake * fee_pct, self.config['costs']['min_fee_usdc'])

        if won:
            shares = stake / entry
            pnl_gross = shares * (1.0 - entry)
        else:
            pnl_gross = -stake

        result.pnl = round(pnl_gross - fee, 4)

        self._log_trade(result)
        return result

    def _log_trade(self, result: TradeResult) -> None:
        """Append trade to trade_log.json."""
        log_entry = {
            'timestamp': result.fill_time or datetime.now(timezone.utc).isoformat(),
            'slot': result.order.slot,
            'direction': result.order.direction,
            'stake': result.order.stake_usdc,
            'entry_price': result.order.entry_price,
            'fill_price': result.fill_price,
            'p_model': result.order.p_model,
            'p_market': result.order.p_market,
            'edge': result.order.edge,
            'kelly': result.order.kelly_fraction,
            'seed_disagreement': result.order.seed_disagreement,
            'status': result.status,
            'outcome': result.outcome,
            'pnl': result.pnl,
        }

        # Append to log file
        log_data = []
        if os.path.exists(self.trade_log_path):
            with open(self.trade_log_path, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = []

        log_data.append(log_entry)

        with open(self.trade_log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def get_trade_history(self, last_n: int = 50) -> list:
        """Load recent trade history."""
        if not os.path.exists(self.trade_log_path):
            return []

        with open(self.trade_log_path, 'r') as f:
            log_data = json.load(f)

        return log_data[-last_n:]

    def get_pnl_summary(self) -> dict:
        """Summarize PnL from trade log."""
        history = self.get_trade_history(last_n=10000)

        settled = [t for t in history if t.get('pnl') is not None]
        if not settled:
            return {'total_pnl': 0, 'n_trades': 0, 'win_rate': 0}

        pnls = [t['pnl'] for t in settled]
        wins = [t for t in settled if t['outcome'] == 1]

        return {
            'total_pnl': round(sum(pnls), 2),
            'n_trades': len(settled),
            'n_wins': len(wins),
            'win_rate': round(len(wins) / len(settled), 4),
            'avg_pnl': round(sum(pnls) / len(pnls), 4),
            'max_win': round(max(pnls), 4),
            'max_loss': round(min(pnls), 4),
        }
