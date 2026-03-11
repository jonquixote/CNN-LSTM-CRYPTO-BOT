"""
monitoring/health.py — System health dashboard checks.

Per spec §16:
    Monitor:
    - Feature pipeline latency (< 2s on VPS)
    - Inference latency (< 500ms on VPS with ONNX)
    - Data freshness (last bar < 10 minutes old)
    - Model staleness (last retrain < 8 days)
    - Bankroll health (drawdown alerts)
    - Filter rejection rates (alert if > 90%)
    - Calibration drift

    Alerts via Telegram for:
    - Feature pipeline failure
    - Inference miss (no prediction within window)
    - Drawdown > 15% from peak
    - Model not retrained in > 7 days
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_system_health(config: Optional[dict] = None) -> dict:
    """
    Run all health checks.

    Returns:
        dict with check results and overall health status
    """
    if config is None:
        config = load_config()

    health = config.get('health', {})
    checks = {}

    # 1. Data freshness
    checks['data_freshness'] = _check_data_freshness(
        health.get('max_data_age_minutes', 10)
    )

    # 2. Model staleness
    checks['model_staleness'] = _check_model_staleness(
        health.get('max_model_age_days', 8)
    )

    # 3. Filter rejection rate
    checks['filter_rejection'] = _check_filter_rejection_rate(
        health.get('max_filter_rejection_pct', 90)
    )

    # 4. Bankroll health
    checks['bankroll'] = _check_bankroll_health(
        health.get('max_drawdown_pct', 15), config
    )

    # 5. Inference latency
    checks['inference_latency'] = _check_inference_latency(
        health.get('max_inference_ms', 500)
    )

    # Overall
    all_ok = all(c.get('ok', False) for c in checks.values())
    alerts = [
        name for name, c in checks.items()
        if not c.get('ok', False)
    ]

    return {
        'healthy': all_ok,
        'checks': checks,
        'alerts': alerts,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def _check_data_freshness(max_age_minutes: int) -> dict:
    """Check if the last data fetch is recent enough."""
    # Read last fetch timestamp from state file
    state_path = os.path.join(
        os.path.dirname(__file__), '..', 'state', 'last_fetch.json'
    )
    if not os.path.exists(state_path):
        return {'ok': False, 'reason': 'no_state_file', 'age_minutes': None}

    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        last_ts = datetime.fromisoformat(state['timestamp'])
        age = datetime.now(timezone.utc) - last_ts
        age_minutes = age.total_seconds() / 60

        return {
            'ok': age_minutes < max_age_minutes,
            'age_minutes': round(age_minutes, 1),
            'threshold': max_age_minutes,
        }
    except Exception as e:
        return {'ok': False, 'reason': str(e), 'age_minutes': None}


def _check_model_staleness(max_age_days: int) -> dict:
    """Check if the model was retrained recently enough."""
    state_path = os.path.join(
        os.path.dirname(__file__), '..', 'state', 'last_retrain.json'
    )
    if not os.path.exists(state_path):
        return {'ok': False, 'reason': 'never_retrained', 'age_days': None}

    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        last_ts = datetime.fromisoformat(state['timestamp'])
        age = datetime.now(timezone.utc) - last_ts
        age_days = age.total_seconds() / 86400

        return {
            'ok': age_days < max_age_days,
            'age_days': round(age_days, 1),
            'threshold': max_age_days,
        }
    except Exception as e:
        return {'ok': False, 'reason': str(e), 'age_days': None}


def _check_filter_rejection_rate(max_rejection_pct: int) -> dict:
    """Check filter rejection rate from trade log."""
    log_path = os.path.join(
        os.path.dirname(__file__), '..', 'trade_log.json'
    )
    if not os.path.exists(log_path):
        return {'ok': True, 'rejection_pct': 0, 'note': 'no_trade_log'}

    try:
        with open(log_path, 'r') as f:
            trades = json.load(f)

        # Look at last 100 ticks
        recent = trades[-100:]
        n_skipped = sum(1 for t in recent if t.get('status') == 'skipped')
        rejection_pct = n_skipped / max(len(recent), 1) * 100

        return {
            'ok': rejection_pct < max_rejection_pct,
            'rejection_pct': round(rejection_pct, 1),
            'threshold': max_rejection_pct,
        }
    except Exception as e:
        return {'ok': True, 'reason': str(e)}


def _check_bankroll_health(max_drawdown_pct: int, config: dict) -> dict:
    """Check bankroll health and drawdown."""
    log_path = os.path.join(
        os.path.dirname(__file__), '..', 'trade_log.json'
    )
    if not os.path.exists(log_path):
        return {'ok': True, 'drawdown_pct': 0, 'note': 'no_trades_yet'}

    try:
        with open(log_path, 'r') as f:
            trades = json.load(f)

        settled = [t for t in trades if t.get('pnl') is not None]
        if not settled:
            return {'ok': True, 'drawdown_pct': 0}

        starting = config['accounts']['starting_bankroll']
        pnls = [t['pnl'] for t in settled]
        equity = starting + sum(pnls)
        peak = starting
        max_dd = 0

        running = starting
        for pnl in pnls:
            running += pnl
            peak = max(peak, running)
            dd = (peak - running) / max(peak, 1) * 100
            max_dd = max(max_dd, dd)

        return {
            'ok': max_dd < max_drawdown_pct,
            'drawdown_pct': round(max_dd, 1),
            'current_equity': round(equity, 2),
            'threshold': max_drawdown_pct,
        }
    except Exception as e:
        return {'ok': False, 'reason': str(e)}


def _check_inference_latency(max_ms: int) -> dict:
    """Check recent inference latency."""
    state_path = os.path.join(
        os.path.dirname(__file__), '..', 'state', 'last_inference.json'
    )
    if not os.path.exists(state_path):
        return {'ok': True, 'latency_ms': None, 'note': 'no_inference_yet'}

    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        latency = state.get('latency_ms', 0)
        return {
            'ok': latency < max_ms,
            'latency_ms': latency,
            'threshold': max_ms,
        }
    except Exception as e:
        return {'ok': True, 'reason': str(e)}
