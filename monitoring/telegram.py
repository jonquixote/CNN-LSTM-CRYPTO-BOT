"""
monitoring/telegram.py — Telegram alerting.

Per spec §19:
    Sends structured health and trade alerts to Telegram.
    Uses TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env.
"""

import os
import json
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def send_telegram(
    message: str,
    parse_mode: str = 'HTML',
) -> bool:
    """
    Send a message via Telegram Bot API.

    Args:
        message: Message text (HTML or Markdown)
        parse_mode: 'HTML' or 'Markdown'

    Returns:
        True if sent successfully
    """
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        logger.warning("Telegram credentials not set. Skipping alert.")
        return False

    url = 'https://api.telegram.org/bot{}/sendMessage'.format(token)
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': parse_mode,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Telegram send failed: {}".format(e))
        return False


def send_trade_alert(trade_info: dict) -> bool:
    """Send trade execution alert."""
    msg = (
        "<b>Trade Executed</b>\n"
        "Slot: {slot}\n"
        "Direction: {direction}\n"
        "Stake: ${stake:.2f}\n"
        "Entry: {entry_price:.4f}\n"
        "Edge: {edge:.4f}\n"
        "Kelly: {kelly:.4f}\n"
        "P(model): {p_model:.4f}"
    ).format(**trade_info)
    return send_telegram(msg)


def send_settlement_alert(trade_info: dict) -> bool:
    """Send trade settlement alert."""
    outcome = "WIN" if trade_info.get('outcome') == 1 else "LOSS"
    emoji = "✅" if outcome == "WIN" else "❌"
    msg = (
        "{emoji} <b>{outcome}</b>\n"
        "Slot: {slot}\n"
        "Direction: {direction}\n"
        "PnL: ${pnl:+.2f}\n"
        "Stake: ${stake:.2f}"
    ).format(emoji=emoji, outcome=outcome, **trade_info)
    return send_telegram(msg)


def send_health_alert(health: dict) -> bool:
    """Send system health alert."""
    if health['healthy']:
        return True  # no alert needed

    alerts = health.get('alerts', [])
    msg = (
        "⚠️ <b>Health Alert</b>\n"
        "Issues: {}\n\n".format(', '.join(alerts))
    )

    for name in alerts:
        check = health['checks'].get(name, {})
        msg += "<b>{}:</b> {}\n".format(name, json.dumps(check, indent=None))

    return send_telegram(msg)


def send_retrain_summary(summary: dict) -> bool:
    """Send weekly retrain summary."""
    msg = (
        "🔄 <b>Retrain Complete</b>\n"
        "Slot: {slot}\n"
        "Folds: {n_folds}\n"
        "Val Sharpe (mean): {val_sharpe_mean:.2f}\n"
        "Val Hit Rate: {val_hit_rate:.2%}\n"
        "ECE: {ece:.4f}\n"
        "Duration: {duration_min:.1f} min"
    ).format(**summary)
    return send_telegram(msg)


def send_daily_pnl(pnl_info: dict) -> bool:
    """Send daily PnL summary."""
    emoji = "📈" if pnl_info.get('total_pnl', 0) >= 0 else "📉"
    msg = (
        "{emoji} <b>Daily PnL Summary</b>\n"
        "Total PnL: ${total_pnl:+.2f}\n"
        "Trades: {n_trades}\n"
        "Wins: {n_wins}\n"
        "Win Rate: {win_rate:.1%}\n"
        "Avg PnL/Trade: ${avg_pnl:+.2f}"
    ).format(emoji=emoji, **pnl_info)
    return send_telegram(msg)
