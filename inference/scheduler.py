"""
inference/scheduler.py — Inference scheduler (cron-driven).

Per spec §13, §14:
    Every 5m bar close (BTC_5m): trigger inference pipeline
    Every 15m bar close (BTC_15m, ETH_5m, ETH_15m): trigger once slots are live

    Pipeline per tick:
    1. Fetch latest bar (fetcher.py)
    2. Append to feature window
    3. Scale (pre-fit RobustScaler)
    4. Run ensemble inference (5 seeds × ONNX)
    5. Calibrate (isotonic.py)
    6. Run filter cascade (filters.py)
    7. Compute Kelly stake (sizing.py)
    8. Execute trade if all filters pass (execution.py)
    9. Log everything

    T4 GPU used for training only. VPS inference via ONNX Runtime (CPU).
"""

import os
import sys
import time
import json
import logging
import pickle
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class InferenceScheduler:
    """
    Manages the inference pipeline for a single slot.

    Runs as a persistent process triggered by cron or systemd timer.
    """

    def __init__(
        self,
        slot: str = 'BTC_5m',
        config: Optional[dict] = None,
        model_dir: str = None,
    ):
        if config is None:
            config = load_config()

        self.config = config
        self.slot = slot
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), '..', 'models', 'saved'
        )

        # Parse slot
        parts = slot.split('_')
        self.symbol = parts[0] + '/USDT'
        self.timeframe = parts[1]

        # Load components
        self._load_models()
        self._load_scaler()
        self._load_calibrator()

        # Feature buffer
        self.feature_buffer = None

    def _load_models(self):
        """Load ONNX models for ensemble inference."""
        try:
            import onnxruntime as ort
            self.sessions = []
            for seed in self.config['model']['ensemble_seeds']:
                model_path = os.path.join(
                    self.model_dir, '{}_{}_seed{}.onnx'.format(self.slot, 'latest', seed)
                )
                if os.path.exists(model_path):
                    session = ort.InferenceSession(model_path)
                    self.sessions.append(session)
                    logger.info("Loaded ONNX model: {}".format(model_path))

            if not self.sessions:
                logger.warning("No ONNX models found. Inference will fail.")
        except ImportError:
            logger.warning("onnxruntime not installed. Using PyTorch fallback.")
            self.sessions = []

    def _load_scaler(self):
        """Load pre-fit RobustScaler."""
        scaler_path = os.path.join(self.model_dir, '{}_scaler.pkl'.format(self.slot))
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded scaler: {}".format(scaler_path))
        else:
            self.scaler = None
            logger.warning("No scaler found. Features will not be scaled.")

    def _load_calibrator(self):
        """Load isotonic calibrator."""
        cal_path = os.path.join(self.model_dir, '{}_calibrator.pkl'.format(self.slot))
        if os.path.exists(cal_path):
            from calibration.isotonic import IsotonicCalibrator
            self.calibrator = IsotonicCalibrator().load(cal_path)
            logger.info("Loaded calibrator: {}".format(cal_path))
        else:
            self.calibrator = None
            logger.warning("No calibrator found. Using raw probabilities.")

    def tick(self) -> dict:
        """
        Run one inference tick.

        Returns:
            dict with inference results, trade decision, and logs
        """
        from data.fetcher import fetch_ohlcv
        from data.features import build_features
        from models.ensemble import ensemble_predict, run_onnx_ensemble_inference
        from strategy.filters import run_filter_cascade
        from strategy.sizing import compute_kelly_stake
        from strategy.execution import TradeExecutor, TradeOrder

        tick_start = time.time()
        result = {
            'slot': self.slot,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': 'skip',
        }

        try:
            # 1. Fetch latest data
            seq_len = self.config['model']['sequence_length']
            fetch_days = max(seq_len / 288 + 5, 10)  # enough bars + margin
            df = fetch_ohlcv(
                self.symbol, self.timeframe,
                since_days=int(fetch_days),
            )

            # 2. Build features
            features = build_features(df)
            feature_array = features.values

            # 3. Scale
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)

            # 4. Get sequence window
            if len(feature_array) < seq_len:
                result['action'] = 'skip'
                result['reason'] = 'insufficient_data'
                return result

            X = feature_array[-seq_len:]  # (seq_len, n_features)

            # 5. Ensemble inference
            if self.sessions:
                ensemble_result = run_onnx_ensemble_inference(self.sessions, X)
            else:
                result['action'] = 'skip'
                result['reason'] = 'no_models_loaded'
                return result

            # 6. Calibrate
            p_up_raw = ensemble_result.p_up
            if self.calibrator:
                p_up_cal = self.calibrator.calibrate_single(p_up_raw)
            else:
                p_up_cal = p_up_raw

            p_down_cal = 1.0 - p_up_cal
            ensemble_result.p_up = p_up_cal
            ensemble_result.p_down = p_down_cal
            ensemble_result.p_up_raw = p_up_raw

            # 7. Get current market price
            # In live: from Polymarket API
            # For now: assumed_market_price = 0.50 (documented per spec)
            p_market = self.config['backtest']['assumed_market_price']

            # 8. Filter cascade
            cascade = run_filter_cascade(
                p_up=p_up_cal,
                p_down=p_down_cal,
                p_market=p_market,
                direction=ensemble_result.direction,
                seed_disagreement=ensemble_result.seed_disagreement,
                config=self.config,
            )

            result['p_up_raw'] = p_up_raw
            result['p_up_calibrated'] = p_up_cal
            result['direction'] = ensemble_result.direction
            result['seed_disagreement'] = ensemble_result.seed_disagreement
            result['filters_passed'] = cascade.passed

            if not cascade.passed:
                result['action'] = 'skip'
                result['reason'] = cascade.rejection_reason
                result['rejection_filter'] = cascade.rejection_filter
                return result

            # 9. Kelly sizing
            p_model = p_up_cal if ensemble_result.direction == "up" else p_down_cal
            bankroll = self.config['accounts']['starting_bankroll']
            sizing = compute_kelly_stake(p_model, p_market, bankroll, self.config)

            if sizing.skip:
                result['action'] = 'skip'
                result['reason'] = sizing.skip_reason
                return result

            # 10. Execute trade
            order = TradeOrder(
                slot=self.slot,
                direction=ensemble_result.direction,
                entry_price=p_market,
                stake_usdc=sizing.stake_usdc,
                p_model=p_model,
                p_market=p_market,
                edge=p_model - p_market,
                kelly_fraction=sizing.kelly_capped,
                seed_disagreement=ensemble_result.seed_disagreement,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            executor = TradeExecutor(self.config)
            trade_result = executor.execute_trade(order)

            result['action'] = 'trade'
            result['trade'] = {
                'direction': order.direction,
                'stake': order.stake_usdc,
                'entry_price': order.entry_price,
                'kelly': sizing.kelly_capped,
                'edge': order.edge,
                'filled': trade_result.filled,
                'status': trade_result.status,
            }

        except Exception as e:
            logger.error("Inference tick error: {}".format(e), exc_info=True)
            result['action'] = 'error'
            result['error'] = str(e)

        result['latency_ms'] = round((time.time() - tick_start) * 1000, 1)
        return result

    def run_loop(self):
        """
        Run inference loop. Triggered by cron/systemd timer.
        For Phase 7 paper trading, runs once per invocation.
        """
        logger.info("Running inference tick for {}".format(self.slot))
        result = self.tick()
        logger.info("Result: {}".format(json.dumps(result, indent=2)))
        return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    slot = sys.argv[1] if len(sys.argv) > 1 else 'BTC_5m'
    scheduler = InferenceScheduler(slot=slot)
    scheduler.run_loop()
