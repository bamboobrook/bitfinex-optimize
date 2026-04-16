import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import json
import os
import sys
import tempfile
from loguru import logger
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
warnings.simplefilter(action='ignore', category=FutureWarning)

# 添加父目录到 path 以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ml_engine.data_processor import DataProcessor
from ml_engine.c3_combo_optimizer import (
    RateCandidate,
    build_anchor_snapshot,
    choose_combo_beam,
    generate_rate_candidates,
)
from ml_engine.execution_features import get_period_window_profile
from ml_engine.system_policy import load_system_policy, get_step_cap_pct, get_policy_version

class EnsemblePredictor:
    """使用集成模型的增强预测器 - 支持GPU加速和并行化"""

    # 类常量配置
    COLD_START_THRESHOLD = 10  # 冷启动数据点阈值
    STALE_DATA_THRESHOLD_HOURS = 2  # 陈旧数据阈值(小时)

    def __init__(self, model_dir=None, max_workers=None):
        # Use absolute path for models directory
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            model_dir = os.path.join(base_dir, "data", "models")
        self.model_dir = model_dir
        cpu_count = os.cpu_count() or 8
        default_workers = max(4, min(cpu_count, 24))
        self.max_workers = int(os.getenv("PREDICT_MAX_WORKERS", str(max_workers or default_workers)))
        self.infer_threads = int(os.getenv("PREDICT_INFER_THREADS", str(min(cpu_count, 24))))
        self.models = {}  # {curr: {model_type: {algo: model}}}
        self.meta_info = {}  # {curr: {model_type: {weights, features, task_type}}}
        self.processor = DataProcessor()
        self._timestamp_deprecation_warned = False  # 废弃警告标志
        self.policy = load_system_policy()
        self.policy_version = get_policy_version(self.policy)
        self.db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "data", "lending_history.db"
        )
        self.refresh_probe_state_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "data", "refresh_probe_state.json"
        )
        self._stale_issues = []
        self._funding_book_cache = {}
        logger.info(
            f"Predictor parallel config: max_workers={self.max_workers}, "
            f"infer_threads={self.infer_threads}"
        )

        # 导入OrderManager用于创建虚拟订单
        try:
            from ml_engine.order_manager import OrderManager
            self.order_manager = OrderManager()
        except ImportError:
            print("Warning: OrderManager not available, virtual order creation disabled")
            self.order_manager = None

        # 加载所有模型
        self.load_all_models()

    def load_ensemble_models(self, currency: str, prefix: str):
        """
        加载单个集成模型的所有组件

        Args:
            currency: 币种 (fUSD, fUST)
            prefix: 模型前缀 (model_execution_prob, model_conservative, etc.)
        """
        models = {}

        # 加载XGBoost
        xgb_path = os.path.join(self.model_dir, f"{currency}_{prefix}_xgb.json")
        if os.path.exists(xgb_path):
            models['xgb'] = xgb.Booster()
            models['xgb'].load_model(xgb_path)
            # Avoid mutating shared booster params during threaded inference.
            # Configure once at load time to reduce runtime races.
            try:
                models['xgb'].set_param({'nthread': self.infer_threads})
            except Exception as e:
                logger.warning(f"Unable to set xgb nthread for {currency}_{prefix}: {e}")

        # 加载LightGBM
        lgb_path = os.path.join(self.model_dir, f"{currency}_{prefix}_lgb.txt")
        if os.path.exists(lgb_path):
            models['lgb'] = lgb.Booster(model_file=lgb_path)

        # 加载CatBoost
        cat_path = os.path.join(self.model_dir, f"{currency}_{prefix}_cat.cbm")
        meta_path = os.path.join(self.model_dir, f"{currency}_{prefix}_meta.json")

        if os.path.exists(cat_path) and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                try:
                    meta = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted metadata JSON for {currency}_{prefix}: {e}")
                    return models, None

            # 验证必需字段
            required_fields = ['task_type', 'weights', 'feature_cols']
            missing_fields = [f for f in required_fields if f not in meta]
            if missing_fields:
                logger.error(
                    f"Invalid metadata for {currency}_{prefix}: "
                    f"missing required fields: {missing_fields}"
                )
                return models, None

            # 验证 task_type 值
            if meta['task_type'] not in ['classification', 'regression']:
                logger.error(
                    f"Invalid task_type '{meta['task_type']}' for {currency}_{prefix}. "
                    f"Must be 'classification' or 'regression'"
                )
                return models, None

            # 根据task_type加载对应的模型类型
            if meta['task_type'] == 'classification':
                models['cat'] = CatBoostClassifier()
            else:
                models['cat'] = CatBoostRegressor()
            models['cat'].load_model(cat_path)

            return models, meta
        else:
            logger.warning(f"Meta file not found for {currency}_{prefix}")
            return models, None

    def load_all_models(self):
        """加载所有币种的所有模型 - 增强版支持新模型"""
        for curr in ['fUSD', 'fUST']:
            self.models[curr] = {}
            self.meta_info[curr] = {}

            # 传统4个模型
            for model_type in ['model_execution_prob', 'model_conservative',
                              'model_aggressive', 'model_balanced']:
                models, meta = self.load_ensemble_models(curr, model_type)

                if models and meta:
                    self.models[curr][model_type] = models
                    self.meta_info[curr][model_type] = meta
                    logger.info(f"Loaded {model_type} for {curr}")
                else:
                    logger.warning(f"Failed to load {model_type} for {curr}")

            # 尝试加载增强模型 (如果存在)
            for enhanced_model_type in ['model_execution_prob_v2', 'model_revenue_optimized']:
                models, meta = self.load_ensemble_models(curr, enhanced_model_type)

                if models and meta:
                    self.models[curr][enhanced_model_type] = models
                    self.meta_info[curr][enhanced_model_type] = meta
                    logger.info(f"✨ Loaded enhanced {enhanced_model_type} for {curr}")
                else:
                    logger.info(f"Enhanced {enhanced_model_type} not available for {curr} (using traditional models)")

    def predict_with_ensemble(self, X: pd.DataFrame, currency: str, model_type: str) -> np.ndarray:
        """
        使用集成模型进行预测

        Args:
            X: 特征数据
            currency: 币种
            model_type: 模型类型
        Returns:
            预测结果数组
        """
        if currency not in self.models or model_type not in self.models[currency]:
            logger.error(f"Model not found: {currency} - {model_type}")
            raise ValueError(f"Model not found: {currency} - {model_type}")

        models = self.models[currency][model_type]
        meta = self.meta_info[currency][model_type]
        weights = meta['weights']

        # B3 FIX: Validate weights for NaN, fallback to equal weights
        if any(np.isnan(v) for v in weights.values()):
            logger.warning(f"NaN detected in ensemble weights for {currency}-{model_type}, using equal weights")
            n_algos = len(weights)
            weights = {k: 1.0 / n_algos for k in weights}

        # 获取各模型预测
        predictions = {}

        # XGBoost预测
        if 'xgb' in models:
            dtest = xgb.DMatrix(X)
            predictions['xgb'] = models['xgb'].predict(dtest)

        # LightGBM预测
        if 'lgb' in models:
            predictions['lgb'] = models['lgb'].predict(X, num_threads=self.infer_threads)

        # CatBoost预测
        if 'cat' in models:
            if meta['task_type'] == 'classification':
                predictions['cat'] = models['cat'].predict_proba(X, thread_count=self.infer_threads)[:, 1]
            else:
                predictions['cat'] = models['cat'].predict(X, thread_count=self.infer_threads)

        # 加权集成
        ensemble_pred = np.zeros(len(X))
        for algo, pred in predictions.items():
            ensemble_pred += pred * weights.get(algo, 0.0)

        return ensemble_pred

    def get_latest_rate_from_db(self, currency: str, period: int) -> tuple:
        """Query database directly for the most recent rate"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query = """
            SELECT close_annual, datetime
            FROM funding_rates
            WHERE currency = ? AND period = ?
              AND close_annual > 0
              AND close_annual <= 50
            ORDER BY datetime DESC
            LIMIT 1
            """
            cursor.execute(query, (currency, period))
            result = cursor.fetchone()

            if result:
                return float(result[0]), result[1]
            return None, None
        finally:
            conn.close()

    def _get_previous_predicted_rate(self, currency: str, period: int) -> Optional[float]:
        """Get the latest historical predicted rate for step-cap control."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT predicted_rate
                FROM virtual_orders
                WHERE currency = ?
                  AND period = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (currency, period),
            )
            row = cursor.fetchone()
            if not row:
                return None
            if row[0] is None:
                return None
            return float(row[0])
        except Exception as e:
            logger.warning(f"Failed to read previous predicted rate for {currency}-{period}: {e}")
            return None
        finally:
            conn.close()

    def _get_latest_prediction_snapshot(self, currency: str, period: int) -> Optional[dict]:
        """Load the latest historical prediction snapshot for recommendation fallback."""
        import sqlite3

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT
                        predicted_rate,
                        execution_probability,
                        strategy,
                        confidence,
                        conservative_rate,
                        balanced_rate,
                        aggressive_rate,
                        trend_factor,
                        current_market_rate,
                        liquidity_score
                    FROM prediction_history
                    WHERE currency = ?
                      AND period = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """,
                    (currency, period),
                ).fetchone()
        except Exception as e:
            logger.warning(f"Failed to load prediction_history snapshot for {currency}-{period}d: {e}")
            return None

        if row is None:
            return None

        liquidity_score = row["liquidity_score"]
        if liquidity_score is None:
            liquidity_level = None
        elif liquidity_score >= 65.0:
            liquidity_level = "high"
        elif liquidity_score >= 40.0:
            liquidity_level = "medium"
        elif liquidity_score >= 20.0:
            liquidity_level = "low"
        else:
            liquidity_level = "insufficient"

        return {
            "currency": currency,
            "period": int(period),
            "current_rate": float(row["current_market_rate"] or 0.0),
            "predicted_rate": float(row["predicted_rate"] or 0.0),
            "execution_probability": float(row["execution_probability"] or 0.0),
            "conservative_rate": float(row["conservative_rate"] or 0.0),
            "aggressive_rate": float(row["aggressive_rate"] or 0.0),
            "balanced_rate": float(row["balanced_rate"] or 0.0),
            "trend_factor": float(row["trend_factor"] or 0.0),
            "strategy": row["strategy"] or "Unknown",
            "confidence": row["confidence"] or "Medium",
            "liquidity_score": float(liquidity_score) if liquidity_score is not None else None,
            "liquidity_level": liquidity_level,
        }

    def _get_days_since_last_execution(self, currency: str, period: int) -> Optional[int]:
        """Returns days since last EXECUTED order for (currency, period), or None if never executed."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT MAX(created_at) FROM virtual_orders "
                    "WHERE currency = ? AND period = ? AND status = 'EXECUTED'",
                    (currency, period)
                ).fetchone()
            if row and row[0]:
                last_exec_dt = datetime.strptime(row[0][:19], '%Y-%m-%d %H:%M:%S')
                return (datetime.now() - last_exec_dt).days
            return None
        except Exception:
            return None

    def _is_zero_liquidity_suspended(self, currency: str, period: int,
                                      exec_threshold: float = 0.05,
                                      min_inactive_days: int = 14) -> bool:
        """
        Returns True when this pair should be suspended from virtual order creation.
        Condition: exec_rate (fast window) < exec_threshold AND days_since_last_exec >= min_inactive_days.
        """
        days_no_exec = self._get_days_since_last_execution(currency, period)
        if days_no_exec is None or days_no_exec < min_inactive_days:
            return False
        try:
            from ml_engine.execution_features import ExecutionFeatures
            fast_days = 21 if period >= 60 else 7
            exec_rate = ExecutionFeatures().calculate_execution_rate(currency, period, fast_days)
            if exec_rate < exec_threshold:
                logger.info(
                    f"Zero-liquidity suspension: {currency}-{period} "
                    f"exec_rate_{fast_days}d={exec_rate:.3f}<{exec_threshold}, "
                    f"{days_no_exec}d since last exec"
                )
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _compute_direction_match(predicted_delta: float, market_signal: float) -> Optional[int]:
        """
        Compare prediction direction with market direction.
        Returns 1 (match), 0 (mismatch), or None when signal is too weak.
        """
        eps = 1e-8
        if abs(market_signal) <= eps:
            return None
        pred_sign = 1 if predicted_delta > eps else (-1 if predicted_delta < -eps else 0)
        market_sign = 1 if market_signal > eps else (-1 if market_signal < -eps else 0)
        if pred_sign == 0:
            return None
        return 1 if pred_sign == market_sign else 0

    def _apply_period_step_cap(
        self,
        currency: str,
        period: int,
        predicted_rate: float,
        current_rate: float = 0.0,
        ma_720: float = 0.0
    ) -> Tuple[float, Optional[float], Optional[float], bool, Optional[float]]:
        """
        Apply per-period one-step cap. Currently strict for 120d by policy.
        Bypass conditions (three-tier):
          1. Market velocity > 25% from 12h MA → full bypass
          2. predicted/current > 1.5x → full bypass
          3. predicted/current > 1.2x → cap * 3 (max 40%)

        Returns:
            (capped_rate, previous_rate, step_change_pct, step_capped, cap_pct)
        """
        cap_pct = get_step_cap_pct(self.policy, period)
        if cap_pct is None:
            return predicted_rate, None, None, False, None

        previous_rate = self._get_previous_predicted_rate(currency, period)
        if previous_rate is None or previous_rate <= 0:
            return predicted_rate, previous_rate, None, False, cap_pct

        # 动态 cap: 三层绕过机制，加速收敛
        if previous_rate > 0 and current_rate > 0:
            divergence_ratio = predicted_rate / current_rate
            # 市场速度信号: current_rate 偏离 12h MA 程度
            velocity = abs(current_rate - ma_720) / (ma_720 + 1e-8) if ma_720 > 0 else 0.0

            if velocity > 0.25:
                # 市场已快速移动（偏离12h均值25%+），完全绕过限制
                effective_cap = 1.0
                logger.info(
                    f"Dynamic cap bypass (velocity) for {currency}-{period}: "
                    f"velocity={velocity:.2f}>0.25, cap released"
                )
            elif divergence_ratio > 1.5:
                # 预测偏离市场 1.5x 以上，完全绕过
                effective_cap = 1.0
                logger.info(
                    f"Dynamic cap bypass for {currency}-{period}: "
                    f"divergence_ratio={divergence_ratio:.2f}>1.5, cap released"
                )
            elif divergence_ratio > 1.2:
                # 中度偏离: cap 翻3倍（最高40%）
                effective_cap = min(cap_pct * 3.0, 0.40)
            else:
                effective_cap = cap_pct
        else:
            effective_cap = cap_pct

        # 零成交加速收敛: 长期无成交时放宽 step_cap，加速价格降至市场水平
        # 注：即使 pair 被 zero-liquidity suspended（不下单），仍允许 predicted_rate 快速追踪市场
        days_no_exec = self._get_days_since_last_execution(currency, period)
        if days_no_exec is not None:
            is_suspended = self._is_zero_liquidity_suspended(currency, period)
            suspend_tag = " [SUSPENDED—rate tracking only]" if is_suspended else ""
            if days_no_exec >= 20:
                effective_cap = max(effective_cap, 1.0)
                logger.info(
                    f"Zero-exec cap bypass {currency}-{period}: "
                    f"{days_no_exec}d since last exec (>=20d), cap fully released{suspend_tag}"
                )
            elif days_no_exec >= 10:
                effective_cap = max(effective_cap, cap_pct * 2.0)
                logger.info(
                    f"Zero-exec cap relaxed {currency}-{period}: "
                    f"{days_no_exec}d since last exec (>=10d), cap x2={effective_cap:.3f}{suspend_tag}"
                )

        lower = previous_rate * (1.0 - effective_cap)
        upper = previous_rate * (1.0 + effective_cap)
        capped_rate = float(np.clip(predicted_rate, lower, upper))
        step_capped = abs(capped_rate - predicted_rate) > 1e-12
        step_change_pct = (capped_rate - previous_rate) / previous_rate
        return capped_rate, previous_rate, float(step_change_pct), step_capped, cap_pct

    def _policy_value(self, section: str, key: str, default):
        return self.policy.get(section, {}).get(key, default)

    def _freshness_thresholds_minutes(self, currency: str = "", period: int = 0) -> Tuple[float, float]:
        warn_minutes = float(self._policy_value("automation", "stale_data_warn_minutes", 60))
        hard_minutes = float(self._policy_value("automation", "stale_data_hard_minutes", 120))
        # fUST has naturally sparse market data (often 4-24h gaps) - use a per-currency override
        if currency:
            per_cur_key = f"stale_data_hard_minutes_{currency}"
            per_cur_hard = self._policy_value("automation", per_cur_key, None)
            if per_cur_hard is not None:
                hard_minutes = float(per_cur_hard)
        # Period-tier override: long periods have sparser data on Bitfinex
        if period > 0:
            tier_config = self._policy_value("automation", "stale_data_hard_minutes_by_period_tier", None)
            if tier_config:
                if period >= 30:
                    tier_hard = tier_config.get("long")
                elif period >= 6:
                    tier_hard = tier_config.get("medium")
                else:
                    tier_hard = tier_config.get("short")
                if tier_hard is not None:
                    hard_minutes = max(hard_minutes, float(tier_hard))
        if hard_minutes <= warn_minutes:
            hard_minutes = warn_minutes + 30.0
        return warn_minutes, hard_minutes

    def _record_stale_issue(self, currency: str, period: int, age_minutes: float, source_ts: str):
        self._stale_issues.append({
            "currency": currency,
            "period": int(period),
            "age_minutes": float(age_minutes),
            "source_timestamp": source_ts,
        })

    def _load_refresh_probe_state(self) -> dict:
        if not os.path.exists(self.refresh_probe_state_path):
            return {"counters": {}, "updated_at": None}
        try:
            with open(self.refresh_probe_state_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return {"counters": {}, "updated_at": None}
            loaded.setdefault("counters", {})
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load refresh probe state: {e}")
            return {"counters": {}, "updated_at": None}

    def _save_refresh_probe_state(self, state: dict):
        try:
            state["updated_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._atomic_write_json(self.refresh_probe_state_path, state, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save refresh probe state: {e}")

    def _json_safe_value(self, value):
        if isinstance(value, np.generic):
            return self._json_safe_value(value.item())
        if isinstance(value, dict):
            return {str(key): self._json_safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_value(item) for item in value]
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return value
        if value is None or isinstance(value, (str, int, bool)):
            return value
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        return str(value)

    def _atomic_write_json(self, path: str, payload: dict, indent: int = 4):
        safe_payload = self._json_safe_value(payload)
        target_dir = os.path.dirname(path) or "."
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-json-", suffix=".json", dir=target_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(safe_payload, f, indent=indent, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _emit_json_result(self, output_path: str, payload: dict, indent: int = 4):
        self._atomic_write_json(output_path, payload, indent=indent)
        print(json.dumps(self._json_safe_value(payload), indent=2, allow_nan=False))

    @staticmethod
    def _get_period_tier(period: int) -> str:
        if period >= 60:
            return "long"
        if period >= 14:
            return "medium"
        return "short"

    def _get_probability_calibration_profile(self, currency: str, period: int) -> dict:
        tier = self._get_period_tier(period)
        profile = {
            "tier": tier,
            "prior": 0.45,
            "sample_scale": 80.0,
            "max_evidence_weight": 0.78,
            "fast_weight": 0.55,
            "slow_weight": 0.25,
            "prior_weight": 0.20,
            "gap_penalty_start": 0.60,
            "gap_penalty_factor": 0.10,
            "gap_penalty_cap": 0.18,
            "collapse_exec_floor": 0.16,
            "collapse_min_orders": 8,
            "collapse_prob_cap": 0.42,
            "days_no_exec_threshold": 14,
            "days_no_exec_prob_cap": 0.35,
            "divergence_penalty_start": 1.20,
            "divergence_penalty_factor": 0.10,
            "divergence_penalty_cap": 0.22,
        }

        if currency == "fUSD":
            if tier == "short":
                profile.update({
                    "prior": 0.42,
                    "sample_scale": 55.0,
                    "max_evidence_weight": 0.82,
                    "fast_weight": 0.60,
                    "slow_weight": 0.25,
                    "prior_weight": 0.15,
                    "gap_penalty_factor": 0.14,
                    "gap_penalty_cap": 0.22,
                    "collapse_exec_floor": 0.18,
                    "collapse_prob_cap": 0.45,
                    "days_no_exec_prob_cap": 0.38,
                })
            elif tier == "medium":
                profile.update({
                    "prior": 0.36,
                    "sample_scale": 42.0,
                    "max_evidence_weight": 0.84,
                    "fast_weight": 0.62,
                    "slow_weight": 0.26,
                    "prior_weight": 0.12,
                    "gap_penalty_start": 0.45,
                    "gap_penalty_factor": 0.18,
                    "gap_penalty_cap": 0.28,
                    "collapse_exec_floor": 0.16,
                    "collapse_prob_cap": 0.36,
                    "days_no_exec_threshold": 10,
                    "days_no_exec_prob_cap": 0.30,
                })
            else:
                profile.update({
                    "prior": 0.30,
                    "sample_scale": 24.0,
                    "max_evidence_weight": 0.88,
                    "fast_weight": 0.62,
                    "slow_weight": 0.28,
                    "prior_weight": 0.10,
                    "gap_penalty_start": 0.35,
                    "gap_penalty_factor": 0.24,
                    "gap_penalty_cap": 0.38,
                    "collapse_exec_floor": 0.20,
                    "collapse_min_orders": 6,
                    "collapse_prob_cap": 0.28,
                    "days_no_exec_threshold": 7,
                    "days_no_exec_prob_cap": 0.22,
                    "divergence_penalty_start": 0.80,
                    "divergence_penalty_factor": 0.14,
                    "divergence_penalty_cap": 0.30,
                })
        else:
            if tier == "short":
                profile.update({
                    "prior": 0.50,
                    "sample_scale": 48.0,
                    "max_evidence_weight": 0.80,
                    "fast_weight": 0.58,
                    "slow_weight": 0.22,
                    "prior_weight": 0.20,
                    "collapse_exec_floor": 0.18,
                    "collapse_prob_cap": 0.48,
                })
            elif tier == "medium":
                profile.update({
                    "prior": 0.44,
                    "sample_scale": 40.0,
                    "max_evidence_weight": 0.82,
                    "fast_weight": 0.56,
                    "slow_weight": 0.24,
                    "prior_weight": 0.20,
                    "gap_penalty_start": 0.55,
                    "gap_penalty_factor": 0.12,
                    "gap_penalty_cap": 0.20,
                    "collapse_exec_floor": 0.14,
                    "collapse_prob_cap": 0.42,
                })
            else:
                profile.update({
                    "prior": 0.36,
                    "sample_scale": 28.0,
                    "max_evidence_weight": 0.84,
                    "fast_weight": 0.58,
                    "slow_weight": 0.27,
                    "prior_weight": 0.15,
                    "gap_penalty_start": 0.45,
                    "gap_penalty_factor": 0.16,
                    "gap_penalty_cap": 0.24,
                    "collapse_exec_floor": 0.12,
                    "collapse_min_orders": 5,
                    "collapse_prob_cap": 0.34,
                    "days_no_exec_threshold": 21,
                    "days_no_exec_prob_cap": 0.28,
                })

        total = profile["fast_weight"] + profile["slow_weight"] + profile["prior_weight"]
        profile["fast_weight"] /= total
        profile["slow_weight"] /= total
        profile["prior_weight"] /= total
        return profile

    def _calibrate_execution_probability(
        self,
        currency: str,
        period: int,
        model_prob: float,
        exec_rate_fast: float,
        exec_rate_slow: float,
        avg_rate_gap: float,
        order_count: int,
        current_rate: float,
    ) -> tuple[float, dict]:
        """
        分币种/分周期执行概率校准。
        fUSD 中长周期更快信任真实执行反馈，避免 60d/120d 在执行率塌陷时仍给出高概率。
        """
        profile = self._get_probability_calibration_profile(currency, period)
        historical_signal = (
            exec_rate_fast * profile["fast_weight"] +
            exec_rate_slow * profile["slow_weight"] +
            profile["prior"] * profile["prior_weight"]
        )
        evidence_weight = min(order_count / profile["sample_scale"], profile["max_evidence_weight"])
        calibrated_prob = (1.0 - evidence_weight) * model_prob + evidence_weight * historical_signal

        gap_ratio = 0.0
        if avg_rate_gap > 0 and current_rate > 0:
            gap_ratio = avg_rate_gap / (current_rate + 1e-8)
            if gap_ratio > profile["gap_penalty_start"]:
                penalty = min(
                    (gap_ratio - profile["gap_penalty_start"]) * profile["gap_penalty_factor"],
                    profile["gap_penalty_cap"]
                )
                calibrated_prob *= (1.0 - penalty)

        if order_count >= profile["collapse_min_orders"] and exec_rate_fast <= profile["collapse_exec_floor"]:
            collapse_cap = max(
                exec_rate_fast + 0.08,
                min(profile["collapse_prob_cap"], exec_rate_slow + 0.06)
            )
            calibrated_prob = min(calibrated_prob, collapse_cap)

        days_no_exec = self._get_days_since_last_execution(currency, period)
        if days_no_exec is not None and days_no_exec >= profile["days_no_exec_threshold"]:
            calibrated_prob = min(calibrated_prob, profile["days_no_exec_prob_cap"])

        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))
        return calibrated_prob, {
            "tier": profile["tier"],
            "historical_signal": float(historical_signal),
            "evidence_weight": float(evidence_weight),
            "gap_ratio": float(gap_ratio),
            "days_no_exec": days_no_exec,
        }

    def _apply_probability_divergence_guard(
        self,
        calibrated_prob: float,
        currency: str,
        period: int,
        final_rate: float,
        current_rate: float,
        exec_rate_fast: float,
        avg_rate_gap: float,
    ) -> tuple[float, dict]:
        """
        最终利率出炉后，再按 market divergence 做一次校准。
        这样推荐排序看到的概率会更贴近“当前这口价”是否可能成交。
        """
        profile = self._get_probability_calibration_profile(currency, period)
        relative_err = 0.0
        if current_rate > 0 and final_rate > current_rate:
            relative_err = (final_rate - current_rate) / (current_rate + 1e-8)
            if relative_err > profile["divergence_penalty_start"]:
                penalty = min(
                    (relative_err - profile["divergence_penalty_start"]) * profile["divergence_penalty_factor"],
                    profile["divergence_penalty_cap"]
                )
                calibrated_prob *= (1.0 - penalty)

        if currency == "fUSD" and period >= 60 and exec_rate_fast < 0.20 and avg_rate_gap > 0:
            calibrated_prob = min(calibrated_prob, max(0.12, exec_rate_fast + 0.06))

        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))
        return calibrated_prob, {
            "relative_err": float(relative_err),
        }

    def predict_single_period(self, row_data: dict, feature_cols: list, currency: str) -> dict:
        """
        对单个period的数据进行预测（用于并行化）- 增强版含执行反馈调整

        Args:
            row_data: 单行数据字典
            feature_cols: 特征列列表
            currency: 币种
        Returns:
            预测结果字典
        """
        # Query database for current rate (get latest data)
        period = int(row_data['period'])
        current_rate_db, data_timestamp = self.get_latest_rate_from_db(currency, period)

        warn_minutes, hard_minutes = self._freshness_thresholds_minutes(currency, period)

        # Use DB rate if available, fallback to feature data
        if current_rate_db is not None:
            if isinstance(data_timestamp, str):
                db_dt = datetime.strptime(data_timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                db_dt = data_timestamp

            data_age = datetime.now() - db_dt
            data_age_minutes = data_age.total_seconds() / 60.0

            if data_age_minutes > hard_minutes:
                logger.error(
                    f"STALE DATA for {currency}-{period}: "
                    f"data is {data_age_minutes:.1f}m old (threshold: {hard_minutes:.1f}m). "
                    f"Refusing to predict."
                )
                self._record_stale_issue(currency, period, data_age_minutes, str(data_timestamp))
                raise ValueError(f"Stale DB data for {currency}-{period}: {data_age_minutes:.1f}m old")
            elif data_age_minutes > warn_minutes:
                logger.warning(
                    f"AGING DATA for {currency}-{period}: "
                    f"data is {data_age_minutes:.1f}m old (warn threshold: {warn_minutes:.1f}m). "
                    f"Confidence will be degraded."
                )

            current_rate = current_rate_db
            actual_timestamp = data_timestamp
            logger.debug(f"Using DB rate for {currency}-{period}: {current_rate} (age: {data_age})")
        else:
            # 验证特征数据时间戳新鲜度
            feature_datetime = row_data.get('datetime')
            if feature_datetime:
                if isinstance(feature_datetime, str):
                    feature_dt = datetime.strptime(feature_datetime, '%Y-%m-%d %H:%M:%S')
                else:
                    feature_dt = feature_datetime

                data_age = datetime.now() - feature_dt
                data_age_minutes = data_age.total_seconds() / 60.0

                if data_age_minutes > hard_minutes:
                    logger.error(
                        f"STALE DATA for {currency}-{period}: "
                        f"DB query failed AND feature data is {data_age_minutes:.1f}m old. "
                        f"Refusing to predict."
                    )
                    self._record_stale_issue(currency, period, data_age_minutes, str(feature_datetime))
                    raise ValueError(f"Stale data for {currency}-{period}: {data_age_minutes:.1f}m old")
                elif data_age_minutes > warn_minutes:
                    logger.warning(
                        f"AGING feature data for {currency}-{period}: {data_age_minutes:.1f}m old"
                    )

                actual_timestamp = feature_datetime
                current_rate = float(row_data['close_annual'])
                logger.warning(
                    f"DB query failed for {currency}-{period}, using feature data from {feature_datetime} "
                    f"(age: {data_age})"
                )
            else:
                logger.error(f"No datetime field in feature data for {currency}-{period}")
                raise ValueError(f"Missing datetime for {currency}-{period}")

        # For confidence degradation logic below
        data_age_hours = data_age_minutes / 60.0
        stale_warn_hours = warn_minutes / 60.0

        # 构造特征DataFrame
        X_single = pd.DataFrame([{col: row_data[col] for col in feature_cols}])

        # 执行4个传统模型预测
        pred_execution_prob = self.predict_with_ensemble(X_single, currency, 'model_execution_prob')[0]
        pred_conservative = self.predict_with_ensemble(X_single, currency, 'model_conservative')[0]
        pred_aggressive = self.predict_with_ensemble(X_single, currency, 'model_aggressive')[0]
        pred_balanced = self.predict_with_ensemble(X_single, currency, 'model_balanced')[0]

        # S1 FIX: 启用 v2 模型参与预测 (如果可用)
        v2_execution_prob = None
        v2_revenue_rate = None

        if currency in self.models:
            # 尝试使用 execution_prob_v2 (基于实际执行结果训练)
            if 'model_execution_prob_v2' in self.models[currency]:
                try:
                    v2_meta = self.meta_info[currency]['model_execution_prob_v2']
                    v2_feature_cols = v2_meta['feature_cols']
                    # v2 模型可能有不同的特征集,需要检查可用性
                    available_cols = [c for c in v2_feature_cols if c in row_data]
                    if len(available_cols) == len(v2_feature_cols):
                        X_v2 = pd.DataFrame([{col: row_data[col] for col in v2_feature_cols}])
                        v2_execution_prob = float(self.predict_with_ensemble(
                            X_v2, currency, 'model_execution_prob_v2'
                        )[0])
                        logger.info(f"v2 execution_prob for {currency}-{period}: {v2_execution_prob:.4f}")
                    else:
                        missing = set(v2_feature_cols) - set(available_cols)
                        logger.info(f"v2 execution_prob skipped for {currency}-{period}: missing features {missing}")
                except Exception as e:
                    logger.warning(f"v2 execution_prob prediction failed for {currency}-{period}: {e}")

            # 尝试使用 revenue_optimized 模型
            if 'model_revenue_optimized' in self.models[currency]:
                try:
                    rev_meta = self.meta_info[currency]['model_revenue_optimized']
                    rev_feature_cols = rev_meta['feature_cols']
                    available_cols = [c for c in rev_feature_cols if c in row_data]
                    if len(available_cols) == len(rev_feature_cols):
                        X_rev = pd.DataFrame([{col: row_data[col] for col in rev_feature_cols}])
                        v2_revenue_rate = float(self.predict_with_ensemble(
                            X_rev, currency, 'model_revenue_optimized'
                        )[0])
                        logger.info(f"v2 revenue_optimized for {currency}-{period}: {v2_revenue_rate:.4f}")
                    else:
                        missing = set(rev_feature_cols) - set(available_cols)
                        logger.info(f"v2 revenue skipped for {currency}-{period}: missing features {missing}")
                except Exception as e:
                    logger.warning(f"v2 revenue prediction failed for {currency}-{period}: {e}")

        # 智能定价策略
        prob = float(pred_execution_prob)
        p_cons = float(pred_conservative)
        p_aggr = float(pred_aggressive)
        p_bal = float(pred_balanced)

        # S1: 融合 v2 执行概率 (如果可用)
        # execution_prob 最终值 = 0.6 × 传统 + 0.4 × v2
        if v2_execution_prob is not None:
            blended_prob = 0.6 * prob + 0.4 * v2_execution_prob
            logger.info(f"v2 blend for {currency}-{period}: traditional={prob:.4f}, v2={v2_execution_prob:.4f}, blended={blended_prob:.4f}")
            prob = blended_prob

        # ========== 校准执行概率 (Bayesian-style Calibration) ==========
        # 根据周期分层窗口动态加权模型概率和历史执行率
        profile = get_period_window_profile(period)
        fast_days = profile["fast_days"]
        slow_days = profile["slow_days"]
        gap_days = profile["gap_days"]

        try:
            from ml_engine.execution_features import ExecutionFeatures
            _exec_calc = ExecutionFeatures()
            exec_rate_fast = _exec_calc.calculate_execution_rate(currency, period, fast_days)
            exec_rate_slow = _exec_calc.calculate_execution_rate(currency, period, slow_days)
            avg_rate_gap = _exec_calc.calculate_avg_rate_gap(currency, period, gap_days)
            order_count = _exec_calc.get_order_count(currency, period)
            logger.debug(
                f"Live execution stats for {currency}-{period}: "
                f"exec_rate_fast={exec_rate_fast:.4f}({fast_days}d), "
                f"exec_rate_slow={exec_rate_slow:.4f}({slow_days}d), "
                f"avg_gap={avg_rate_gap:.4f}({gap_days}d), "
                f"orders={order_count}"
            )
        except Exception as e:
            logger.warning(f"Failed to get live execution stats for {currency}-{period}: {e}, using feature data")
            exec_rate_fast = row_data.get('exec_rate_fast', row_data.get('exec_rate_7d', 0.6))
            exec_rate_slow = row_data.get('exec_rate_slow', row_data.get('exec_rate_30d', 0.6))
            avg_rate_gap = row_data.get('avg_rate_gap_failed_profile', row_data.get('avg_rate_gap_failed_7d', 0.0))
            order_count = 0

        # Backward-compatible aliases for existing logs/outputs
        exec_rate_7d = exec_rate_fast
        exec_rate_30d = exec_rate_slow

        calibrated_prob, calibration_meta = self._calibrate_execution_probability(
            currency=currency,
            period=period,
            model_prob=prob,
            exec_rate_fast=exec_rate_fast,
            exec_rate_slow=exec_rate_slow,
            avg_rate_gap=avg_rate_gap,
            order_count=order_count,
            current_rate=current_rate,
        )

        # 诊断日志: 记录关键决策变量
        logger.debug(
            f"Strategy decision for {currency}-{period}: "
            f"prob={prob:.4f}, exec_rate_fast={exec_rate_fast:.4f}, "
            f"calibrated_prob={calibrated_prob:.4f}, "
            f"hist_signal={calibration_meta['historical_signal']:.4f}, "
            f"evidence={calibration_meta['evidence_weight']:.4f}"
        )

        # 连续插值策略选择 (消除硬阈值跳变)
        # w_cons: calibrated_prob < 0.45 时为1, > 0.45 时线性降至0 (在0.10宽度内)
        # w_aggr: calibrated_prob > 0.55 时开始上升, 0.90 时为1
        # w_bal: 填补中间区域
        w_cons = float(np.clip((0.45 - calibrated_prob) / 0.35, 0, 1))
        w_aggr = float(np.clip((calibrated_prob - 0.55) / 0.35, 0, 1))
        w_bal = 1.0 - w_cons - w_aggr
        base_rate = w_cons * p_cons + w_bal * p_bal + w_aggr * p_aggr

        # 策略标签和置信度
        if w_cons > 0.5:
            strategy_desc = "Conservative-leaning"
            confidence = "Low"
        elif w_aggr > 0.5:
            strategy_desc = "Aggressive-leaning"
            confidence = "High"
        else:
            strategy_desc = "Balanced"
            confidence = "Medium"

        # 修改2.4: Confidence degradation based on data age
        if data_age_hours > stale_warn_hours:
            # Degrade confidence by one level
            confidence_map = {"High": "Medium", "Medium": "Low", "Low": "Low"}
            old_confidence = confidence
            confidence = confidence_map.get(confidence, confidence)
            if old_confidence != confidence:
                logger.info(f"Confidence degraded for {currency}-{period}: {old_confidence} -> {confidence} (data age: {data_age_hours:.1f}h)")

        # ========== 执行反馈调整 (Unified Adjustment) ==========
        # exec_rate_fast, exec_rate_slow, avg_rate_gap 已在上方从DB实时获取
        # 统一调整因子: 消除原来三因子(execution_adj × risk_adj × currency_factor)的重复计算

        adjustment = self._get_unified_adjustment(
            exec_rate_fast, exec_rate_slow, avg_rate_gap, base_rate, period, currency,
            current_rate=current_rate
        )

        # 应用调整
        model_rate = base_rate * adjustment
        # 市场锚定: 基于 confidence 动态权重,不再对 current_rate 乘调整系数
        conf_weight = {"High": 0.65, "Medium": 0.50, "Low": 0.35}.get(confidence, 0.50)

        # 当调整因子偏离时,渐进增加模型权重,避免市场锚定抵消纠偏
        # v5.2: 门槛从1.15/0.85降到1.08/0.92,渐进增强而非突变
        abs_dev = abs(adjustment - 1.0)
        if abs_dev > 0.08:
            extreme_boost = min(abs_dev - 0.08, 0.20)
            conf_weight = min(conf_weight + extreme_boost, 0.85)

        # 分层锚定: 长周期偏稳健, 短中周期偏敏感
        anchor_minutes = profile["anchor_minutes"]
        if anchor_minutes >= 10080:
            anchor_rate = float(row_data.get('robust_ma_10080', row_data.get('ma_10080', current_rate)))
        elif anchor_minutes >= 1440:
            anchor_rate = float(row_data.get('robust_ma_1440', row_data.get('ma_1440', current_rate)))
        elif anchor_minutes >= 720:
            anchor_rate = float(row_data.get('robust_ma_720', row_data.get('ma_720', current_rate)))
        else:
            anchor_rate = current_rate

        if period >= 60:
            conf_weight = min(conf_weight + 0.10, 0.90)  # 长周期增加模型权重

        # 锚点覆盖保护: 调整因子明确下调(adj<0.90)而锚点仍远高于当前市场时，
        # 将锚点上限设为 max(model_rate×1.1, current_rate×1.5)，
        # 防止历史高位 MA 逆向拉升、抵消执行率反馈的纠偏效果
        if adjustment < 0.90 and anchor_rate > current_rate * 2.0 and current_rate > 0:
            capped_anchor = min(anchor_rate, max(model_rate * 1.1, current_rate * 1.5))
            logger.debug(
                f"Anchor capped {currency}-{period}: {anchor_rate:.4f}->{capped_anchor:.4f} "
                f"(adj={adjustment:.3f} current={current_rate:.4f})"
            )
            anchor_rate = capped_anchor

        # Fix2: 市场快速变化时动态增加模型权重，减弱历史 MA 锚点的滞后拉力
        if current_rate > 0 and anchor_rate > 0:
            anchor_deviation = abs(current_rate - anchor_rate) / anchor_rate
            if anchor_deviation > 0.40:
                # 市场已偏离锚点 40%+，强烈信任模型而非历史锚点
                conf_weight = min(conf_weight + 0.25, 0.90)
                logger.debug(
                    f"Anchor deviation boost {currency}-{period}: "
                    f"deviation={anchor_deviation:.2f}>0.40, conf_weight→{conf_weight:.2f}"
                )
            elif anchor_deviation > 0.20:
                # 市场偏离 20-40%，适度增加模型权重
                boost = (anchor_deviation - 0.20) / 0.20 * 0.15
                conf_weight = min(conf_weight + boost, 0.85)

        adjusted_rate = conf_weight * model_rate + (1 - conf_weight) * anchor_rate

        # S1: v2 revenue_optimized 模型已禁用混合
        # 原因: revenue_optimized 模型的训练目标是 close_annual × revenue_reward (0~2范围的奖励分数)
        # 但此处将其作为利率(5~18范围)按30%权重混入，导致所有预测被系统性拉低~30%
        # 保留日志记录用于监控，不参与计算
        if v2_revenue_rate is not None:
            logger.debug(f"v2 revenue (disabled, for monitoring only) {currency}-{period}: {v2_revenue_rate:.4f}")
        # ====================================================================

        # 趋势修正 — 按周期选择趋势窗口和权重,减少短期噪音对长周期的影响
        if period >= 90:
            trend = 0.0                                                # 90d/120d: 完全禁用趋势修正
            trend_weight = 0.0
        elif period >= 60:
            trend = float(row_data.get('rate_chg_1440', 0))            # 24h窗口
            trend_weight = 0.01                                        # 从0.03降到0.01
        elif period >= 20:
            trend = float(row_data.get('rate_chg_240', 0))   # 4h窗口
            trend_weight = 0.05
        else:
            trend = float(row_data.get('rate_chg_60', 0))    # 1h窗口
            trend_weight = 0.08
        # 趋势归一化: 按周期调整基数,长周期24h变化量级更大
        if period >= 60:
            trend_norm = 10.0
        elif period >= 20:
            trend_norm = 7.0
        else:
            trend_norm = 5.0
        trend_factor = np.clip(trend / trend_norm, -1.0, 1.0)
        trend_adjustment = trend_factor * trend_weight * adjusted_rate
        final_rate = adjusted_rate + trend_adjustment

        # B4 FIX: Check for NaN in final_rate before clipping
        if np.isnan(final_rate):
            logger.error(f"NaN detected in final_rate for {currency}-{period}, skipping prediction")
            raise ValueError(f"NaN in final_rate for {currency}-{period}")

        # 动态安全边界: 分层锚点 + 执行率感知 floor，避免长周期卡死高位
        ma_720 = float(row_data.get('robust_ma_720', row_data.get('ma_720', current_rate)))
        ma_1440 = float(row_data.get('robust_ma_1440', row_data.get('ma_1440', ma_720)))
        ma_10080 = float(row_data.get('robust_ma_10080', row_data.get('ma_10080', ma_1440)))
        if period >= 60:
            bound_base = 0.7 * ma_10080 + 0.3 * ma_1440
        elif period >= 20:
            bound_base = 0.6 * ma_1440 + 0.4 * ma_720
        else:
            bound_base = 0.55 * ma_720 + 0.45 * current_rate

        # 市场崩塌修复: exec_rate 极低时将 current_rate 混入 bound_base，
        # 防止历史高位 MA 把 floor 拉得远高于市场
        if exec_rate_fast < 0.20 and current_rate > 0:
            alpha = 1.0 - (exec_rate_fast / 0.20)        # exec_rate=0 → alpha=1.0
            blend_weight = alpha * 0.35                   # 最大混入 35% current_rate
            bound_base = (1.0 - blend_weight) * bound_base + blend_weight * (current_rate * 1.2)

        if calibrated_prob > 0.8:
            floor_base = 0.50
            max_bound = bound_base * 1.5
            strategy_label = "aggressive"
        elif calibrated_prob > 0.5:
            floor_base = 0.60
            max_bound = bound_base * 1.25
            strategy_label = "balanced"
        else:
            floor_base = 0.65
            max_bound = bound_base * 1.0
            strategy_label = "conservative"

        if period >= 60:
            exec_penalty = np.clip((0.35 - exec_rate_fast) / 0.35, 0.0, 1.0) * 0.22
            trend_penalty = 0.06 if exec_rate_fast < exec_rate_slow * 0.75 else 0.0
            floor_factor = max(0.35, floor_base - exec_penalty - trend_penalty)
        elif period >= 20:
            exec_penalty = np.clip((0.35 - exec_rate_fast) / 0.35, 0.0, 1.0) * 0.12
            floor_factor = max(0.45, floor_base - exec_penalty)
        else:
            exec_penalty = np.clip((0.35 - exec_rate_fast) / 0.35, 0.0, 1.0) * 0.08
            floor_factor = max(0.50, floor_base - exec_penalty)

        min_bound = max(bound_base * floor_factor, 0.01)

        clipped_rate = np.clip(final_rate, min_bound, max_bound)
        was_clipped = (clipped_rate != final_rate)

        if was_clipped:
            reduction_pct = abs((final_rate - clipped_rate) / final_rate * 100) if final_rate != 0 else 0
            logger.info(
                f"Rate clipped for {currency}-{period}: "
                f"{final_rate:.4f} -> {clipped_rate:.4f} "
                f"(exec_prob={prob:.2f}, bounds: {min_bound:.4f} to {max_bound:.4f}, "
                f"strategy={strategy_label}, reduction={reduction_pct:.1f}%)"
            )
        final_rate = clipped_rate

        # Enforce policy step cap for selected periods (120d by default).
        final_rate, previous_rate, step_change_pct, step_capped, step_cap_pct = self._apply_period_step_cap(
            currency, period, final_rate, current_rate=current_rate, ma_720=ma_720
        )
        if step_capped and previous_rate is not None:
            logger.info(
                f"Step cap applied for {currency}-{period}: "
                f"{clipped_rate:.4f} -> {final_rate:.4f} "
                f"(prev={previous_rate:.4f}, cap={step_cap_pct:.2%})"
            )

        # Closed-loop diagnostics: market following error and direction alignment.
        market_follow_error = float(final_rate - current_rate)
        if period >= 60:
            market_signal = float(row_data.get('rate_chg_1440', 0.0))
        elif period >= 20:
            market_signal = float(row_data.get('rate_chg_240', 0.0))
        else:
            market_signal = float(row_data.get('rate_chg_60', 0.0))
        direction_match = self._compute_direction_match(final_rate - current_rate, market_signal)

        calibrated_prob, post_guard_meta = self._apply_probability_divergence_guard(
            calibrated_prob=calibrated_prob,
            currency=currency,
            period=period,
            final_rate=final_rate,
            current_rate=current_rate,
            exec_rate_fast=exec_rate_fast,
            avg_rate_gap=avg_rate_gap,
        )

        # ========== 结构化诊断日志 ==========
        logger.info(
            f"PREDICTION_DIAG {currency}-{period}d: "
            f"current={current_rate:.4f} base={base_rate:.4f} "
            f"adj={adjustment:.4f} model_rate={model_rate:.4f} "
            f"conf_w={conf_weight:.2f} adjusted={adjusted_rate:.4f} "
            f"trend_adj={trend_adjustment:.4f} final={final_rate:.4f} "
            f"exec_fast={exec_rate_fast:.3f}({fast_days}d) exec_slow={exec_rate_slow:.3f}({slow_days}d) "
            f"calib_prob={calibrated_prob:.3f} strategy={strategy_desc} "
            f"clipped={was_clipped} step_capped={step_capped} follow_err={market_follow_error:.4f} "
            f"floor={floor_factor:.3f} bounds=[{min_bound:.4f},{max_bound:.4f}]"
        )

        volume_ratio = row_data.get('volume_ratio', 1.0)
        liq_score, liq_level = self._calc_liquidity_score(
            calibrated_prob=calibrated_prob,
            volume_ratio=volume_ratio,
            data_age_minutes=data_age_minutes,
            currency=currency
        )

        return {
            "currency": currency,
            "period": int(row_data['period']),
            "current_rate": current_rate,
            "predicted_rate": float(final_rate),
            "execution_probability": prob,
            "calibrated_execution_prob": float(calibrated_prob),  # exec_rate校准后的成交概率（反映真实流动性）
            "exec_rate_raw": float(exec_rate_fast),  # 原始历史成交率（用于流动性评分，不受模型滞后影响）
            "liquidity_score": liq_score,
            "liquidity_level": liq_level,
            "order_count": int(order_count),
            "avg_rate_gap_failed": float(avg_rate_gap),
            "calibration_tier": calibration_meta["tier"],
            "calibration_historical_signal": float(calibration_meta["historical_signal"]),
            "calibration_evidence_weight": float(calibration_meta["evidence_weight"]),
            "calibration_gap_ratio": float(calibration_meta["gap_ratio"]),
            "calibration_days_no_exec": calibration_meta["days_no_exec"],
            "probability_relative_err": float(post_guard_meta["relative_err"]),
            "conservative_rate": p_cons,
            "aggressive_rate": p_aggr,
            "balanced_rate": p_bal,
            "trend_factor": trend,
            "strategy": strategy_desc,
            "confidence": confidence,
            "data_timestamp": actual_timestamp,  # Actual timestamp of rate data
            "prediction_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # When prediction was made
            "data_age_minutes": float(data_age_minutes),
            # 新增调试信息
            "execution_rate_7d": exec_rate_7d,
            "execution_rate_slow": exec_rate_slow,
            "exec_rate_fast_window_days": fast_days,
            "exec_rate_slow_window_days": slow_days,
            "avg_gap_window_days": gap_days,
            "execution_adjustment_applied": adjustment,
            "market_follow_error": market_follow_error,
            "direction_match": direction_match,
            "step_change_pct": step_change_pct,
            "step_capped": step_capped,
            "policy_step_cap_pct": step_cap_pct,
            # v2 model info (S1)
            "v2_execution_prob": v2_execution_prob,
            "v2_revenue_rate": v2_revenue_rate,
            # Rate clipping 元数据
            "was_clipped": was_clipped,
            "clipping_strategy": strategy_label,
            "clipping_bounds": {"min": float(min_bound), "max": float(max_bound)}
        }

    def _get_period_sensitivity(self, period: int):
        """Returns (max_up, max_down) for the given period tier."""
        if period >= 60:
            return 0.30, 0.30
        elif period >= 20:
            return 0.28, 0.24
        elif period >= 8:
            return 0.32, 0.20
        else:
            return 0.18, 0.16

    def _calc_exec_rate_signal(self, exec_rate_7d: float, max_up: float, max_down: float) -> float:
        """exec_rate 偏离 0.50 目标的响应，使用平滑非线性映射。"""
        target = 0.50
        if exec_rate_7d <= 0:
            return -max_down  # 完全无成交：极限下调
        deviation = exec_rate_7d - target
        if deviation > 0:
            norm = min(deviation / target, 1.0)
            return max_up * (norm ** 0.7)  # 上调：幂函数加速响应
        else:
            norm = min(-deviation / target, 1.0)
            return -max_down * norm  # 下调：线性

    def _calc_trend_signal(self, trend: float, exec_rate_7d: float, max_up: float, max_down: float) -> float:
        """趋势微调：紧急模式（exec<30%）只允许上调方向。"""
        if exec_rate_7d < 0.30:
            # 紧急模式：已有 exec_signal 大幅下调，趋势只允许反向微修复
            return max_up * 0.03 if trend > 1.2 else 0.0
        else:
            if trend > 1.2:
                return max_up * 0.03
            elif trend < 0.8:
                return -max_down * 0.03
            return 0.0

    def _calc_liquidity_score(
        self,
        calibrated_prob: float,
        volume_ratio: float,
        data_age_minutes: float,
        currency: str
    ) -> tuple:
        """计算 (liquidity_score: float 0-100, liquidity_level: str)"""
        warn_min, hard_min = self._freshness_thresholds_minutes(currency)
        if data_age_minutes <= warn_min:
            freshness_signal = 1.0
        else:
            freshness_signal = max(0.2, 1.0 - 0.8 * min(
                (data_age_minutes - warn_min) / max(hard_min - warn_min, 1.0), 1.0
            ))
        # 成交量信号（volume_ratio 上限 3x）
        volume_signal = min(float(volume_ratio) if volume_ratio else 1.0, 3.0) / 3.0
        # 综合评分：成交率 50% + 成交量 30% + 新鲜度 20%
        score = (calibrated_prob * 0.50 + volume_signal * 0.30 + freshness_signal * 0.20) * 100.0
        score = float(np.clip(score, 0.0, 100.0))
        if score >= 65.0:
            level = "high"
        elif score >= 40.0:
            level = "medium"
        elif score >= 20.0:
            level = "low"
        else:
            level = "insufficient"
        return round(score, 1), level

    def _fetch_bitfinex_public_json(self, path: str, params: Optional[dict] = None,
                                    cache_ttl_seconds: int = 20):
        """Fetch Bitfinex public JSON with a short in-memory cache."""
        params = params or {}
        query = urllib_parse.urlencode(sorted(params.items()))
        url = f"https://api-pub.bitfinex.com{path}"
        cache_key = f"{url}?{query}" if query else url
        now_ts = datetime.now().timestamp()

        cached = self._funding_book_cache.get(cache_key)
        if cached and (now_ts - cached["ts"]) < cache_ttl_seconds:
            return cached["data"]

        if query:
            url = f"{url}?{query}"

        req = urllib_request.Request(
            url,
            headers={"User-Agent": "optimize-liquidity-check/1.0"}
        )
        try:
            with urllib_request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and len(data) >= 2 and data[0] == "error":
                logger.warning(f"Bitfinex public API returned error payload for {url}: {data}")
                return None
            self._funding_book_cache[cache_key] = {"ts": now_ts, "data": data}
            return data
        except (urllib_error.URLError, TimeoutError, ValueError) as e:
            logger.warning(f"Bitfinex public API fetch failed for {url}: {e}")
            return None

    @staticmethod
    def _annualize_rate(rate: float) -> float:
        if rate is None:
            return 0.0
        return float(rate) * 365.0 * 100.0

    @staticmethod
    def _clip_unit(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _calc_fillability_signal(self, bid_levels: list) -> float:
        """
        Evaluate whether medium-size non-2d funding offers can be filled quickly.
        Targets follow the user's current operational range: 10k / 50k / 100k.
        """
        if not bid_levels:
            return 0.0

        sorted_levels = sorted(bid_levels, key=lambda x: x["rate"], reverse=True)
        total_depth = sum(level["amount"] for level in sorted_levels)

        def target_score(target: float) -> float:
            if total_depth <= 0:
                return 0.0

            cumulative = 0.0
            levels_needed = len(sorted_levels)
            for idx, level in enumerate(sorted_levels, start=1):
                cumulative += level["amount"]
                if cumulative >= target:
                    levels_needed = idx
                    break

            coverage = self._clip_unit(cumulative / target)
            # Require progressively fewer levels to earn a high score.
            level_efficiency = self._clip_unit(1.0 - min(max(levels_needed - 1, 0), 10) / 12.0)
            return coverage * level_efficiency

        signal = (
            target_score(10_000.0) * 0.25 +
            target_score(50_000.0) * 0.35 +
            target_score(100_000.0) * 0.40
        )
        return self._clip_unit(signal)

    def _calc_book_structure_factor(self, bid_levels: list) -> float:
        """Penalize books whose non-2d depth is overly concentrated in one period."""
        if not bid_levels:
            return 0.55

        period_amounts = {}
        total_depth = 0.0
        for level in bid_levels:
            period = int(level["period"])
            amount = float(level["amount"])
            total_depth += amount
            period_amounts[period] = period_amounts.get(period, 0.0) + amount

        if total_depth <= 0:
            return 0.55

        max_share = max(period_amounts.values()) / total_depth
        distinct_ratio = self._clip_unit(len(period_amounts) / 8.0)
        concentration_signal = self._clip_unit(1.0 - max(0.0, max_share - 0.55) / 0.40)

        # Keep some book value, but suppress single-period pileups.
        return float(np.clip(
            0.55 + distinct_ratio * 0.25 + concentration_signal * 0.20,
            0.55,
            1.0
        ))

    def _get_realtime_non2d_liquidity_signal(self, currency: str) -> dict:
        """
        Use realtime Bitfinex funding book to judge whether non-2d orders can be
        executed now. Internal-only helper; no extra output fields are exposed.
        """
        book = self._fetch_bitfinex_public_json(
            f"/v2/book/{currency}/P0",
            params={"len": 100},
            cache_ttl_seconds=20
        )
        if not isinstance(book, list):
            return {
                "available": False,
                "fillability_signal": 0.0,
                "depth_signal": 0.0,
            }

        bid_levels = []
        for row in book:
            if not isinstance(row, list) or len(row) < 4:
                continue
            try:
                rate = float(row[0])
                period = int(row[1])
                amount = float(row[3])
            except (TypeError, ValueError):
                continue

            # Funding book: amount < 0 means bid. Ignore 2d to avoid short-term masking.
            if amount >= 0 or period <= 2:
                continue
            bid_levels.append({
                "rate": self._annualize_rate(rate),
                "period": period,
                "amount": abs(amount),
            })

        if not bid_levels:
            return {
                "available": True,
                "fillability_signal": 0.0,
                "depth_signal": 0.0,
            }

        total_depth = sum(level["amount"] for level in bid_levels)
        fillability_signal = self._calc_fillability_signal(bid_levels)
        # Use a tighter scaling so sub-million depth no longer looks close to max.
        depth_signal = self._clip_unit(
            np.log1p(total_depth / 100_000.0) / np.log1p(50.0)
        )
        structure_factor = self._calc_book_structure_factor(bid_levels)

        return {
            "available": True,
            "fillability_signal": fillability_signal,
            "depth_signal": float(depth_signal),
            "structure_factor": structure_factor,
        }

    def _get_unified_adjustment(self, exec_rate_7d: float, exec_rate_30d: float,
                                avg_gap: float, base_rate: float,
                                period: int, currency: str,
                                current_rate: float = 0.0) -> float:
        """
        统一调整因子 — 三信号加法叠加，消除串联乘法的耦合振荡

        信号1: exec_rate 偏离目标的非线性响应
        信号2: 利率差距反馈（带证据权重，防死锁）
        信号3: 趋势动量微调

        Returns:
            调整系数，已 clip 到 [1-max_down, 1+max_up]
        """
        max_up, max_down = self._get_period_sensitivity(period)

        # 信号1: exec_rate 偏离目标
        exec_signal = self._calc_exec_rate_signal(exec_rate_7d, max_up, max_down)

        # 信号2: 利率差距反馈（avg_gap>0 说明定价过高，仅下调方向有效）
        # 证据权重防死锁：数据不足时缩小惩罚
        # 独立激活：市场已崩塌但 exec_rate 尚未反应时（avg_gap/current_rate > 2.0），强制激活
        large_divergence = avg_gap > 0 and current_rate > 0 and (avg_gap / current_rate) > 2.0
        if avg_gap > 0 and (exec_signal < 0 or large_divergence):
            gap_contribution = -min(avg_gap / (base_rate + 1e-8), 0.12)
            gap_signal = gap_contribution
        else:
            gap_signal = 0.0

        # 信号3: 趋势动量微调
        trend = exec_rate_7d / (exec_rate_30d + 1e-8)
        trend_signal = self._calc_trend_signal(trend, exec_rate_7d, max_up, max_down)

        # 负信号比例缩放: 防止多信号同时为负时超过 max_down 被 clip 丢失比例信息
        total_neg = min(exec_signal, 0) + min(gap_signal, 0) + min(trend_signal, 0)
        if total_neg < -max_down:
            scale = max_down / (-total_neg)
            if exec_signal < 0:
                exec_signal *= scale
            if gap_signal < 0:
                gap_signal *= scale
            if trend_signal < 0:
                trend_signal *= scale

        # 加法叠加，一次性 clip
        adjustment = 1.0 + exec_signal + gap_signal + trend_signal
        return float(np.clip(adjustment, 1.0 - max_down, 1.0 + max_up))

    def _calc_market_liquidity(self, preds: list) -> dict:
        """按 currency 聚合流动性评分，同时查询 24h vs 30d 成交量比率"""
        from collections import defaultdict
        groups = defaultdict(list)
        for p in preds:
            groups[p['currency']].append(p)

        result = {}
        for currency, items in groups.items():
            non2d_items = [x for x in items if int(x.get('period', 0)) != 2]
            ref_items = non2d_items or items
            avg_exec = sum(x.get('exec_rate_raw', x.get('calibrated_execution_prob', 0.5)) for x in ref_items) / len(ref_items)
            avg_age = sum(float(x.get('data_age_minutes', 0.0) or 0.0) for x in ref_items) / len(ref_items)
            warn_min, hard_min = self._freshness_thresholds_minutes(currency)
            if avg_age <= warn_min:
                freshness_signal = 1.0
            else:
                freshness_signal = max(0.2, 1.0 - 0.8 * min(
                    (avg_age - warn_min) / max(hard_min - warn_min, 1.0), 1.0
                ))

            # 查 funding_rates 24h vs 30d 平均成交量（用 period=30 作代表性基准）
            try:
                import sqlite3
                with sqlite3.connect(self.db_path) as conn:
                    row = conn.execute("""
                        SELECT
                            AVG(CASE WHEN datetime >= strftime('%Y-%m-%d %H:%M:%S','now','-1 day')
                                     THEN volume ELSE NULL END) AS vol_24h,
                            AVG(CASE WHEN datetime >= strftime('%Y-%m-%d %H:%M:%S','now','-7 days')
                                     THEN volume ELSE NULL END) AS vol_7d,
                            AVG(volume) AS vol_30d
                        FROM funding_rates
                        WHERE currency = ?
                          AND period = 30
                          AND datetime >= strftime('%Y-%m-%d %H:%M:%S','now','-30 days')
                    """, (currency,)).fetchone()
                vol_24h = row[0] or 0.0
                vol_7d  = row[1] or 1e-8
                vol_30d = row[2] or 1e-8
                # 双窗口 min 策略：取 24h/7d 与 24h/30d 的最小值（更悲观的那个）
                # 7d 基线捕捉突发崩盘，30d 基线捕捉持续萎缩，两者取严
                ratio_vs_7d  = vol_24h / (vol_7d  + 1e-8)
                ratio_vs_30d = vol_24h / (vol_30d + 1e-8)
                volume_ratio_24h = round(min(ratio_vs_7d, ratio_vs_30d), 3)
            except Exception:
                volume_ratio_24h = None

            book_signal = self._get_realtime_non2d_liquidity_signal(currency)
            if book_signal["available"]:
                fillability_signal = float(book_signal.get("fillability_signal", 0.0) or 0.0)
                fast_score = self._clip_unit(
                    avg_exec * 0.45 +
                    fillability_signal * 0.30 +
                    book_signal.get("depth_signal", 0.0) * 0.15 +
                    freshness_signal * 0.10
                ) * book_signal.get("structure_factor", 1.0)
                base_score = (
                    avg_exec * 0.40 +
                    freshness_signal * 0.20 +
                    fillability_signal * 0.20 +
                    book_signal["depth_signal"] * 0.20
                )
                score = base_score * book_signal.get("structure_factor", 1.0) * 100.0
            else:
                fillability_signal = self._clip_unit(avg_exec * 0.75 + freshness_signal * 0.25)
                fast_score = self._clip_unit(avg_exec * 0.70 + freshness_signal * 0.30)
                score = (
                    avg_exec * 0.70 +
                    freshness_signal * 0.30
                ) * 100.0
            score = round(float(np.clip(score, 0.0, 100.0)), 1)

            if score >= 65:
                level = "high"
            elif score >= 40:
                level = "medium"
            elif score >= 20:
                level = "low"
            else:
                level = "insufficient"

            result[currency] = {
                "level": level,
                "score": score,
                "avg_exec_rate": round(avg_exec, 3),
                "fast_score": round(float(np.clip(fast_score, 0.0, 1.0)), 3),
                "fillability_signal": round(float(np.clip(fillability_signal, 0.0, 1.0)), 3),
                "volume_ratio_24h": volume_ratio_24h,
            }
        return result

    def _estimate_rank6_reference_rate(self, preds: list) -> float:
        """Use fixed fUSD-2d as the terminal fallback reference for path scoring."""
        best_rate = None
        for pred in preds or []:
            if pred.get("currency") == "fUSD" and int(pred.get("period", 0) or 0) == 2:
                rate = float(pred.get("predicted_rate", 0.0) or 0.0)
                if rate > 0:
                    best_rate = rate
                    break

        if best_rate is None:
            snapshot = self._get_latest_prediction_snapshot("fUSD", 2)
            if snapshot is not None:
                best_rate = float(snapshot.get("predicted_rate", 0.0) or 0.0)

        if best_rate is None or best_rate <= 0:
            latest_rate, _ = self.get_latest_rate_from_db("fUSD", 2)
            if latest_rate:
                best_rate = float(latest_rate)

        if best_rate and best_rate > 0:
            return float(best_rate)
        # 动态兜底：取当前批次所有 current_rate 的中位数
        all_rates = [
            float(p.get("current_rate", 0.0) or 0.0)
            for p in (preds or [])
            if float(p.get("current_rate", 0.0) or 0.0) > 0
        ]
        if all_rates:
            import numpy as np
            best_rate = float(np.median(all_rates))
            logger.warning(f"rank6 reference: 无 fUSD-2d 数据，使用 current_rate 中位数={best_rate:.2f}")
        else:
            best_rate = 5.0
            logger.warning("rank6 reference: 无任何利率数据，使用硬编码兜底 5.0")
        return float(best_rate)

    def _estimate_frr_proxy_rate(self, currency: str, current_rate: float) -> float:
        """
        First-pass FRR proxy:
        use the latest same-currency 120d market rate; if unavailable or stale (>7d), fall back to current_rate.
        """
        try:
            proxy_rate, proxy_ts = self.get_latest_rate_from_db(currency, 120)
        except Exception:
            proxy_rate, proxy_ts = None, None
        if proxy_rate is None or float(proxy_rate or 0.0) <= 0.0:
            return float(current_rate or 0.0)
        # 新鲜度检查：120d 利率 >7 天则降级为 current_rate
        if proxy_ts:
            try:
                from datetime import datetime as _dt
                if isinstance(proxy_ts, str):
                    ts_dt = _dt.fromisoformat(proxy_ts)
                elif isinstance(proxy_ts, (int, float)):
                    ts_val = proxy_ts / 1000.0 if proxy_ts > 1e10 else proxy_ts
                    ts_dt = _dt.fromtimestamp(ts_val)
                else:
                    ts_dt = proxy_ts
                age_days = (_dt.now() - ts_dt).total_seconds() / 86400.0
                if age_days > 7:
                    logger.debug(
                        f"FRR proxy {currency}-120d 已 {age_days:.1f} 天未更新(>7d)，降级为 current_rate={current_rate}"
                    )
                    return float(current_rate or 0.0)
            except Exception:
                pass
        return float(proxy_rate)

    def _get_pending_order_pressure(self, currency: str, period: int) -> float:
        """Estimate how crowded this slot already is in pending virtual orders."""
        import sqlite3

        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN currency = ? AND period = ? AND status = 'PENDING' THEN 1 ELSE 0 END) AS pair_pending,
                        SUM(CASE WHEN currency = ? AND status = 'PENDING' THEN 1 ELSE 0 END) AS currency_pending
                    FROM virtual_orders
                    """,
                    (currency, period, currency),
                ).fetchone()
        except Exception:
            return 0.0

        pair_pending = float((row[0] if row else 0.0) or 0.0)
        currency_pending = float((row[1] if row else 0.0) or 0.0)
        return self._clip_unit(pair_pending / 4.0 * 0.75 + currency_pending / 18.0 * 0.25)

    def _calculate_path_value_score(self, pred: dict, rank6_rate: float) -> float:
        """
        Score the external execution path value:
        fixed-rate hanging window -> FRR proxy window -> fixed rank6 fallback.
        """
        rate = float(pred.get("predicted_rate", 0.0) or 0.0)
        current_rate = float(pred.get("current_rate", 0.0) or 0.0)
        period = int(pred.get("period", 0) or 0)
        exec_prob = self._clip_unit(
            pred.get("calibrated_execution_prob", pred.get("execution_probability", 0.0) or 0.0)
        )
        exec_rate_fast = self._clip_unit(pred.get("execution_rate_7d", pred.get("exec_rate_raw", exec_prob)))
        exec_rate_slow = self._clip_unit(pred.get("execution_rate_slow", exec_rate_fast))
        fast_liquidity_score = self._clip_unit(pred.get("fast_liquidity_score", pred.get("liquidity_score", 0.0) / 100.0))
        pending_pressure = self._clip_unit(pred.get("pending_order_pressure", 0.0) or 0.0)
        frr_proxy_rate = float(pred.get("frr_proxy_rate", current_rate) or current_rate or 0.0)

        market_follow_error = float(pred.get("market_follow_error", 0.0) or 0.0)
        avg_gap_failed = float(pred.get("avg_rate_gap_failed", 0.0) or 0.0)
        follow_gap_ratio = market_follow_error / (current_rate + 1e-8) if market_follow_error > 0 and current_rate > 0 else 0.0
        failed_gap_ratio = avg_gap_failed / (current_rate + 1e-8) if avg_gap_failed > 0 and current_rate > 0 else 0.0
        premium_ratio = (rate - current_rate) / (current_rate + 1e-8) if rate > current_rate and current_rate > 0 else 0.0

        stage1_fill_probability = self._clip_unit(
            (
                exec_prob * 0.50 +
                exec_rate_fast * 0.20 +
                fast_liquidity_score * 0.20 +
                (1.0 - pending_pressure) * 0.10
            ) *
            max(
                0.18,
                1.0 - follow_gap_ratio * 0.22 - failed_gap_ratio * 0.12 - max(premium_ratio - 0.35, 0.0) * 0.10
            )
        )

        ladder_decay_rate = rate * (0.99 ** 6)
        fallback_anchor = max(rank6_rate, frr_proxy_rate, current_rate, 0.01)
        rank6_fallback_penalty = float(
            np.clip(1.0 - max(rate - fallback_anchor, 0.0) / fallback_anchor * 0.20, 0.30, 1.0)
        )
        if pred.get("currency") == "fUST":
            # Default guarded fUST scenarios should inherit only a limited portion of
            # the 120d proxy upside; recent fake bursts in illiquid books were still
            # lifting rank2-rank4 too easily.
            frr_guard = (
                0.20 +
                fast_liquidity_score * 0.16 +
                stage1_fill_probability * 0.12 +
                max(min(exec_rate_fast, stage1_fill_probability) - 0.35, 0.0) * 0.10
            )
            if exec_rate_slow > 0:
                frr_guard += min(max(exec_rate_fast / (exec_rate_slow + 1e-8) - 1.0, 0.0), 1.0) * 0.04
            frr_guard -= pending_pressure * 0.12
            frr_guard -= min(max(failed_gap_ratio - 0.22, 0.0), 1.0) * 0.08
            if period >= 14:
                frr_guard -= 0.03
            if period >= 30:
                frr_guard -= 0.03
            if rate >= 11.0 and stage1_fill_probability >= 0.70 and fast_liquidity_score >= 0.72:
                frr_guard += 0.08
            frr_guard = float(np.clip(frr_guard, 0.16, 0.52))
            frr_fallback_value = rank6_rate + max(frr_proxy_rate - rank6_rate, 0.0) * frr_guard
        else:
            frr_fallback_value = (
                frr_proxy_rate * 0.72 +
                rank6_rate * 0.28
            )
        residual_path_rate = (
            ladder_decay_rate * 0.35 +
            frr_fallback_value * 0.40 +
            rank6_rate * 0.25
        )
        period_bonus = 1.0 + min(max(period, 0), 120) / 120.0 * 0.05
        if pred.get("currency") == "fUSD" and period == 120 and max(rate, current_rate) >= 12.0:
            period_bonus += 0.10

        if stage1_fill_probability >= 0.67:
            expected_terminal_mode = "stage1_fixed"
        elif frr_fallback_value >= rank6_rate:
            expected_terminal_mode = "stage2_frr"
        else:
            expected_terminal_mode = "rank6_fixed"

        path_value_score = (
            stage1_fill_probability * rate +
            (1.0 - stage1_fill_probability) * residual_path_rate
        ) * period_bonus

        pred["stage1_fill_probability"] = float(stage1_fill_probability)
        pred["frr_proxy_rate"] = float(frr_proxy_rate)
        pred["frr_fallback_value"] = float(frr_fallback_value)
        pred["rank6_fallback_penalty"] = rank6_fallback_penalty
        pred["expected_terminal_mode"] = expected_terminal_mode
        pred["_path_meta"] = {
            "follow_gap_ratio": float(follow_gap_ratio),
            "failed_gap_ratio": float(failed_gap_ratio),
            "premium_ratio": float(premium_ratio),
            "fallback_anchor": float(fallback_anchor),
            "stage1_fill_probability": float(stage1_fill_probability),
            "period_bonus": float(period_bonus),
            "frr_fallback_value": float(frr_fallback_value),
            "rank6_fallback_penalty": rank6_fallback_penalty,
            "expected_terminal_mode": expected_terminal_mode,
        }
        return float(path_value_score)

    def _calculate_fast_liquidity_score(self, pred: dict, market_liquidity: dict) -> float:
        """Estimate fast fill odds for the first stage of the external execution path."""
        currency = pred.get("currency", "")
        market_meta = market_liquidity.get(currency, {})
        exec_prob = self._clip_unit(
            pred.get("calibrated_execution_prob", pred.get("execution_probability", 0.0) or 0.0)
        )
        exec_rate_fast = self._clip_unit(pred.get("execution_rate_7d", pred.get("exec_rate_raw", exec_prob)))
        market_score = float(market_meta.get("score", pred.get("liquidity_score", 0.0)) or 0.0)
        market_score_signal = self._clip_unit(market_score / 100.0)
        market_fast_score = self._clip_unit(
            market_meta.get("fast_score", market_score_signal)
        )
        fillability_signal = self._clip_unit(
            market_meta.get("fillability_signal", market_fast_score)
        )
        volume_ratio = market_meta.get("volume_ratio_24h")
        if volume_ratio is None:
            volume_signal = 0.60
            volume_penalty = 1.0
        else:
            vr = float(volume_ratio or 0.0)
            volume_signal = self._clip_unit(vr / 1.20)
            # volume_penalty: 当 volume_ratio < 0.4 时线性惩罚至 0.75x
            if vr >= 0.4:
                volume_penalty = 1.0
            else:
                volume_penalty = 0.75 + 0.625 * (vr / 0.4)  # vr=0.4时=1.0, vr=0时=0.75

        pending_pressure = self._get_pending_order_pressure(currency, int(pred.get("period", 0) or 0))
        fast_score = (
            exec_prob * 0.34 +
            exec_rate_fast * 0.22 +
            market_fast_score * 0.20 +
            fillability_signal * 0.14 +
            market_score_signal * 0.10
        ) * (0.88 + 0.12 * volume_signal) * (1.0 - 0.20 * pending_pressure) * volume_penalty

        pred["pending_order_pressure"] = float(pending_pressure)
        pred["_liquidity_meta"] = {
            "market_score": float(market_score),
            "market_score_signal": float(market_score_signal),
            "market_fast_score": float(market_fast_score),
            "fillability_signal": float(fillability_signal),
            "volume_ratio_24h": None if volume_ratio is None else float(volume_ratio),
            "volume_signal": float(volume_signal),
            "volume_penalty": float(volume_penalty),
            "pending_order_pressure": float(pending_pressure),
        }
        return self._clip_unit(fast_score)

    def _calculate_currency_regime_multiplier(self, pred: dict, path_meta: dict, liquidity_meta: dict) -> float:
        """Bias ranking toward the default fUSD regime unless fUST shows real burst + liquidity."""
        currency = str(pred.get("currency", ""))
        period = int(pred.get("period", 0) or 0)
        rate = float(pred.get("predicted_rate", 0.0) or 0.0)
        current_rate = float(pred.get("current_rate", 0.0) or 0.0)
        avg_gap = float(pred.get("avg_rate_gap_failed", 0.0) or 0.0)
        pending_pressure = self._clip_unit(pred.get("pending_order_pressure", 0.0) or 0.0)
        market_fast_score = float(liquidity_meta.get("market_fast_score", 0.0) or 0.0)
        fillability_signal = float(liquidity_meta.get("fillability_signal", 0.0) or 0.0)
        volume_ratio = liquidity_meta.get("volume_ratio_24h")
        volume_ratio = float(volume_ratio or 0.0) if volume_ratio is not None else 0.0
        stage1_fill_probability = float(path_meta.get("stage1_fill_probability", 0.0) or 0.0)
        premium_abs = max(rate - current_rate, 0.0)

        if currency == "fUSD":
            multiplier = 1.04
            state = "fusd_preferred"
            if period == 120 and max(rate, current_rate) >= 12.0:
                multiplier += 0.14
                state = "fusd_120d_high_yield"
            elif period >= 60 and rate >= 10.0:
                multiplier += 0.05
                state = "fusd_long_support"
            if market_fast_score < 0.35:
                multiplier *= 0.95
        else:
            burst_ready = (
                premium_abs >= 2.5 and
                rate >= 11.0 and
                stage1_fill_probability >= 0.72 and
                market_fast_score >= 0.78 and
                fillability_signal >= 0.78 and
                volume_ratio >= 1.10
            )
            if burst_ready:
                multiplier = 1.02 + min((premium_abs - 2.5) / 10.0, 0.08)
                state = "fust_burst"
            else:
                gap_ratio = avg_gap / (current_rate + 1e-8) if avg_gap > 0 and current_rate > 0 else 0.0
                multiplier = min(
                    0.84,
                    0.72 +
                    market_fast_score * 0.05 +
                    stage1_fill_probability * 0.04 +
                    min(volume_ratio, 0.8) * 0.01
                )
                multiplier -= max(pending_pressure - 0.35, 0.0) * 0.12
                multiplier -= min(max(gap_ratio - 0.25, 0.0), 1.0) * 0.06
                if volume_ratio >= 1.20 and market_fast_score < 0.60:
                    multiplier -= 0.05
                if period >= 30:
                    multiplier -= 0.02
                if period >= 120:
                    multiplier -= 0.03
                state = "fust_guarded"

        pred["currency_regime_state"] = state
        return float(np.clip(multiplier, 0.70, 1.20))

    def _calculate_safety_multiplier(self, pred: dict, path_meta: dict) -> float:
        """Suppress fake premiums that do not survive the full execution path."""
        current_rate = float(pred.get("current_rate", 0.0) or 0.0)
        rate = float(pred.get("predicted_rate", 0.0) or 0.0)
        period = int(pred.get("period", 0) or 0)
        exec_rate_fast = self._clip_unit(pred.get("execution_rate_7d", pred.get("exec_rate_raw", 0.0)))
        follow_gap_ratio = float(path_meta.get("follow_gap_ratio", 0.0) or 0.0)
        failed_gap_ratio = float(path_meta.get("failed_gap_ratio", 0.0) or 0.0)
        premium_ratio = float(path_meta.get("premium_ratio", 0.0) or 0.0)
        rank6_fallback_penalty = float(path_meta.get("rank6_fallback_penalty", 1.0) or 1.0)

        multiplier = rank6_fallback_penalty
        multiplier *= max(0.25, 1.0 - max(follow_gap_ratio - 0.30, 0.0) * 0.22)
        multiplier *= max(0.35, 1.0 - max(failed_gap_ratio - 0.20, 0.0) * 0.16)
        multiplier *= max(0.55, 1.0 - max(premium_ratio - 0.50, 0.0) * 0.10)

        if current_rate > 0 and rate > current_rate * 2.5:
            multiplier *= 0.85
        if period >= 60 and exec_rate_fast < 0.22:
            multiplier *= 0.92
        if pred.get("currency") == "fUSD" and period == 120 and max(rate, current_rate) >= 12.0:
            multiplier *= 1.03

        return float(np.clip(multiplier, 0.20, 1.08))

    def _priority_bucket(self, value: float, step: float) -> int:
        value = float(value or 0.0)
        step = float(step or 0.0)
        if not np.isfinite(value):
            return 0
        if step <= 0:
            return int(round(value * 1000))
        return int(np.floor((value + 1e-12) / step))

    def _priority_revenue_score(self, pred: dict) -> float:
        path_value_score = float(pred.get("path_value_score", 0.0) or 0.0)
        currency_regime_multiplier = float(pred.get("currency_regime_multiplier", 1.0) or 1.0)
        safety_multiplier = float(pred.get("safety_multiplier", 1.0) or 1.0)
        revenue_score = path_value_score * currency_regime_multiplier * safety_multiplier
        return float(revenue_score) if np.isfinite(revenue_score) else 0.0

    def _prediction_hard_rank_key(self, pred: dict) -> tuple:
        combo_cfg = self.policy.get("combo_optimizer", {})
        revenue_step = float(combo_cfg.get("hard_sort_revenue_step", 0.10) or 0.10)
        fill_step = float(combo_cfg.get("hard_sort_fill_step", 0.02) or 0.02)
        revenue_score = self._priority_revenue_score(pred)
        fill_score = float(
            pred.get(
                "stage1_fill_probability",
                pred.get("fast_liquidity_score", pred.get("execution_probability", 0.0)),
            ) or 0.0
        )
        period = int(pred.get("period", 0) or 0)
        currency_priority = 1 if str(pred.get("currency", "")) == "fUSD" else 0

        pred["revenue_priority_bucket"] = self._priority_bucket(revenue_score, revenue_step)
        pred["fill_priority_bucket"] = self._priority_bucket(fill_score, fill_step)

        return (
            int(pred["revenue_priority_bucket"]),
            int(pred["fill_priority_bucket"]),
            period,
            currency_priority,
            revenue_score,
            fill_score,
            float(pred.get("final_rank_score", pred.get("weighted_score", 0.0)) or 0.0),
            float(pred.get("predicted_rate", 0.0) or 0.0),
        )

    def _apply_path_ranking(self, preds: list, market_liquidity: dict, fusd_2d_pred: Optional[dict]) -> list:
        """Rank predictions by external path value instead of static weighted score."""
        if not preds:
            return []

        rank6_inputs = list(preds)
        if fusd_2d_pred is not None:
            rank6_inputs.append(fusd_2d_pred)
        rank6_rate = self._estimate_rank6_reference_rate(rank6_inputs)

        ranked_preds = []
        for pred in preds:
            pred["frr_proxy_rate"] = self._estimate_frr_proxy_rate(
                pred.get("currency", ""),
                float(pred.get("current_rate", 0.0) or 0.0),
            )
            pred["fast_liquidity_score"] = self._calculate_fast_liquidity_score(pred, market_liquidity)
            pred["path_value_score"] = self._calculate_path_value_score(pred, rank6_rate)
            path_meta = pred.get("_path_meta", {})
            liquidity_meta = pred.get("_liquidity_meta", {})
            pred["currency_regime_multiplier"] = self._calculate_currency_regime_multiplier(
                pred, path_meta, liquidity_meta
            )
            pred["safety_multiplier"] = self._calculate_safety_multiplier(pred, path_meta)
            pred["liquidity_multiplier"] = float(0.68 + 0.32 * pred["fast_liquidity_score"])

            final_rank_score = (
                pred["path_value_score"] *
                pred["currency_regime_multiplier"] *
                pred["safety_multiplier"] *
                pred["liquidity_multiplier"]
            )
            pred["final_rank_score"] = float(final_rank_score)
            pred["weighted_score"] = float(final_rank_score)
            ranked_preds.append(pred)

        ranked_preds.sort(
            key=self._prediction_hard_rank_key,
            reverse=True,
        )
        return ranked_preds

    def _get_recommendation_regime_multiplier(self, pred: dict) -> float:
        """
        推荐排序的二次风控。
        针对 review 里暴露出的 fUSD 中长周期退化场景，继续压低高 gap / 低执行率组合。
        """
        currency = str(pred.get("currency", ""))
        period = int(pred.get("period", 0) or 0)
        exec_rate_fast = float(pred.get("execution_rate_7d", 0.0) or 0.0)
        order_count = int(pred.get("order_count", 0) or 0)
        avg_gap = float(pred.get("avg_rate_gap_failed", 0.0) or 0.0)
        current_rate = float(pred.get("current_rate", 0.0) or 0.0)
        follow_err = float(pred.get("market_follow_error", 0.0) or 0.0)

        multiplier = 1.0
        gap_ratio = avg_gap / (current_rate + 1e-8) if avg_gap > 0 and current_rate > 0 else 0.0
        relative_err = follow_err / (current_rate + 1e-8) if follow_err > 0 and current_rate > 0 else 0.0

        if currency == "fUSD":
            if period >= 60:
                if exec_rate_fast < 0.25 and order_count >= 6:
                    multiplier *= max(0.35, 0.55 + 0.45 * (exec_rate_fast / 0.25))
                if gap_ratio > 0.80:
                    multiplier *= max(0.40, 1.0 - 0.18 * (gap_ratio - 0.80))
                if relative_err > 1.00:
                    multiplier *= max(0.35, 1.0 - 0.16 * (relative_err - 1.00))
            elif period >= 10:
                if exec_rate_fast < 0.20 and order_count >= 8:
                    multiplier *= max(0.50, 0.70 + 0.30 * (exec_rate_fast / 0.20))
                if gap_ratio > 1.00:
                    multiplier *= max(0.50, 1.0 - 0.10 * (gap_ratio - 1.00))

        return float(np.clip(multiplier, 0.25, 1.0))

    def _ensure_prediction_history_schema(self):
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_timestamp TEXT NOT NULL,
                    update_cycle_id TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    period INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    recommendation_rank INTEGER,
                    rank_weight REAL,
                    candidate_id TEXT,
                    decision_mode TEXT,
                    predicted_rate REAL NOT NULL,
                    execution_probability REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    conservative_rate REAL,
                    balanced_rate REAL,
                    aggressive_rate REAL,
                    trend_factor REAL,
                    current_market_rate REAL NOT NULL,
                    rate_premium_pct REAL,
                    ma_60 REAL,
                    ma_1440 REAL,
                    volatility_60 REAL,
                    volume_ma_60 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("PRAGMA table_info(prediction_history)")
            existing = {row[1] for row in cursor.fetchall()}
            required_columns = {
                "calibrated_execution_probability": "REAL",
                "weighted_score": "REAL",
                "liquidity_score": "REAL",
                "data_age_minutes": "REAL",
                "market_follow_error": "REAL",
                "stale_data": "INTEGER",
                "path_value_score": "REAL",
                "stage1_fill_probability": "REAL",
                "frr_proxy_rate": "REAL",
                "frr_fallback_value": "REAL",
                "rank6_fallback_penalty": "REAL",
                "fast_liquidity_score": "REAL",
                "currency_regime_state": "TEXT",
                "final_rank_score": "REAL",
                "recommendation_rank": "INTEGER",
                "rank_weight": "REAL",
                "candidate_id": "TEXT",
                "decision_mode": "TEXT",
                "expected_terminal_mode": "TEXT",
            }
            for column, col_type in required_columns.items():
                if column not in existing:
                    cursor.execute(f"ALTER TABLE prediction_history ADD COLUMN {column} {col_type}")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_prediction_history_cycle "
                "ON prediction_history(update_cycle_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_prediction_history_created "
                "ON prediction_history(created_at DESC)"
            )
            conn.commit()

    def _default_rank_weight(self, recommendation_rank: int) -> float:
        if recommendation_rank == 1:
            return 0.60
        if recommendation_rank <= 6:
            return 0.10
        return 0.0

    def _build_candidate_id(self, pred: dict) -> str:
        liquidity_level = str(pred.get("liquidity_level") or "unknown").lower()
        liquidity_alias = {
            "medium": "mid",
            "high": "high",
            "low": "low",
        }.get(liquidity_level, liquidity_level or "unknown")
        return f"{pred['currency']}-{int(pred['period'])}-balanced-{liquidity_alias}"

    def _build_shadow_candidate_id(self, pred: dict) -> str:
        candidate_band = str(pred.get("candidate_band") or "balanced_mid").replace("_", "-")
        return f"{pred['currency']}-{int(pred['period'])}-{candidate_band}"

    def _load_market_anchor_rows(self, currency: str, period: int) -> list[dict]:
        import sqlite3

        market_rows = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                close_rows = conn.execute(
                    """
                    SELECT close_annual
                    FROM funding_rates
                    WHERE currency = ?
                      AND period = ?
                      AND close_annual IS NOT NULL
                      AND close_annual > 0
                    ORDER BY rowid DESC
                    LIMIT 12
                    """,
                    (currency, period),
                ).fetchall()
                executed_rows = conn.execute(
                    """
                    SELECT execution_rate
                    FROM virtual_orders
                    WHERE currency = ?
                      AND period = ?
                      AND status = 'EXECUTED'
                      AND execution_rate IS NOT NULL
                    ORDER BY rowid DESC
                    LIMIT 12
                    """,
                    (currency, period),
                ).fetchall()
        except Exception:
            return []

        for row in close_rows:
            market_rows.append({"close": float(row[0]), "executed": None})
        for row in executed_rows:
            market_rows.append({"close": None, "executed": float(row[0])})
        return market_rows

    def _score_shadow_candidate(self, base_pred: dict, candidate: RateCandidate, market_liquidity: dict, rank6_rate: float):
        candidate_pred = dict(base_pred)
        candidate_pred["predicted_rate"] = float(candidate.rate)
        candidate_pred["candidate_band"] = str(candidate.band)
        candidate_pred["candidate_id"] = self._build_shadow_candidate_id(candidate_pred)
        candidate_pred["market_follow_error"] = float(
            candidate_pred["predicted_rate"] - float(candidate_pred.get("current_rate", 0.0) or 0.0)
        )
        candidate_pred["frr_proxy_rate"] = self._estimate_frr_proxy_rate(
            candidate_pred.get("currency", ""),
            float(candidate_pred.get("current_rate", 0.0) or 0.0),
        )
        candidate_pred["fast_liquidity_score"] = self._calculate_fast_liquidity_score(candidate_pred, market_liquidity)
        candidate_pred["path_value_score"] = self._calculate_path_value_score(candidate_pred, rank6_rate)
        path_meta = candidate_pred.get("_path_meta", {})
        liquidity_meta = candidate_pred.get("_liquidity_meta", {})
        candidate_pred["currency_regime_multiplier"] = self._calculate_currency_regime_multiplier(
            candidate_pred, path_meta, liquidity_meta
        )
        candidate_pred["safety_multiplier"] = self._calculate_safety_multiplier(candidate_pred, path_meta)
        candidate_pred["liquidity_multiplier"] = float(0.68 + 0.32 * candidate_pred["fast_liquidity_score"])
        candidate_pred["final_rank_score"] = float(
            candidate_pred["path_value_score"] *
            candidate_pred["currency_regime_multiplier"] *
            candidate_pred["safety_multiplier"] *
            candidate_pred["liquidity_multiplier"]
        )
        candidate_pred["weighted_score"] = float(candidate_pred["final_rank_score"])
        return candidate_pred

    def _build_shadow_combo(self, ranked_predictions: list, update_cycle_id: str, beam_width: int):
        candidates = []
        scored_candidates = {}
        candidate_lookup = {}
        market_liquidity = self._calc_market_liquidity(ranked_predictions)
        rank6_rate = self._estimate_rank6_reference_rate(ranked_predictions)
        anchor_backed_pairs = set()

        for pred in ranked_predictions:
            currency = str(pred.get("currency", ""))
            period = int(pred.get("period", 0) or 0)
            if currency == "fUSD" and period == 2:
                continue

            market_rows = self._load_market_anchor_rows(currency, period)
            pair_candidates = []
            pair_anchor_backed = False
            if market_rows:
                try:
                    anchor = build_anchor_snapshot(currency=currency, period=period, market_rows=market_rows)
                    pair_candidates = generate_rate_candidates(
                        currency=currency,
                        period=period,
                        anchor=anchor,
                        hard_cap_pct=float(get_step_cap_pct(self.policy, period) or 0.05),
                        max_candidates=5,
                    )
                    if pair_candidates:
                        pair_anchor_backed = True
                        anchor_backed_pairs.add((currency, period))
                except Exception as e:
                    logger.warning(f"Anchor candidate generation failed for {currency}-{period}: {e}")

            if not pair_candidates:
                rate = float(pred.get("predicted_rate", 0.0) or 0.0)
                candidate_band = str(pred.get("candidate_band") or "balanced_mid")
                pair_candidates = [
                    RateCandidate(
                        currency=currency,
                        period=period,
                        rate=rate,
                        band=candidate_band,
                        distance_from_mid=float(rate - float(pred.get("current_rate", 0.0) or 0.0)),
                    )
                ]

            seen_keys = set()
            for candidate in pair_candidates:
                key = (candidate.currency, candidate.period, float(candidate.rate))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidate_pred = self._score_shadow_candidate(pred, candidate, market_liquidity, rank6_rate)
                candidate_pred["anchor_backed"] = pair_anchor_backed
                candidates.append(candidate)
                candidate_lookup[key] = candidate_pred
                scored_candidates[key] = {
                    "candidate_path_ev": self._priority_revenue_score(candidate_pred),
                    "fill_quality": float(
                        candidate_pred.get(
                            "stage1_fill_probability",
                            candidate_pred.get("fast_liquidity_score", candidate_pred.get("execution_probability", 0.0))
                        ) or 0.0
                    ),
                    "tenor_value": float(int(candidate_pred.get("period", 0) or 0)),
                    "currency_priority": float(1.0 if candidate_pred.get("currency") == "fUSD" else 0.0),
                    "anchor_backed": pair_anchor_backed,
                }

        if not candidates:
            return [], {}

        shadow_candidates = choose_combo_beam(candidates, scored_candidates, beam_width)
        if not shadow_candidates:
            return [], {}

        shadow_combo = []
        combo_revenue_ev = 0.0
        combo_fill_quality = 0.0
        for rank, candidate in enumerate(shadow_candidates, start=1):
            key = (candidate.currency, candidate.period, candidate.rate)
            metrics = scored_candidates[key]
            shadow_pred = dict(candidate_lookup[key])
            shadow_pred["update_cycle_id"] = update_cycle_id
            shadow_pred["recommendation_rank"] = rank
            shadow_pred["rank_weight"] = self._default_rank_weight(rank)
            shadow_pred["candidate_band"] = shadow_pred.get("candidate_band") or candidate.band
            shadow_pred["candidate_id"] = self._build_shadow_candidate_id(shadow_pred)
            shadow_pred["decision_mode"] = "exploit"
            shadow_pred["anchor_backed"] = bool(metrics.get("anchor_backed"))
            shadow_combo.append(shadow_pred)

            combo_revenue_ev += shadow_pred["rank_weight"] * metrics["candidate_path_ev"]
            combo_fill_quality += shadow_pred["rank_weight"] * metrics["fill_quality"]

        return shadow_combo, {
            "beam_width": int(beam_width),
            "combo_revenue_ev": float(combo_revenue_ev),
            "combo_fill_quality": float(combo_fill_quality),
            "anchor_backed_pair_count": len(anchor_backed_pairs),
        }

    def _enrich_prediction_identity(self, ranked_predictions: list, update_cycle_id: Optional[str] = None):
        cycle_id = update_cycle_id or datetime.now().strftime('%Y%m%d_%H%M%S')

        for rank, pred in enumerate(ranked_predictions, start=1):
            if not pred.get("update_cycle_id"):
                pred["update_cycle_id"] = cycle_id
            if pred.get("decision_mode") == "probe":
                if pred.get("rank_weight") is None:
                    pred["rank_weight"] = 0.0
            elif pred.get("recommendation_rank") is None:
                pred["recommendation_rank"] = rank
            if pred.get("decision_mode") != "probe" and pred.get("rank_weight") is None:
                pred["rank_weight"] = self._default_rank_weight(int(pred["recommendation_rank"]))
            if not pred.get("candidate_id"):
                if pred.get("candidate_band"):
                    pred["candidate_id"] = self._build_shadow_candidate_id(pred)
                else:
                    pred["candidate_id"] = self._build_candidate_id(pred)
            if not pred.get("decision_mode"):
                pred["decision_mode"] = "exploit"

        return ranked_predictions

    def _build_live_execution_predictions(
        self,
        sorted_preds: list,
        combo_top5: list,
        fusd_2d_pred: Optional[dict],
        update_cycle_id: str,
    ) -> list:
        execution_preds = []
        seen_pairs = set()

        for rank, pred in enumerate(combo_top5[:5], start=1):
            exploit_pred = dict(pred)
            exploit_pred["update_cycle_id"] = update_cycle_id
            exploit_pred["recommendation_rank"] = rank
            exploit_pred["rank_weight"] = self._default_rank_weight(rank)
            exploit_pred["decision_mode"] = "exploit"
            execution_preds.append(exploit_pred)
            seen_pairs.add((exploit_pred["currency"], int(exploit_pred["period"])))

        probe_sources = []
        if fusd_2d_pred is not None:
            probe_sources.append(fusd_2d_pred)
        for pred in sorted_preds:
            probe_sources.append(pred)

        for pred in probe_sources:
            pair = (pred["currency"], int(pred["period"]))
            if pair in seen_pairs:
                continue
            probe_pred = dict(pred)
            probe_pred["update_cycle_id"] = update_cycle_id
            probe_pred["recommendation_rank"] = None
            probe_pred["rank_weight"] = 0.0
            probe_pred["decision_mode"] = "probe"
            execution_preds.append(probe_pred)
            seen_pairs.add(pair)

        return self._enrich_prediction_identity(execution_preds, update_cycle_id=update_cycle_id)

    def _cleanup_live_cycle_state(self, update_cycle_id: str):
        import sqlite3

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM prediction_history WHERE update_cycle_id = ?",
                    (update_cycle_id,),
                )
                conn.execute(
                    "DELETE FROM virtual_orders WHERE update_cycle_id = ?",
                    (update_cycle_id,),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cleanup live cycle state for {update_cycle_id}: {e}")

    def _write_live_fail_closed_result(
        self,
        output_path: str,
        result: dict,
        error_message: str,
        update_cycle_id: Optional[str] = None,
    ):
        result["status"] = "error"
        result["fail_closed"] = True
        result["error"] = error_message
        result["recommendations"] = []
        if update_cycle_id:
            self._cleanup_live_cycle_state(update_cycle_id)
        self._emit_json_result(output_path, result, indent=4)
        logger.error(error_message)

    def _persist_prediction_history(self, ranked_predictions: list):
        if not ranked_predictions:
            return

        import sqlite3

        self._ensure_prediction_history_schema()
        update_cycle_id = ranked_predictions[0].get("update_cycle_id") or datetime.now().strftime('%Y%m%d_%H%M%S')
        self._enrich_prediction_identity(ranked_predictions, update_cycle_id=update_cycle_id)
        prediction_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stale_flag = int(bool(self._stale_issues))

        rows = []
        for rank, pred in enumerate(ranked_predictions, start=1):
            current_rate = float(pred.get("current_rate", 0.0) or 0.0)
            predicted_rate = float(pred.get("predicted_rate", 0.0) or 0.0)
            rate_premium_pct = None
            if current_rate > 0:
                rate_premium_pct = (predicted_rate - current_rate) / current_rate * 100.0

            rows.append((
                prediction_timestamp,
                pred["update_cycle_id"],
                pred["currency"],
                int(pred["period"]),
                rank,
                pred["recommendation_rank"],
                pred["rank_weight"],
                pred["candidate_id"],
                pred.get("decision_mode", "exploit"),
                predicted_rate,
                float(pred.get("execution_probability", 0.0) or 0.0),
                pred.get("strategy", "Unknown"),
                pred.get("confidence", "Medium"),
                float(pred.get("conservative_rate", 0.0) or 0.0),
                float(pred.get("balanced_rate", 0.0) or 0.0),
                float(pred.get("aggressive_rate", 0.0) or 0.0),
                float(pred.get("trend_factor", 0.0) or 0.0),
                current_rate,
                rate_premium_pct,
                None,
                None,
                None,
                None,
                float(pred.get("calibrated_execution_prob", 0.0) or 0.0),
                float(pred.get("weighted_score", 0.0) or 0.0),
                float(pred.get("liquidity_score", 0.0) or 0.0),
                float(pred.get("data_age_minutes", 0.0) or 0.0),
                float(pred.get("market_follow_error", 0.0) or 0.0),
                stale_flag,
                float(pred.get("path_value_score", 0.0) or 0.0),
                float(pred.get("stage1_fill_probability", 0.0) or 0.0),
                float(pred.get("frr_proxy_rate", current_rate) or current_rate or 0.0),
                pred.get("frr_fallback_value"),
                float(pred.get("rank6_fallback_penalty", 0.0) or 0.0),
                float(pred.get("fast_liquidity_score", 0.0) or 0.0),
                pred.get("currency_regime_state"),
                float(pred.get("final_rank_score", pred.get("weighted_score", 0.0)) or 0.0),
                pred.get("expected_terminal_mode"),
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO prediction_history (
                    prediction_timestamp, update_cycle_id, currency, period, rank,
                    recommendation_rank, rank_weight, candidate_id, decision_mode,
                    predicted_rate, execution_probability, strategy, confidence,
                    conservative_rate, balanced_rate, aggressive_rate, trend_factor,
                    current_market_rate, rate_premium_pct, ma_60, ma_1440,
                    volatility_60, volume_ma_60, calibrated_execution_probability,
                    weighted_score, liquidity_score, data_age_minutes,
                    market_follow_error, stale_data, path_value_score,
                    stage1_fill_probability, frr_proxy_rate, frr_fallback_value,
                    rank6_fallback_penalty, fast_liquidity_score,
                    currency_regime_state, final_rank_score, expected_terminal_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows
            )
            conn.commit()
        logger.info(
            f"Persisted prediction_history snapshot: cycle={update_cycle_id}, rows={len(rows)}, stale={bool(stale_flag)}"
        )

    def get_latest_predictions(self):
        """获取最新预测结果 - 并行化版本"""
        all_predictions = []

        for curr in ['fUSD', 'fUST']:
            if not self.models.get(curr):
                logger.warning(f"No models loaded for {curr}")
                continue

            logger.info(f"Fetching data for {curr}...")
            df = self.processor.load_data(curr)

            if df.empty:
                continue

            # 特征工程
            def process_group(group):
                return self.processor.add_technical_indicators(group)

            df_features = df.groupby('period', group_keys=False).apply(process_group)

            # 获取每个period的最新数据
            latest_data = df_features.groupby('period').tail(1).copy()

            # 准备特征
            meta = self.meta_info[curr]['model_conservative']
            feature_cols = meta['feature_cols']

            # 并行化预测
            logger.info(f"Starting parallel predictions for {curr} with {self.max_workers} workers...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 创建预测任务
                futures = []
                for idx, row in latest_data.iterrows():
                    row_dict = row.to_dict()
                    future = executor.submit(self.predict_single_period, row_dict, feature_cols, curr)
                    futures.append(future)

                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_predictions.append(result)
                    except Exception as e:
                        msg = str(e)
                        if "Stale data" in msg or "Stale DB data" in msg:
                            logger.warning(f"Prediction skipped due to stale market data: {msg}")
                        else:
                            logger.error(f"Prediction failed: {e}")

            logger.info(f"Completed predictions for {curr}: {len(all_predictions)} results")

        # 排序结果以保证确定性输出
        all_predictions.sort(key=lambda x: (x['currency'], x['period']))

        return all_predictions

    def generate_recommendations(self, output_path=None):
        """生成推荐结果并保存 - 增强版含虚拟订单创建"""
        # Use absolute path by default
        if output_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            output_path = os.path.join(base_dir, "data", "optimal_combination.json")
        self._stale_issues = []
        combo_policy = self.policy.get("combo_optimizer", {})
        combo_mode = combo_policy.get("combo_mode", "shadow")
        preds = self.get_latest_predictions()
        if not preds:
            stale_minutes = (
                max((float(item.get("age_minutes", 0.0)) for item in self._stale_issues), default=0.0)
                if self._stale_issues else 0.0
            )
            live_fail_closed = combo_mode == "live" and bool(self._stale_issues)
            stale_result = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "error" if live_fail_closed else ("stale_data" if self._stale_issues else "failed"),
                "strategy_info": "AI-Optimized Multi-Model Ensemble with Execution Feedback Loop",
                "policy_version": self.policy_version,
                "stale_data": bool(self._stale_issues),
                "stale_minutes": int(round(stale_minutes)),
                "stale_reason": "Market data freshness gate blocked prediction" if self._stale_issues else "No predictions generated",
                "stale_issues": self._stale_issues[:20],
                "recommendations": [],
            }
            if live_fail_closed:
                stale_result["fail_closed"] = True
                stale_result["error"] = "C3 live mode fail-closed: no valid fresh predictions"
            self._emit_json_result(output_path, stale_result, indent=4)
            if live_fail_closed:
                logger.error("C3 live mode fail-closed: no valid fresh predictions")
            else:
                logger.warning("No valid predictions generated; stale-data result persisted.")
            return

        valid_preds = [p for p in preds if p['predicted_rate'] > 1.0]

        market_liquidity = self._calc_market_liquidity(preds)
        gated_currencies = {
            currency for currency, info in market_liquidity.items()
            if info.get("score", 100.0) < 40.0 and
            (info.get("volume_ratio_24h") or 0.0) < 0.1
        }
        if gated_currencies:
            logger.warning(
                f"Liquidity gate triggered for {sorted(gated_currencies)}: "
                f"score < 40 and volume_ratio_24h < 0.1, non-2d orders will downgrade to 2d"
            )

        stale_pairs_set = set(
            (item['currency'], item['period']) for item in self._stale_issues
        ) if self._stale_issues else set()

        live_fusd_2d_pred = next(
            (p for p in valid_preds if p['currency'] == 'fUSD' and int(p['period']) == 2),
            None
        )
        fusd_2d_pred = live_fusd_2d_pred
        if fusd_2d_pred is None and combo_mode != "live":
            fusd_2d_pred = self._get_latest_prediction_snapshot('fUSD', 2)
            if fusd_2d_pred is not None:
                logger.warning(
                    "Current fUSD-2d prediction unavailable, using latest prediction_history snapshot for fixed rank6."
                )

        if combo_mode == "live":
            fail_closed_reasons = []
            if live_fusd_2d_pred is None:
                fail_closed_reasons.append("missing fresh fUSD-2d prediction")
            if stale_pairs_set:
                stale_pairs = ", ".join(
                    f"{currency}-{period}d" for currency, period in sorted(stale_pairs_set)
                )
                fail_closed_reasons.append(f"stale pairs detected: {stale_pairs}")
            if fail_closed_reasons:
                fail_result = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "status": "error",
                    "strategy_info": "AI-Optimized Multi-Model Ensemble with Execution Feedback Loop",
                    "policy_version": self.policy_version,
                    "stale_data": bool(self._stale_issues),
                    "stale_minutes": int(
                        round(max((float(item.get("age_minutes", 0.0)) for item in self._stale_issues), default=0.0))
                    ) if self._stale_issues else 0,
                    "stale_reason": "partial_stale_data_skipped" if self._stale_issues else "",
                    "stale_issues": self._stale_issues[:20] if self._stale_issues else [],
                    "market_liquidity": market_liquidity,
                    "recommendations": [],
                }
                self._write_live_fail_closed_result(
                    output_path,
                    fail_result,
                    f"C3 live mode fail-closed: {'; '.join(fail_closed_reasons)}",
                )
                return

        raw_ranked_preds = self._apply_path_ranking(valid_preds, market_liquidity, fusd_2d_pred)
        update_cycle_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._enrich_prediction_identity(raw_ranked_preds, update_cycle_id=update_cycle_id)

        sorted_preds = [
            p for p in raw_ranked_preds
            if not (p['currency'] in gated_currencies and int(p['period']) != 2)
        ]
        gated_2d_preds = []
        for currency in sorted(gated_currencies):
            pred_2d = next(
                (p for p in sorted_preds if p['currency'] == currency and int(p['period']) == 2),
                None
            )
            if pred_2d is not None:
                gated_2d_preds.append(pred_2d)

        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "status": 'success',
            "strategy_info": "AI-Optimized Multi-Model Ensemble with Execution Feedback Loop",
            "policy_version": self.policy_version,
            "stale_data": bool(self._stale_issues),
            "stale_minutes": int(
                round(max((float(item.get("age_minutes", 0.0)) for item in self._stale_issues), default=0.0))
            ) if self._stale_issues else 0,
            "stale_reason": "partial_stale_data_skipped" if self._stale_issues else "",
            "stale_issues": self._stale_issues[:20] if self._stale_issues else [],
            "market_liquidity": market_liquidity,
            "recommendations": []
        }

        combo_top5 = None
        combo_build_failed = False
        history_preds = raw_ranked_preds
        order_source_preds = sorted_preds
        if combo_mode in {"shadow", "live"}:
            try:
                raw_beam_width = combo_policy.get("beam_width", 24)
                beam_width = int(24 if raw_beam_width is None else raw_beam_width)
                if beam_width <= 0:
                    raise ValueError("beam_width must be positive")
                shadow_combo, shadow_metrics = self._build_shadow_combo(sorted_preds, update_cycle_id, beam_width)
            except Exception as e:
                logger.warning(f"{combo_mode} combo skipped: {e}")
                combo_build_failed = combo_mode == "live"
            else:
                if shadow_combo:
                    if combo_mode == "shadow":
                        logger.info(
                            "Shadow combo diagnostics generated internally: "
                            f"beam_width={shadow_metrics.get('beam_width')} "
                            f"combo_revenue_ev={shadow_metrics.get('combo_revenue_ev'):.4f} "
                            f"combo_fill_quality={shadow_metrics.get('combo_fill_quality'):.4f}"
                        )
                    else:
                        if int(shadow_metrics.get("anchor_backed_pair_count", 0) or 0) < 5:
                            result["status"] = "error"
                            result["fail_closed"] = True
                            result["error"] = "C3 live mode fail-closed: insufficient anchor-backed candidate pool"
                            result["recommendations"] = []
                            self._emit_json_result(output_path, result, indent=4)
                            logger.error("C3 live mode fail-closed: insufficient anchor-backed candidate pool")
                            return
                        if len(shadow_combo) >= 5:
                            combo_top5 = shadow_combo[:5]
                            if any(not bool(item.get("anchor_backed")) for item in combo_top5):
                                result["status"] = "error"
                                result["fail_closed"] = True
                                result["error"] = "C3 live mode fail-closed: combo contains non-anchor-backed candidate"
                                result["recommendations"] = []
                                self._emit_json_result(output_path, result, indent=4)
                                logger.error("C3 live mode fail-closed: combo contains non-anchor-backed candidate")
                                return
                        else:
                            result["status"] = "error"
                            result["fail_closed"] = True
                            result["error"] = "C3 live mode fail-closed: incomplete combo"
                            result["recommendations"] = []
                            self._emit_json_result(output_path, result, indent=4)
                            logger.error("C3 live mode fail-closed: incomplete combo")
                            return
                elif combo_mode == "live":
                    combo_build_failed = True

        if combo_mode == "live" and (combo_build_failed or combo_top5 is None):
            result["status"] = "error"
            result["fail_closed"] = True
            result["error"] = "C3 live mode fail-closed: combo build failed"
            result["recommendations"] = []
            self._emit_json_result(output_path, result, indent=4)
            logger.error("C3 live mode fail-closed: combo build failed")
            return

        # rank 1-5: top 5 from sorted_preds excluding fUSD-2d
        # gated 货币的 2d 订单参与正常评分排序，不再强制置顶（避免低利率 2d 异常 rank1）
        if combo_top5 is None:
            top5_candidates = [
                p for p in sorted_preds
                if not (p['currency'] == 'fUSD' and p['period'] == 2)
            ]
            top5 = top5_candidates[:5]
        else:
            top5 = combo_top5
            history_preds = self._build_live_execution_predictions(
                sorted_preds=sorted_preds,
                combo_top5=combo_top5,
                fusd_2d_pred=fusd_2d_pred,
                update_cycle_id=update_cycle_id,
            )
            order_source_preds = history_preds
            suspended_live_pairs = [
                f"{pred['currency']}-{int(pred['period'])}d"
                for pred in combo_top5
                if self._is_zero_liquidity_suspended(pred['currency'], int(pred['period']))
            ]
            if suspended_live_pairs:
                self._write_live_fail_closed_result(
                    output_path,
                    result,
                    "C3 live mode fail-closed: suspended live combo pairs: "
                    + ", ".join(suspended_live_pairs),
                )
                return
        recommendations_to_add = top5 + ([fusd_2d_pred] if fusd_2d_pred else [])

        # Build the recommendations list
        for i, pred in enumerate(recommendations_to_add[:6]):
            result["recommendations"].append({
                "rank": i + 1,
                "type": "optimal" if i == 0 else "alternative",
                "currency": pred['currency'],
                "period": pred['period'],
                "rate": round(pred['predicted_rate'], 4),
                "confidence": pred['confidence'],
                "liquidity_score": pred.get('liquidity_score'),
                "liquidity_level": pred.get('liquidity_level'),
                "details": {
                    "current": round(pred['current_rate'], 4),
                    "execution_probability": round(pred['execution_probability'], 4),
                    "conservative_floor": round(pred['conservative_rate'], 4),
                    "aggressive_target": round(pred['aggressive_rate'], 4),
                    "balanced_target": round(pred['balanced_rate'], 4),
                    "trend_1h": round(pred['trend_factor'], 4),
                    "strategy": pred['strategy']
                }
            })

        try:
            self._persist_prediction_history(history_preds)
        except Exception as e:
            if combo_mode == "live":
                self._write_live_fail_closed_result(
                    output_path,
                    result,
                    "C3 live mode fail-closed: prediction_history persist failed",
                    update_cycle_id=update_cycle_id,
                )
                logger.error(f"prediction_history persist error: {e}")
                return
            logger.warning(f"Failed to persist prediction_history snapshot: {e}")

        emit_result_after_orders = combo_mode == "live" and self.order_manager is not None
        if not emit_result_after_orders:
            logger.info(f"Recommendations saved to {output_path}")
            self._emit_json_result(output_path, result, indent=4)

        # ========== 创建虚拟订单 with Stratified Sampling by Period Tiers ==========
        if self.order_manager is not None:
            try:
                if self._stale_issues:
                    stale_pairs = [(i['currency'], i['period']) for i in self._stale_issues]
                    print(f"\n⚠️  Detected stale market data for {len(stale_pairs)} pair(s): {stale_pairs}")
                    print("    Continuing with virtual order creation for non-stale pairs.")
                if gated_currencies:
                    print(
                        f"\n⚠️  Liquidity gate active for {sorted(gated_currencies)}: "
                        "score<40 and volume_ratio_24h<0.1, non-2d orders downgraded to 2d."
                    )

                # Stratified sampling configuration by period tiers - 均衡采样
                # 目标是覆盖全期限，各层级数量均衡
                # 周期层级配置
                # 注意: 交易所不存在周期8-9
                # 可用周期: [2,3,4,5,6,7, 10,14,15,20,30, 60,90,120]
                period_policy = self.policy.get('period_policy', {})
                TIER_CONFIG = {
                    'short': {
                        'periods': period_policy.get('short_periods', [2, 3, 4, 5, 6, 7]),
                        'min_orders': 7,
                        'target_per_period': 1
                    },
                    'medium': {
                        'periods': period_policy.get('medium_periods', [10, 14, 15, 20, 30]),
                        'min_orders': 7,
                        'target_per_period': 1
                    },
                    'long': {
                        'periods': period_policy.get('long_periods', [60, 90, 120]),
                        'min_orders': 6,
                        'target_per_period': 2
                    }
                }
                TOP_N = 24                 # 增加总订单数到24，确保各层级都有足够覆盖

                # Helper function to get tier for a period
                def get_tier(period):
                    for tier_name, config in TIER_CONFIG.items():
                        if period in config['periods']:
                            return tier_name
                    return None

                # Group predictions by tier
                tier_preds = {'short': [], 'medium': [], 'long': []}
                suspended_pairs = []  # (currency, period) tuples skipped due to zero liquidity

                for pred in order_source_preds:
                    tier = get_tier(pred['period'])
                    if tier:
                        # Skip zero-liquidity pairs (exec_rate_21d < 5% for 14+ days)
                        if self._is_zero_liquidity_suspended(pred['currency'], pred['period']):
                            suspended_pairs.append((pred['currency'], pred['period']))
                            continue
                        # Check if cold start
                        order_count = self.order_manager.get_order_count(
                            pred['currency'],
                            pred['period']
                        )
                        pred['is_cold_start'] = order_count < self.COLD_START_THRESHOLD
                        pred['order_count'] = order_count
                        pred['tier'] = tier
                        tier_preds[tier].append(pred)

                # Select orders from each tier
                selected_preds = []
                tier_shortfall = 0  # Track how many slots we couldn't fill

                for tier_name, config in TIER_CONFIG.items():
                    min_orders = config['min_orders']
                    available = tier_preds[tier_name]

                    # Take up to min_orders from this tier
                    selected = available[:min_orders]
                    selected_preds.extend(selected)

                    # Calculate shortfall if tier doesn't have enough
                    if len(selected) < min_orders:
                        tier_shortfall += min_orders - len(selected)

                # Fill remaining slots with best remaining predictions (regardless of tier)
                remaining_slots = TOP_N - len(selected_preds) + tier_shortfall

                # Collect all remaining predictions not yet selected
                selected_keys = {(p['currency'], p['period']) for p in selected_preds}
                remaining_preds = []
                for tier_name in ['long', 'medium', 'short']:  # Prefer longer periods for remaining
                    for pred in tier_preds[tier_name]:
                        if (pred['currency'], pred['period']) not in selected_keys:
                            remaining_preds.append(pred)

                # Fill remaining slots
                selected_preds.extend(remaining_preds[:remaining_slots])

                # Per-period coverage guarantee: ensure each period has at least 1 order
                all_periods = [p for cfg in TIER_CONFIG.values() for p in cfg['periods']]
                covered_periods = {p['period'] for p in selected_preds}
                for period in all_periods:
                    if period not in covered_periods:
                        # Find best candidate for this period from tier_preds
                        candidate = None
                        for tier_name in ['long', 'medium', 'short']:
                            for p in tier_preds[tier_name]:
                                if p['period'] == period:
                                    candidate = p
                                    break
                            if candidate:
                                break
                        if candidate:
                            if len(selected_preds) >= TOP_N:
                                # Replace lowest-scored order to keep total count stable
                                min_idx = min(
                                    range(len(selected_preds)),
                                    key=lambda i: selected_preds[i].get('weighted_score', 0)
                                )
                                selected_preds[min_idx] = candidate
                            else:
                                selected_preds.append(candidate)

                # If liquidity gate is active, force the downgraded 2d order into execution set.
                selected_keys = {(p['currency'], p['period']) for p in selected_preds}
                forced_keys = {(p['currency'], p['period']) for p in gated_2d_preds}
                for gate_pred in gated_2d_preds:
                    key = (gate_pred['currency'], gate_pred['period'])
                    if key in selected_keys:
                        continue
                    if len(selected_preds) >= TOP_N:
                        replace_candidates = [
                            i for i, pred in enumerate(selected_preds)
                            if (pred['currency'], pred['period']) not in forced_keys
                        ]
                        if replace_candidates:
                            min_idx = min(
                                replace_candidates,
                                key=lambda i: selected_preds[i].get('weighted_score', 0)
                            )
                            selected_preds[min_idx] = gate_pred
                        else:
                            continue
                    else:
                        selected_preds.append(gate_pred)
                    selected_keys.add(key)

                # Create virtual orders
                if suspended_pairs:
                    print(f"\n⏸  Zero-liquidity suspension: {len(suspended_pairs)} pair(s) skipped (no exec for 14+d, exec_rate<5%):")
                    for cur, per in suspended_pairs:
                        print(f"   {cur}-{per}d")
                probe_state = self._load_refresh_probe_state()
                counters = probe_state.setdefault("counters", {})
                lookback_hours = int(self._policy_value("automation", "refresh_probe_lookback_hours", 24))
                min_validations = int(self._policy_value("automation", "refresh_probe_min_validations", 1))
                trigger_cycles = int(self._policy_value("automation", "refresh_probe_trigger_cycles", 6))
                max_probe_per_cycle = int(self._policy_value("automation", "refresh_probe_max_per_cycle", 4))
                probe_created = 0

                created_orders = []
                for pred in selected_preds[:TOP_N]:
                    key = f"{pred['currency']}-{pred['period']}"
                    if (
                        combo_mode == "live"
                        and pred.get("decision_mode") == "exploit"
                        and pred.get("update_cycle_id") == update_cycle_id
                        and pred.get("recommendation_rank") is not None
                        and int(pred["recommendation_rank"]) <= 5
                    ):
                        pred["force_create"] = True

                    # Fix7: 跳过数据陈旧的 pair，避免创建基于过期市场数据的虚拟单
                    if stale_pairs_set and (pred['currency'], pred['period']) in stale_pairs_set:
                        logger.info(
                            f"Skipping virtual order for stale pair {pred['currency']}-{pred['period']}d "
                            f"(market data too old)"
                        )
                        continue

                    recent_feedback_ok = not self.order_manager.needs_refresh_probe(
                        pred['currency'], pred['period'], lookback_hours, min_validations
                    )

                    order_id = self.order_manager.create_virtual_order(pred)
                    probe_type = "normal"

                    duplicate_skipped = isinstance(order_id, str) and order_id.startswith("DUPLICATE_SKIPPED")
                    if duplicate_skipped:
                        if recent_feedback_ok:
                            counters[key] = 0
                        else:
                            counters[key] = int(counters.get(key, 0)) + 1

                            # Insert a low-weight refresh probe after sustained no-feedback cycles.
                            if counters[key] >= trigger_cycles and probe_created < max_probe_per_cycle:
                                probe_pred = dict(pred)
                                probe_pred["probe_type"] = "refresh_probe"
                                probe_pred["force_create"] = True
                                probe_pred["data_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                probe_order_id = self.order_manager.create_virtual_order(probe_pred)
                                if not (isinstance(probe_order_id, str) and probe_order_id.startswith("DUPLICATE_SKIPPED")):
                                    order_id = probe_order_id
                                    probe_type = "refresh_probe"
                                    probe_created += 1
                                    counters[key] = 0
                    else:
                        counters[key] = 0

                    created_orders.append({
                        'order_id': order_id,
                        'currency': pred['currency'],
                        'period': pred['period'],
                        'is_cold_start': pred.get('is_cold_start', False),
                        'order_count': pred.get('order_count', 0),
                        'tier': pred.get('tier', 'unknown'),
                        'probe_type': probe_type,
                    })

                self._save_refresh_probe_state(probe_state)

                # Print summary
                tier_counts = {'short': 0, 'medium': 0, 'long': 0}
                cold_count = 0
                probe_count = 0
                for order in created_orders:
                    tier_counts[order['tier']] += 1
                    if order['is_cold_start']:
                        cold_count += 1
                    if order.get('probe_type') == 'refresh_probe':
                        probe_count += 1

                print(f"\n=== Virtual Order Creation Summary ===")
                print(f"Total orders created: {len(created_orders)}")
                print(f"Tier distribution:")
                print(f"  - Short (2-7d): {tier_counts['short']} orders")
                print(f"  - Medium (10-30d): {tier_counts['medium']} orders")
                print(f"  - Long (60-120d): {tier_counts['long']} orders")
                print(f"Cold start combinations: {cold_count}")
                print(f"Refresh probe orders: {probe_count}")
                print(f"Sampling strategy: Stratified by Period Tiers")

                # Print details
                for order in created_orders:
                    probe_tag = " [PROBE]" if order.get('probe_type') == 'refresh_probe' else ""
                    status = "COLD START" if order['is_cold_start'] else f"WARM ({order['order_count']} orders)"
                    print(f"  {order['currency']}-{order['period']}d [{order['tier']}]"
                          f"{probe_tag}: {status}")

            except Exception as e:
                if combo_mode == "live":
                    self._write_live_fail_closed_result(
                        output_path,
                        result,
                        "C3 live mode fail-closed: virtual order creation failed",
                        update_cycle_id=update_cycle_id,
                    )
                    logger.error(f"virtual order creation error: {e}")
                    return
                print(f"Failed to create virtual orders: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: OrderManager not available, skipping virtual order creation")
        # ============================================================

        if emit_result_after_orders:
            logger.info(f"Recommendations saved to {output_path}")
            self._emit_json_result(output_path, result, indent=4)

        # ========== 收集并输出系统指标 ==========
        try:
            from ml_engine.metrics import MetricsCollector, save_metrics_to_file

            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            db_path = os.path.join(base_dir, 'data', 'lending_history.db')

            metrics_collector = MetricsCollector(db_path)

            # 收集所有指标（包括预测的 clipping 信息）
            all_metrics = metrics_collector.get_all_metrics(predictions=preds)

            # 打印到日志和控制台
            metrics_collector.print_metrics_summary(predictions=preds)

            # 保存到文件
            metrics_output = os.path.join(base_dir, 'data', 'system_metrics.json')
            save_metrics_to_file(all_metrics, metrics_output)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            import traceback
            traceback.print_exc()
        # ============================================================

if __name__ == "__main__":
    # 子进程也写入主日志文件，确保 v2 模型活动可追踪
    import pathlib
    _log_path = pathlib.Path(__file__).resolve().parent.parent / "log" / "ml_optimizer.log"
    logger.add(str(_log_path), retention='7 days', rotation="10 MB")

    predictor = EnsemblePredictor()
    predictor.generate_recommendations()
