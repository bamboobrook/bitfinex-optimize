import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import json
import os
import sys
from loguru import logger
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
warnings.simplefilter(action='ignore', category=FutureWarning)

# 添加父目录到 path 以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ml_engine.data_processor import DataProcessor
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
        predicted_rate: float
    ) -> Tuple[float, Optional[float], Optional[float], bool, Optional[float]]:
        """
        Apply per-period one-step cap. Currently strict for 120d by policy.

        Returns:
            (capped_rate, previous_rate, step_change_pct, step_capped, cap_pct)
        """
        cap_pct = get_step_cap_pct(self.policy, period)
        if cap_pct is None:
            return predicted_rate, None, None, False, None

        previous_rate = self._get_previous_predicted_rate(currency, period)
        if previous_rate is None or previous_rate <= 0:
            return predicted_rate, previous_rate, None, False, cap_pct

        lower = previous_rate * (1.0 - cap_pct)
        upper = previous_rate * (1.0 + cap_pct)
        capped_rate = float(np.clip(predicted_rate, lower, upper))
        step_capped = abs(capped_rate - predicted_rate) > 1e-12
        step_change_pct = (capped_rate - previous_rate) / previous_rate
        return capped_rate, previous_rate, float(step_change_pct), step_capped, cap_pct

    def _policy_value(self, section: str, key: str, default):
        return self.policy.get(section, {}).get(key, default)

    def _freshness_thresholds_minutes(self, currency: str = "") -> Tuple[float, float]:
        warn_minutes = float(self._policy_value("automation", "stale_data_warn_minutes", 60))
        hard_minutes = float(self._policy_value("automation", "stale_data_hard_minutes", 120))
        # fUST has naturally sparse market data (often 4-24h gaps) - use a per-currency override
        if currency:
            per_cur_key = f"stale_data_hard_minutes_{currency}"
            per_cur_hard = self._policy_value("automation", per_cur_key, None)
            if per_cur_hard is not None:
                hard_minutes = float(per_cur_hard)
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
            with open(self.refresh_probe_state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save refresh probe state: {e}")

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

        warn_minutes, hard_minutes = self._freshness_thresholds_minutes(currency)

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

        # 贝叶斯风格校准: 根据数据充分度动态加权
        # order_count 少时信任模型,多时信任历史执行率
        # sqrt曲线: 50单→ew=0.50, 100单→ew=0.65(上限), 200单→ew=0.65
        evidence_weight = min(np.sqrt(order_count / 200.0), 0.65)
        calibrated_prob = (1.0 - evidence_weight) * prob + evidence_weight * exec_rate_fast
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

        # 诊断日志: 记录关键决策变量
        logger.debug(
            f"Strategy decision for {currency}-{period}: "
            f"prob={prob:.4f}, exec_rate_fast={exec_rate_fast:.4f}, "
            f"calibrated_prob={calibrated_prob:.4f}"
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
            exec_rate_fast, exec_rate_slow, avg_rate_gap, base_rate, period, currency
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
            currency, period, final_rate
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
            "liquidity_score": liq_score,
            "liquidity_level": liq_level,
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

    def _get_unified_adjustment(self, exec_rate_7d: float, exec_rate_30d: float,
                                avg_gap: float, base_rate: float,
                                period: int, currency: str) -> float:
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
        if avg_gap > 0 and exec_signal < 0:
            gap_contribution = -min(avg_gap / (base_rate + 1e-8), 0.12)
            gap_signal = gap_contribution
        else:
            gap_signal = 0.0

        # 信号3: 趋势动量微调
        trend = exec_rate_7d / (exec_rate_30d + 1e-8)
        trend_signal = self._calc_trend_signal(trend, exec_rate_7d, max_up, max_down)

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
            avg_score = sum(x.get('liquidity_score', 50) for x in items) / len(items)
            avg_exec = sum(x.get('calibrated_execution_prob', 0.5) for x in items) / len(items)

            # 查 funding_rates 24h vs 30d 平均成交量（用 period=30 作代表性基准）
            try:
                import sqlite3
                with sqlite3.connect(self.db_path) as conn:
                    row = conn.execute("""
                        SELECT
                            AVG(CASE WHEN datetime >= strftime('%Y-%m-%d %H:%M:%S','now','-1 day')
                                     THEN volume ELSE NULL END) AS vol_24h,
                            AVG(volume) AS vol_30d
                        FROM funding_rates
                        WHERE currency = ?
                          AND period = 30
                          AND datetime >= strftime('%Y-%m-%d %H:%M:%S','now','-30 days')
                    """, (currency,)).fetchone()
                vol_24h = row[0] or 0.0
                vol_30d = row[1] or 1e-8
                volume_ratio_24h = round(vol_24h / (vol_30d + 1e-8), 3)
            except Exception:
                volume_ratio_24h = None

            score = round(avg_score, 1)
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
                "volume_ratio_24h": volume_ratio_24h,
            }
        return result

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
        preds = self.get_latest_predictions()
        if not preds:
            stale_minutes = (
                max((float(item.get("age_minutes", 0.0)) for item in self._stale_issues), default=0.0)
                if self._stale_issues else 0.0
            )
            stale_result = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "stale_data" if self._stale_issues else "failed",
                "strategy_info": "AI-Optimized Multi-Model Ensemble with Execution Feedback Loop",
                "policy_version": self.policy_version,
                "stale_data": bool(self._stale_issues),
                "stale_minutes": int(round(stale_minutes)),
                "stale_reason": "Market data freshness gate blocked prediction" if self._stale_issues else "No predictions generated",
                "stale_issues": self._stale_issues[:20],
                "recommendations": [],
            }
            with open(output_path, 'w') as f:
                json.dump(stale_result, f, indent=4)
            logger.warning("No valid predictions generated; stale-data result persisted.")
            print(json.dumps(stale_result, indent=2))
            return

        # 排序逻辑：
        # 1. 优先考虑成交概率 >= 0.5 的选项
        # 2. 在成交概率合格的基础上，按预测利率降序排列
        valid_preds = [p for p in preds if p['predicted_rate'] > 1.0]

        market_liquidity = self._calc_market_liquidity(preds)

        # Revenue-optimized scoring: 收益率优先,兼顾成交和长周期
        def calculate_weighted_score(pred):
            """
            收益率优先评分 (v6):
            - 40% effective_rate (calibrated_prob * rate - 期望收益)
            - 30% raw_rate (高利率偏好)
            - 20% exec_prob (成交保障，使用校准概率)
            - 10% revenue_factor (period_days/120 - 长周期仅作微调)
            + execution floor: calibrated_prob < 0.35 gets 0.4x penalty
            + data_age multiplier: 数据越老（流动性越低）评分越低，warn~hard 线性衰减到 0.6x
            + divergence multiplier: 预测利率/当前市场 > 2.5x 时线性降权至 0.5x
            """
            calib_prob = pred.get('calibrated_execution_prob', pred['execution_probability'])
            rate = pred['predicted_rate']
            period = pred['period']
            data_age = pred.get('data_age_minutes', 0.0)
            currency = pred.get('currency', '')

            # 1. 标准化利率到0-1范围 (假设最高利率40%)
            normalized_rate = min(rate / 40.0, 1.0)

            # 2. Effective rate = rate * calibrated_prob (期望收益，反映真实成交率)
            effective_rate = normalized_rate * calib_prob

            # 3. Revenue factor based on period (continuous, not tiered)
            revenue_factor = min(period / 120.0, 1.0)

            # 4. Execution floor: calibrated_prob < 0.35 时线性衰减惩罚（覆盖零流动性场景）
            # 线性衰减避免 0.35 处硬跳变，确保低流动性周期仍有订单进入市场（闭环不断裂）
            # calib_prob=0.26 → 0.4 + 0.6×0.74 = 0.844x（而非原来 0.4x）
            if calib_prob >= 0.35:
                exec_floor_multiplier = 1.0
            else:
                exec_floor_multiplier = 0.4 + 0.6 * (calib_prob / 0.35)

            # 5. Data-age multiplier: 数据老旧 = 流动性可能枯竭，推荐权重线性衰减
            warn_min, hard_min = self._freshness_thresholds_minutes(currency)
            if data_age <= warn_min:
                age_multiplier = 1.0
            else:
                # warn~hard 区间线性从 1.0 衰减到 0.6
                age_multiplier = 1.0 - 0.4 * min((data_age - warn_min) / max(hard_min - warn_min, 1.0), 1.0)

            # 6. Market divergence multiplier: 预测利率远高于当前市场时降权
            # 防止"历史高利率残留exec_rate"掩盖当前市场崩塌（如 fUST-60: 预测13% vs 市场2.9%）
            # 使用相对偏差而非绝对值，避免对高利率市场误惩罚
            follow_err = pred.get('market_follow_error', 0.0)
            current_rate_val = pred.get('current_rate', rate)
            if follow_err > 0 and current_rate_val > 0:
                relative_err = follow_err / current_rate_val
                if relative_err > 1.5:
                    # relative_err 1.5→4.0 线性映射到 multiplier 1.0→0.5
                    divergence_multiplier = max(0.5, 1.0 - 0.2 * (relative_err - 1.5))
                else:
                    divergence_multiplier = 1.0
            else:
                divergence_multiplier = 1.0

            # 7. 最终分数 = 40%期望收益 + 30%原始利率 + 20%执行概率 + 10%周期
            final_score = (
                effective_rate * 0.40 +
                normalized_rate * 0.30 +
                calib_prob * 0.20 +
                revenue_factor * 0.10
            ) * exec_floor_multiplier * age_multiplier * divergence_multiplier

            return final_score

        sorted_preds = sorted(
            valid_preds,
            key=calculate_weighted_score,
            reverse=True
        )

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

        # Build recommendations: rank 1-5 = top 5 (excluding fUSD-2d), rank 6 = fUSD-2d (fixed)
        fusd_2d_pred = None
        for pred in sorted_preds:
            if pred['currency'] == 'fUSD' and pred['period'] == 2:
                fusd_2d_pred = pred
                break

        # rank 1-5: top 5 from sorted_preds excluding fUSD-2d
        top5 = [p for p in sorted_preds if not (p['currency'] == 'fUSD' and p['period'] == 2)][:5]
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

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)

        logger.info(f"Recommendations saved to {output_path}")
        print(json.dumps(result, indent=2))

        # ========== 创建虚拟订单 with Stratified Sampling by Period Tiers ==========
        if self.order_manager is not None:
            try:
                if self._stale_issues:
                    print("\n⚠️  Detected stale market data in this cycle. Skip virtual order creation.")
                    return

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

                for pred in sorted_preds:
                    tier = get_tier(pred['period'])
                    if tier:
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

                # Create virtual orders
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
                print(f"Failed to create virtual orders: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: OrderManager not available, skipping virtual order creation")
        # ============================================================

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
