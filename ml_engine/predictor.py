import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import json
import os
import sys
from loguru import logger
from datetime import datetime
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
warnings.simplefilter(action='ignore', category=FutureWarning)

# 添加父目录到 path 以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ml_engine.data_processor import DataProcessor

class EnsemblePredictor:
    """使用集成模型的增强预测器 - 支持GPU加速和并行化"""

    # 类常量配置
    COLD_START_THRESHOLD = 10  # 冷启动数据点阈值
    STALE_DATA_THRESHOLD_HOURS = 2  # 陈旧数据阈值(小时)

    def __init__(self, model_dir=None, max_workers=8):
        # Use absolute path for models directory
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            model_dir = os.path.join(base_dir, "data", "models")
        self.model_dir = model_dir
        self.max_workers = max_workers  # 并行worker数量
        self.models = {}  # {curr: {model_type: {algo: model}}}
        self.meta_info = {}  # {curr: {model_type: {weights, features, task_type}}}
        self.processor = DataProcessor()
        self._timestamp_deprecation_warned = False  # 废弃警告标志

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
            predictions['lgb'] = models['lgb'].predict(X)

        # CatBoost预测
        if 'cat' in models:
            if meta['task_type'] == 'classification':
                predictions['cat'] = models['cat'].predict_proba(X)[:, 1]
            else:
                predictions['cat'] = models['cat'].predict(X)

        # 加权集成
        ensemble_pred = np.zeros(len(X))
        for algo, pred in predictions.items():
            ensemble_pred += pred * weights.get(algo, 0.0)

        return ensemble_pred

    def get_latest_rate_from_db(self, currency: str, period: int) -> tuple:
        """Query database directly for the most recent rate"""
        import sqlite3
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "data", "lending_history.db"
        )

        conn = sqlite3.connect(db_path)
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

        # Use DB rate if available, fallback to feature data
        if current_rate_db is not None:
            # 修改2.4: Graduated freshness check with confidence degradation
            from datetime import timedelta
            if isinstance(data_timestamp, str):
                db_dt = datetime.strptime(data_timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                db_dt = data_timestamp

            data_age = datetime.now() - db_dt
            data_age_hours = data_age.total_seconds() / 3600

            # fUST long periods (90d/120d) have lower liquidity, relaxed thresholds
            is_low_liquidity = (currency == 'fUST' and period in [90, 120])
            stale_warn_hours = 8 if is_low_liquidity else 4
            stale_error_hours = 12 if is_low_liquidity else 8

            if data_age_hours > stale_error_hours:
                logger.error(
                    f"STALE DATA for {currency}-{period}: "
                    f"data is {data_age_hours:.1f}h old (threshold: {stale_error_hours}h). "
                    f"Refusing to predict."
                )
                raise ValueError(f"Stale DB data for {currency}-{period}: {data_age_hours:.1f}h old")
            elif data_age_hours > stale_warn_hours:
                logger.warning(
                    f"AGING DATA for {currency}-{period}: "
                    f"data is {data_age_hours:.1f}h old (warn threshold: {stale_warn_hours}h). "
                    f"Confidence will be degraded."
                )

            current_rate = current_rate_db
            actual_timestamp = data_timestamp
            logger.debug(f"Using DB rate for {currency}-{period}: {current_rate} (age: {data_age})")
        else:
            # 验证特征数据时间戳新鲜度
            feature_datetime = row_data.get('datetime')
            if feature_datetime:
                from datetime import timedelta
                if isinstance(feature_datetime, str):
                    feature_dt = datetime.strptime(feature_datetime, '%Y-%m-%d %H:%M:%S')
                else:
                    feature_dt = feature_datetime

                # Apply same graduated freshness logic as DB path
                data_age = datetime.now() - feature_dt
                data_age_hours = data_age.total_seconds() / 3600
                is_low_liquidity = (currency == 'fUST' and period in [90, 120])
                stale_warn_hours = 8 if is_low_liquidity else 4
                stale_error_hours = 12 if is_low_liquidity else 8

                if data_age_hours > stale_error_hours:
                    logger.error(
                        f"STALE DATA for {currency}-{period}: "
                        f"DB query failed AND feature data is {data_age_hours:.1f}h old. "
                        f"Refusing to predict."
                    )
                    raise ValueError(f"Stale data for {currency}-{period}: {data_age_hours:.1f}h old")
                elif data_age_hours > stale_warn_hours:
                    logger.warning(
                        f"AGING feature data for {currency}-{period}: {data_age_hours:.1f}h old"
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

            # Ensure data_age_hours is set for confidence degradation below
            # (already set in both DB and feature branches above)

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

        # ========== 校准执行概率 (Calibrated Probability) ==========
        # 温和校准: 50%原始概率 + 50%历史校准, 避免过度压低
        exec_rate_7d = row_data.get('exec_rate_7d', 0.6)
        raw_calibrated = prob * (exec_rate_7d / 0.7)
        calibrated_prob = 0.5 * prob + 0.5 * raw_calibrated  # 混合校准,保留模型信心
        calibrated_prob = np.clip(calibrated_prob, 0.0, 1.0)

        # 根据校准后概率动态选择策略 (收益率优先)
        if calibrated_prob < 0.40:
            # 仅在校准概率极低时才走保守
            base_rate = p_cons * 0.6 + p_bal * 0.4
            strategy_desc = "High Certainty (Low Calibrated Prob)"
            confidence = "Low"
        elif calibrated_prob > 0.75:
            # 高概率时偏激进,最大化收益
            base_rate = p_aggr * 0.6 + p_bal * 0.3 + p_cons * 0.1
            strategy_desc = "High Yield (High Prob)"
            confidence = "High"
        else:
            base_rate = p_bal
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

        # ========== 执行反馈调整 (Execution Feedback Adjustment) ==========
        # exec_rate_7d already fetched above for calibration
        exec_rate_30d = row_data.get('exec_rate_30d', 0.6)
        avg_rate_gap = row_data.get('avg_rate_gap_failed_7d', 0.0)
        risk_adjustment = row_data.get('risk_adjustment_factor', 1.0)

        # 计算执行调整系数
        execution_adjustment = self._calculate_execution_adjustment(
            exec_rate_7d, exec_rate_30d, avg_rate_gap, base_rate
        )

        # ⭐ 新增:币种差异化因子
        currency_factor = self._get_currency_adjustment_factor(currency, period, exec_rate_7d)

        # 应用调整 (执行反馈 × 风险因子 × 币种因子)
        adjusted_rate = base_rate * execution_adjustment * risk_adjustment * currency_factor

        # S1: 融合 v2 revenue_optimized 预测 (如果可用)
        if v2_revenue_rate is not None and v2_revenue_rate > 0:
            # 加权混合: 70% 传统调整后 + 30% v2 收益优化
            adjusted_rate = 0.7 * adjusted_rate + 0.3 * v2_revenue_rate
            logger.debug(f"v2 revenue blend for {currency}-{period}: adjusted={adjusted_rate:.4f}")
        # ====================================================================

        # 趋势修正 (降低趋势影响权重从0.15到0.10)
        trend = float(row_data.get('rate_chg_60', 0))
        trend_factor = np.clip(trend / 5.0, -1.0, 1.0)
        trend_adjustment = trend_factor * 0.10 * adjusted_rate
        final_rate = adjusted_rate + trend_adjustment

        # B4 FIX: Check for NaN in final_rate before clipping
        if np.isnan(final_rate):
            logger.error(f"NaN detected in final_rate for {currency}-{period}, skipping prediction")
            raise ValueError(f"NaN in final_rate for {currency}-{period}")

        # 动态安全边界: 根据执行概率调整激进程度 (修改2.2: 收紧bounds)
        # 高概率(>0.8): 宽边界 - 信任模型预测
        # 中等概率(0.5-0.8): 收紧边界 - 适度控制 (0.45x-1.2x → 0.50x-1.10x)
        # 低概率(<0.5): 窄边界 - 保守策略 (0.5x-1.05x → 0.55x-1.0x)
        if prob > 0.8:
            # 高执行概率 = 激进策略
            min_bound = max(current_rate * 0.4, 0.01)
            max_bound = current_rate * 1.5
            strategy_label = "aggressive"
        elif prob > 0.5:
            # 中等概率 = 收紧平衡策略
            min_bound = max(current_rate * 0.50, 0.01)
            max_bound = current_rate * 1.10
            strategy_label = "balanced"
        else:
            # 低概率 = 收紧保守策略
            min_bound = max(current_rate * 0.55, 0.01)
            max_bound = current_rate * 1.0
            strategy_label = "conservative"

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

        return {
            "currency": currency,
            "period": int(row_data['period']),
            "current_rate": current_rate,
            "predicted_rate": float(final_rate),
            "execution_probability": prob,
            "conservative_rate": p_cons,
            "aggressive_rate": p_aggr,
            "balanced_rate": p_bal,
            "trend_factor": trend,
            "strategy": strategy_desc,
            "confidence": confidence,
            "data_timestamp": actual_timestamp,  # Actual timestamp of rate data
            "prediction_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # When prediction was made
            # 新增调试信息
            "execution_rate_7d": exec_rate_7d,
            "execution_adjustment_applied": execution_adjustment,
            # v2 model info (S1)
            "v2_execution_prob": v2_execution_prob,
            "v2_revenue_rate": v2_revenue_rate,
            # Rate clipping 元数据
            "was_clipped": was_clipped,
            "clipping_strategy": strategy_label,
            "clipping_bounds": {"min": float(min_bound), "max": float(max_bound)}
        }

    def _calculate_execution_adjustment(self, exec_rate_7d: float, exec_rate_30d: float,
                                         avg_gap: float, base_rate: float) -> float:
        """
        根据成交历史计算利率调整系数 - v3 收益率优先版

        调整逻辑(温和版 - 避免过度压低利率):
        - 成交率 < 10%: 降低15% (极端低成交,仅温和调整)
        - 成交率 < 30%: 降低10%
        - 成交率 < 50%: 降低5%
        - 成交率 50-70%: 基本不调整
        - 成交率 > 70%: 适度提高,鼓励更高收益
        - 成交率 > 90%: 提高8%

        Args:
            exec_rate_7d: 7日成交率
            exec_rate_30d: 30日成交率
            avg_gap: 失败订单平均利率差距
            base_rate: 基础预测利率

        Returns:
            调整系数 (0.82-1.18)
        """
        # 基础调整 - 温和版,收益率优先
        if exec_rate_7d < 0.1:
            adjustment = 0.85      # 极端低成交,温和降低
        elif exec_rate_7d < 0.3:
            adjustment = 0.90      # 低成交,适度降低
        elif exec_rate_7d < 0.5:
            adjustment = 0.95      # 中低成交,微调
        elif exec_rate_7d < 0.7:
            adjustment = 1.0       # 正常区间,不调整
        elif exec_rate_7d > 0.9:
            adjustment = 1.08      # 高成交,鼓励提高利率
        elif exec_rate_7d > 0.7:
            adjustment = 1.04      # 较高成交,适度提高
        else:
            adjustment = 1.0

        # 利率差距惩罚 (温和版,上限10%)
        if avg_gap > 0:
            gap_penalty = min(avg_gap / (base_rate + 1e-8), 0.10)
            adjustment *= (1.0 - gap_penalty)

        # 趋势因素
        exec_trend = exec_rate_7d / (exec_rate_30d + 1e-8)
        if exec_trend < 0.8:  # 成交率显著恶化
            adjustment *= 0.98
        elif exec_trend > 1.2:  # 成交率改善
            adjustment *= 1.02

        # 安全边界 (收窄范围,避免极端调整)
        adjustment = np.clip(adjustment, 0.82, 1.18)

        return adjustment

    def _get_currency_adjustment_factor(self, currency: str, period: int, exec_rate: float) -> float:
        """
        币种和周期差异化调整因子 - v3 收益率优先版

        策略:
        - 基于exec_rate与目标(0.50)的偏差,使用连续函数
        - 高于目标: 线性放大至+12% (鼓励高执行率组合追求更高收益)
        - 低于目标: 线性缩减至-12% (温和惩罚,不过度压低)
        - 长周期奖励: period >= 60d 额外+5% (鼓励长周期)
        - 最终clamp到[0.88, 1.18]

        Args:
            currency: 币种 (fUSD/fUST)
            period: 期限 (天)
            exec_rate: 执行成交率

        Returns:
            调整因子 (0.88-1.18)
        """
        target_exec_rate = 0.50
        deviation = exec_rate - target_exec_rate

        if deviation >= 0:
            # Above target: scale up to +12% (at exec_rate=1.0, deviation=0.50 -> +12%)
            adjustment = 1.0 + (deviation / 0.50) * 0.12
        else:
            # Below target: scale down to -12% (at exec_rate=0.0, deviation=-0.50 -> -12%)
            adjustment = 1.0 + (deviation / 0.50) * 0.12

        # Long period bonus: +5% for period >= 60d (不依赖exec_rate,鼓励长周期)
        if period >= 60:
            adjustment += 0.05

        # Clamp to safe range (比之前收窄,避免极端调整)
        adjustment = np.clip(adjustment, 0.88, 1.18)

        return adjustment

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
        preds = self.get_latest_predictions()
        if not preds:
            logger.warning("No predictions generated.")
            return

        # 排序逻辑：
        # 1. 优先考虑成交概率 >= 0.5 的选项
        # 2. 在成交概率合格的基础上，按预测利率降序排列
        valid_preds = [p for p in preds if p['predicted_rate'] > 1.0]

        # Revenue-optimized scoring: 收益率优先,兼顾成交和长周期
        def calculate_weighted_score(pred):
            """
            收益率优先评分 (v3):
            - 30% raw_rate (高利率偏好,直接反映收益能力)
            - 25% effective_rate (rate * prob - 期望收益)
            - 20% exec_prob (成交保障)
            - 25% revenue_factor (period_days/120 - 长周期收益)
            + execution floor: exec_prob < 0.35 gets 0.4x penalty (仅极低概率才惩罚)
            """
            prob = pred['execution_probability']
            rate = pred['predicted_rate']
            period = pred['period']

            # 1. 标准化利率到0-1范围 (假设最高利率40%)
            normalized_rate = min(rate / 40.0, 1.0)

            # 2. Effective rate = rate * prob (期望收益)
            effective_rate = normalized_rate * prob

            # 3. Revenue factor based on period (continuous, not tiered)
            revenue_factor = min(period / 120.0, 1.0)

            # 4. Execution floor: 仅在极低概率时惩罚
            exec_floor_multiplier = 0.4 if prob < 0.35 else 1.0

            # 5. 最终分数 = 30%原始利率 + 25%有效利率 + 20%执行概率 + 25%周期收益
            final_score = (
                normalized_rate * 0.30 +
                effective_rate * 0.25 +
                prob * 0.20 +
                revenue_factor * 0.25
            ) * exec_floor_multiplier

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
            "recommendations": []
        }

        # Build recommendations: ensure top 5 + fUSD-2d if not already included
        recommendations_to_add = []
        fusd_2d_pred = None

        # First, add top 5 predictions
        for pred in sorted_preds[:5]:
            recommendations_to_add.append(pred)
            if pred['currency'] == 'fUSD' and pred['period'] == 2:
                fusd_2d_pred = pred

        # If fUSD-2d is not in top 5, find it and add as 6th
        if fusd_2d_pred is None:
            for pred in sorted_preds:
                if pred['currency'] == 'fUSD' and pred['period'] == 2:
                    fusd_2d_pred = pred
                    recommendations_to_add.append(pred)
                    break

        # If still not found (shouldn't happen), add 6th best instead
        if fusd_2d_pred is None and len(sorted_preds) > 5:
            recommendations_to_add.append(sorted_preds[5])

        # Build the recommendations list
        for i, pred in enumerate(recommendations_to_add[:6]):
            result["recommendations"].append({
                "rank": i + 1,
                "type": "optimal" if i == 0 else "alternative",
                "currency": pred['currency'],
                "period": pred['period'],
                "rate": round(pred['predicted_rate'], 4),
                "confidence": pred['confidence'],
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
                # Stratified sampling configuration by period tiers - 均衡采样
                # 目标是覆盖全期限，各层级数量均衡
                # 周期层级配置
                # 注意: 交易所不存在周期8-9
                # 可用周期: [2,3,4,5,6,7, 10,14,15,20,30, 60,90,120]
                TIER_CONFIG = {
                    'short': {'periods': [2, 3, 4, 5, 6, 7], 'min_orders': 7, 'target_per_period': 1},
                    'medium': {'periods': [10, 14, 15, 20, 30], 'min_orders': 7, 'target_per_period': 1},
                    'long': {'periods': [60, 90, 120], 'min_orders': 6, 'target_per_period': 2}
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

                # Create virtual orders
                created_orders = []
                for pred in selected_preds[:TOP_N]:
                    order_id = self.order_manager.create_virtual_order(pred)
                    created_orders.append({
                        'order_id': order_id,
                        'currency': pred['currency'],
                        'period': pred['period'],
                        'is_cold_start': pred.get('is_cold_start', False),
                        'order_count': pred.get('order_count', 0),
                        'tier': pred.get('tier', 'unknown')
                    })

                # Print summary
                tier_counts = {'short': 0, 'medium': 0, 'long': 0}
                cold_count = 0
                for order in created_orders:
                    tier_counts[order['tier']] += 1
                    if order['is_cold_start']:
                        cold_count += 1

                print(f"\n=== Virtual Order Creation Summary ===")
                print(f"Total orders created: {len(created_orders)}")
                print(f"Tier distribution:")
                print(f"  - Short (2-7d): {tier_counts['short']} orders")
                print(f"  - Medium (10-30d): {tier_counts['medium']} orders")
                print(f"  - Long (60-120d): {tier_counts['long']} orders")
                print(f"Cold start combinations: {cold_count}")
                print(f"Sampling strategy: Stratified by Period Tiers")

                # Print details
                for order in created_orders:
                    status = "COLD START" if order['is_cold_start'] else f"WARM ({order['order_count']} orders)"
                    print(f"  {order['currency']}-{order['period']}d [{order['tier']}]: {status}")

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
    predictor = EnsemblePredictor()
    predictor.generate_recommendations()
