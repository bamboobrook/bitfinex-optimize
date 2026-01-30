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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.data_processor import DataProcessor

class EnsemblePredictor:
    """使用集成模型的增强预测器 - 支持GPU加速和并行化"""

    def __init__(self, model_dir=None, max_workers=8):
        # Use absolute path for models directory
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "data", "models")
        self.model_dir = model_dir
        self.max_workers = max_workers  # 并行worker数量
        self.models = {}  # {curr: {model_type: {algo: model}}}
        self.meta_info = {}  # {curr: {model_type: {weights, features, task_type}}}
        self.processor = DataProcessor()

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
                meta = json.load(f)

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
        """加载所有币种的所有模型"""
        for curr in ['fUSD', 'fUST']:
            self.models[curr] = {}
            self.meta_info[curr] = {}

            for model_type in ['model_execution_prob', 'model_conservative',
                              'model_aggressive', 'model_balanced']:
                models, meta = self.load_ensemble_models(curr, model_type)

                if models and meta:
                    self.models[curr][model_type] = models
                    self.meta_info[curr][model_type] = meta
                    logger.info(f"Loaded {model_type} for {curr}")
                else:
                    logger.warning(f"Failed to load {model_type} for {curr}")

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
            return np.array([])

        models = self.models[currency][model_type]
        meta = self.meta_info[currency][model_type]
        weights = meta['weights']

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
        # 构造特征DataFrame
        X_single = pd.DataFrame([{col: row_data[col] for col in feature_cols}])

        # 执行4个模型预测
        pred_execution_prob = self.predict_with_ensemble(X_single, currency, 'model_execution_prob')[0]
        pred_conservative = self.predict_with_ensemble(X_single, currency, 'model_conservative')[0]
        pred_aggressive = self.predict_with_ensemble(X_single, currency, 'model_aggressive')[0]
        pred_balanced = self.predict_with_ensemble(X_single, currency, 'model_balanced')[0]

        # 智能定价策略
        prob = float(pred_execution_prob)
        p_cons = float(pred_conservative)
        p_aggr = float(pred_aggressive)
        p_bal = float(pred_balanced)

        # 根据成交概率动态选择策略
        if prob < 0.5:
            base_rate = p_cons
            strategy_desc = "High Certainty (Low Prob)"
            confidence = "Low"
        elif prob > 0.8:
            base_rate = p_aggr * 0.7 + p_bal * 0.3
            strategy_desc = "High Yield (High Prob)"
            confidence = "High"
        else:
            base_rate = p_bal
            strategy_desc = "Balanced"
            confidence = "Medium"

        # ========== 执行反馈调整 (Execution Feedback Adjustment) ==========
        exec_rate_7d = row_data.get('exec_rate_7d', 0.7)
        exec_rate_30d = row_data.get('exec_rate_30d', 0.7)
        avg_rate_gap = row_data.get('avg_rate_gap_failed_7d', 0.0)
        risk_adjustment = row_data.get('risk_adjustment_factor', 1.0)

        # 计算执行调整系数
        execution_adjustment = self._calculate_execution_adjustment(
            exec_rate_7d, exec_rate_30d, avg_rate_gap, base_rate
        )

        # 应用调整
        adjusted_rate = base_rate * execution_adjustment * risk_adjustment
        # ====================================================================

        # 趋势修正 (降低趋势影响权重从0.15到0.10)
        trend = float(row_data.get('rate_chg_60', 0))
        trend_factor = np.clip(trend / 5.0, -1.0, 1.0)
        trend_adjustment = trend_factor * 0.10 * adjusted_rate
        final_rate = adjusted_rate + trend_adjustment

        # 安全边界检查 (更严格: 降低上限从1.1到1.05)
        current_rate = float(row_data['close_annual'])
        final_rate = np.clip(final_rate, current_rate * 0.50, current_rate * 1.05)

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
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Use current time, not historical data time
            # 新增调试信息
            "execution_rate_7d": exec_rate_7d,
            "execution_adjustment_applied": execution_adjustment
        }

    def _calculate_execution_adjustment(self, exec_rate_7d: float, exec_rate_30d: float,
                                         avg_gap: float, base_rate: float) -> float:
        """
        根据成交历史计算利率调整系数

        调整逻辑:
        - 成交率 < 50%: 降低10%
        - 成交率 50-60%: 降低5%
        - 成交率 60-70%: 降低2%
        - 成交率 70-90%: 不调整
        - 成交率 > 90%: 可提高1% (弱化马太效应)

        额外考虑:
        - 失败订单平均利率差距大,进一步降低
        - 成交率趋势恶化,额外降低3%

        Args:
            exec_rate_7d: 7日成交率
            exec_rate_30d: 30日成交率
            avg_gap: 失败订单平均利率差距
            base_rate: 基础预测利率

        Returns:
            调整系数 (0.85-1.05)
        """
        # 基础调整
        if exec_rate_7d < 0.5:
            adjustment = 0.90
        elif exec_rate_7d < 0.6:
            adjustment = 0.95
        elif exec_rate_7d < 0.7:
            adjustment = 0.98
        elif exec_rate_7d > 0.90:  # Raise threshold from 0.85 to 0.90
            adjustment = 1.01       # Lower reward from 1.02 to 1.01
        else:
            adjustment = 1.0

        # 利率差距惩罚
        if avg_gap > 0:
            gap_penalty = min(avg_gap / (base_rate + 1e-8), 0.08)  # 最多降低8%
            adjustment *= (1.0 - gap_penalty)

        # 趋势因素
        exec_trend = exec_rate_7d / (exec_rate_30d + 1e-8)
        if exec_trend < 0.8:  # 成交率显著恶化
            adjustment *= 0.97
        elif exec_trend > 1.2:  # 成交率改善
            adjustment *= 1.01

        # 安全边界
        adjustment = np.clip(adjustment, 0.85, 1.05)

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

        return all_predictions

    def generate_recommendations(self, output_path="../data/optimal_combination.json"):
        """生成推荐结果并保存 - 增强版含虚拟订单创建"""
        preds = self.get_latest_predictions()
        if not preds:
            logger.warning("No predictions generated.")
            return

        # 排序逻辑：
        # 1. 优先考虑成交概率 >= 0.5 的选项
        # 2. 在成交概率合格的基础上，按预测利率降序排列
        valid_preds = [p for p in preds if p['predicted_rate'] > 1.0]

        # 按成交概率和预测利率综合排序
        sorted_preds = sorted(
            valid_preds,
            key=lambda x: (x['execution_probability'] * x['predicted_rate']),
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

        # ========== 创建虚拟订单 with Stratified Sampling (Create Virtual Orders) ==========
        if self.order_manager is not None:
            try:
                # Sampling configuration
                TOP_N = 10                 # Total virtual orders to create
                COLD_START_SLOTS = 3       # Reserve 3 slots for cold start combinations
                COLD_START_THRESHOLD = 10  # Combinations with <10 orders are cold start

                # Identify cold start vs warm combinations
                cold_start_preds = []
                warm_preds = []

                for pred in sorted_preds:
                    order_count = self.order_manager.get_order_count(
                        pred['currency'],
                        pred['period']
                    )

                    if order_count < COLD_START_THRESHOLD:
                        pred['is_cold_start'] = True
                        pred['order_count'] = order_count
                        cold_start_preds.append(pred)
                    else:
                        pred['is_cold_start'] = False
                        pred['order_count'] = order_count
                        warm_preds.append(pred)

                # Stratified sampling: allocate slots
                selected_preds = []

                # 1. Select top warm predictions (exploiting known good combinations)
                warm_slots = TOP_N - COLD_START_SLOTS
                selected_preds.extend(warm_preds[:warm_slots])

                # 2. Select cold start predictions (exploring new combinations)
                selected_preds.extend(cold_start_preds[:COLD_START_SLOTS])

                # 3. Fill remaining slots if cold start pool is insufficient
                remaining = TOP_N - len(selected_preds)
                if remaining > 0:
                    # Use remaining warm predictions
                    selected_preds.extend(warm_preds[warm_slots:warm_slots + remaining])

                # Create virtual orders
                created_orders = []
                for pred in selected_preds[:TOP_N]:
                    order_id = self.order_manager.create_virtual_order(pred)
                    created_orders.append({
                        'order_id': order_id,
                        'currency': pred['currency'],
                        'period': pred['period'],
                        'is_cold_start': pred.get('is_cold_start', False),
                        'order_count': pred.get('order_count', 0)
                    })

                # Print summary
                cold_count = sum(1 for o in created_orders if o['is_cold_start'])
                warm_count = len(created_orders) - cold_count

                print(f"\n=== Virtual Order Creation Summary ===")
                print(f"Total orders created: {len(created_orders)}")
                print(f"  - Warm combinations (exploiting): {warm_count}")
                print(f"  - Cold start combinations (exploring): {cold_count}")
                print(f"Sampling strategy: Stratified (70% exploit + 30% explore)")

                # Print details
                for order in created_orders:
                    status = "COLD START" if order['is_cold_start'] else f"WARM ({order['order_count']} orders)"
                    print(f"  {order['currency']}-{order['period']}d: {status}")

            except Exception as e:
                print(f"Failed to create virtual orders: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: OrderManager not available, skipping virtual order creation")
        # ============================================================

if __name__ == "__main__":
    predictor = EnsemblePredictor()
    predictor.generate_recommendations()
