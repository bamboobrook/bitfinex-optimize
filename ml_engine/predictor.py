import pandas as pd
import xgboost as xgb
import json
import os
import sys
from loguru import logger
from datetime import datetime
import numpy as np

# 添加父目录到 path 以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.data_processor import DataProcessor

class LendingPredictor:
    def __init__(self, model_dir="data/models"):
        self.model_dir = model_dir
        self.models = {}
        self.features = {}
        self.processor = DataProcessor()
        
        # 加载模型
        for curr in ['fUSD', 'fUST']:
            model_path = os.path.join(model_dir, f"{curr}_model.json")
            feature_path = os.path.join(model_dir, f"{curr}_features.json")
            
            if os.path.exists(model_path) and os.path.exists(feature_path):
                self.models[curr] = xgb.Booster()
                self.models[curr].load_model(model_path)
                with open(feature_path, 'r') as f:
                    self.features[curr] = json.load(f)
                logger.info(f"Loaded model for {curr}")
            else:
                logger.warning(f"Model not found for {curr}")

    def get_latest_predictions(self):
        all_predictions = []
        
        for curr in ['fUSD', 'fUST']:
            if curr not in self.models:
                continue
                
            # 1. 获取最近数据 (需要足够长以计算24h lag)
            # 这里的 load_data 会读取全部，为了效率，我们应该在 DataProcessor 加一个 limit
            # 但考虑到用户内存巨大且本地库不算大，复用现有逻辑最稳妥
            logger.info(f"Fetching data for {curr}...")
            df = self.processor.load_data(curr)
            
            if df.empty:
                continue
                
            # 2. 特征工程
            # 我们需要按 Period 分组计算特征
            # 为了防止 Look-ahead bias，我们计算特征时不应该包含未来的 Target 构建逻辑，
            # 只计算 Input Features (Lags, Rolling)
            # 复用 add_technical_indicators 但忽略 Target 部分的 NaN
            
            def process_group(group):
                # 只要过去的数据够长，最后一行就是有效的
                return self.processor.add_technical_indicators(group)

            df_features = df.groupby('period', group_keys=False).apply(process_group)
            
            # 3. 提取每个 Period 的最后一行 (Latest State)
            latest_data = df_features.groupby('period').tail(1).copy()
            
            # 4. 准备预测输入
            # 确保列顺序与训练时一致
            feature_cols = self.features[curr]
            
            # 检查是否有缺失列 (Target列除外)
            missing_cols = [c for c in feature_cols if c not in latest_data.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {curr}: {missing_cols}")
                continue
                
            X_latest = latest_data[feature_cols]
            dtest = xgb.DMatrix(X_latest)
            
            # 5. 预测
            # 注意：模型预测的是 'future_q20_120'
            preds = self.models[curr].predict(dtest)
            
            latest_data['predicted_rate'] = preds
            
            # 6. 收集结果
            for idx, row in latest_data.iterrows():
                # 简单的置信度/评分逻辑
                # Score = 预测利率 * 成交量权重 (如果成交量太低，稍微降分)
                # 这里为了简单直接用预测利率，因为我们已经是 conservative prediction
                
                # 记录原始数据用于展示
                all_predictions.append({
                    "currency": curr,
                    "period": int(row['period']),
                    "current_rate": float(row['close_annual']),
                    "predicted_rate": float(row['predicted_rate']),
                    "volume": float(row['volume']), # 最近一根K线的量
                    "timestamp": row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return all_predictions

    def generate_recommendations(self, output_path="data/optimal_combination.json"):
        preds = self.get_latest_predictions()
        if not preds:
            logger.warning("No predictions generated.")
            return
            
        # 排序逻辑：按预测收益率降序
        sorted_preds = sorted(preds, key=lambda x: x['predicted_rate'], reverse=True)
        
        # 构造输出格式
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "market_analysis": "Based on XGBoost predictions of future 2h conservative execution rates.",
            "recommendations": []
        }
        
        # 最优组合 (Top 1)
        if len(sorted_preds) > 0:
            best = sorted_preds[0]
            result["recommendations"].append({
                "rank": 1,
                "type": "optimal",
                "currency": best['currency'],
                "period": best['period'],
                "rate": round(best['predicted_rate'], 4),
                "confidence": "High", # 既然是 q20，我们认为成交概率高
                "details": f"Current: {best['current_rate']:.4f}%"
            })
            
        # 次优组合 (Top 2-5)
        for i, pred in enumerate(sorted_preds[1:5]):
            result["recommendations"].append({
                "rank": i + 2,
                "type": "alternative",
                "currency": pred['currency'],
                "period": pred['period'],
                "rate": round(pred['predicted_rate'], 4),
                "confidence": "Medium",
                "details": f"Current: {pred['current_rate']:.4f}%"
            })
            
        # 保存
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"Recommendations saved to {output_path}")
        # Print summary to console
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    predictor = LendingPredictor()
    predictor.generate_recommendations()
