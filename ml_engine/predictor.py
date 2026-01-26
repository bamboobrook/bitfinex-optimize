import pandas as pd
import xgboost as xgb
import json
import os
import sys
from loguru import logger
from datetime import datetime
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 添加父目录到 path 以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_engine.data_processor import DataProcessor

class LendingPredictor:
    def __init__(self, model_dir="../data/models"):
        self.model_dir = model_dir
        self.models = {} # {curr: {'conservative': model, 'aggressive': model}}
        self.features = {}
        self.processor = DataProcessor()
        
        # 加载模型
        for curr in ['fUSD', 'fUST']:
            self.models[curr] = {}
            for m_type in ['model_conservative', 'model_aggressive']:
                model_path = os.path.join(model_dir, f"{curr}_{m_type}.json")
                if os.path.exists(model_path):
                    # key simplified to 'conservative' or 'aggressive'
                    key = m_type.split('_')[1] 
                    self.models[curr][key] = xgb.Booster()
                    self.models[curr][key].load_model(model_path)
            
            feature_path = os.path.join(model_dir, f"{curr}_features.json")
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.features[curr] = json.load(f)
            
            logger.info(f"Loaded models for {curr}: {list(self.models[curr].keys())}")

    def get_latest_predictions(self):
        all_predictions = []
        
        for curr in ['fUSD', 'fUST']:
            if not self.models[curr]:
                continue
                
            logger.info(f"Fetching data for {curr}...")
            df = self.processor.load_data(curr)
            
            if df.empty:
                continue
                
            # 特征工程
            def process_group(group):
                return self.processor.add_technical_indicators(group)

            df_features = df.groupby('period', group_keys=False).apply(process_group)
            
            # 最新��态
            latest_data = df_features.groupby('period').tail(1).copy()
            feature_cols = self.features[curr]
            
            X_latest = latest_data[feature_cols]
            dtest = xgb.DMatrix(X_latest)
            
            # 双模型预测
            pred_cons = self.models[curr]['conservative'].predict(dtest)
            pred_aggr = self.models[curr]['aggressive'].predict(dtest)
            
            # 策略融合计算
            for idx, row in latest_data.iterrows():
                i = latest_data.index.get_loc(idx)
                p_cons = float(pred_cons[i])
                p_aggr = float(pred_aggr[i])
                
                # 动态策略逻辑：
                # 1. 计算动量趋势 (rate_chg_60)
                #    如果最近1小时利率在涨 (trend > 0)，我们更倾向于 aggressive
                #    如果最近1小时利率在跌 (trend < 0)，我们回退到 conservative
                trend = float(row.get('rate_chg_60', 0))
                
                # 归一化 Trend 因子 (假设 +- 5% 年化变化是强趋势)
                trend_factor = np.clip(trend / 5.0, -1.0, 1.0) # -1 to 1
                
                # 基础权重：默认偏激进 (0.6 Aggressive, 0.4 Conservative) 以满足 "收益足够高"
                base_weight_aggr = 0.6
                
                # 根据趋势调整权重
                # 趋势向上 -> 增加 Aggressive 权重
                # 趋势向下 -> 减少 Aggressive 权重
                final_weight_aggr = np.clip(base_weight_aggr + (trend_factor * 0.3), 0.0, 1.0)
                
                # 混合定价
                target_rate = (p_aggr * final_weight_aggr) + (p_cons * (1.0 - final_weight_aggr))
                
                # 兜底逻辑：如果 Aggressive 比 Conservative 还低 (倒挂)，说明市场极度看空
                # 此时直接取最小值，保证成交
                if p_aggr < p_cons:
                    target_rate = p_aggr
                    strategy_desc = "Bearish Escape"
                else:
                    strategy_desc = f"Dynamic (Aggr Weight: {final_weight_aggr:.2f})"
                
                all_predictions.append({
                    "currency": curr,
                    "period": int(row['period']),
                    "current_rate": float(row['close_annual']),
                    "predicted_rate": float(target_rate),
                    "conservative_rate": p_cons,
                    "aggressive_rate": p_aggr,
                    "trend_factor": trend,
                    "strategy": strategy_desc,
                    "timestamp": row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return all_predictions

    def generate_recommendations(self, output_path="../data/optimal_combination.json"):
        preds = self.get_latest_predictions()
        if not preds:
            logger.warning("No predictions generated.")
            return
            
        # 排序逻辑优化：
        # 1. 过滤掉 "过低" 的利率 (比如 < 当前市场价的 50%，或者绝对值太低)
        # 2. 按 "Predicted Rate" (收益) 降序排列 -> 满足 "收益足够高"
        
        # 简单过滤
        valid_preds = [p for p in preds if p['predicted_rate'] > 1.0] # 至少大于 1%
        
        # 排序
        sorted_preds = sorted(valid_preds, key=lambda x: x['predicted_rate'], reverse=True)
        
        result = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "status": 'success',
            "strategy_info": "High Yield Priority with Dynamic Execution Safety",
            "recommendations": []
        }
        
        for i, pred in enumerate(sorted_preds[:5]):
            result["recommendations"].append({
                "rank": i + 1,
                "type": "optimal" if i == 0 else "alternative",
                "currency": pred['currency'],
                "period": pred['period'],
                "rate": round(pred['predicted_rate'], 4),
                "confidence": "Dynamic",
                "details": {
                    "current": round(pred['current_rate'], 4),
                    "conservative_floor": round(pred['conservative_rate'], 4),
                    "aggressive_target": round(pred['aggressive_rate'], 4),
                    "trend_1h": round(pred['trend_factor'], 4)
                }
            })
            
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        logger.info(f"Recommendations saved to {output_path}")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    predictor = LendingPredictor()
    predictor.generate_recommendations()