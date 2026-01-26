import xgboost as xgb
import pandas as pd
import numpy as np
import os
from loguru import logger
import json
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelTrainer:
    def __init__(self, data_dir="data/processed", model_dir="data/models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # GPU 参数配置 (专为 RTX 5090 优化)
        self.params = {
            'objective': 'reg:absoluteerror',  # 使用绝对误差，减少异常值影响
            'tree_method': 'hist',            # 使用直方图算法
            'device': 'cuda',                 # 启用 GPU
            'learning_rate': 0.03,            # 较小的学习率
            'max_depth': 10,                  # 较深的模型以捕捉复杂模式
            'subsample': 0.8,                 # 样本采样
            'colsample_bytree': 0.8,          # 特征采样
            'eval_metric': ['mae', 'rmse'],
            'n_jobs': 24,                      # CPU 线程数辅助数据处理
        }

    def load_data(self, currency):
        path = os.path.join(self.data_dir, f"{currency}_features.parquet")
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            return None
        return pd.read_parquet(path)

    def train(self, currency: str, target_col: str = 'future_q20_120'):
        logger.info(f"Starting training for {currency} aiming at {target_col}...")
        
        df = self.load_data(currency)
        if df is None:
            return

        # 准备特征 (排除非数值列和Target列)
        exclude_cols = ['currency', 'timestamp', 'datetime', 'year_month', 'candle_size', 
                       'future_min_120', 'future_q20_120', 'future_mean_120']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # 严格的时间序列切分 (Time Series Split)
        # 最后 10% 数据作为验证集 (Validation Set)
        split_idx = int(len(df) * 0.9)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        logger.info(f"Using GPU for training...")

        # 转换为 DMatrix (XGBoost 专用格式)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 训练模型
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=5000,          # 最大轮数
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,     # 早停机制
            verbose_eval=100               # 每100轮打印一次
        )
        
        # 评估
        preds = model.predict(dval)
        mae = mean_absolute_error(y_val, preds)
        logger.info(f"Validation MAE: {mae:.6f}")
        
        # 保存模型
        model_path = os.path.join(self.model_dir, f"{currency}_model.json")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # 保存特征列表 (预测时需要保证顺序一致)
        feature_path = os.path.join(self.model_dir, f"{currency}_features.json")
        with open(feature_path, 'w') as f:
            json.dump(feature_cols, f)
            
        return model

if __name__ == "__main__":
    trainer = ModelTrainer()
    # 针对两个币种分别训练
    for curr in ['fUSD', 'fUST']:
        trainer.train(curr)
