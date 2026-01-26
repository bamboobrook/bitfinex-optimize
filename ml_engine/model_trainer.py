import xgboost as xgb
import pandas as pd
import numpy as np
import os
from loguru import logger
import json
from sklearn.metrics import mean_absolute_error

class ModelTrainer:
    def __init__(self, data_dir="../data/processed", model_dir="../data/models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # GPU 参数配置
        self.params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'learning_rate': 0.03,
            'max_depth': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': 24,
            'objective': 'reg:absoluteerror', 
            'eval_metric': 'mae'
        }

    def load_data(self, currency):
        path = os.path.join(self.data_dir, f"{currency}_features.parquet")
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            return None
        return pd.read_parquet(path)

    def train_single_target(self, currency: str, df: pd.DataFrame, target_name: str, output_name: str):
        """训练单个目标模型"""
        logger.info(f"Training {output_name} for {currency} (Target: {target_name})...")
        
        # 排除列
        exclude_cols = ['currency', 'timestamp', 'datetime', 'year_month', 'candle_size', 
                       'future_min_120', 'future_conservative', 'future_aggressive']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_name]
        
        # Split
        split_idx = int(len(df) * 0.9)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        mae = mean_absolute_error(y_val, model.predict(dval))
        logger.info(f"[{output_name}] Validation MAE: {mae:.6f}")
        
        model.save_model(os.path.join(self.model_dir, f"{currency}_{output_name}.json"))
        
        # 保存特征 (只保存一次即可)
        feat_path = os.path.join(self.model_dir, f"{currency}_features.json")
        if not os.path.exists(feat_path):
            with open(feat_path, 'w') as f:
                json.dump(feature_cols, f)

    def run_training(self):
        for curr in ['fUSD', 'fUST']:
            df = self.load_data(curr)
            if df is None: continue
            
            # 1. 训练保守模型 (High Certainty)
            self.train_single_target(curr, df, 'future_conservative', 'model_conservative')
            
            # 2. 训练激进模型 (High Yield)
            self.train_single_target(curr, df, 'future_aggressive', 'model_aggressive')

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training()