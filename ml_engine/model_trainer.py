import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import os
from loguru import logger
import json
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class EnsembleModelTrainer:
    def __init__(self, data_dir="../data/processed", model_dir="../data/models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # XGBoost GPU 参数配置
        self.xgb_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'learning_rate': 0.03,
            'max_depth': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': 24,
        }

        # LightGBM CPU 参数配置 (使用CPU多线程 - GPU需要OpenCL支持)
        self.lgb_params = {
            'device': 'cpu',
            'learning_rate': 0.03,
            'max_depth': 10,
            'num_leaves': 127,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': 24,
            'verbose': -1
        }

        # CatBoost GPU 参数配置 (使用GPU加速)
        self.catboost_params = {
            'task_type': 'GPU',
            'devices': '0',
            'learning_rate': 0.03,
            'depth': 10,
            'bootstrap_type': 'Bernoulli',  # GPU模式下使用Bernoulli支持subsample
            'subsample': 0.8,
            'thread_count': 24,
            'verbose': False,
            'allow_writing_files': False
        }

    def load_data(self, currency):
        path = os.path.join(self.data_dir, f"{currency}_features.parquet")
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            return None
        return pd.read_parquet(path)

    def prepare_features(self, df: pd.DataFrame):
        """准备特征和排除列"""
        exclude_cols = [
            'currency', 'timestamp', 'datetime', 'year_month', 'candle_size',
            'future_conservative', 'future_aggressive', 'future_balanced', 'future_execution_prob',
            'period'  # period作为标识列，不参与训练
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols

    def train_xgboost_regression(self, X_train, y_train, X_val, y_val):
        """训练XGBoost回归模型"""
        params = self.xgb_params.copy()
        params['objective'] = 'reg:absoluteerror'
        params['eval_metric'] = 'mae'

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        pred_val = model.predict(dval)
        mae = mean_absolute_error(y_val, pred_val)
        return model, mae

    def train_xgboost_classification(self, X_train, y_train, X_val, y_val):
        """训练XGBoost分类模型"""
        params = self.xgb_params.copy()
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        pred_val = model.predict(dval)
        auc = roc_auc_score(y_val, pred_val)
        return model, auc

    def train_lightgbm_regression(self, X_train, y_train, X_val, y_val):
        """训练LightGBM回归模型"""
        params = self.lgb_params.copy()
        params['objective'] = 'mae'
        params['metric'] = 'mae'

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)
        return model, mae

    def train_lightgbm_classification(self, X_train, y_train, X_val, y_val):
        """训练LightGBM分类模型"""
        params = self.lgb_params.copy()
        params['objective'] = 'binary'
        params['metric'] = 'auc'

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )

        pred_val = model.predict(X_val)
        auc = roc_auc_score(y_val, pred_val)
        return model, auc

    def train_catboost_regression(self, X_train, y_train, X_val, y_val):
        """训练CatBoost回归模型"""
        params = self.catboost_params.copy()
        params['loss_function'] = 'MAE'

        model = CatBoostRegressor(**params, iterations=2000, early_stopping_rounds=50)

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)
        return model, mae

    def train_catboost_classification(self, X_train, y_train, X_val, y_val):
        """训练CatBoost分类模型"""
        params = self.catboost_params.copy()
        params['loss_function'] = 'Logloss'

        model = CatBoostClassifier(**params, iterations=2000, early_stopping_rounds=50)

        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        pred_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred_val)
        return model, auc

    def train_ensemble_for_target(self, currency: str, df: pd.DataFrame, target_name: str,
                                  task_type: str, output_prefix: str):
        """
        为单个目标训练集成模型

        Args:
            currency: 币种
            df: 数据框
            target_name: 目标列名
            task_type: 'regression' or 'classification'
            output_prefix: 输出文件前缀 (如 'model_conservative')
        """
        logger.info(f"Training {output_prefix} for {currency} (Target: {target_name}, Type: {task_type})...")

        feature_cols = self.prepare_features(df)
        X = df[feature_cols]
        y = df[target_name]

        # 时间序列划分 (90/10)
        split_idx = int(len(df) * 0.9)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        models = {}
        scores = {}

        # 训练三个模型
        if task_type == 'regression':
            # XGBoost
            models['xgb'], scores['xgb'] = self.train_xgboost_regression(X_train, y_train, X_val, y_val)
            logger.info(f"[{output_prefix}] XGBoost MAE: {scores['xgb']:.6f}")

            # LightGBM
            models['lgb'], scores['lgb'] = self.train_lightgbm_regression(X_train, y_train, X_val, y_val)
            logger.info(f"[{output_prefix}] LightGBM MAE: {scores['lgb']:.6f}")

            # CatBoost
            models['cat'], scores['cat'] = self.train_catboost_regression(X_train, y_train, X_val, y_val)
            logger.info(f"[{output_prefix}] CatBoost MAE: {scores['cat']:.6f}")

        else:  # classification
            # XGBoost
            models['xgb'], scores['xgb'] = self.train_xgboost_classification(X_train, y_train, X_val, y_val)
            logger.info(f"[{output_prefix}] XGBoost AUC: {scores['xgb']:.6f}")

            # LightGBM
            models['lgb'], scores['lgb'] = self.train_lightgbm_classification(X_train, y_train, X_val, y_val)
            logger.info(f"[{output_prefix}] LightGBM AUC: {scores['lgb']:.6f}")

            # CatBoost
            models['cat'], scores['cat'] = self.train_catboost_classification(X_train, y_train, X_val, y_val)
            logger.info(f"[{output_prefix}] CatBoost AUC: {scores['cat']:.6f}")

        # 计算集成权重 (基于验证集表现)
        # 对于回归：权重 = 1/MAE，对于分类：权重 = AUC
        if task_type == 'regression':
            weights = {k: 1.0 / v for k, v in scores.items()}
        else:
            weights = scores.copy()

        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        logger.info(f"[{output_prefix}] Ensemble weights: {normalized_weights}")

        # 保存模型
        self.save_ensemble_models(currency, output_prefix, models, normalized_weights, feature_cols, task_type)

        return models, normalized_weights

    def save_ensemble_models(self, currency: str, prefix: str, models: dict, weights: dict,
                            feature_cols: list, task_type: str):
        """保存集成模型的所有组件"""
        # 保存XGBoost
        models['xgb'].save_model(os.path.join(self.model_dir, f"{currency}_{prefix}_xgb.json"))

        # 保存LightGBM
        models['lgb'].save_model(os.path.join(self.model_dir, f"{currency}_{prefix}_lgb.txt"))

        # 保存CatBoost
        models['cat'].save_model(os.path.join(self.model_dir, f"{currency}_{prefix}_cat.cbm"))

        # 保存元信息
        meta_info = {
            'weights': weights,
            'feature_cols': feature_cols,
            'task_type': task_type
        }
        meta_path = os.path.join(self.model_dir, f"{currency}_{prefix}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)

        logger.info(f"Saved ensemble models and meta info to {self.model_dir}")

    def run_training(self):
        """运行完整的训练流程"""
        for curr in ['fUSD', 'fUST']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {curr}")
            logger.info(f"{'='*60}\n")

            df = self.load_data(curr)
            if df is None:
                continue

            # 1. 训练成交概率模型 (二分类)
            self.train_ensemble_for_target(
                curr, df, 'future_execution_prob', 'classification', 'model_execution_prob'
            )

            # 2. 训练保守模型 (回归)
            self.train_ensemble_for_target(
                curr, df, 'future_conservative', 'regression', 'model_conservative'
            )

            # 3. 训练激进模型 (回归)
            self.train_ensemble_for_target(
                curr, df, 'future_aggressive', 'regression', 'model_aggressive'
            )

            # 4. 训练平衡模型 (回归)
            self.train_ensemble_for_target(
                curr, df, 'future_balanced', 'regression', 'model_balanced'
            )

            logger.info(f"Completed training for {curr}\n")

if __name__ == "__main__":
    trainer = EnsembleModelTrainer()
    trainer.run_training()
