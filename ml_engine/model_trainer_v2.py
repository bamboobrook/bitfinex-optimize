"""
增强版模型训练器 - 支持闭环自优化

新特性:
1. 使用 TrainingDataBuilder 融合市场数据 + 虚拟订单执行结果
2. 支持基于实际执行结果的新训练目标
3. 支持收益优化模型训练

作者: 闭环自优化系统
日期: 2026-02-07
"""

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from training_data_builder import TrainingDataBuilder


class EnhancedModelTrainer:
    """
    增强版模型训练器

    支持:
    1. 传统训练目标 (conservative, aggressive, balanced, execution_prob)
    2. 新增执行结果训练目标 (execution_prob_v2)
    3. 新增收益优化目标 (revenue_optimized)
    """

    def __init__(self, db_path: str = 'data/lending_history.db', model_dir: str = None):
        """
        初始化

        Args:
            db_path: 数据库路径
            model_dir: 模型保存目录
        """
        self.db_path = db_path

        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            model_dir = os.path.join(base_dir, "data", "models")

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

        # LightGBM CPU 参数配置
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

        # CatBoost GPU 参数配置
        self.catboost_params = {
            'task_type': 'GPU',
            'devices': '0',
            'learning_rate': 0.03,
            'depth': 10,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'thread_count': 24,
            'verbose': False,
            'allow_writing_files': False
        }

    def prepare_training_data(
        self,
        start_date: str,
        end_date: str,
        use_execution_feedback: bool = True
    ) -> pd.DataFrame:
        """
        准备训练数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            use_execution_feedback: 是否使用执行反馈数据

        Returns:
            训练数据 DataFrame
        """
        print("\n" + "="*60)
        print("📊 准备训练数据")
        print("="*60)

        builder = TrainingDataBuilder(self.db_path)

        df = builder.build_training_data(
            start_date=start_date,
            end_date=end_date,
            include_execution_results=use_execution_feedback
        )

        # 添加传统训练目标
        df = self._add_traditional_targets(df)

        return df

    def _add_traditional_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加传统训练目标

        目标:
        1. future_conservative: 未来30%分位数
        2. future_aggressive: 未来60%分位数
        3. future_balanced: 未来70%分位数
        4. future_execution_prob: 是否<=未来80%分位数 (二分类)

        Args:
            df: 原始数据

        Returns:
            添加目标后的数据
        """
        print("\n添加传统训练目标...")

        # 按 currency + period 分组计算
        df = df.sort_values(['currency', 'period', 'datetime'])

        def compute_targets(group):
            # 使用固定前向窗口 (120小时)
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=120)

            # Target 1: 保守利率 (30%分位数)
            group['future_conservative'] = group['low_annual'].rolling(window=indexer).quantile(0.3)

            # Target 2: 激进利率 (60%分位数)
            group['future_aggressive'] = group['close_annual'].rolling(window=indexer).quantile(0.6)

            # Target 3: 平衡利率 (70%分位数)
            group['future_balanced'] = group['close_annual'].rolling(window=indexer).quantile(0.7)

            # Target 4: 成交概率 (二分类标签)
            future_80pct = group['close_annual'].rolling(window=indexer).quantile(0.8)
            group['future_execution_prob'] = (group['close_annual'] <= future_80pct).astype(int)

            return group

        df = df.groupby(['currency', 'period'], group_keys=False).apply(compute_targets)

        # 清理 NaN
        initial_rows = len(df)
        df = df.dropna(subset=['future_conservative', 'future_aggressive',
                               'future_balanced', 'future_execution_prob'])
        final_rows = len(df)

        print(f"✓ 传统目标添加完成 (保留 {final_rows:,}/{initial_rows:,} 行)")

        return df

    def prepare_features(self, df: pd.DataFrame) -> list:
        """
        准备特征列表

        排除:
        - 标识列: currency, period, timestamp, datetime
        - 目标列: future_*, actual_*, revenue_*, rate_competitiveness
        - 其他元数据

        Args:
            df: 数据框

        Returns:
            特征列列表
        """
        exclude_cols = [
            'currency', 'timestamp', 'datetime', 'period',
            'future_conservative', 'future_aggressive', 'future_balanced', 'future_execution_prob',
            'actual_execution_binary', 'revenue_reward', 'rate_competitiveness',
            'status', 'order_timestamp', 'predicted_rate', 'execution_confidence',
            'total_score', 'market_median', 'execution_rate',
            'revenue_optimized_target',
        ]

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        print(f"\n特征数量: {len(feature_cols)}")
        return feature_cols

    def train_xgboost_regression(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练XGBoost回归模型"""
        params = self.xgb_params.copy()
        params['objective'] = 'reg:absoluteerror'
        params['eval_metric'] = 'mae'

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
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

    def train_xgboost_classification(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练XGBoost分类模型"""
        params = self.xgb_params.copy()
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
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

    def train_lightgbm_regression(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练LightGBM回归模型"""
        params = self.lgb_params.copy()
        params['objective'] = 'mae'
        params['metric'] = 'mae'

        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
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

    def train_lightgbm_classification(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练LightGBM分类模型"""
        params = self.lgb_params.copy()
        params['objective'] = 'binary'
        params['metric'] = 'auc'

        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
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

    def train_catboost_regression(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练CatBoost回归模型"""
        params = self.catboost_params.copy()
        params['loss_function'] = 'MAE'

        model = CatBoostRegressor(**params, iterations=2000, early_stopping_rounds=50)

        train_pool = Pool(X_train, y_train, weight=sample_weight)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred_val)
        return model, mae

    def train_catboost_classification(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练CatBoost分类模型"""
        params = self.catboost_params.copy()
        params['loss_function'] = 'Logloss'

        model = CatBoostClassifier(**params, iterations=2000, early_stopping_rounds=50)

        train_pool = Pool(X_train, y_train, weight=sample_weight)
        val_pool = Pool(X_val, y_val)

        model.fit(train_pool, eval_set=val_pool, verbose=False)

        pred_val = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred_val)
        return model, auc

    def train_single_target(
        self,
        currency: str,
        df: pd.DataFrame,
        target_name: str,
        task_type: str,
        output_prefix: str
    ):
        """
        为单个目标训练集成模型

        Args:
            currency: 币种
            df: 数据框
            target_name: 目标列名
            task_type: 'regression' or 'classification'
            output_prefix: 输出文件前缀
        """
        print(f"\n{'='*60}")
        print(f"训练 {output_prefix} ({currency})")
        print(f"目标: {target_name}, 类型: {task_type}")
        print("="*60)

        # 检查目标列是否存在
        if target_name not in df.columns:
            print(f"⚠️  目标列 '{target_name}' 不存在,跳过")
            return None

        # 过滤有效样本
        valid_df = df[df[target_name].notna()].copy()
        if len(valid_df) < 100:
            print(f"⚠️  有效样本不足 ({len(valid_df)} < 100),跳过")
            return None

        print(f"有效样本: {len(valid_df):,}")

        # S4: 计算时间衰减权重 (半衰期30天)
        sample_weights = None
        if 'datetime' in valid_df.columns:
            try:
                now = pd.Timestamp.now()
                dt_series = pd.to_datetime(valid_df['datetime'])
                days_ago = (now - dt_series).dt.total_seconds() / 86400.0
                sample_weights = np.power(0.5, days_ago / 30.0)  # 30天半衰期
                sample_weights = sample_weights.values
                print(f"时间衰减权重: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
            except Exception as e:
                print(f"⚠️  时间衰减计算失败: {e}, 使用均等权重")
                sample_weights = None

        # 准备特征
        feature_cols = self.prepare_features(valid_df)
        X = valid_df[feature_cols]
        y = valid_df[target_name]

        # 时间序列划分 (90/10)
        split_idx = int(len(valid_df) * 0.9)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # S4: Split sample weights accordingly
        w_train = sample_weights[:split_idx] if sample_weights is not None else None
        w_val = sample_weights[split_idx:] if sample_weights is not None else None

        print(f"训练集: {len(X_train):,}, 验证集: {len(X_val):,}")

        models = {}
        scores = {}

        # 训练三个模型 (传入 sample_weight)
        if task_type == 'regression':
            # XGBoost
            models['xgb'], scores['xgb'] = self.train_xgboost_regression(
                X_train, y_train, X_val, y_val, sample_weight=w_train
            )
            print(f"  XGBoost MAE: {scores['xgb']:.6f}")

            # LightGBM
            models['lgb'], scores['lgb'] = self.train_lightgbm_regression(
                X_train, y_train, X_val, y_val, sample_weight=w_train
            )
            print(f"  LightGBM MAE: {scores['lgb']:.6f}")

            # CatBoost
            models['cat'], scores['cat'] = self.train_catboost_regression(
                X_train, y_train, X_val, y_val, sample_weight=w_train
            )
            print(f"  CatBoost MAE: {scores['cat']:.6f}")

        else:  # classification
            # XGBoost
            models['xgb'], scores['xgb'] = self.train_xgboost_classification(
                X_train, y_train, X_val, y_val, sample_weight=w_train
            )
            print(f"  XGBoost AUC: {scores['xgb']:.6f}")

            # LightGBM
            models['lgb'], scores['lgb'] = self.train_lightgbm_classification(
                X_train, y_train, X_val, y_val, sample_weight=w_train
            )
            print(f"  LightGBM AUC: {scores['lgb']:.6f}")

            # CatBoost
            models['cat'], scores['cat'] = self.train_catboost_classification(
                X_train, y_train, X_val, y_val, sample_weight=w_train
            )
            print(f"  CatBoost AUC: {scores['cat']:.6f}")

        # 计算集成权重
        if task_type == 'regression':
            weights = {k: 1.0 / v for k, v in scores.items()}
        else:
            weights = scores.copy()

        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        print(f"\n集成权重: {normalized_weights}")

        # 保存模型
        self.save_ensemble_models(currency, output_prefix, models, normalized_weights,
                                 feature_cols, task_type)

        return models, normalized_weights

    def save_ensemble_models(
        self,
        currency: str,
        prefix: str,
        models: dict,
        weights: dict,
        feature_cols: list,
        task_type: str
    ):
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

        print(f"✓ 模型已保存到 {self.model_dir}")

    def train_all_models(
        self,
        start_date: str,
        end_date: str,
        use_execution_feedback: bool = True
    ):
        """
        训练所有模型

        流程:
        1. 准备训练数据 (融合市场数据 + 执行结果)
        2. 按币种分别训练
        3. 训练6个目标模型 (传统4个 + 新增2个)

        Args:
            start_date: 开始日期
            end_date: 结束日期
            use_execution_feedback: 是否使用执行反馈
        """
        print("\n" + "="*80)
        print(" "*20 + "🚀 增强版模型训练器启动")
        print("="*80)

        # 准备训练数据
        df = self.prepare_training_data(start_date, end_date, use_execution_feedback)

        # 按币种训练
        for currency in ['fUSD', 'fUST']:
            print("\n" + "#"*80)
            print(f"# 训练 {currency} 模型")
            print("#"*80)

            curr_df = df[df['currency'] == currency].copy()
            print(f"\n{currency} 数据: {len(curr_df):,} 行")

            if len(curr_df) < 100:
                print(f"⚠️  {currency} 数据不足,跳过")
                continue

            # 1. 训练成交概率模型 (传统)
            self.train_single_target(
                currency, curr_df, 'future_execution_prob', 'classification', 'model_execution_prob'
            )

            # 2. 训练保守模型
            self.train_single_target(
                currency, curr_df, 'future_conservative', 'regression', 'model_conservative'
            )

            # 3. 训练激进模型
            self.train_single_target(
                currency, curr_df, 'future_aggressive', 'regression', 'model_aggressive'
            )

            # 4. 训练平衡模型
            self.train_single_target(
                currency, curr_df, 'future_balanced', 'regression', 'model_balanced'
            )

            # === 新增模型 (基于执行反馈) ===

            # 5. 训练执行概率v2模型 (基于实际执行结果)
            if 'actual_execution_binary' in curr_df.columns:
                valid_count = curr_df['actual_execution_binary'].notna().sum()
                if valid_count >= 100:
                    print(f"\n✨ 包含 {valid_count:,} 条实际执行结果,训练 execution_prob_v2")
                    self.train_single_target(
                        currency, curr_df, 'actual_execution_binary', 'classification',
                        'model_execution_prob_v2'
                    )
                else:
                    print(f"\n⚠️  实际执行结果不足 ({valid_count} < 100),跳过 execution_prob_v2")

            # 6. 训练收益优化模型
            if 'revenue_reward' in curr_df.columns and 'actual_execution_binary' in curr_df.columns:
                # 创建收益优化目标: predicted_rate × revenue_reward
                # 注意: 这里需要有 predicted_rate,但 curr_df 可能没有
                # 改用 close_annual 作为代理
                curr_df['revenue_optimized_target'] = curr_df['close_annual'] * curr_df['revenue_reward']
                valid_count = curr_df['revenue_optimized_target'].notna().sum()

                if valid_count >= 100:
                    print(f"\n✨ 训练收益优化模型 (有效样本: {valid_count:,})")
                    self.train_single_target(
                        currency, curr_df, 'revenue_optimized_target', 'regression',
                        'model_revenue_optimized'
                    )
                else:
                    print(f"\n⚠️  收益优化样本不足 ({valid_count} < 100),跳过")

            print(f"\n✅ {currency} 模型训练完成")

        print("\n" + "="*80)
        print(" "*20 + "✅ 所有模型训练完成")
        print("="*80)


def main():
    """
    测试脚本
    """
    import argparse

    parser = argparse.ArgumentParser(description='增强版模型训练器')
    parser.add_argument('--start-date', type=str, default='2026-01-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2026-02-08',
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--use-execution-feedback', action='store_true', default=True,
                       help='是否使用执行反馈')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='模型保存目录')

    args = parser.parse_args()

    trainer = EnhancedModelTrainer(
        db_path='data/lending_history.db',
        model_dir=args.model_dir
    )

    trainer.train_all_models(
        start_date=args.start_date,
        end_date=args.end_date,
        use_execution_feedback=args.use_execution_feedback
    )


if __name__ == '__main__':
    main()
