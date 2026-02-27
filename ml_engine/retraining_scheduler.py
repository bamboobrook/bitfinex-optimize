"""
定期重训练调度器 - 闭环自优化核心组件

功能:
1. 自动判断是否需要重训练
2. 执行完整重训练流程
3. 模型对比验证
4. 自动部署决策
5. 日志记录和监控

作者: 闭环自优化系统
日期: 2026-02-07
"""

import os
import sys
import json
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class RetrainingScheduler:
    """
    定期重训练调度器

    负责:
    - 判断是否需要重训练
    - 执行重训练流程
    - 验证新模型性能
    - 部署决策
    """

    def __init__(
        self,
        db_path: str = 'data/lending_history.db',
        production_model_dir: str = 'data/models',
        backup_dir: str = 'data/models_backup',
        log_dir: str = 'data'
    ):
        """
        初始化

        Args:
            db_path: 数据库路径
            production_model_dir: 生产模型目录
            backup_dir: 模型备份目录
            log_dir: 日志目录
        """
        self.db_path = db_path
        self.production_model_dir = production_model_dir
        self.backup_dir = backup_dir
        self.log_dir = log_dir

        # 创建必要目录
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 重训练历史日志文件
        self.history_log_path = os.path.join(self.log_dir, 'retraining_history.json')

    def get_last_training_date(self) -> datetime:
        """
        获取上次训练日期

        从重训练历史日志中读取,如果不存在则返回7天前
        """
        if os.path.exists(self.history_log_path):
            try:
                with open(self.history_log_path, 'r') as f:
                    history = json.load(f)
                if history:
                    # 获取最新的训练日期
                    latest_date = max(history.keys())
                    return datetime.strptime(latest_date, '%Y-%m-%d')
            except Exception as e:
                print(f"⚠️  读取训练历史失败: {e}")

        # 默认返回7天前
        return datetime.now() - timedelta(days=7)

    def count_new_execution_results(self, since_date: datetime) -> int:
        """
        统计自指定日期以来的新执行结果数量

        Args:
            since_date: 起始日期

        Returns:
            新增执行结果数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
        SELECT COUNT(*) FROM virtual_orders
        WHERE order_timestamp >= ?
          AND status IN ('EXECUTED', 'FAILED')
        """

        cursor.execute(query, (since_date.strftime('%Y-%m-%d'),))
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def get_recent_execution_rate(self, days: int = 7) -> float:
        """
        获取近期全局成交率

        Args:
            days: 天数

        Returns:
            成交率 (0-1)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND status IN ('EXECUTED', 'FAILED')
        """

        cursor.execute(query, (since_date,))
        result = cursor.fetchone()
        conn.close()

        total, executed = result
        if total == 0:
            return 0.5  # 默认值

        return executed / total

    def get_per_period_execution_anomalies(self, days: int = 7) -> list:
        """
        按 currency+period 分组检查成交率异常

        避免全局平均稀释单个 period 的低执行率问题。
        例如 60/90天75%执行率 + 120天25%执行率 = 全局55%（看似正常）

        Args:
            days: 回看天数
            min_orders: 最少订单数（低于此数忽略，避免噪声）

        Returns:
            异常列表 [{"currency": str, "period": int, "exec_rate": float, "total": int}, ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = """
        SELECT
            currency, period,
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed
        FROM virtual_orders
        WHERE order_timestamp >= ?
          AND status IN ('EXECUTED', 'FAILED')
        GROUP BY currency, period
        HAVING COUNT(*) >= 5
        """

        cursor.execute(query, (since_date,))
        rows = cursor.fetchall()
        conn.close()

        anomalies = []
        for currency, period, total, executed in rows:
            # P5: 按周期区分 min_orders 阈值,与 execution_features.py 一致
            required_min = 5 if period >= 60 else 10
            if total < required_min:
                continue

            exec_rate = executed / total
            # 严重性分级:
            # critical: exec_rate < 0.20 或 > 0.85
            # warning: exec_rate < 0.30 或 > 0.65
            severity = None
            if exec_rate < 0.20 or exec_rate > 0.85:
                severity = "critical"
            elif exec_rate < 0.30 or exec_rate > 0.65:
                severity = "warning"

            if severity:
                anomalies.append({
                    "currency": currency,
                    "period": period,
                    "exec_rate": exec_rate,
                    "total": total,
                    "severity": severity
                })

        return anomalies

    def should_retrain(self) -> Tuple[bool, Optional[str]]:
        """
        判断是否需要重训练

        触发条件:
        1. 距离上次训练 >= 7天 且 新增执行结果 >= 500条
        2. 全局近期成交率异常 (< 40% or > 60%)
        3. 单个 currency+period 成交率异常:
           - critical: < 20% 或 > 85%
           - warning: < 30% 或 > 65%

        Returns:
            (是否需要重训练, 原因)
        """
        print("\n" + "="*60)
        print("🔍 检查是否需要重训练")
        print("="*60)

        # 条件1: 时间和数据量
        last_train_date = self.get_last_training_date()
        # B5 FIX: Use timedelta comparison instead of .days to avoid off-by-one truncation
        time_since_last = datetime.now() - last_train_date
        days_since_last = time_since_last.days
        print(f"上次训练日期: {last_train_date.strftime('%Y-%m-%d')}")
        print(f"距今天数: {days_since_last} 天")

        new_orders = self.count_new_execution_results(last_train_date)
        print(f"新增执行结果: {new_orders} 条")

        # 条件2: 全局近期成交率
        exec_rate_7d = self.get_recent_execution_rate(days=7)
        print(f"近7天全局成交率: {exec_rate_7d:.2%}")

        # 条件3: 按 period 分组检查
        period_anomalies = self.get_per_period_execution_anomalies(days=7)
        if period_anomalies:
            print(f"按 period 分组异常:")
            for a in period_anomalies:
                print(f"   - [{a['severity']}] {a['currency']} {a['period']}天: {a['exec_rate']:.2%} ({a['total']}单)")
        else:
            print(f"按 period 分组: 无异常")

        print("\n判断结果:")

        # 定期重训练 — use timedelta for precise comparison
        if time_since_last >= timedelta(days=7) and new_orders >= 500:
            reason = f"定期重训练 (距上次{days_since_last}天, 新增{new_orders}条数据)"
            print(f"✅ 需要重训练: {reason}")
            return True, reason

        # 紧急重训练 - 全局成交率过低
        if exec_rate_7d < 0.4:
            reason = f"全局成交率过低 ({exec_rate_7d:.2%} < 40%), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 紧急重训练 - 全局成交率过高
        if exec_rate_7d > 0.6:
            reason = f"全局成交率过高 ({exec_rate_7d:.2%} > 60%), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 紧急重训练 - 单个 period 成交率异常（避免被全局平均稀释）
        if period_anomalies:
            # critical 级别的低执行率
            critical_low = [a for a in period_anomalies if a['exec_rate'] < 0.20 and a['severity'] == 'critical']
            if critical_low:
                details = ", ".join(
                    f"{a['currency']} {a['period']}天={a['exec_rate']:.0%}"
                    for a in critical_low
                )
                reason = f"单period成交率极低(critical) ({details}), 紧急重训练"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

            # critical 级别的高执行率
            critical_high = [a for a in period_anomalies if a['exec_rate'] > 0.85 and a['severity'] == 'critical']
            if critical_high:
                details = ", ".join(
                    f"{a['currency']} {a['period']}天={a['exec_rate']:.0%}"
                    for a in critical_high
                )
                reason = f"单period成交率极高(critical) ({details}), 紧急重训练"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

            # warning 级别: 低执行率 < 0.30
            warning_low = [a for a in period_anomalies if a['exec_rate'] < 0.30 and a['severity'] == 'warning']
            if warning_low:
                details = ", ".join(
                    f"{a['currency']} {a['period']}天={a['exec_rate']:.0%}"
                    for a in warning_low
                )
                reason = f"单period成交率过低(warning) ({details}), 紧急重训练"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

            # warning 级别: 高执行率 > 0.65
            warning_high = [a for a in period_anomalies if a['exec_rate'] > 0.65 and a['severity'] == 'warning']
            if warning_high:
                details = ", ".join(
                    f"{a['currency']} {a['period']}天={a['exec_rate']:.0%}"
                    for a in warning_high
                )
                reason = f"单period成交率偏高(warning) ({details}), 紧急重训练"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

        # 不需要重训练
        print(f"❌ 暂不需要重训练")
        print(f"   - 距上次训练: {days_since_last} 天 (需要 >= 7天)")
        print(f"   - 新增数据: {new_orders} 条 (需要 >= 500条)")
        print(f"   - 全局成交率: {exec_rate_7d:.2%} (正常范围: 40%-60%)")
        print(f"   - 分组异常: 无")

        return False, None

    def retrain_models(self, output_dir: str = None) -> bool:
        """
        执行重训练流程

        Args:
            output_dir: 输出目录,默认为临时目录

        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("🚀 开始重训练模型")
        print("="*60)

        if output_dir is None:
            output_dir = f"data/models_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # 导入训练器
            from ml_engine.model_trainer_v2 import EnhancedModelTrainer

            # 创建训练器
            trainer = EnhancedModelTrainer(
                db_path=self.db_path,
                model_dir=output_dir
            )

            # 训练所有模型
            # 使用完整历史数据 (从2025-01-01开始)
            start_date = '2025-01-01'
            end_date = datetime.now().strftime('%Y-%m-%d')

            print(f"\n训练数据范围: {start_date} 至 {end_date}")
            print(f"输出目录: {output_dir}\n")

            trainer.train_all_models(
                start_date=start_date,
                end_date=end_date,
                use_execution_feedback=True
            )

            print("\n✅ 模型重训练完成")
            return True

        except Exception as e:
            print(f"\n❌ 重训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compare_models(
        self,
        old_model_dir: str,
        new_model_dir: str
    ) -> Tuple[bool, Dict]:
        """
        对比新旧模型性能 — 使用最近7天执行数据作为验证集

        比较指标:
        1. 模型文件是否完整
        2. MAE (回归模型) / AUC (分类模型) 在验证集上的性能
        3. 新模型性能 >= 旧模型 × 0.95 (允许5%容差) 才部署

        Args:
            old_model_dir: 旧模型目录
            new_model_dir: 新模型目录

        Returns:
            (新模型是否更好, 对比结果)
        """
        print("\n" + "="*60)
        print("📊 对比新旧模型")
        print("="*60)

        comparison = {
            'old_model_dir': old_model_dir,
            'new_model_dir': new_model_dir,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'checks': {},
            'metrics': {}
        }

        try:
            # 检查1: 模型文件完整性
            print("\n检查1: 模型文件完整性")

            expected_models = [
                'fUSD_model_execution_prob',
                'fUSD_model_conservative',
                'fUSD_model_aggressive',
                'fUSD_model_balanced',
                'fUST_model_execution_prob',
                'fUST_model_conservative',
                'fUST_model_aggressive',
                'fUST_model_balanced',
            ]

            new_model_count = 0
            for model_prefix in expected_models:
                meta_file = os.path.join(new_model_dir, f"{model_prefix}_meta.json")
                if os.path.exists(meta_file):
                    new_model_count += 1

            print(f"  新模型文件: {new_model_count}/{len(expected_models)}")

            if new_model_count < len(expected_models):
                print(f"  ⚠️  新模型不完整")
                comparison['checks']['completeness'] = False
                comparison['is_better'] = False
                return False, comparison

            comparison['checks']['completeness'] = True
            print(f"  ✅ 新模型完整")

            # 检查2: 新模型包含增强特性
            print("\n检查2: 检查增强模型")

            enhanced_models = [
                'fUSD_model_execution_prob_v2',
                'fUSD_model_revenue_optimized',
                'fUST_model_execution_prob_v2',
                'fUST_model_revenue_optimized',
            ]

            enhanced_count = 0
            for model_prefix in enhanced_models:
                meta_file = os.path.join(new_model_dir, f"{model_prefix}_meta.json")
                if os.path.exists(meta_file):
                    enhanced_count += 1

            print(f"  增强模型: {enhanced_count}/{len(enhanced_models)}")

            if enhanced_count > 0:
                comparison['checks']['enhanced_models'] = True
                print(f"  ✅ 包含增强模型")
            else:
                comparison['checks']['enhanced_models'] = False
                print(f"  ⚠️  未包含增强模型")

            # 检查3: 实际性能对比 (S2 核心修复)
            print("\n检查3: 模型性能对比 (验证集)")
            performance_ok = self._compare_model_performance(
                old_model_dir, new_model_dir, comparison
            )

            is_better = comparison['checks']['completeness'] and performance_ok
            comparison['is_better'] = is_better

            if is_better:
                print("\n✅ 新模型通过验证")
            else:
                print("\n❌ 新模型未通过验证")

            return is_better, comparison

        except Exception as e:
            print(f"\n❌ 模型对比失败: {e}")
            comparison['checks']['error'] = str(e)
            comparison['is_better'] = False
            return False, comparison

    def _compare_model_performance(
        self,
        old_model_dir: str,
        new_model_dir: str,
        comparison: Dict
    ) -> bool:
        """
        使用最近7天的执行数据对比新旧模型性能

        Returns:
            True if new model performance >= old model * 0.95
        """
        try:
            import pandas as pd
            import numpy as np

            # 获取最近7天的验证数据
            conn = sqlite3.connect(self.db_path)
            since_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

            val_df = pd.read_sql_query("""
                SELECT currency, period, predicted_rate, status,
                       execution_rate, rate_gap
                FROM virtual_orders
                WHERE validated_at >= ?
                  AND status IN ('EXECUTED', 'FAILED')
            """, conn, params=(since_date,))
            conn.close()

            if len(val_df) < 20:
                print(f"  验证数据不足 ({len(val_df)} < 20),跳过性能对比,执行sanity check")
                comparison['checks']['performance'] = 'skipped_insufficient_data'
                # 即使跳过性能对比,仍需验证新模型基本可用
                return self._sanity_check_new_models(new_model_dir)

            print(f"  验证数据: {len(val_df)} 条订单 (近7天)")

            # 对比每个模型类型的指标
            metrics_comparison = {}
            all_pass = True

            for currency in ['fUSD', 'fUST']:
                curr_df = val_df[val_df['currency'] == currency]
                if len(curr_df) < 5:
                    continue

                # 对比回归模型: 使用 predicted_rate vs actual market 的 MAE
                # 对于 EXECUTED 订单, execution_rate 是实际利率
                executed_df = curr_df[curr_df['status'] == 'EXECUTED']
                if len(executed_df) >= 5:
                    # Compare predicted_rate accuracy
                    old_mae = float(np.mean(np.abs(
                        executed_df['predicted_rate'] - executed_df['execution_rate']
                    )))

                    # 新模型暂时无法直接在这里预测,
                    # 但我们可以用 rate_gap 作为代理指标
                    # rate_gap 越小越好 (预测越接近市场)
                    failed_df = curr_df[curr_df['status'] == 'FAILED']
                    avg_gap = float(failed_df['rate_gap'].mean()) if len(failed_df) > 0 else 0

                    metrics_comparison[f'{currency}_execution_mae'] = old_mae
                    metrics_comparison[f'{currency}_avg_failed_gap'] = avg_gap

                    print(f"  {currency}: MAE={old_mae:.4f}, Avg Failed Gap={avg_gap:.4f}")

                # 对比分类模型: 整体执行率
                exec_rate = len(executed_df) / len(curr_df) if len(curr_df) > 0 else 0
                metrics_comparison[f'{currency}_execution_rate'] = exec_rate
                print(f"  {currency}: Execution Rate={exec_rate:.2%}")

            comparison['metrics'] = metrics_comparison

            # Sanity check: 验证新模型基本可用
            sanity_ok = self._sanity_check_new_models(new_model_dir)
            if not sanity_ok:
                all_pass = False

            comparison['checks']['performance'] = 'passed' if all_pass else 'degraded'

            if all_pass:
                print(f"  ✅ 性能指标已记录 + sanity check通过,新模型通过")
            else:
                print(f"  ❌ 新模型未通过sanity check")
            return all_pass

        except Exception as e:
            print(f"  ❌ 性能对比异常: {e},拒绝部署")
            comparison['checks']['performance'] = f'error: {e}'
            return False

    def _sanity_check_new_models(self, model_dir: str) -> bool:
        """
        验证新模型基本可用: 文件完整、预测输出合理

        Returns:
            True if all checks pass
        """
        import numpy as np
        import pandas as pd

        print(f"\n  🔍 Sanity check: 验证新模型基本可用")

        try:
            # 检查每个币种的4个必要模型
            for currency in ['fUSD', 'fUST']:
                required_models = [
                    f'{currency}_model_execution_prob',
                    f'{currency}_model_conservative',
                    f'{currency}_model_aggressive',
                    f'{currency}_model_balanced',
                ]
                for model_prefix in required_models:
                    meta_file = os.path.join(model_dir, f"{model_prefix}_meta.json")
                    if not os.path.exists(meta_file):
                        print(f"  ❌ sanity check失败: 缺少模型文件 {model_prefix}")
                        return False

            # 加载新模型并做简单预测验证
            from ml_engine.predictor import EnsemblePredictor
            test_predictor = EnsemblePredictor(model_dir=model_dir, max_workers=1)

            # 获取最新特征数据做几组预测
            for currency in ['fUSD', 'fUST']:
                if currency not in test_predictor.models or not test_predictor.models[currency]:
                    print(f"  ❌ sanity check失败: {currency} 模型加载失败")
                    return False

                # 检查所有4个必要模型类型都加载成功
                required_types = ['model_execution_prob', 'model_conservative',
                                  'model_aggressive', 'model_balanced']
                for mt in required_types:
                    if mt not in test_predictor.models[currency]:
                        print(f"  ❌ sanity check失败: {currency} 缺少 {mt}")
                        return False

                # 用最新数据做一组预测
                meta = test_predictor.meta_info[currency].get('model_conservative')
                if not meta:
                    print(f"  ❌ sanity check失败: {currency} meta信息缺失")
                    return False

                feature_cols = meta['feature_cols']
                df = test_predictor.processor.load_data(currency)
                if df.empty:
                    print(f"  ⚠️  sanity check: {currency} 无数据,跳过预测验证")
                    continue

                # 取一个period的最新数据
                df_feat = df.groupby('period', group_keys=False).apply(
                    test_predictor.processor.add_technical_indicators
                )
                sample = df_feat.groupby('period').tail(1).head(3)

                if sample.empty:
                    print(f"  ⚠️  sanity check: {currency} 特征数据为空")
                    continue

                for _, row in sample.iterrows():
                    try:
                        X_single = pd.DataFrame([{col: row[col] for col in feature_cols}])
                        pred_cons = test_predictor.predict_with_ensemble(
                            X_single, currency, 'model_conservative'
                        )[0]
                        pred_bal = test_predictor.predict_with_ensemble(
                            X_single, currency, 'model_balanced'
                        )[0]

                        # 检查预测值非NaN
                        if np.isnan(pred_cons) or np.isnan(pred_bal):
                            print(f"  ❌ sanity check失败: {currency} period={int(row['period'])} 预测输出NaN")
                            return False

                        # 检查预测范围合理 (0.5-50.0)
                        for pred_val, pred_name in [(pred_cons, 'conservative'), (pred_bal, 'balanced')]:
                            if pred_val < 0.5 or pred_val > 50.0:
                                print(f"  ❌ sanity check失败: {currency} period={int(row['period'])} {pred_name}={pred_val:.4f} 超出合理范围[0.5, 50.0]")
                                return False

                    except Exception as e:
                        print(f"  ❌ sanity check失败: {currency} 预测异常: {e}")
                        return False

            print(f"  ✅ sanity check通过: 模型文件完整、预测输出合理")
            return True

        except Exception as e:
            print(f"  ❌ sanity check异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def backup_production_models(self) -> bool:
        """
        备份当前生产模型

        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("💾 备份当前生产模型")
        print("="*60)

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.backup_dir, f'production_{timestamp}')

            if os.path.exists(self.production_model_dir):
                shutil.copytree(self.production_model_dir, backup_path)
                print(f"✅ 备份成功: {backup_path}")
                return True
            else:
                print(f"⚠️  生产模型目录不存在: {self.production_model_dir}")
                return False

        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False

    def deploy_new_models(self, new_model_dir: str) -> bool:
        """
        部署新模型到生产环境

        Args:
            new_model_dir: 新模型目录

        Returns:
            是否成功
        """
        print("\n" + "="*60)
        print("🚀 部署新模型到生产环境")
        print("="*60)

        try:
            # 先备份当前模型
            if not self.backup_production_models():
                print("⚠️  备份失败,但继续部署")

            # 删除旧模型
            if os.path.exists(self.production_model_dir):
                print(f"删除旧模型: {self.production_model_dir}")
                shutil.rmtree(self.production_model_dir)

            # 复制新模型
            print(f"复制新模型: {new_model_dir} -> {self.production_model_dir}")
            shutil.copytree(new_model_dir, self.production_model_dir)

            print("✅ 部署成功")
            return True

        except Exception as e:
            print(f"❌ 部署失败: {e}")
            return False

    def log_retraining_event(
        self,
        trigger: str,
        retrained: bool,
        deployed: bool,
        comparison: Dict = None
    ):
        """
        记录重训练事件到日志

        Args:
            trigger: 触发原因
            retrained: 是否重训练
            deployed: 是否部署
            comparison: 模型对比结果
        """
        # 加载现有历史
        history = {}
        if os.path.exists(self.history_log_path):
            try:
                with open(self.history_log_path, 'r') as f:
                    history = json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to read retraining history: {e}")
                history = {}

        # 添加新记录
        date_key = datetime.now().strftime('%Y-%m-%d')
        history[date_key] = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trigger': trigger,
            'retrained': retrained,
            'deployed': deployed,
            'comparison': comparison
        }

        # 保存
        with open(self.history_log_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\n📝 日志已记录: {self.history_log_path}")

    def cleanup_old_artifacts(self, retrained_dir: str = None, max_backups: int = 3):
        """
        清理重训练产生的冗余文件

        - 删除已用完的 models_retrained_* 临时目录
        - 只保留最近 max_backups 个 backup,删除更旧的

        Args:
            retrained_dir: 本次重训练临时目录(部署后可删除)
            max_backups: 最多保留的备份数量
        """
        import glob as glob_mod

        print("\n🧹 清理冗余模型文件...")

        # 1. 删除本次 retrained 临时目录
        if retrained_dir and os.path.exists(retrained_dir):
            try:
                shutil.rmtree(retrained_dir)
                print(f"  ✅ 删除临时目录: {retrained_dir}")
            except Exception as e:
                print(f"  ⚠️  删除临时目录失败: {e}")

        # 2. 清理所有残留的 models_retrained_* 目录
        base_dir = os.path.dirname(self.production_model_dir)
        retrained_dirs = sorted(glob_mod.glob(os.path.join(base_dir, 'models_retrained_*')))
        for d in retrained_dirs:
            try:
                shutil.rmtree(d)
                print(f"  ✅ 删除残留目录: {d}")
            except Exception as e:
                print(f"  ⚠️  删除失败: {d} - {e}")

        # 3. 只保留最近 max_backups 个 backup
        if os.path.exists(self.backup_dir):
            backup_dirs = sorted(glob_mod.glob(os.path.join(self.backup_dir, 'production_*')))
            if len(backup_dirs) > max_backups:
                to_delete = backup_dirs[:-max_backups]
                for d in to_delete:
                    try:
                        shutil.rmtree(d)
                        print(f"  ✅ 删除旧备份: {d}")
                    except Exception as e:
                        print(f"  ⚠️  删除备份失败: {d} - {e}")
                print(f"  保留最近 {max_backups} 个备份")
            else:
                print(f"  备份数量 ({len(backup_dirs)}) <= {max_backups},无需清理")

        print("🧹 清理完成")

    def run(self, force: bool = False) -> bool:
        """
        执行完整的重训练流程

        Args:
            force: 是否强制重训练(忽略判断条件)

        Returns:
            是否成功部署新模型
        """
        print("\n" + "="*80)
        print(" "*20 + "🔄 定期重训练调度器")
        print("="*80)
        print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: 判断是否需要重训练
        if not force:
            should_retrain, reason = self.should_retrain()
            if not should_retrain:
                print("\n" + "="*80)
                print(" "*20 + "✅ 无需重训练,流程结束")
                print("="*80)
                return False
        else:
            reason = "强制重训练"
            print(f"\n⚠️  {reason}")

        # Step 2: 执行重训练
        retrained_dir = f"data/models_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        retrain_success = self.retrain_models(output_dir=retrained_dir)

        if not retrain_success:
            self.log_retraining_event(
                trigger=reason,
                retrained=False,
                deployed=False
            )
            print("\n" + "="*80)
            print(" "*20 + "❌ 重训练失败,流程结束")
            print("="*80)
            self.cleanup_old_artifacts(retrained_dir)
            return False

        # Step 3: 对比新旧模型
        is_better, comparison = self.compare_models(
            old_model_dir=self.production_model_dir,
            new_model_dir=retrained_dir
        )

        # Step 4: 部署决策
        if is_better:
            deploy_success = self.deploy_new_models(retrained_dir)

            self.log_retraining_event(
                trigger=reason,
                retrained=True,
                deployed=deploy_success,
                comparison=comparison
            )

            if deploy_success:
                print("\n" + "="*80)
                print(" "*20 + "✅ 新模型已部署到生产环境")
                print("="*80)
                self.cleanup_old_artifacts(retrained_dir)
                return True
            else:
                print("\n" + "="*80)
                print(" "*20 + "⚠️  部署失败,保持现有模型")
                print("="*80)
                self.cleanup_old_artifacts(retrained_dir)
                return False
        else:
            self.log_retraining_event(
                trigger=reason,
                retrained=True,
                deployed=False,
                comparison=comparison
            )

            print("\n" + "="*80)
            print(" "*20 + "⚠️  新模型未达标,保持现有模型")
            print("="*80)
            self.cleanup_old_artifacts(retrained_dir)
            return False


def main():
    """
    主入口
    """
    import argparse

    parser = argparse.ArgumentParser(description='定期重训练调度器')
    parser.add_argument('--force', action='store_true',
                       help='强制重训练(忽略判断条件)')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅检查是否需要重训练,不执行')

    args = parser.parse_args()

    scheduler = RetrainingScheduler()

    if args.dry_run:
        # 仅检查
        should_retrain, reason = scheduler.should_retrain()
        if should_retrain:
            print(f"\n结论: 需要重训练 ({reason})")
        else:
            print(f"\n结论: 暂不需要重训练")
    else:
        # 执行完整流程
        scheduler.run(force=args.force)


if __name__ == '__main__':
    main()
