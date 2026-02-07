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
        log_dir: str = 'log'
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
        获取近期成交率

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

    def should_retrain(self) -> Tuple[bool, Optional[str]]:
        """
        判断是否需要重训练

        触发条件:
        1. 距离上次训练 >= 7天 且 新增执行结果 >= 500条
        2. 近期成交率异常 (< 40% or > 60%)

        Returns:
            (是否需要重训练, 原因)
        """
        print("\n" + "="*60)
        print("🔍 检查是否需要重训练")
        print("="*60)

        # 条件1: 时间和数据量
        last_train_date = self.get_last_training_date()
        days_since_last = (datetime.now() - last_train_date).days
        print(f"上次训练日期: {last_train_date.strftime('%Y-%m-%d')}")
        print(f"距今天数: {days_since_last} 天")

        new_orders = self.count_new_execution_results(last_train_date)
        print(f"新增执行结果: {new_orders} 条")

        # 条件2: 近期成交率
        exec_rate_7d = self.get_recent_execution_rate(days=7)
        print(f"近7天成交率: {exec_rate_7d:.2%}")

        print("\n判断结果:")

        # 定期重训练
        if days_since_last >= 7 and new_orders >= 500:
            reason = f"定期重训练 (距上次{days_since_last}天, 新增{new_orders}条数据)"
            print(f"✅ 需要重训练: {reason}")
            return True, reason

        # 紧急重训练 - 成交率过低
        if exec_rate_7d < 0.4:
            reason = f"成交率过低 ({exec_rate_7d:.2%} < 40%), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 紧急重训练 - 成交率过高
        if exec_rate_7d > 0.6:
            reason = f"成交率过高 ({exec_rate_7d:.2%} > 60%), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 不需要重训练
        print(f"❌ 暂不需要重训练")
        print(f"   - 距上次训练: {days_since_last} 天 (需要 >= 7天)")
        print(f"   - 新增数据: {new_orders} 条 (需要 >= 500条)")
        print(f"   - 成交率: {exec_rate_7d:.2%} (正常范围: 40%-60%)")

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
        对比新旧模型性能

        比较指标:
        1. 模型文件是否完整
        2. 特征数量是否一致
        3. (可选) 在验证集上的性能

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
            'checks': {}
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

            # 简单判断: 如果新模型完整,则认为更好
            # (更复杂的对比可以在实际验证集上测试)
            is_better = comparison['checks']['completeness']
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
            except:
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
                return True
            else:
                print("\n" + "="*80)
                print(" "*20 + "⚠️  部署失败,保持现有模型")
                print("="*80)
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
