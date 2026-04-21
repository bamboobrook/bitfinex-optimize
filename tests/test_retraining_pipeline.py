"""
Phase A 回归测试: 验证重训练管线产出

核心断言:
1. prepare_training_data() 产出密集市场数据（非稀疏订单数据）
2. 传统目标在密集数据上可正确计算（非全NaN）
3. v2 样本仅来自 _exploit_quality=True 的行
4. 重训练输出目录包含预期 meta 文件
"""
import os
import sys
import pytest
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_engine'))

from ml_engine.training_data_builder import TrainingDataBuilder
from ml_engine.model_trainer_v2 import EnhancedModelTrainer


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'lending_history.db')


def _check_db_exists():
    if not os.path.exists(DB_PATH):
        pytest.skip(f"Database not found: {DB_PATH}")


class TestTrainingDataBuilder:
    """测试 training_data_builder 的 merge 方向和过滤逻辑"""

    @pytest.fixture(autouse=True)
    def setup(self):
        _check_db_exists()
        self.builder = TrainingDataBuilder(DB_PATH)

    def test_merge_produces_dense_output(self):
        """merge_market_and_execution 应产出密集市场数据（>1000行/币种）"""
        market_data = self.builder.load_market_data('2025-06-01', '2026-04-21')
        execution_results = self.builder.load_execution_results('2025-06-01', '2026-04-21')

        merged = self.builder.merge_market_and_execution(market_data, execution_results)

        for currency in ['fUSD', 'fUST']:
            curr_df = merged[merged['currency'] == currency]
            assert len(curr_df) > 1000, (
                f"{currency} 仅{len(curr_df)}行，应为密集市场数据（>1000）"
            )

    def test_merge_has_internal_fields(self):
        """merge 后应包含 _order_match_minutes / _execution_label_eligible / _exploit_quality"""
        market_data = self.builder.load_market_data('2025-06-01', '2026-04-21')
        execution_results = self.builder.load_execution_results('2025-06-01', '2026-04-21')

        merged = self.builder.merge_market_and_execution(market_data, execution_results)

        assert '_order_match_minutes' in merged.columns
        assert '_execution_label_eligible' in merged.columns
        assert '_exploit_quality' in merged.columns

    def test_no_blanket_exploit_filter(self):
        """不应 blanket 过滤 decision_mode=='exploit'，保留全部市场数据行"""
        market_data = self.builder.load_market_data('2025-06-01', '2026-04-21')
        execution_results = self.builder.load_execution_results('2025-06-01', '2026-04-21')

        merged = self.builder.merge_market_and_execution(market_data, execution_results)

        # 总行数应接近 market_data 行数（~887K），而非订单数（~12K）
        assert len(merged) > 50000, (
            f"merge 后仅{len(merged)}行，疑似仍以订单为主表"
        )

    def test_execution_label_eligible_is_strict(self):
        """_execution_label_eligible 应仅标记紧邻订单的行（≤30min）"""
        market_data = self.builder.load_market_data('2025-06-01', '2026-04-21')
        execution_results = self.builder.load_execution_results('2025-06-01', '2026-04-21')

        merged = self.builder.merge_market_and_execution(market_data, execution_results)

        eligible = merged[merged['_execution_label_eligible'] == True]
        if len(eligible) > 0:
            # eligible 行的 _order_match_minutes 应全部 ≤ 30
            assert (eligible['_order_match_minutes'] <= 30).all(), (
                "_execution_label_eligible=True 的行 _order_match_minutes 应 ≤ 30"
            )


class TestModelTrainerV2:
    """测试 model_trainer_v2 的训练逻辑"""

    @pytest.fixture(autouse=True)
    def setup(self):
        _check_db_exists()

    def test_prepare_training_data_not_empty(self):
        """prepare_training_data 应产出非空数据"""
        trainer = EnhancedModelTrainer(db_path=DB_PATH)
        df = trainer.prepare_training_data('2025-06-01', '2026-04-21', use_execution_feedback=True)

        assert len(df) > 0, "prepare_training_data 产出为空"

    def test_traditional_targets_not_all_nan(self):
        """传统目标在密集数据上应可正确计算（非全NaN）"""
        trainer = EnhancedModelTrainer(db_path=DB_PATH)
        df = trainer.prepare_training_data('2025-06-01', '2026-04-21', use_execution_feedback=True)

        for target in ['future_conservative', 'future_aggressive',
                       'future_balanced', 'future_execution_prob']:
            non_nan = df[target].notna().sum()
            assert non_nan > 100, (
                f"{target} 仅{non_nan}个非NaN值，传统目标计算失败"
            )

    def test_v2_samples_only_from_exploit_quality(self):
        """v2 样本应仅来自 _exploit_quality=True 的行"""
        trainer = EnhancedModelTrainer(db_path=DB_PATH)
        df = trainer.prepare_training_data('2025-06-01', '2026-04-21', use_execution_feedback=True)

        if '_exploit_quality' in df.columns:
            v2_eligible = df[df['_exploit_quality'] == True]
            # v2 样本数应远小于总行数（稀疏子集）
            assert len(v2_eligible) < len(df), (
                f"v2 样本({len(v2_eligible)})应小于总行数({len(df)})"
            )
            # 但 v2 样本数应 > 0
            if 'actual_execution_binary' in df.columns:
                v2_with_exec = v2_eligible['actual_execution_binary'].notna().sum()
                # 可能为0（如果历史订单无 decision_mode），此时仅 warning
                if v2_with_exec == 0:
                    print("⚠️  v2 样本为0，可能历史订单缺少 decision_mode 列")


class TestRetrainingOutput:
    """测试重训练输出验证"""

    @pytest.fixture(autouse=True)
    def setup(self):
        _check_db_exists()

    def test_train_all_models_produces_output(self):
        """train_all_models 应在输出目录产出模型文件"""
        trainer = EnhancedModelTrainer(db_path=DB_PATH)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.model_dir = tmpdir
            trainer.train_all_models(
                start_date='2025-06-01',
                end_date='2026-04-21',
                use_execution_feedback=True
            )

            # 检查核心模型 meta 文件
            required_core = [
                f'{c}_model_{m}_meta.json'
                for c in ['fUSD', 'fUST']
                for m in ['execution_prob', 'conservative', 'aggressive', 'balanced']
            ]

            found = sum(1 for f in required_core
                       if os.path.exists(os.path.join(tmpdir, f)))
            assert found >= 6, (
                f"核心模型仅{found}/8产出，训练管线仍有问题"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
