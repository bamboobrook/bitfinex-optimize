"""
训练数据构建器 - 闭环自优化核心模块

功能:
1. 融合历史市场数据 (funding_rates) + 虚拟订单执行结果 (virtual_orders)
2. 生成增强训练标签 (actual_execution, revenue_reward)
3. 为模型重训练提供高质量训练数据

作者: 闭环自优化系统
日期: 2026-02-07
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class TrainingDataBuilder:
    """
    训练数据构建器

    将历史市场数据与虚拟订单执行结果融合,生成用于模型训练的增强数据集
    """

    def __init__(self, db_path: str = 'data/lending_history.db'):
        """
        初始化

        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path

    def _get_virtual_orders_columns(self) -> List[str]:
        """Get virtual_orders column names for backward-compatible SQL building."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(virtual_orders)")
            return [row[1] for row in cursor.fetchall()]
        finally:
            conn.close()

    def load_execution_results(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 virtual_orders 表加载执行结果

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame with columns:
            - order_timestamp: 订单时间
            - currency: 币种
            - period: 期限
            - predicted_rate: 预测利率
            - status: 执行状态 (EXECUTED/FAILED)
            - execution_confidence: 成交置信度
            - total_score: 混合评分
            - market_median: 市场中位数
            - execution_rate: 实际成交利率
        """
        conn = sqlite3.connect(self.db_path)
        table_cols = set(self._get_virtual_orders_columns())

        base_cols = [
            'order_timestamp',
            'currency',
            'period',
            'predicted_rate',
            'status',
            'execution_confidence',
            'total_score',
            'market_median',
            'execution_rate',
        ]
        selected_cols = [
            col if col in table_cols else f"NULL AS {col}"
            for col in base_cols
        ]
        optional_cols = [
            'follow_error_at_order',
            'gate_reject_reason',
            'direction_match',
            'step_change_pct',
            'step_capped',
            'policy_step_cap_pct',
            'probe_type',
            'path_value_score',
            'stage1_fill_probability',
            'stage2_frr_proxy_rate',
            'terminal_mode',
            'update_cycle_id',
            'recommendation_rank',
            'rank_weight',
            'candidate_id',
            'decision_mode',
            'data_quality_label',
            'validation_label',
            'realized_terminal_mode',
            'realized_terminal_value',
            'realized_wait_hours',
        ]
        selected_cols.extend([c for c in optional_cols if c in table_cols])

        query = f"""
        SELECT
            {', '.join(selected_cols)}
        FROM virtual_orders
        WHERE order_timestamp >= ? AND order_timestamp < ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
        ORDER BY order_timestamp
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        # 转换时间字段
        if len(df) > 0:
            df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])

        expired_count = (df['status'] == 'EXPIRED').sum() if len(df) > 0 else 0
        print(f"✓ 加载执行结果: {len(df)} 条 (EXECUTED: {(df['status']=='EXECUTED').sum()}, FAILED: {(df['status']=='FAILED').sum()}, EXPIRED: {expired_count})")

        return df

    def load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 funding_rates 表加载历史市场数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with market data
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            currency,
            period,
            timestamp,
            datetime,
            open_annual,
            close_annual,
            high_annual,
            low_annual,
            volume,
            hour,
            day_of_week
        FROM funding_rates
        WHERE datetime >= ? AND datetime < ?
        ORDER BY datetime, currency, period
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        if len(df) > 0:
            df['datetime'] = pd.to_datetime(df['datetime'])

        print(f"✓ 加载市场数据: {len(df)} 条")

        return df

    def merge_market_and_execution(
        self,
        market_data: pd.DataFrame,
        execution_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        时间对齐: 以 dense market_data 为主表，匹配最近的执行结果

        策略:
        - 使用 merge_asof 以 market_data 为 LEFT（密集 ~887K 行）
        - execution_results 为 RIGHT（稀疏 ~12K 行），无匹配时订单列为 NaN
        - direction='backward': 只匹配过去/当前订单，防止前视偏差
        - 匹配容忍度: 2小时
        - 按 currency + period 分组匹配

        新增内部字段:
        - _order_match_minutes: 订单与市场行的时间差（分钟），NaN 表示无匹配
        - _execution_label_eligible: 订单紧邻且非前视的资格标记
        - _exploit_quality: exploit + STRONG + _execution_label_eligible

        Args:
            market_data: 市场数据（主表）
            execution_results: 执行结果（右表）

        Returns:
            融合后的DataFrame,包含增强标签
        """
        if len(execution_results) == 0:
            print("⚠️  无执行结果数据,返回原始市场数据")
            market_data = market_data.copy()
            for col in ['actual_execution_binary', 'revenue_reward', 'rate_competitiveness',
                        '_order_match_minutes', '_execution_label_eligible', '_exploit_quality']:
                market_data[col] = np.nan
            return market_data

        # 确保时间字段格式正确
        market_data = market_data.copy()
        execution_results = execution_results.copy()

        if 'datetime' not in market_data.columns:
            print("⚠️  market_data 缺少 'datetime' 列")
            return market_data

        market_data['datetime'] = pd.to_datetime(market_data['datetime'])
        execution_results['order_timestamp'] = pd.to_datetime(execution_results['order_timestamp'])

        # 按时间和币种/期限排序
        market_data = market_data.sort_values('datetime')
        execution_results = execution_results.sort_values('order_timestamp')

        # 时间对齐合并：market_data 为 LEFT（密集），execution_results 为 RIGHT（稀疏）
        merge_cols = [
            'order_timestamp', 'currency', 'period', 'status',
            'predicted_rate', 'execution_confidence', 'total_score',
            'market_median', 'execution_rate'
        ]
        for c in ['follow_error_at_order', 'gate_reject_reason', 'direction_match',
                  'step_change_pct', 'step_capped', 'policy_step_cap_pct', 'probe_type',
                  'path_value_score', 'stage1_fill_probability', 'stage2_frr_proxy_rate',
                  'terminal_mode', 'update_cycle_id', 'recommendation_rank',
                  'rank_weight', 'candidate_id', 'decision_mode',
                  'data_quality_label', 'validation_label',
                  'realized_terminal_mode', 'realized_terminal_value',
                  'realized_wait_hours']:
            if c in execution_results.columns:
                merge_cols.append(c)

        merged = pd.merge_asof(
            market_data.sort_values('datetime'),                            # 主表：市场数据（密集）
            execution_results[merge_cols].sort_values('order_timestamp'),  # 右表：订单（稀疏）
            left_on='datetime',
            right_on='order_timestamp',
            by=['currency', 'period'],
            direction='backward',  # 只匹配过去/当前订单，防止前视偏差
            tolerance=pd.Timedelta('2h')  # 严格限制在2小时内
        )

        # 计算 _order_match_minutes: 订单与市场行的时间差
        if 'order_timestamp' in merged.columns:
            time_diff = merged['datetime'] - merged['order_timestamp']
            merged['_order_match_minutes'] = time_diff.dt.total_seconds() / 60.0
            # 无订单匹配时 order_timestamp 为 NaT → _order_match_minutes 为 NaN
        else:
            merged['_order_match_minutes'] = np.nan

        # _execution_label_eligible: 订单紧邻（≤30min）且非前视
        # backward merge 保证 order_timestamp <= datetime，只需检查时间差
        merged['_execution_label_eligible'] = (
            merged['_order_match_minutes'].notna() &
            (merged['_order_match_minutes'] >= 0) &
            (merged['_order_match_minutes'] <= 30) &
            merged.get('status', pd.Series(dtype=str)).notna()
        )

        # 生成增强标签（不再 blanket 过滤 decision_mode）
        merged = self._generate_enhanced_labels(merged)

        # 统计
        has_order = merged['_order_match_minutes'].notna()
        eligible_count = merged['_execution_label_eligible'].sum()
        exploit_quality_count = merged.get('_exploit_quality', pd.Series(dtype=bool)).sum()
        exec_label_count = merged['actual_execution_binary'].notna().sum()
        exec_count = (merged['actual_execution_binary'] == 1).sum()
        failed_count = (merged['actual_execution_binary'] == 0).sum()

        print(f"✓ 数据融合完成: 总样本 {len(merged):,}, "
              f"有订单匹配 {has_order.sum():,}, "
              f"eligible {eligible_count:,}, "
              f"exploit_quality {exploit_quality_count:,}")
        print(f"  执行标签覆盖: {exec_label_count:,} (成交 {exec_count:,}, 失败 {failed_count:,})")

        return merged

    def _generate_enhanced_labels(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        生成增强训练标签

        新增标签:
        1. actual_execution_binary: 实际是否成交 (0/1)
        2. revenue_reward: 收益奖励分数
        3. rate_competitiveness: 利率竞争力

        Args:
            merged_df: 融合后的数据

        Returns:
            添加增强标签后的DataFrame
        """
        df = merged_df.copy()

        # 不再 blanket 过滤 decision_mode=='exploit'（那会把 dense market frame 压成稀疏订单级）
        # 改为标记 _exploit_quality 供 v2 模型训练时使用
        has_decision_mode = 'decision_mode' in df.columns
        has_data_quality = 'data_quality_label' in df.columns
        has_eligible = '_execution_label_eligible' in df.columns

        if has_decision_mode and has_data_quality and has_eligible:
            df['_exploit_quality'] = (
                (df['decision_mode'] == 'exploit') &
                (df['data_quality_label'] == 'STRONG') &
                df['_execution_label_eligible']
            )
        elif has_eligible:
            df['_exploit_quality'] = df['_execution_label_eligible']
        else:
            df['_exploit_quality'] = False

        # 标签1: 实际成交二元标签（仅在有订单匹配的行上设置）
        has_order = df.get('status', pd.Series(dtype=str)).notna()
        df['actual_execution_binary'] = np.nan
        if 'status' in df.columns:
            df.loc[has_order & (df['status'] == 'EXECUTED'), 'actual_execution_binary'] = 1.0
            df.loc[has_order & (df['status'] == 'FAILED'), 'actual_execution_binary'] = 0.0
            # EXPIRED: 出价太高、市场无机会成交 → 负样本，但降低权重避免过拟合
            expired_mask = has_order & (df['status'] == 'EXPIRED')
            df.loc[expired_mask, 'actual_execution_binary'] = 0.0
            df['_expired_weight'] = np.where(expired_mask, 0.5, 1.0)
        else:
            df['_expired_weight'] = 1.0

        # 标签2: 利率竞争力 (predicted_rate / market_median)
        df['rate_competitiveness'] = df.apply(
            lambda row: row['predicted_rate'] / (row['market_median'] + 1e-8)
            if pd.notna(row['market_median']) and pd.notna(row['predicted_rate'])
            else np.nan,
            axis=1
        )

        # 标签3: 收益奖励
        df['revenue_reward'] = df.apply(self._compute_revenue_reward, axis=1)

        # 路径标签: 默认值兜底，兼容旧表结构
        for col in [
            'path_value_score',
            'stage1_fill_probability',
            'stage2_frr_proxy_rate',
            'terminal_mode',
            'realized_terminal_mode',
            'realized_terminal_value',
            'realized_wait_hours',
        ]:
            if col not in df.columns:
                df[col] = np.nan

        # SQLite 在整列皆为 NULL 时会把数值列读成 object，后续 fillna 会触发 pandas FutureWarning。
        for col in [
            'predicted_rate',
            'market_median',
            'execution_rate',
            'stage2_frr_proxy_rate',
            'realized_terminal_value',
            'realized_wait_hours',
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        executed_mask = (df.get('status', pd.Series(dtype=str)) == 'EXECUTED') if 'status' in df.columns else pd.Series(False, index=df.index)
        fallback_terminal_value = pd.Series(
            np.where(
                executed_mask,
                df.get('execution_rate', np.nan).fillna(df.get('predicted_rate', np.nan)),
                df.get('stage2_frr_proxy_rate', np.nan)
            ),
            index=df.index,
        )
        realized_terminal_value_available = (
            df['realized_terminal_value'].notna()
        ) & (df['realized_terminal_value'] != '')
        df['path_terminal_value'] = df['realized_terminal_value'].where(
            realized_terminal_value_available,
            fallback_terminal_value
        )

        fallback_stage1_success = (executed_mask & (df.get('terminal_mode', pd.Series(dtype=str)) == 'FIXED')).astype(float)
        realized_terminal_mode_available = (
            df['realized_terminal_mode'].notna()
        ) & (df['realized_terminal_mode'] != '')
        df['path_stage1_success'] = np.where(
            realized_terminal_mode_available,
            (df['realized_terminal_mode'] == 'FIXED').astype(float),
            fallback_stage1_success
        )
        df['path_wait_hours'] = df['realized_wait_hours']

        strong_quality_mask = (
            df['data_quality_label'] == 'STRONG'
            if 'data_quality_label' in df.columns
            else pd.Series(True, index=df.index)
        )
        df.loc[~strong_quality_mask, ['path_terminal_value', 'path_stage1_success', 'path_wait_hours']] = np.nan

        # 闭环诊断标签: 跟随误差、方向一致性、单步变化
        if 'follow_error_at_order' in df.columns:
            df['follow_error'] = df['follow_error_at_order']
        else:
            df['follow_error'] = df.apply(
                lambda row: row.get('predicted_rate', np.nan) - row.get('market_median', np.nan)
                if pd.notna(row.get('predicted_rate')) and pd.notna(row.get('market_median'))
                else np.nan,
                axis=1
            )

        if 'direction_match' not in df.columns:
            df['direction_match'] = np.nan
        if 'step_change_pct' not in df.columns:
            df['step_change_pct'] = np.nan
        if 'step_capped' not in df.columns:
            df['step_capped'] = np.nan
        if 'probe_type' not in df.columns:
            df['probe_type'] = 'normal'
        else:
            df['probe_type'] = df['probe_type'].fillna('normal')

        return df

    def _compute_revenue_reward(self, row: pd.Series) -> float:
        """
        计算收益奖励

        奖励公式:
        - 未成交: 0
        - 成交: rate_competitiveness × market_factor

        市场因子:
        - fUST 或 高成交周期 (15/60/90): 1.2
        - 其他: 1.0

        Args:
            row: DataFrame 行

        Returns:
            收益奖励分数 (0-2.0)
        """
        # 未成交或缺失数据
        if pd.isna(row.get('status')) or row.get('status') != 'EXECUTED':
            return 0.0

        # 利率竞争力
        if pd.isna(row.get('rate_competitiveness')):
            return 0.0

        rate_comp = row['rate_competitiveness']

        # 市场因子: 鼓励在高成交率场景提高利率
        currency = row.get('currency', '')
        period = row.get('period', 0)

        if currency == 'fUST' or period in [15, 60, 90]:
            market_factor = 1.2  # 高成交场景奖励
        else:
            market_factor = 1.0

        reward = rate_comp * market_factor

        # 上限 2.0
        return min(reward, 2.0)

    def build_training_data(
        self,
        start_date: str,
        end_date: str,
        include_execution_results: bool = True
    ) -> pd.DataFrame:
        """
        主入口: 生成融合训练数据

        流程:
        1. 加载历史市场数据 (funding_rates)
        2. 加载虚拟订单执行结果 (virtual_orders)
        3. 时间对齐融合
        4. 生成增强标签

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            include_execution_results: 是否包含执行结果 (默认True)

        Returns:
            融合训练数据 DataFrame
        """
        print("\n" + "="*60)
        print("🚀 训练数据构建器启动")
        print("="*60)
        print(f"📅 时间范围: {start_date} 至 {end_date}")
        print(f"📊 融合执行结果: {'是' if include_execution_results else '否'}")
        print()

        # Step 1: 加载市场数据
        print("[1/3] 加载历史市场数据...")
        market_data = self.load_market_data(start_date, end_date)

        if not include_execution_results:
            print("\n⚠️  仅使用市场数据,不融合执行结果")
            return market_data

        # Step 2: 加载执行结果
        print("\n[2/3] 加载虚拟订单执行结果...")
        execution_results = self.load_execution_results(start_date, end_date)

        # Step 3: 融合数据
        print("\n[3/3] 融合数据并生成增强标签...")
        merged_data = self.merge_market_and_execution(market_data, execution_results)

        # 数据质量报告
        self._print_data_quality_report(merged_data)

        print("\n" + "="*60)
        print("✅ 训练数据构建完成")
        print("="*60)

        return merged_data

    def _print_data_quality_report(self, df: pd.DataFrame):
        """
        打印数据质量报告

        Args:
            df: 训练数据
        """
        print("\n" + "-"*60)
        print("📊 数据质量报告")
        print("-"*60)

        total = len(df)
        print(f"总样本数: {total:,}")

        if 'actual_execution_binary' in df.columns:
            matched = df['actual_execution_binary'].notna().sum()
            exec_count = (df['actual_execution_binary'] == 1).sum()
            failed_count = (df['actual_execution_binary'] == 0).sum()

            print(f"\n执行结果覆盖:")
            coverage = 100 * matched / total if total > 0 else 0.0
            print(f"  - 包含执行结果: {matched:,} ({coverage:.1f}%)")
            print(f"  - 成交样本: {exec_count:,}")
            print(f"  - 失败样本: {failed_count:,}")

            if matched > 0:
                exec_rate = exec_count / matched
                print(f"  - 成交率: {exec_rate:.2%}")

        if 'revenue_reward' in df.columns:
            valid_rewards = df['revenue_reward'].dropna()
            if len(valid_rewards) > 0:
                print(f"\n收益奖励统计:")
                print(f"  - 有效样本: {len(valid_rewards):,}")
                print(f"  - 范围: [{valid_rewards.min():.2f}, {valid_rewards.max():.2f}]")
                print(f"  - 平均值: {valid_rewards.mean():.3f}")
                print(f"  - 中位数: {valid_rewards.median():.3f}")

        if 'rate_competitiveness' in df.columns:
            valid_comp = df['rate_competitiveness'].dropna()
            if len(valid_comp) > 0:
                print(f"\n利率竞争力统计:")
                print(f"  - 有效样本: {len(valid_comp):,}")
                print(f"  - 范围: [{valid_comp.min():.2f}, {valid_comp.max():.2f}]")
                print(f"  - 平均值: {valid_comp.mean():.3f}")

        if 'follow_error' in df.columns:
            valid_follow = df['follow_error'].dropna()
            if len(valid_follow) > 0:
                print(f"\n市场跟随误差统计 (pred - market_median):")
                print(f"  - 有效样本: {len(valid_follow):,}")
                print(f"  - MAE: {valid_follow.abs().mean():.4f}")
                print(f"  - 中位数: {valid_follow.median():.4f}")

        if 'direction_match' in df.columns:
            valid_match = df['direction_match'].dropna()
            if len(valid_match) > 0:
                print(f"\n方向一致性:")
                print(f"  - 有效样本: {len(valid_match):,}")
                print(f"  - 一致率: {100.0 * valid_match.mean():.2f}%")

        print("-"*60)


def main():
    """
    测试脚本
    """
    builder = TrainingDataBuilder('data/lending_history.db')

    # 测试: 构建训练数据
    df = builder.build_training_data(
        start_date='2026-01-01',
        end_date='2026-02-08',
        include_execution_results=True
    )

    print("\n" + "="*60)
    print("📋 数据预览")
    print("="*60)
    print(df[['datetime', 'currency', 'period', 'close_annual',
             'actual_execution_binary', 'revenue_reward',
             'rate_competitiveness']].head(10))

    # 保存样本数据
    output_path = 'data/training_sample.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ 样本数据已保存: {output_path}")


if __name__ == '__main__':
    main()
