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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from ml_engine.data_processor import DataProcessor
from ml_engine.predictor import EnsemblePredictor
from ml_engine.system_policy import load_system_policy


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
        self.policy = load_system_policy()

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

    def _virtual_orders_columns(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(virtual_orders)")
            return [row[1] for row in cursor.fetchall()]
        finally:
            conn.close()

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
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
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
        HAVING COUNT(*) >= 3
        """

        cursor.execute(query, (since_date,))
        rows = cursor.fetchall()
        conn.close()

        anomalies = []
        for currency, period, total, executed in rows:
            # P5: 按周期区分 min_orders 阈值,与 execution_features.py 一致
            required_min = 3 if period >= 60 else 5
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

    def _check_market_divergence_trigger(self) -> bool:
        """
        检测多数活跃 (currency, period) 的预测利率是否系统性高于市场 2 倍以上。
        市场崩塌后 Blend Zone 和 exec_rate 滞后时的补充保险触发器。
        若 >= 50% 的活跃组合 avg(predicted_rate)/avg(market_median) > 2.0，返回 True。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            since_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT currency, period,
                       AVG(predicted_rate) as avg_pred,
                       AVG(market_median) as avg_market
                FROM virtual_orders
                WHERE order_timestamp >= ?
                  AND market_median IS NOT NULL
                  AND market_median > 0
                  AND predicted_rate IS NOT NULL
                GROUP BY currency, period
                HAVING COUNT(*) >= 3
            """, (since_date,))
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return False

        overpriced = sum(
            1 for r in rows
            if r[2] is not None and r[3] is not None and (r[2] / (r[3] + 1e-8)) > 2.0
        )
        ratio = overpriced / len(rows)
        if ratio >= 0.5:
            logger.info(
                f"Market divergence trigger: {overpriced}/{len(rows)} pairs overpriced >2x "
                f"({ratio:.0%})"
            )
            return True
        return False

    def _check_zero_liquidity_anomaly(self) -> list:
        """检测某(currency, period)组合7天内虚拟订单极少(<2条)的情况。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT currency, period, MAX(order_timestamp) as last_order, COUNT(*) as cnt
                FROM virtual_orders
                WHERE order_timestamp >= datetime('now', '-7 days')
                GROUP BY currency, period
                HAVING cnt < 2
            """)
            return cursor.fetchall()
        finally:
            conn.close()

    def _trigger_thresholds(self) -> Dict[str, float]:
        cfg = self.policy.get("retrain_trigger", {})
        return {
            "score_threshold": float(cfg.get("score_threshold", 1.0)),
            "follow_mae_ratio_threshold": float(cfg.get("follow_mae_ratio_threshold", 0.65)),
            "direction_match_threshold": float(cfg.get("direction_match_threshold", 0.40)),
            "p120_step_p95_threshold": float(cfg.get("p120_step_p95_threshold", 0.05)),
            "global_exec_low": float(cfg.get("global_exec_low", 0.40)),
            "global_exec_high": float(cfg.get("global_exec_high", 0.60)),
        }

    def _compute_retrain_trigger_score(
        self,
        exec_rate_7d: float,
        period_anomalies: List[Dict],
        follow_metrics: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-signal retraining score. Higher score means stronger retrain urgency.
        """
        th = self._trigger_thresholds()
        components = {
            "exec_rate_component": 0.0,
            "period_anomaly_component": 0.0,
            "follow_component": 0.0,
            "direction_component": 0.0,
            "p120_stability_component": 0.0,
        }

        # Global execution anomaly component.
        if exec_rate_7d < th["global_exec_low"]:
            components["exec_rate_component"] = min(
                (th["global_exec_low"] - exec_rate_7d) / max(th["global_exec_low"], 1e-8),
                1.0
            )
        elif exec_rate_7d > th["global_exec_high"]:
            components["exec_rate_component"] = min(
                (exec_rate_7d - th["global_exec_high"]) / max(1.0 - th["global_exec_high"], 1e-8),
                1.0
            )

        # Per-period anomaly component (critical weighted higher).
        critical = sum(1 for x in period_anomalies if x.get("severity") == "critical")
        warning = sum(1 for x in period_anomalies if x.get("severity") == "warning")
        if critical or warning:
            components["period_anomaly_component"] = min(critical * 0.5 + warning * 0.2, 1.0)

        samples = int(follow_metrics.get("samples", 0) or 0)
        p120_samples = int(follow_metrics.get("p120_samples", 0) or 0)
        follow_ratio = float(follow_metrics.get("follow_mae_ratio", 0.0) or 0.0)
        direction_rate = float(follow_metrics.get("direction_match_rate", 0.0) or 0.0)
        p120_p95 = float(follow_metrics.get("p120_step_p95", 0.0) or 0.0)

        if samples >= 40 and follow_ratio > th["follow_mae_ratio_threshold"]:
            components["follow_component"] = min(
                (follow_ratio - th["follow_mae_ratio_threshold"]) /
                max(th["follow_mae_ratio_threshold"], 1e-8),
                1.0
            )

        if samples >= 40 and direction_rate > 0 and direction_rate < th["direction_match_threshold"]:
            components["direction_component"] = min(
                (th["direction_match_threshold"] - direction_rate) /
                max(th["direction_match_threshold"], 1e-8),
                1.0
            )

        if p120_samples >= 10 and p120_p95 > th["p120_step_p95_threshold"]:
            components["p120_stability_component"] = min(
                (p120_p95 - th["p120_step_p95_threshold"]) /
                max(th["p120_step_p95_threshold"], 1e-8),
                1.0
            )

        score = (
            0.25 * components["exec_rate_component"] +
            0.20 * components["period_anomaly_component"] +
            0.25 * components["follow_component"] +
            0.15 * components["direction_component"] +
            0.15 * components["p120_stability_component"]
        )
        return float(score), components

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

        # 条件4: 跟随误差与120d稳定性 (闭环主质量指标)
        follow_metrics = self._get_follow_stability_metrics(days=7)
        if follow_metrics["samples"] > 0:
            print(
                f"跟随误差(近7天): MAE={follow_metrics['follow_mae']:.4f}, "
                f"MAE比率={follow_metrics['follow_mae_ratio']:.3f}, "
                f"方向一致率={follow_metrics['direction_match_rate']:.2%}"
            )
            if follow_metrics["p120_samples"] > 0:
                print(
                    f"120d稳定性: p95(|step_change|)={follow_metrics['p120_step_p95']:.2%} "
                    f"(样本={follow_metrics['p120_samples']})"
                )
        else:
            print("跟随误差指标: 数据不足")

        print("\n判断结果:")
        th = self._trigger_thresholds()

        # 定期重训练 — 双路径: 超时强制 或 常规积累
        if time_since_last >= timedelta(days=14) and new_orders >= 20:
            reason = f"超时强制重训练 (距上次{days_since_last}天, 新增{new_orders}条)"
            print(f"✅ 需要重训练: {reason}")
            return True, reason
        if time_since_last >= timedelta(days=7) and new_orders >= 100:
            reason = f"定期重训练 (距上次{days_since_last}天, 新增{new_orders}条数据)"
            print(f"✅ 需要重训练: {reason}")
            return True, reason

        # 简单直接触发: 全局成交率严重异常 (优先于多信号score，不依赖样本积累)
        exec_low = th.get("global_exec_low", 0.30)
        exec_high = th.get("global_exec_high", 0.60)
        if exec_rate_7d < exec_low:
            reason = f"全局成交率过低 ({exec_rate_7d:.2%} < {exec_low:.0%}), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason
        if exec_rate_7d > exec_high:
            reason = f"全局成交率过高 ({exec_rate_7d:.2%} > {exec_high:.0%}), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 执行率快速下滑检测 (14d→7d 趋势漂移)
        exec_rate_14d = self.get_recent_execution_rate(days=14)
        if exec_rate_14d is not None and exec_rate_7d is not None:
            drift = exec_rate_14d - exec_rate_7d  # 正值 = 近期恶化
            if drift > 0.15 and exec_rate_7d < exec_high:
                reason = f"执行率快速下滑: 14d={exec_rate_14d:.1%}→7d={exec_rate_7d:.1%} (跌幅={drift:.1%})"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

        # 货币对零流动性检测
        zero_liq = self._check_zero_liquidity_anomaly()
        if zero_liq:
            currencies = [(r[0], r[1]) for r in zero_liq]
            reason = f"货币对零流动性 (7天内<2单): {currencies}"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # Multi-signal trigger score (execution + follow + stability + per-period anomalies).
        trigger_score, components = self._compute_retrain_trigger_score(
            exec_rate_7d=exec_rate_7d,
            period_anomalies=period_anomalies,
            follow_metrics=follow_metrics,
        )
        print(
            f"触发分数: {trigger_score:.3f}/{th['score_threshold']:.2f} "
            f"(exec={components['exec_rate_component']:.2f}, "
            f"period={components['period_anomaly_component']:.2f}, "
            f"follow={components['follow_component']:.2f}, "
            f"direction={components['direction_component']:.2f}, "
            f"p120={components['p120_stability_component']:.2f})"
        )
        if trigger_score >= th["score_threshold"]:
            reason = (
                "多信号触发重训练 "
                f"(score={trigger_score:.2f} >= {th['score_threshold']:.2f})"
            )
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

        # 2天短窗口: 检测急剧崩溃（避免7天窗口稀释0%信号）
        short_anomalies = self.get_per_period_execution_anomalies(days=2)
        zero_rate_2d = [a for a in short_anomalies if a['exec_rate'] == 0.0 and a['total'] >= 2]
        if zero_rate_2d:
            details = ", ".join(f"{a['currency']}-{a['period']}d" for a in zero_rate_2d)
            reason = f"2天内执行率归零 ({details}), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 市场偏离触发: 多数 pair 预测利率系统性高于市场 2 倍（market regime change）
        if self._check_market_divergence_trigger():
            reason = "市场偏离触发: >=50% 活跃组合预测利率高于市场中位 2 倍以上"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 不需要重训练
        print(f"❌ 暂不需要重训练")
        print(f"   - 距上次训练: {days_since_last} 天 (需要 >= 7天)")
        print(f"   - 新增数据: {new_orders} 条 (需要 >= 100条)")
        print(
            f"   - 全局成交率: {exec_rate_7d:.2%} "
            f"(正常范围: {th['global_exec_low']:.0%}-{th['global_exec_high']:.0%})"
        )
        print(f"   - 分组异常: 无")
        if follow_metrics["samples"] > 0:
            print(
                f"   - 跟随误差MAE比率: {follow_metrics['follow_mae_ratio']:.2f} "
                f"(阈值: <= {th['follow_mae_ratio_threshold']:.2f})"
            )
            if follow_metrics["p120_samples"] > 0:
                print(
                    f"   - 120d稳定性p95: {follow_metrics['p120_step_p95']:.2%} "
                    f"(阈值: <= {th['p120_step_p95_threshold']:.2%})"
                )

        return False, None

    def _get_follow_stability_metrics(self, days: int = 7) -> Dict[str, float]:
        """
        Calculate closed-loop quality metrics from recent validated orders.

        Metrics:
        - follow_mae: mean(|predicted - market_median|)
        - follow_mae_ratio: follow_mae / mean(market_median)
        - direction_match_rate: mean(direction_match)
        - p120_step_p95: 95th percentile of abs(step_change_pct) for 120d
        """
        conn = sqlite3.connect(self.db_path)
        columns = set(self._virtual_orders_columns())
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        try:
            select_cols = ["predicted_rate", "market_median", "period"]
            if "direction_match" in columns:
                select_cols.append("direction_match")
            if "step_change_pct" in columns:
                select_cols.append("step_change_pct")

            query = f"""
                SELECT {", ".join(select_cols)}
                FROM virtual_orders
                WHERE validated_at >= ?
                  AND status IN ('EXECUTED', 'FAILED')
                  AND market_median IS NOT NULL
            """
            cursor.execute(query, (since_date,))
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "samples": 0,
                "follow_mae": 0.0,
                "follow_mae_ratio": 0.0,
                "direction_match_rate": 0.0,
                "p120_samples": 0,
                "p120_step_p95": 0.0,
            }

        col_idx = {name: idx for idx, name in enumerate(select_cols)}
        abs_errors = []
        medians = []
        direction_vals = []
        p120_steps = []

        for row in rows:
            pred = row[col_idx["predicted_rate"]]
            median = row[col_idx["market_median"]]
            period = row[col_idx["period"]]
            if pred is not None and median is not None:
                abs_errors.append(abs(float(pred) - float(median)))
                medians.append(abs(float(median)))

            if "direction_match" in col_idx:
                dm = row[col_idx["direction_match"]]
                if dm is not None:
                    direction_vals.append(float(dm))

            if "step_change_pct" in col_idx and int(period) == 120:
                step = row[col_idx["step_change_pct"]]
                if step is not None:
                    p120_steps.append(abs(float(step)))

        follow_mae = float(sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0
        denom = float(sum(medians) / len(medians)) if medians else 0.0
        follow_mae_ratio = (follow_mae / denom) if denom > 1e-8 else 0.0
        direction_match_rate = (
            float(sum(direction_vals) / len(direction_vals))
            if direction_vals else 0.0
        )
        p120_step_p95 = (
            float(np.percentile(p120_steps, 95))
            if p120_steps else 0.0
        )

        return {
            "samples": len(abs_errors),
            "follow_mae": follow_mae,
            "follow_mae_ratio": follow_mae_ratio,
            "direction_match_rate": direction_match_rate,
            "p120_samples": len(p120_steps),
            "p120_step_p95": p120_step_p95,
        }

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
            True if new model passes champion/challenger and quality gates
        """
        try:
            val_data = self._prepare_champion_validation_data(days=7, warmup_days=21)
            val_rows = sum(len(df) for df in val_data.values())
            if val_rows < 200:
                print(f"  验证切片不足 ({val_rows} < 200),跳过性能对比,仅执行sanity check")
                comparison['checks']['performance'] = 'skipped_insufficient_data'
                return self._sanity_check_new_models(new_model_dir)

            print(f"  验证切片样本: {val_rows} 行 (近7天)")

            old_eval = self._evaluate_model_dir_on_validation(old_model_dir, val_data)
            new_eval = self._evaluate_model_dir_on_validation(new_model_dir, val_data)

            metrics_comparison = {
                "validation_rows": val_rows,
                "old_overall_score": old_eval["overall_score"],
                "new_overall_score": new_eval["overall_score"],
                "overall_score_delta": new_eval["overall_score"] - old_eval["overall_score"],
                "old_currency_scores": old_eval["currency_scores"],
                "new_currency_scores": new_eval["currency_scores"],
                "old_metrics": old_eval["metrics"],
                "new_metrics": new_eval["metrics"],
            }
            comparison['metrics'] = metrics_comparison

            all_pass = True
            old_score = old_eval["overall_score"]
            new_score = new_eval["overall_score"]

            # New model should not degrade aggregated score by more than 2%.
            if old_score > 0 and new_score < old_score * 0.98:
                print(f"  ❌ 综合分数下降过多: old={old_score:.4f}, new={new_score:.4f}")
                all_pass = False

            # Per-currency guardrail: not worse than 5%.
            for currency in ['fUSD', 'fUST']:
                old_curr = old_eval["currency_scores"].get(currency, 0.0)
                new_curr = new_eval["currency_scores"].get(currency, 0.0)
                if old_curr > 0 and new_curr < old_curr * 0.95:
                    print(f"  ❌ {currency} 分数下降超过5%: old={old_curr:.4f}, new={new_curr:.4f}")
                    all_pass = False

            # Sanity check: 验证新模型基本可用
            sanity_ok = self._sanity_check_new_models(new_model_dir)
            if not sanity_ok:
                all_pass = False

            # Closed-loop quality gate: follow error + 120d stability.
            follow_ok, follow_metrics = self._evaluate_follow_and_stability(days=7)
            metrics_comparison.update(follow_metrics)
            if not follow_ok:
                print("  ❌ 跟随误差/稳定性未达标,拒绝部署")
                all_pass = False

            comparison['checks']['performance'] = 'passed' if all_pass else 'degraded'

            if all_pass:
                print(f"  ✅ 新模型通过同集对比 + sanity check + 闭环质量门禁")
            else:
                print(f"  ❌ 新模型未通过对比门禁")
            return all_pass

        except Exception as e:
            print(f"  ❌ 性能对比异常: {e},拒绝部署")
            comparison['checks']['performance'] = f'error: {e}'
            return False

    def _prepare_champion_validation_data(self, days: int = 7, warmup_days: int = 21) -> Dict[str, pd.DataFrame]:
        """
        Build a shared validation slice for old/new model comparison.
        """
        now = datetime.now()
        start = (now - timedelta(days=days + warmup_days)).strftime('%Y-%m-%d')
        end = now.strftime('%Y-%m-%d')
        since_dt = now - timedelta(days=days)

        data_by_currency: Dict[str, pd.DataFrame] = {}
        processor = DataProcessor(self.db_path)

        for currency in ['fUSD', 'fUST']:
            df = processor.load_data(currency)
            if df.empty:
                continue
            df = df[df['datetime'] >= pd.to_datetime(start)].copy()
            if df.empty:
                continue
            df_feat = df.groupby('period', group_keys=False).apply(processor.add_technical_indicators)
            df_feat = df_feat.sort_values(['currency', 'period', 'datetime'])

            # Traditional targets (same definition as trainer).
            def _compute_targets(group: pd.DataFrame) -> pd.DataFrame:
                group = group.copy()
                indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=120)
                group['future_conservative'] = group['low_annual'].rolling(window=indexer).quantile(0.3)
                group['future_aggressive'] = group['close_annual'].rolling(window=indexer).quantile(0.6)
                group['future_balanced'] = group['close_annual'].rolling(window=indexer).quantile(0.7)
                fut80 = group['close_annual'].rolling(window=indexer).quantile(0.8)
                group['future_execution_prob'] = (group['close_annual'] <= fut80).astype(float)
                return group

            df_feat = df_feat.groupby(['currency', 'period'], group_keys=False).apply(_compute_targets)

            val_df = df_feat[df_feat['datetime'] >= since_dt].copy()
            val_df = val_df.dropna(subset=[
                'future_conservative',
                'future_aggressive',
                'future_balanced',
                'future_execution_prob',
            ])
            if not val_df.empty:
                data_by_currency[currency] = val_df

        return data_by_currency

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        AUC without external dependencies. Returns 0.5 when class is single.
        """
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
        pos_rank_sum = ranks[y_true == 1].sum()
        auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(max(0.0, min(1.0, auc)))

    def _evaluate_model_dir_on_validation(
        self,
        model_dir: str,
        val_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Evaluate one model directory on shared validation slice.
        """
        predictor = EnsemblePredictor(model_dir=model_dir, max_workers=1)
        currency_scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, float]] = {}

        tasks = [
            ('model_execution_prob', 'future_execution_prob', 'classification', 0.25),
            ('model_conservative', 'future_conservative', 'regression', 0.20),
            ('model_aggressive', 'future_aggressive', 'regression', 0.20),
            ('model_balanced', 'future_balanced', 'regression', 0.35),
        ]

        for currency, df in val_data.items():
            metric_vals: Dict[str, float] = {}
            score_parts = []

            for model_type, target, task_type, weight in tasks:
                if currency not in predictor.meta_info or model_type not in predictor.meta_info[currency]:
                    continue
                feature_cols = predictor.meta_info[currency][model_type]['feature_cols']
                missing = [c for c in feature_cols if c not in df.columns]
                if missing:
                    continue

                subset = df[feature_cols + [target]].dropna()
                if len(subset) < 40:
                    continue
                X = subset[feature_cols].copy()
                y = subset[target].values.astype(float)
                y_pred = predictor.predict_with_ensemble(X, currency, model_type)

                if task_type == 'classification':
                    auc = self._safe_auc(y, y_pred)
                    metric_vals[f"{model_type}_auc"] = float(auc)
                    score_parts.append(weight * auc)
                else:
                    mae = float(np.mean(np.abs(y_pred - y)))
                    metric_vals[f"{model_type}_mae"] = mae
                    # Convert MAE to score in (0,1], higher is better.
                    score_parts.append(weight * (1.0 / (1.0 + mae)))

            if score_parts:
                currency_scores[currency] = float(sum(score_parts) / sum(w for _, _, _, w in tasks))
            else:
                currency_scores[currency] = 0.0
            details[currency] = metric_vals

        valid_scores = [v for v in currency_scores.values() if v > 0]
        overall = float(np.mean(valid_scores)) if valid_scores else 0.0
        return {
            "overall_score": overall,
            "currency_scores": currency_scores,
            "metrics": details,
        }

    def _evaluate_follow_and_stability(self, days: int = 7) -> Tuple[bool, Dict]:
        """
        Deployment gate for closed-loop quality.

        Pass criteria:
        - follow_mae_ratio <= policy threshold (when enough samples)
        - direction_match_rate >= policy threshold (when enough samples)
        - p120_step_p95 <= policy threshold (when enough 120d samples)
        """
        metrics = self._get_follow_stability_metrics(days=days)
        th = self._trigger_thresholds()
        output = {
            "follow_mae_7d": metrics["follow_mae"],
            "follow_mae_ratio_7d": metrics["follow_mae_ratio"],
            "direction_match_rate_7d": metrics["direction_match_rate"],
            "p120_step_p95_7d": metrics["p120_step_p95"],
        }

        passed = True
        if metrics["samples"] >= 40 and metrics["follow_mae_ratio"] > th["follow_mae_ratio_threshold"]:
            passed = False
        if metrics["samples"] >= 40 and metrics["direction_match_rate"] > 0 and metrics["direction_match_rate"] < th["direction_match_threshold"]:
            passed = False
        if metrics["p120_samples"] >= 10 and metrics["p120_step_p95"] > th["p120_step_p95_threshold"]:
            passed = False

        return passed, output

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
