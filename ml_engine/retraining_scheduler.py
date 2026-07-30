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

    def _get_production_model_deployed_at(self) -> Optional[datetime]:
        """
        获取当前生产模型部署时间。

        Returns:
            最近一次成功部署事件时间；旧历史不可用时回退到 meta mtime。
        """
        for entry in reversed(self._load_retraining_history_entries()):
            if entry.get('deployed') is not True:
                continue
            timestamp = entry.get('timestamp')
            if not timestamp:
                continue
            try:
                return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            except (TypeError, ValueError):
                continue

        if not os.path.exists(self.production_model_dir):
            return None

        try:
            meta_files = [
                os.path.join(self.production_model_dir, name)
                for name in os.listdir(self.production_model_dir)
                if name.endswith('_meta.json')
            ]
        except Exception:
            return None

        if not meta_files:
            return None

        newest_mtime = max(os.path.getmtime(path) for path in meta_files)
        return datetime.fromtimestamp(newest_mtime)

    def _get_production_model_age_days(self) -> int:
        """
        获取当前生产模型年龄（以最新 meta 文件为准）。

        Returns:
            距今的天数；若生产模型不存在或无 meta 文件，返回一个很大的值以触发重训。
        """
        deployed_at = self._get_production_model_deployed_at()
        if deployed_at is None:
            return 999

        return max(0, (datetime.now() - deployed_at).days)

    def _load_retraining_history_entries(self) -> List[Dict]:
        """
        兼容读取旧(dict keyed by date)与新(list append-only)格式的重训练历史。
        """
        if not os.path.exists(self.history_log_path):
            return []

        try:
            with open(self.history_log_path, 'r') as f:
                raw = json.load(f)
        except Exception as e:
            print(f"⚠️  读取训练历史失败: {e}")
            return []

        entries: List[Dict] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    entries.append(item)
        elif isinstance(raw, dict):
            for key, value in raw.items():
                if not isinstance(value, dict):
                    continue
                item = dict(value)
                item.setdefault('history_date', key)
                entries.append(item)

        def _entry_ts(entry: Dict) -> datetime:
            ts_text = entry.get('timestamp')
            if ts_text:
                try:
                    return datetime.strptime(ts_text, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass
            date_text = entry.get('history_date')
            if date_text:
                try:
                    return datetime.strptime(date_text, '%Y-%m-%d')
                except Exception:
                    pass
            return datetime.min

        entries.sort(key=_entry_ts)
        return entries

    def get_last_training_date(self) -> datetime:
        """
        获取上次训练日期

        从重训练历史日志中读取,如果不存在则返回7天前
        """
        history = self._load_retraining_history_entries()
        if history:
            latest = history[-1]
            ts_text = latest.get('timestamp')
            if ts_text:
                try:
                    return datetime.strptime(ts_text, '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"⚠️  解析训练时间失败: {e}")
            date_text = latest.get('history_date')
            if date_text:
                try:
                    return datetime.strptime(date_text, '%Y-%m-%d')
                except Exception as e:
                    print(f"⚠️  解析训练日期失败: {e}")

        # 默认返回7天前
        return datetime.now() - timedelta(days=7)

    @staticmethod
    def _effective_since_dt(days: int, since_dt: Optional[datetime] = None) -> datetime:
        base_since = datetime.now() - timedelta(days=days)
        if since_dt is None:
            return base_since
        return max(base_since, since_dt)

    def _count_orders_since(self, since_dt: datetime) -> int:
        """
        统计某个时间点之后的订单结果数量，用于判断新模型观察样本是否足够。

        `order_timestamp` 是预测使用的市场数据时间，可能早于模型部署/
        订单落库时间。部署后成熟度必须按 `created_at` 归因，否则新模型
        首轮订单会被漏计。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM virtual_orders
                WHERE created_at >= ?
                  AND status IN ('EXECUTED', 'FAILED')
                """,
                (since_dt.strftime('%Y-%m-%d %H:%M:%S'),)
            )
            row = cursor.fetchone()
            return int(row[0] or 0) if row else 0
        finally:
            conn.close()

    def _count_long_window_orders_since(
        self,
        since_dt: datetime,
        min_window_hours: int = 72,
    ) -> int:
        """Count uncensored outcomes from long validation windows."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM virtual_orders
                WHERE created_at >= ?
                  AND validation_window_hours >= ?
                  AND status IN ('EXECUTED', 'FAILED')
                """,
                (
                    since_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    int(min_window_hours),
                ),
            )
            row = cursor.fetchone()
            return int(row[0] or 0) if row else 0
        finally:
            conn.close()

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
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
        """

        cursor.execute(query, (since_date.strftime('%Y-%m-%d'),))
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def get_recent_execution_rate(self, days: int = 7, since_dt: Optional[datetime] = None) -> float:
        """
        获取近期全局成交率

        Args:
            days: 天数
            since_dt: 可选起始时间；若提供，则取 max(now-days, since_dt)

        Returns:
            成交率 (0-1)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = self._effective_since_dt(days=days, since_dt=since_dt).strftime('%Y-%m-%d %H:%M:%S')
        time_column = "created_at" if since_dt is not None else "order_timestamp"

        query = """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed
        FROM virtual_orders
        WHERE {time_column} >= ?
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
        """

        cursor.execute(query.format(time_column=time_column), (since_date,))
        result = cursor.fetchone()
        conn.close()

        total, executed = result
        if total == 0:
            return 0.0  # 零订单表示无成交，不应伪装健康

        return executed / total

    def get_per_period_execution_anomalies(
        self,
        days: int = 7,
        since_dt: Optional[datetime] = None
    ) -> list:
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

        since_date = self._effective_since_dt(days=days, since_dt=since_dt).strftime('%Y-%m-%d %H:%M:%S')
        time_column = "created_at" if since_dt is not None else "order_timestamp"

        query = """
        SELECT
            currency, period,
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed
        FROM virtual_orders
        WHERE {time_column} >= ?
          AND status IN ('EXECUTED', 'FAILED')
        GROUP BY currency, period
        HAVING COUNT(*) >= 3
        """

        cursor.execute(query.format(time_column=time_column), (since_date,))
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

    def _check_market_divergence_trigger(self, since_dt: Optional[datetime] = None) -> bool:
        """
        检测多数活跃 (currency, period) 的预测利率是否系统性高于市场 2 倍以上。
        市场崩塌后 Blend Zone 和 exec_rate 滞后时的补充保险触发器。
        若 >= 50% 的活跃组合 avg(predicted_rate)/avg(market_median) > 2.0，返回 True。
        """
        columns = set(self._virtual_orders_columns())
        if "market_median" not in columns:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            since_date = self._effective_since_dt(days=7, since_dt=since_dt).strftime('%Y-%m-%d %H:%M:%S')
            time_column = "created_at" if since_dt is not None else "order_timestamp"
            cursor.execute("""
                SELECT currency, period,
                       AVG(predicted_rate) as avg_pred,
                       AVG(market_median) as avg_market
                FROM virtual_orders
                WHERE {time_column} >= ?
                  AND market_median IS NOT NULL
                  AND market_median > 0
                  AND predicted_rate IS NOT NULL
                GROUP BY currency, period
                HAVING COUNT(*) >= 3
            """.format(time_column=time_column), (since_date,))
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
            print(
                f"⚠️  Market divergence trigger: {overpriced}/{len(rows)} pairs overpriced >2x "
                f"({ratio:.0%})"
            )
            return True
        return False

    def _check_zero_liquidity_anomaly(self, since_dt: Optional[datetime] = None) -> list:
        """
        检测“有足够成熟结果但一次都未成交”的组合。

        旧实现统计的是生成订单数，并把长周期尚未到验证窗口的 PENDING
        订单误判成零流动性；这里只看已决结果，并按 created_at 归因到
        当前模型。没有生成订单的组合由 refresh-probe/暂停机制处理，不触发
        重训练。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            since_date = self._effective_since_dt(days=7, since_dt=since_dt).strftime('%Y-%m-%d %H:%M:%S')
            min_decided = int(
                self.policy.get("retrain_trigger", {}).get("zero_liq_min_decided_orders", 5)
            )
            cursor.execute("""
                SELECT
                    currency,
                    period,
                    MAX(created_at) AS last_created,
                    SUM(CASE WHEN status IN ('EXECUTED', 'FAILED') THEN 1 ELSE 0 END) AS decided,
                    SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) AS executed
                FROM virtual_orders
                WHERE created_at >= ?
                GROUP BY currency, period
                HAVING decided >= ? AND executed = 0
            """, (since_date, min_decided))
            return cursor.fetchall()
        finally:
            conn.close()

    def _trigger_thresholds(self) -> Dict[str, float]:
        cfg = self.policy.get("retrain_trigger", {})
        return {
            "score_threshold": float(cfg.get("score_threshold", 0.5)),
            "follow_mae_ratio_threshold": float(cfg.get("follow_mae_ratio_threshold", 0.65)),
            "direction_match_threshold": float(cfg.get("direction_match_threshold", 0.40)),
            "p120_step_p95_threshold": float(cfg.get("p120_step_p95_threshold", 0.05)),
            "global_exec_low": float(cfg.get("global_exec_low", 0.30)),
            "global_exec_high": float(cfg.get("global_exec_high", 0.60)),
            "post_deploy_grace_hours": float(cfg.get("post_deploy_grace_hours", 12.0)),
            "post_deploy_min_orders": int(cfg.get("post_deploy_min_orders", 40)),
            "post_deploy_high_exec_grace_hours": float(
                cfg.get("post_deploy_high_exec_grace_hours", 72.0)
            ),
            "post_deploy_high_exec_min_orders": int(
                cfg.get("post_deploy_high_exec_min_orders", 120)
            ),
            "post_deploy_high_exec_min_72h_orders": int(
                cfg.get("post_deploy_high_exec_min_72h_orders", 20)
            ),
            "post_deploy_zero_liq_min_hours": float(
                cfg.get("post_deploy_zero_liq_min_hours", 72.0)
            ),
            "post_deploy_zero_liq_min_orders": int(
                cfg.get("post_deploy_zero_liq_min_orders", 120)
            ),
            "zero_liq_min_decided_orders": int(
                cfg.get("zero_liq_min_decided_orders", 5)
            ),
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
        0. 生产模型过期 >14天（最高优先级）
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

        # 条件0（最高优先级）: 生产模型过期
        model_age = self._get_production_model_age_days()
        print(f"生产模型年龄: {model_age} 天")
        if model_age > 14:
            reason = f"生产模型已{model_age}天未更新，强制重训"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 条件1: 时间和数据量
        last_train_date = self.get_last_training_date()
        # B5 FIX: Use timedelta comparison instead of .days to avoid off-by-one truncation
        time_since_last = datetime.now() - last_train_date
        days_since_last = time_since_last.days
        print(f"上次训练日期: {last_train_date.strftime('%Y-%m-%d')}")
        print(f"距今天数: {days_since_last} 天")

        new_orders = self.count_new_execution_results(last_train_date)
        print(f"新增执行结果: {new_orders} 条")

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

        # 新模型部署后，质量信号只看“当前生产模型部署后的订单”，避免旧模型坏账反复触发重训。
        deployed_at = self._get_production_model_deployed_at()
        quality_since_dt = deployed_at
        post_deploy_age_hours = None
        post_deploy_orders = None
        post_deploy_72h_orders = None
        high_exec_maturity_ready = True
        if deployed_at is not None:
            post_deploy_age_hours = (datetime.now() - deployed_at).total_seconds() / 3600.0
            post_deploy_orders = self._count_orders_since(deployed_at)
            post_deploy_72h_orders = self._count_long_window_orders_since(deployed_at)
            high_exec_maturity_ready = (
                post_deploy_age_hours >= th["post_deploy_high_exec_grace_hours"] and
                post_deploy_orders >= th["post_deploy_high_exec_min_orders"] and
                post_deploy_72h_orders >= th["post_deploy_high_exec_min_72h_orders"]
            )
            print(f"当前生产模型部署时间: {deployed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(
                f"部署后已观察: {post_deploy_age_hours:.1f} 小时, "
                f"订单结果: {post_deploy_orders} 条, 72h结果: {post_deploy_72h_orders} 条"
            )

            if (
                post_deploy_age_hours < th["post_deploy_grace_hours"] or
                post_deploy_orders < th["post_deploy_min_orders"]
            ):
                reasons = []
                if post_deploy_age_hours < th["post_deploy_grace_hours"]:
                    reasons.append(
                        f"观察时间不足 {post_deploy_age_hours:.1f}h < {th['post_deploy_grace_hours']:.1f}h"
                    )
                if post_deploy_orders < th["post_deploy_min_orders"]:
                    reasons.append(
                        f"部署后样本不足 {post_deploy_orders} < {th['post_deploy_min_orders']}"
                    )
                print("\n判断结果:")
                print(
                    "⏳ 暂不重训练: 当前生产模型仍在观察窗口内, "
                    + "，".join(reasons)
                )
                return False, None

        # 条件2: 全局近期成交率（若新模型刚部署，则仅看部署后的订单）
        exec_rate_7d = self.get_recent_execution_rate(days=7, since_dt=quality_since_dt)
        print(f"近7天全局成交率: {exec_rate_7d:.2%}")

        # 条件3: 按 period 分组检查
        period_anomalies = self.get_per_period_execution_anomalies(days=7, since_dt=quality_since_dt)
        if period_anomalies:
            print(f"按 period 分组异常:")
            for a in period_anomalies:
                print(f"   - [{a['severity']}] {a['currency']} {a['period']}天: {a['exec_rate']:.2%} ({a['total']}单)")
        else:
            print(f"按 period 分组: 无异常")

        # 条件4: 跟随误差与120d稳定性 (闭环主质量指标)
        follow_metrics = self._get_follow_stability_metrics(days=7, since_dt=quality_since_dt)
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

        # 简单直接触发: 全局成交率严重异常 (优先于多信号score，不依赖样本积累)
        exec_low = th.get("global_exec_low", 0.30)
        exec_high = th.get("global_exec_high", 0.60)
        if exec_rate_7d < exec_low:
            reason = f"全局成交率过低 ({exec_rate_7d:.2%} < {exec_low:.0%}), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason
        if exec_rate_7d > exec_high:
            if high_exec_maturity_ready:
                reason = f"全局成交率过高 ({exec_rate_7d:.2%} > {exec_high:.0%}), 紧急重训练"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason
            print(
                "⏳ 高成交率信号暂缓: 等待完整验证窗口 "
                f"({post_deploy_age_hours:.1f}h/{th['post_deploy_high_exec_grace_hours']:.0f}h, "
                f"{post_deploy_orders}/{th['post_deploy_high_exec_min_orders']}单, "
                f"72h={post_deploy_72h_orders}/{th['post_deploy_high_exec_min_72h_orders']}单)"
            )

        # 执行率快速下滑检测 (14d→7d 趋势漂移)
        exec_rate_14d = self.get_recent_execution_rate(days=14, since_dt=quality_since_dt)
        if exec_rate_14d is not None and exec_rate_7d is not None:
            drift = exec_rate_14d - exec_rate_7d  # 正值 = 近期恶化
            if drift > 0.15 and exec_rate_7d < exec_high:
                reason = f"执行率快速下滑: 14d={exec_rate_14d:.1%}→7d={exec_rate_7d:.1%} (跌幅={drift:.1%})"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

        # 货币对零流动性检测
        zero_liq_ready = (
            deployed_at is None or
            post_deploy_age_hours is None or
            (
                post_deploy_age_hours >= th["post_deploy_zero_liq_min_hours"]
                and post_deploy_orders >= th["post_deploy_zero_liq_min_orders"]
            )
        )
        if zero_liq_ready:
            zero_liq = self._check_zero_liquidity_anomaly(since_dt=quality_since_dt)
            if zero_liq:
                currencies = [
                    (r[0], r[1], int(r[3] or 0), int(r[4] or 0))
                    for r in zero_liq
                ]
                reason = (
                    "货币对成熟结果零成交 "
                    f"(至少{th['zero_liq_min_decided_orders']}条已决): {currencies}"
                )
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

        # Multi-signal trigger score (execution + follow + stability + per-period anomalies).
        score_exec_rate = exec_rate_7d
        score_period_anomalies = period_anomalies
        if not high_exec_maturity_ready:
            score_exec_rate = min(exec_rate_7d, exec_high)
            score_period_anomalies = [
                anomaly for anomaly in period_anomalies
                if anomaly.get("exec_rate", 0.0) <= exec_high
            ]

        trigger_score, components = self._compute_retrain_trigger_score(
            exec_rate_7d=score_exec_rate,
            period_anomalies=score_period_anomalies,
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
            if critical_high and high_exec_maturity_ready:
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
            if warning_high and high_exec_maturity_ready:
                details = ", ".join(
                    f"{a['currency']} {a['period']}天={a['exec_rate']:.0%}"
                    for a in warning_high
                )
                reason = f"单period成交率偏高(warning) ({details}), 紧急重训练"
                print(f"⚠️  需要重训练: {reason}")
                return True, reason

        # 2天短窗口: 检测急剧崩溃（避免7天窗口稀释0%信号）
        short_anomalies = self.get_per_period_execution_anomalies(days=2, since_dt=quality_since_dt)
        zero_rate_2d = [a for a in short_anomalies if a['exec_rate'] == 0.0 and a['total'] >= 2]
        if zero_rate_2d:
            details = ", ".join(f"{a['currency']}-{a['period']}d" for a in zero_rate_2d)
            reason = f"2天内执行率归零 ({details}), 紧急重训练"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 市场偏离触发: 多数 pair 预测利率系统性高于市场 2 倍（market regime change）
        if zero_liq_ready and self._check_market_divergence_trigger(since_dt=quality_since_dt):
            reason = "市场偏离触发: >=50% 活跃组合预测利率高于市场中位 2 倍以上"
            print(f"⚠️  需要重训练: {reason}")
            return True, reason

        # 不需要重训练
        print(f"❌ 暂不需要重训练")
        print(f"   - 距上次训练: {days_since_last} 天 (需要 >= 7天)")
        print(f"   - 新增数据: {new_orders} 条 (需要 >= 100条)")
        if exec_rate_7d > exec_high and not high_exec_maturity_ready:
            print(
                f"   - 全局成交率: {exec_rate_7d:.2%} "
                f"(偏高，等待 {th['post_deploy_high_exec_grace_hours']:.0f}h/"
                f"{th['post_deploy_high_exec_min_orders']}条已决/"
                f"{th['post_deploy_high_exec_min_72h_orders']}条72h结果后再判断)"
            )
        else:
            print(
                f"   - 全局成交率: {exec_rate_7d:.2%} "
                f"(正常范围: {th['global_exec_low']:.0%}-{th['global_exec_high']:.0%})"
            )
        deferred_high_anomalies = (
            [
                anomaly for anomaly in period_anomalies
                if anomaly.get("exec_rate", 0.0) > exec_high
            ]
            if not high_exec_maturity_ready
            else []
        )
        if deferred_high_anomalies:
            print(
                f"   - 分组异常: {len(deferred_high_anomalies)}条高成交率信号等待成熟"
            )
        elif period_anomalies:
            print(f"   - 分组异常: {len(period_anomalies)}条，均未达到重训条件")
        else:
            print("   - 分组异常: 无")
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

    def _get_follow_stability_metrics(
        self,
        days: int = 7,
        since_dt: Optional[datetime] = None
    ) -> Dict[str, float]:
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

        since_date = self._effective_since_dt(days=days, since_dt=since_dt).strftime('%Y-%m-%d %H:%M:%S')
        time_column = "created_at" if since_dt is not None else "validated_at"

        if "market_median" not in columns:
            conn.close()
            return {
                "samples": 0,
                "follow_mae": 0.0,
                "follow_mae_ratio": 0.0,
                "direction_match_rate": 0.0,
                "p120_samples": 0,
                "p120_step_p95": 0.0,
            }

        try:
            select_cols = ["predicted_rate", "market_median", "period"]
            if "direction_match" in columns:
                select_cols.append("direction_match")
            if "step_change_pct" in columns:
                select_cols.append("step_change_pct")

            query = f"""
                SELECT {", ".join(select_cols)}
                FROM virtual_orders
                WHERE {time_column} >= ?
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

            # 保留最近7天作为真正的 out-of-time champion/challenger 验证集。
            # Challenger 不能先在这批数据上训练，再用同一批数据决定部署。
            training_end = datetime.now() - timedelta(days=7)
            training_start = training_end - timedelta(days=90)
            start_date = training_start.strftime('%Y-%m-%d')
            end_date = training_end.strftime('%Y-%m-%d')

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
            missing_enhanced_models = []
            retained_count = 0
            for model_prefix in enhanced_models:
                old_meta_file = os.path.join(old_model_dir, f"{model_prefix}_meta.json")
                new_meta_file = os.path.join(new_model_dir, f"{model_prefix}_meta.json")

                if os.path.exists(new_meta_file):
                    enhanced_count += 1
                else:
                    # 增强模型是当前闭环能力的必备输出；新模型缺失即拒绝部署。
                    missing_enhanced_models.append(model_prefix)

                if os.path.exists(old_meta_file) and os.path.exists(new_meta_file):
                    retained_count += 1

            print(f"  增强模型: {enhanced_count}/{len(enhanced_models)} (旧模型保留: {retained_count})")

            if missing_enhanced_models:
                print(f"  ⚠️  增强模型缺失: {', '.join(missing_enhanced_models)}（拒绝部署）")
                comparison['checks']['enhanced_models'] = False
                comparison['checks']['enhanced_model_retention'] = False
                comparison['missing_enhanced_models'] = missing_enhanced_models
            else:
                comparison['checks']['enhanced_model_retention'] = True
                comparison['checks']['enhanced_models'] = True
                comparison['missing_enhanced_models'] = []
                print(f"  ✅ 增强模型检查通过")

            # 检查3: 实际性能对比 (S2 核心修复)
            print("\n检查3: 模型性能对比 (验证集)")
            performance_ok = self._compare_model_performance(
                old_model_dir, new_model_dir, comparison
            )

            is_better = (
                comparison['checks']['completeness'] and
                comparison['checks'].get('enhanced_models', False) and
                performance_ok
            )
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
            feedback_by_target = {
                'actual_execution_binary': 0,
                'revenue_optimized_target': 0,
            }
            for frame in val_data.values():
                if not isinstance(frame, pd.DataFrame):
                    continue
                for target in feedback_by_target:
                    if target in frame.columns:
                        feedback_by_target[target] += int(
                            frame[target].notna().sum()
                        )
            feedback_rows = max(feedback_by_target.values(), default=0)
            if val_rows < 200 and feedback_rows < 40:
                print(
                    f"  验证切片不足 (rows={val_rows}, feedback={feedback_rows}),"
                    "跳过性能对比,仅执行sanity check"
                )
                comparison['checks']['performance'] = 'skipped_insufficient_data'
                return self._sanity_check_new_models(new_model_dir)

            print(
                f"  验证切片样本: {val_rows} 行, "
                f"增强标签最多 {feedback_rows} 条 (近7天)"
            )

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

            enhanced_checks = 0
            enhanced_pass = True
            model_gate = self.policy.get("model_gate", {})
            min_v2_auc = float(
                model_gate.get("min_execution_v2_auc", 0.50)
            )
            max_v2_brier = float(
                model_gate.get("max_execution_v2_brier", 0.25)
            )
            max_revenue_mae = float(
                model_gate.get("max_revenue_mae", 5.0)
            )
            for currency in ['fUSD', 'fUST']:
                old_metrics = old_eval["metrics"].get(currency, {})
                new_metrics = new_eval["metrics"].get(currency, {})

                v2_samples = int(
                    new_metrics.get(
                        "model_execution_prob_v2_eligible_samples",
                        old_metrics.get(
                            "model_execution_prob_v2_eligible_samples",
                            0,
                        ),
                    )
                )
                if v2_samples >= 40:
                    new_auc = new_metrics.get("model_execution_prob_v2_auc")
                    new_brier = new_metrics.get("model_execution_prob_v2_brier")
                    old_auc = old_metrics.get("model_execution_prob_v2_auc")
                    old_brier = old_metrics.get("model_execution_prob_v2_brier")
                    if (
                        new_auc is None
                        or new_brier is None
                        or not np.isfinite(float(new_auc))
                        or not np.isfinite(float(new_brier))
                    ):
                        print(
                            f"  ❌ {currency} execution_prob_v2 有 {v2_samples} 条验证样本，"
                            "但 challenger 无有效指标"
                        )
                        all_pass = False
                        enhanced_pass = False
                    else:
                        new_auc = float(new_auc)
                        new_brier = float(new_brier)
                        if old_auc is not None and not np.isfinite(float(old_auc)):
                            old_auc = None
                        if old_brier is not None and not np.isfinite(float(old_brier)):
                            old_brier = None
                        enhanced_checks += 1
                        if new_auc < min_v2_auc:
                            print(
                                f"  ❌ {currency} execution_prob_v2 AUC "
                                f"低于绝对门槛: {new_auc:.4f} < {min_v2_auc:.4f}"
                            )
                            all_pass = False
                            enhanced_pass = False
                        if new_brier > max_v2_brier:
                            print(
                                f"  ❌ {currency} execution_prob_v2 Brier "
                                f"高于绝对门槛: {new_brier:.4f} > {max_v2_brier:.4f}"
                            )
                            all_pass = False
                            enhanced_pass = False
                        if old_auc is not None and new_auc < old_auc - 0.02:
                            print(
                                f"  ❌ {currency} execution_prob_v2 AUC 下降超过0.02: "
                                f"old={old_auc:.4f}, new={new_auc:.4f}"
                            )
                            all_pass = False
                            enhanced_pass = False
                        if (
                            old_brier is not None
                            and old_brier > 0
                            and new_brier > old_brier * 1.05
                        ):
                            print(
                                f"  ❌ {currency} execution_prob_v2 Brier 退化超过5%: "
                                f"old={old_brier:.4f}, new={new_brier:.4f}"
                            )
                            all_pass = False
                            enhanced_pass = False

                revenue_samples = int(
                    new_metrics.get(
                        "model_revenue_optimized_eligible_samples",
                        old_metrics.get(
                            "model_revenue_optimized_eligible_samples",
                            0,
                        ),
                    )
                )
                if revenue_samples >= 40:
                    new_mae = new_metrics.get("model_revenue_optimized_mae")
                    old_mae = old_metrics.get("model_revenue_optimized_mae")
                    if new_mae is None or not np.isfinite(float(new_mae)):
                        print(
                            f"  ❌ {currency} revenue_optimized 有 {revenue_samples} 条验证样本，"
                            "但 challenger 无有效指标"
                        )
                        all_pass = False
                        enhanced_pass = False
                    else:
                        new_mae = float(new_mae)
                        if old_mae is not None and not np.isfinite(float(old_mae)):
                            old_mae = None
                        enhanced_checks += 1
                        if new_mae > max_revenue_mae:
                            print(
                                f"  ❌ {currency} revenue_optimized MAE "
                                f"高于绝对门槛: {new_mae:.4f} > "
                                f"{max_revenue_mae:.4f}"
                            )
                            all_pass = False
                            enhanced_pass = False
                        if (
                            old_mae is not None
                            and old_mae > 0
                            and new_mae > old_mae * 1.05
                        ):
                            print(
                                f"  ❌ {currency} revenue_optimized MAE 退化超过5%: "
                                f"old={old_mae:.4f}, new={new_mae:.4f}"
                            )
                            all_pass = False
                            enhanced_pass = False

            comparison['checks']['enhanced_performance'] = (
                'passed'
                if enhanced_checks and enhanced_pass
                else 'degraded'
                if not enhanced_pass
                else 'skipped_insufficient_data'
            )

            # Sanity check: 验证新模型基本可用
            sanity_ok = self._sanity_check_new_models(new_model_dir)
            if not sanity_ok:
                all_pass = False

            # Live follow/stability is a retraining trigger signal, not a hard
            # deploy gate for the challenger. Otherwise stale production metrics
            # can permanently block any new model from replacing a degraded incumbent.
            follow_ok, follow_metrics = self._evaluate_follow_and_stability(days=7)
            metrics_comparison.update(follow_metrics)
            if not follow_ok:
                print("  ⚠️  当前线上跟随误差/稳定性未达标，仅记录告警，不阻断新模型部署")
                comparison['checks']['live_follow_stability'] = 'warning'
            else:
                comparison['checks']['live_follow_stability'] = 'passed'

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
        Build one shared OOT slice for traditional and feedback models.

        Dense market rows provide forward market targets. Sparse order rows
        provide execution/revenue outcomes, with technical features copied from
        their backward-matched market snapshot.
        """
        now = datetime.now()
        start_dt = now - timedelta(days=days + warmup_days)
        since_dt = now - timedelta(days=days)
        start = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end = now.strftime('%Y-%m-%d %H:%M:%S')

        from ml_engine.model_trainer_v2 import (
            EnhancedModelTrainer,
            add_training_technical_features,
        )
        from ml_engine.training_data_builder import TrainingDataBuilder

        processor = DataProcessor(self.db_path)
        builder = TrainingDataBuilder(self.db_path)
        market_data = builder.load_market_data(start, end)
        execution_results = builder.load_execution_results(start, end)
        combined = builder.merge_market_and_execution(
            market_data,
            execution_results,
        )
        if combined.empty:
            return {}

        featured = add_training_technical_features(combined, processor)
        featured = EnhancedModelTrainer._add_traditional_targets(featured)
        if 'path_terminal_value' in featured.columns:
            featured['revenue_optimized_target'] = featured['path_terminal_value']
        elif 'revenue_reward' in featured.columns:
            featured['revenue_optimized_target'] = (
                featured['close_annual'] * featured['revenue_reward']
            )
        else:
            featured['revenue_optimized_target'] = np.nan

        roles = featured.get(
            '_sample_role',
            pd.Series('market_dense', index=featured.index),
        )
        market_times = pd.to_datetime(featured['datetime'], errors='coerce')
        order_times = pd.to_datetime(
            featured.get('order_timestamp'),
            errors='coerce',
        )
        market_validation = roles.eq('market_dense') & (market_times >= since_dt)
        order_validation = roles.eq('order_supervision') & (order_times >= since_dt)
        if '_exploit_quality' in featured.columns:
            order_validation &= featured['_exploit_quality'].fillna(False)
        validation_frame = featured.loc[
            market_validation | order_validation
        ].copy()

        data_by_currency: Dict[str, pd.DataFrame] = {}
        for currency in ['fUSD', 'fUST']:
            val_df = validation_frame[
                validation_frame['currency'] == currency
            ].copy()
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
        order = np.argsort(y_score, kind='mergesort')
        sorted_scores = y_score[order]
        ranks = np.empty(len(y_score), dtype=float)
        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            # Mann-Whitney AUC requires average ranks for tied predictions.
            average_rank = ((start + 1) + end) / 2.0
            ranks[order[start:end]] = average_rank
            start = end
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
            ('model_execution_prob_v2', 'actual_execution_binary', 'classification', 0.25),
            ('model_revenue_optimized', 'revenue_optimized_target', 'regression', 0.25),
        ]

        for currency, df in val_data.items():
            metric_vals: Dict[str, float] = {}
            score_parts = []
            evaluated_weight = 0.0

            for model_type, target, task_type, weight in tasks:
                if target not in df.columns:
                    metric_vals[f"{model_type}_eligible_samples"] = 0
                    continue
                eligible_rows = int(df[target].notna().sum())
                metric_vals[f"{model_type}_eligible_samples"] = eligible_rows
                if currency not in predictor.meta_info or model_type not in predictor.meta_info[currency]:
                    continue
                feature_cols = predictor.meta_info[currency][model_type]['feature_cols']
                missing = [c for c in feature_cols if c not in df.columns]
                if missing:
                    continue

                subset = df[feature_cols + [target]].dropna(subset=[target])
                if len(subset) < 40:
                    continue
                X = subset[feature_cols].copy()
                y = subset[target].values.astype(float)
                y_pred = predictor.predict_with_ensemble(X, currency, model_type)
                y_pred = np.asarray(y_pred, dtype=float)
                if (
                    len(y_pred) != len(y)
                    or not np.isfinite(y).all()
                    or not np.isfinite(y_pred).all()
                ):
                    metric_vals[f"{model_type}_invalid_predictions"] = 1
                    continue

                if task_type == 'classification':
                    auc = self._safe_auc(y, y_pred)
                    brier = float(np.mean(np.square(y_pred - y)))
                    metric_vals[f"{model_type}_auc"] = float(auc)
                    metric_vals[f"{model_type}_brier"] = brier
                    score_parts.append(weight * auc)
                else:
                    mae = float(np.mean(np.abs(y_pred - y)))
                    metric_vals[f"{model_type}_mae"] = mae
                    # Convert MAE to score in (0,1], higher is better.
                    score_parts.append(weight * (1.0 / (1.0 + mae)))
                evaluated_weight += weight

            if score_parts and evaluated_weight > 0:
                currency_scores[currency] = float(
                    sum(score_parts) / evaluated_weight
                )
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

    @staticmethod
    def _check_model_artifact_set(
        model_dir: str,
        model_prefix: str,
    ) -> Tuple[bool, str]:
        """Validate every positively weighted ensemble component on disk."""
        meta_path = os.path.join(model_dir, f"{model_prefix}_meta.json")
        if not os.path.exists(meta_path):
            return False, f"缺少 {model_prefix}_meta.json"
        try:
            with open(meta_path, 'r', encoding='utf-8') as meta_file:
                meta = json.load(meta_file)
        except Exception as exc:
            return False, f"{model_prefix} meta无法读取: {exc}"

        weights = meta.get('weights')
        if not isinstance(weights, dict) or not weights:
            return False, f"{model_prefix} weights无效"
        suffixes = {
            'xgb': '_xgb.json',
            'lgb': '_lgb.txt',
            'cat': '_cat.cbm',
        }
        positive_components = 0
        for algorithm, raw_weight in weights.items():
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                return False, f"{model_prefix} {algorithm}权重无效"
            if not np.isfinite(weight) or weight < 0:
                return False, f"{model_prefix} {algorithm}权重无效"
            if weight <= 1e-12:
                continue
            positive_components += 1
            suffix = suffixes.get(algorithm)
            if suffix is None:
                return False, f"{model_prefix} 未知组件 {algorithm}"
            component_path = os.path.join(
                model_dir,
                f"{model_prefix}{suffix}",
            )
            if (
                not os.path.exists(component_path)
                or os.path.getsize(component_path) <= 0
            ):
                return False, f"{model_prefix} 缺少正权重组件 {algorithm}"

        if positive_components == 0:
            return False, f"{model_prefix} 没有正权重组件"
        return True, ""

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
            required_types = [
                'model_execution_prob',
                'model_conservative',
                'model_aggressive',
                'model_balanced',
                'model_execution_prob_v2',
                'model_revenue_optimized',
            ]

            # 检查每个币种的核心与增强模型
            for currency in ['fUSD', 'fUST']:
                required_models = [
                    f'{currency}_{model_type}'
                    for model_type in required_types
                ]
                for model_prefix in required_models:
                    complete, reason = self._check_model_artifact_set(
                        model_dir,
                        model_prefix,
                    )
                    if not complete:
                        print(f"  ❌ sanity check失败: {reason}")
                        return False

            # 加载新模型并做简单预测验证
            from ml_engine.predictor import EnsemblePredictor
            test_predictor = EnsemblePredictor(model_dir=model_dir, max_workers=1)

            # 获取最新特征数据做几组预测
            for currency in ['fUSD', 'fUST']:
                if currency not in test_predictor.models or not test_predictor.models[currency]:
                    print(f"  ❌ sanity check失败: {currency} 模型加载失败")
                    return False

                # 检查所有必要模型类型都加载成功
                for mt in required_types:
                    if mt not in test_predictor.models[currency]:
                        print(f"  ❌ sanity check失败: {currency} 缺少 {mt}")
                        return False

                df = test_predictor.processor.load_data(currency)
                if df.empty:
                    print(f"  ⚠️  sanity check: {currency} 无数据,跳过预测验证")
                    continue

                # 取一个period的最新数据
                df = df.sort_values(['period', 'datetime'], kind='mergesort')
                feature_groups = [
                    test_predictor.processor.add_technical_indicators(group)
                    for _, group in df.groupby('period', sort=False)
                ]
                df_feat = pd.concat(
                    feature_groups,
                    ignore_index=True,
                    sort=False,
                )
                sample = df_feat.groupby('period').tail(1).head(3)

                if sample.empty:
                    print(f"  ⚠️  sanity check: {currency} 特征数据为空")
                    continue

                for _, row in sample.iterrows():
                    for model_type in required_types:
                        try:
                            meta = test_predictor.meta_info[currency].get(
                                model_type
                            )
                            if not meta:
                                print(
                                    f"  ❌ sanity check失败: {currency} "
                                    f"{model_type} meta信息缺失"
                                )
                                return False
                            feature_cols = meta['feature_cols']
                            missing = [
                                col for col in feature_cols
                                if col not in row.index
                            ]
                            if missing:
                                print(
                                    f"  ❌ sanity check失败: {currency} "
                                    f"{model_type} 缺少特征 {missing[:5]}"
                                )
                                return False
                            X_single = pd.DataFrame(
                                [{col: row[col] for col in feature_cols}]
                            )
                            pred_val = float(
                                test_predictor.predict_with_ensemble(
                                    X_single,
                                    currency,
                                    model_type,
                                )[0]
                            )
                            if not np.isfinite(pred_val):
                                print(
                                    f"  ❌ sanity check失败: {currency} "
                                    f"period={int(row['period'])} "
                                    f"{model_type} 输出非有限值"
                                )
                                return False

                            if 'execution_prob' in model_type:
                                lower, upper = 0.0, 1.0
                            else:
                                lower, upper = 0.5, 50.0
                            if pred_val < lower or pred_val > upper:
                                print(
                                    f"  ❌ sanity check失败: {currency} "
                                    f"period={int(row['period'])} "
                                    f"{model_type}={pred_val:.4f} "
                                    f"超出合理范围[{lower}, {upper}]"
                                )
                                return False
                        except Exception as e:
                            print(
                                f"  ❌ sanity check失败: {currency} "
                                f"{model_type} 预测异常: {e}"
                            )
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

            prod_dir = self.production_model_dir

            # 合并部署：保留旧模型中存在但新模型中未重新训练的增强模型
            enhanced_suffixes = ['_v2_', '_revenue_optimized_']
            retained_files = []
            if os.path.exists(prod_dir):
                for fname in os.listdir(prod_dir):
                    src_path = os.path.join(prod_dir, fname)
                    dst_path = os.path.join(new_model_dir, fname)
                    if not os.path.exists(dst_path) and os.path.isfile(src_path):
                        if any(s in fname for s in enhanced_suffixes):
                            shutil.copy2(src_path, dst_path)
                            retained_files.append(fname)

                # 删除旧模型
                print(f"删除旧模型: {prod_dir}")
                shutil.rmtree(prod_dir)

            # 复制新模型（含保留的增强模型）
            print(f"复制新模型: {new_model_dir} -> {prod_dir}")
            shutil.copytree(new_model_dir, prod_dir)

            if retained_files:
                print(f"  📎 保留旧增强模型: {', '.join(retained_files)}")

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
        history = self._load_retraining_history_entries()
        history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trigger': trigger,
            'retrained': retrained,
            'deployed': deployed,
            'comparison': comparison
        })

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

    def run(self, force: bool = False) -> str:
        """
        执行完整的重训练流程

        Args:
            force: 是否强制重训练(忽略判断条件)

        Returns:
            状态码: 'deployed' | 'trained_not_deployed' | 'trained_not_better' | 'not_needed' | 'train_failed'
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
                return 'not_needed'
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
            return 'train_failed'

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
                return 'deployed'
            else:
                print("\n" + "="*80)
                print(" "*20 + "⚠️  部署失败,保持现有模型")
                print("="*80)
                self.cleanup_old_artifacts(retrained_dir)
                return 'trained_not_deployed'
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
            return 'trained_not_better'


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
        if not args.force:
            should_retrain, reason = scheduler.should_retrain()
            if not should_retrain:
                print("\n" + "="*80)
                print(" "*20 + "✅ 无需重训练,流程结束")
                print("="*80)
                raise SystemExit(0)

        ok = scheduler.run(force=args.force)
        # exit code: 0=部署成功或训练成功但未部署, 1=训练失败, 2=无需重训练
        exit_code = {
            'deployed': 0,
            'trained_not_deployed': 0,
            'trained_not_better': 0,
            'not_needed': 2,
            'train_failed': 1,
        }.get(ok, 1)
        print(f"\n重训练结果: {ok} (exit code: {exit_code})")
        raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
