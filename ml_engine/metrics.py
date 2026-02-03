"""
监控指标收集和计算模块
用于追踪系统性能、rate clipping 影响、执行表现等关键指标
"""
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """系统指标收集器"""

    def __init__(self, db_path: str):
        """
        初始化指标收集器

        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path

    def get_clipping_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算 rate clipping 相关指标

        Args:
            predictions: 预测结果列表，每个元素包含 was_clipped, clipping_strategy 等字段

        Returns:
            Dict 包含:
            - clipping_rate: 触发 clipping 的预测比例
            - avg_clip_reduction: 平均削减幅度
            - strategy_distribution: 各策略使用分布
        """
        if not predictions:
            return {
                'clipping_rate': 'N/A',
                'avg_clip_reduction': 'N/A',
                'strategy_distribution': {}
            }

        total = len(predictions)
        clipped_count = 0
        total_reduction = 0.0
        strategy_counts = {'aggressive': 0, 'balanced': 0, 'conservative': 0}

        for pred in predictions:
            # 统计 clipping 策略分布
            strategy = pred.get('clipping_strategy', 'unknown')
            if strategy in strategy_counts:
                strategy_counts[strategy] += 1

            # 统计被削减的预测
            if pred.get('was_clipped', False):
                clipped_count += 1

                # 尝试计算削减幅度（如果有原始预测值）
                # 注意: 当前返回值中没有存储原始预测，这里从边界估算
                final_rate = pred.get('predicted_rate', 0)
                bounds = pred.get('clipping_bounds', {})
                max_bound = bounds.get('max', 0)
                min_bound = bounds.get('min', 0)

                # 估算削减幅度（假设原始值超出边界）
                if final_rate == max_bound and max_bound > 0:
                    # 被上限削减，估算原始值至少超出 10%
                    estimated_reduction = 10.0
                    total_reduction += estimated_reduction
                elif final_rate == min_bound and min_bound > 0:
                    # 被下限削减
                    estimated_reduction = 10.0
                    total_reduction += estimated_reduction

        return {
            'clipping_rate': f"{clipped_count / total * 100:.1f}%" if total > 0 else "0%",
            'clipped_count': clipped_count,
            'total_predictions': total,
            'avg_clip_reduction': f"{total_reduction / clipped_count:.1f}%" if clipped_count > 0 else "0%",
            'strategy_distribution': {
                k: f"{v / total * 100:.1f}%" for k, v in strategy_counts.items()
            }
        }

    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        计算执行相关指标

        Returns:
            Dict 包含:
            - execution_rate: 整体执行率
            - status_breakdown: 各状态订单数量
            - avg_rate_gap_failed: 失败订单的平均利率差
            - cold_start_coverage: Cold start 组合占比
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 1. 执行率统计
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM virtual_orders
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())

            total_validated = status_counts.get('EXECUTED', 0) + status_counts.get('FAILED', 0)
            exec_rate = status_counts.get('EXECUTED', 0) / total_validated * 100 if total_validated > 0 else 0

            # 2. FAILED 订单的平均 rate gap
            cursor.execute("""
                SELECT AVG(rate_gap) as avg_gap
                FROM virtual_orders
                WHERE status='FAILED'
            """)
            avg_gap_row = cursor.fetchone()
            avg_gap = avg_gap_row[0] if avg_gap_row[0] is not None else 0

            # 3. Cold start 覆盖率统计
            # 查询每个 currency-period 组合的订单数
            cursor.execute("""
                SELECT
                    currency || '-' || period as combo,
                    COUNT(*) as order_count
                FROM virtual_orders
                WHERE status IN ('EXECUTED', 'FAILED')
                GROUP BY currency, period
            """)
            combo_counts = cursor.fetchall()

            cold_start_threshold = 10
            cold_start_combos = sum(1 for _, count in combo_counts if count < cold_start_threshold)
            total_combos = len(combo_counts)
            cold_start_pct = cold_start_combos / total_combos * 100 if total_combos > 0 else 0

            conn.close()

            return {
                'execution_rate': f"{exec_rate:.1f}%",
                'total_executed': status_counts.get('EXECUTED', 0),
                'total_failed': status_counts.get('FAILED', 0),
                'total_pending': status_counts.get('PENDING', 0),
                'avg_rate_gap_failed': f"{avg_gap:.4f}%" if avg_gap > 0 else "N/A",
                'cold_start_coverage': f"{cold_start_pct:.1f}%",
                'cold_start_combos': cold_start_combos,
                'total_combos': total_combos
            }

        except Exception as e:
            logger.error(f"Failed to collect execution metrics: {e}")
            return {
                'error': str(e)
            }

    def get_all_metrics(self, predictions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        获取所有监控指标

        Args:
            predictions: 可选的预测结果列表

        Returns:
            包含所有指标的 Dict
        """
        metrics = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'execution_metrics': self.get_execution_metrics()
        }

        if predictions:
            metrics['clipping_metrics'] = self.get_clipping_metrics(predictions)

        return metrics

    def print_metrics_summary(self, predictions: Optional[List[Dict[str, Any]]] = None):
        """
        打印指标摘要到日志

        Args:
            predictions: 可选的预测结果列表
        """
        metrics = self.get_all_metrics(predictions)

        logger.info("=" * 60)
        logger.info("SYSTEM METRICS SUMMARY")
        logger.info("=" * 60)

        # 执行指标
        exec_metrics = metrics.get('execution_metrics', {})
        logger.info(f"Execution Rate: {exec_metrics.get('execution_rate', 'N/A')}")
        logger.info(f"Orders - EXECUTED: {exec_metrics.get('total_executed', 0)}, "
                   f"FAILED: {exec_metrics.get('total_failed', 0)}, "
                   f"PENDING: {exec_metrics.get('total_pending', 0)}")
        logger.info(f"Cold Start Coverage: {exec_metrics.get('cold_start_coverage', 'N/A')} "
                   f"({exec_metrics.get('cold_start_combos', 0)}/{exec_metrics.get('total_combos', 0)} combos)")

        # Clipping 指标
        if 'clipping_metrics' in metrics:
            clip_metrics = metrics['clipping_metrics']
            logger.info(f"Rate Clipping Rate: {clip_metrics.get('clipping_rate', 'N/A')} "
                       f"({clip_metrics.get('clipped_count', 0)}/{clip_metrics.get('total_predictions', 0)} predictions)")
            logger.info(f"Strategy Distribution: {clip_metrics.get('strategy_distribution', {})}")

        logger.info("=" * 60)


def save_metrics_to_file(metrics: Dict[str, Any], output_path: str):
    """
    将指标保存到 JSON 文件

    Args:
        metrics: 指标字典
        output_path: 输出文件路径
    """
    import json

    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
