#!/usr/bin/env python3
"""
强制验证旧的PENDING订单(忽略验证窗口检查)

这个脚本专门用于回补验证那些validation_window已经过期但仍处于PENDING状态的订单
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.execution_validator import ExecutionValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_old_pending_orders(days_old=1):
    """获取N天前的PENDING订单"""
    db_path = 'data/lending_history.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cutoff_date = (datetime.now() - timedelta(days=days_old)).strftime('%Y-%m-%d')

    cursor.execute("""
        SELECT *
        FROM virtual_orders
        WHERE status = 'PENDING'
          AND order_timestamp < ?
        ORDER BY order_timestamp DESC
    """, (cutoff_date,))

    orders = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return orders


def force_validate_order(validator, order):
    """强制验证订单,修改验证逻辑以处理过期的validation_window"""
    try:
        order_id = order['order_id']
        order_time = datetime.fromisoformat(order['order_timestamp'])
        window_hours = order['validation_window_hours']
        window_end = order_time + timedelta(hours=window_hours)

        # 使用当前时间作为验证窗口结束时间(因为validation_window已过期)
        now = datetime.now()
        actual_window_end = min(window_end, now)

        # 查询市场数据
        market_data = validator.query_market_rates(
            order['currency'],
            order['period'],
            order_time,
            actual_window_end
        )

        if not market_data:
            # 标记为EXPIRED
            update_data = {'status': 'EXPIRED'}
            validator.update_order_status(order_id, update_data)
            return {'status': 'EXPIRED', 'reason': 'No market data'}

        # 检查执行 - FIXED LOGIC (Plan B: Percentile-based)
        # 在Lending市场中，Lender设置的利率越低，越容易被Borrower选中
        # 因此只有当预测利率 <= 市场25分位数时，才认为订单成交
        predicted_rate = order['predicted_rate']
        executed = False
        execution_details = {}

        # Collect all market rates during validation window
        market_close_rates = []
        market_data_timestamped = []

        for row in market_data:
            timestamp_str, high_annual, close_annual = row
            close_annual = close_annual or 0.0
            high_annual = high_annual or 0.0

            market_close_rates.append(close_annual)
            market_data_timestamped.append((timestamp_str, close_annual, high_annual))

        if not market_close_rates:
            # No valid market data
            update_data = {'status': 'EXPIRED'}
            validator.update_order_status(order_id, update_data)
            return {'status': 'EXPIRED', 'reason': 'No market data'}

        # Calculate market percentiles
        percentile_25 = np.percentile(market_close_rates, 25)
        min_rate = np.min(market_close_rates)
        max_rate = np.max(market_close_rates)
        median_rate = np.median(market_close_rates)

        # Execution logic: predicted_rate <= 25th percentile
        if predicted_rate <= percentile_25:
            executed = True

            # Find the first timestamp where market rate is close to predicted rate
            execution_timestamp = None
            execution_rate = None

            for timestamp_str, close_rate, high_rate in market_data_timestamped:
                if close_rate >= predicted_rate * 0.95:  # Within 5% tolerance
                    execution_timestamp = timestamp_str
                    execution_rate = close_rate
                    break

            # Fallback: use first datapoint if no match found
            if execution_timestamp is None:
                execution_timestamp = market_data_timestamped[0][0]
                execution_rate = market_data_timestamped[0][1]

            row_time = datetime.fromisoformat(execution_timestamp)
            delay_minutes = int((row_time - order_time).total_seconds() / 60)

            execution_details = {
                'status': 'EXECUTED',
                'executed_at': execution_timestamp,
                'execution_rate': execution_rate,
                'execution_delay_minutes': delay_minutes,
                'max_market_rate': max_rate
            }
        else:
            # Order failed - predicted rate too high (not competitive)
            execution_details = {
                'status': 'FAILED',
                'max_market_rate': max_rate,
                'rate_gap': predicted_rate - percentile_25  # Gap to 25th percentile
            }

        validator.update_order_status(order_id, execution_details)
        return execution_details

    except Exception as e:
        logger.error(f"Error: {e}")
        return {'status': 'ERROR', 'error': str(e)}


def main():
    logger.info("=" * 70)
    logger.info("强制验证旧的PENDING订单")
    logger.info("=" * 70)

    orders = get_old_pending_orders(days_old=1)
    total = len(orders)

    if total == 0:
        logger.info("没有需要验证的旧订单")
        return

    logger.info(f"找到 {total} 个旧PENDING订单需要验证")

    validator = ExecutionValidator()
    results = {
        'EXECUTED': 0,
        'FAILED': 0,
        'EXPIRED': 0,
        'ERROR': 0
    }

    for i, order in enumerate(orders, 1):
        order_id = order['order_id']
        currency = order['currency']
        period = order['period']
        order_time = order['order_timestamp']

        if i % 50 == 0 or i == 1:
            logger.info(f"\n进度: [{i}/{total}] {100*i/total:.1f}%")

        try:
            result = force_validate_order(validator, order)
            status = result.get('status', 'ERROR')
            results[status] = results.get(status, 0) + 1

            if i <= 5 or i % 100 == 0:
                logger.info(f"  订单 {order_id[:8]}... -> {status}")

        except Exception as e:
            results['ERROR'] += 1
            logger.error(f"  验证失败: {order_id[:8]}... - {e}")

    # 统计结果
    logger.info("\n" + "=" * 70)
    logger.info("验证完成 - 统计结果")
    logger.info("=" * 70)
    logger.info(f"总订单数: {total}")
    logger.info(f"EXECUTED: {results['EXECUTED']} ({100*results['EXECUTED']/total:.1f}%)")
    logger.info(f"FAILED:   {results['FAILED']} ({100*results['FAILED']/total:.1f}%)")
    logger.info(f"EXPIRED:  {results['EXPIRED']} ({100*results['EXPIRED']/total:.1f}%)")
    logger.info(f"ERROR:    {results['ERROR']} ({100*results['ERROR']/total:.1f}%)")
    logger.info("=" * 70)

    # 检查未验证的EXPIRED订单
    db_path = 'data/lending_history.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM virtual_orders
        WHERE status='EXPIRED' AND validated_at IS NULL
    """)
    unvalidated_expired = cursor.fetchone()[0]

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed,
            ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN status IN ('EXECUTED', 'FAILED') THEN 1 ELSE 0 END), 0), 1) as exec_rate
        FROM virtual_orders
        WHERE order_timestamp >= date('now', '-7 days')
          AND status IN ('EXECUTED', 'FAILED')
    """)

    stats = cursor.fetchone()
    conn.close()

    logger.info(f"\n未验证的EXPIRED订单: {unvalidated_expired}")
    logger.info(f"近7天执行率: {stats[3]:.1f}% (EXECUTED: {stats[1]}, FAILED: {stats[2]})")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
