#!/usr/bin/env python3
"""
回补未验证的EXPIRED订单

功能:
1. 查询所有 status='EXPIRED' AND validated_at IS NULL 的订单
2. 将它们的状态重置为PENDING
3. 使用ExecutionValidator.validate_single_order重新验证
4. 根据市场数据正确分类为EXECUTED/FAILED/EXPIRED
5. 显示回补统计结果

使用方法:
    python ml_engine/backfill_expired_orders.py
"""

import sys
import os
import sqlite3
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.execution_validator import ExecutionValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_unvalidated_expired_orders():
    """获取所有未验证的EXPIRED订单"""
    db_path = 'data/lending_history.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            order_id,
            currency,
            period,
            predicted_rate,
            order_timestamp,
            validation_window_hours,
            created_at
        FROM virtual_orders
        WHERE status = 'EXPIRED'
          AND validated_at IS NULL
        ORDER BY order_timestamp DESC
    """)

    orders = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return orders


def reset_order_to_pending(order_id):
    """将订单状态重置为PENDING,以便重新验证,并返回更新后的订单"""
    db_path = 'data/lending_history.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE virtual_orders
        SET status = 'PENDING',
            validated_at = NULL,
            execution_rate = NULL,
            executed_at = NULL,
            execution_delay_minutes = NULL,
            max_market_rate = NULL,
            rate_gap = NULL
        WHERE order_id = ?
    """, (order_id,))

    conn.commit()

    # 查询更新后的订单数据
    cursor.execute("""
        SELECT * FROM virtual_orders WHERE order_id = ?
    """, (order_id,))

    updated_order = dict(cursor.fetchone())
    conn.close()

    return updated_order


def backfill_orders():
    """回补验证所有未验证的EXPIRED订单"""
    logger.info("=" * 70)
    logger.info("开始回补未验证的EXPIRED订单")
    logger.info("=" * 70)

    # 获取未验证的订单
    orders = get_unvalidated_expired_orders()
    total_orders = len(orders)

    if total_orders == 0:
        logger.info("没有需要回补的订单")
        return

    logger.info(f"找到 {total_orders} 个需要回补验证的订单")

    # 统计结果
    results = {
        'EXECUTED': 0,
        'FAILED': 0,
        'EXPIRED': 0,
        'ERROR': 0
    }

    # 初始化验证器
    validator = ExecutionValidator()

    # 逐个验证订单
    for i, order in enumerate(orders, 1):
        order_id = order['order_id']
        currency = order['currency']
        period = order['period']
        order_time = order['order_timestamp']

        try:
            logger.info(f"\n[{i}/{total_orders}] 验证订单 {order_id}")
            logger.info(f"  币种: {currency}-{period}d")
            logger.info(f"  订单时间: {order_time}")
            logger.info(f"  预测利率: {order['predicted_rate']:.4f}%")

            # 重置订单状态为PENDING,并获取更新后的订单数据
            updated_order = reset_order_to_pending(order_id)
            logger.info(f"  已重置为PENDING状态")

            # 重新验证 - 传递完整的order字典
            result = validator.validate_single_order(updated_order)

            if result:
                status = result.get('status', 'UNKNOWN')
                results[status] = results.get(status, 0) + 1

                logger.info(f"  ✓ 验证完成: {status}")
                if status == 'EXECUTED':
                    logger.info(f"    执行利率: {result.get('execution_rate', 'N/A'):.4f}%")
                    logger.info(f"    执行时间: {result.get('execution_timestamp', 'N/A')}")
                elif status == 'FAILED':
                    logger.info(f"    失败原因: {result.get('reason', 'N/A')}")
                elif status == 'EXPIRED':
                    logger.info(f"    过期原因: {result.get('reason', 'N/A')}")
            else:
                results['ERROR'] += 1
                logger.error(f"  ✗ 验证失败: 无返回结果")

        except Exception as e:
            results['ERROR'] += 1
            logger.error(f"  ✗ 验证异常: {str(e)}")

    # 打印统计结果
    logger.info("\n" + "=" * 70)
    logger.info("回补验证完成 - 统计结果")
    logger.info("=" * 70)
    logger.info(f"总订单数: {total_orders}")
    logger.info(f"EXECUTED (成功执行): {results['EXECUTED']} ({100*results['EXECUTED']/total_orders:.1f}%)")
    logger.info(f"FAILED (未执行):     {results['FAILED']} ({100*results['FAILED']/total_orders:.1f}%)")
    logger.info(f"EXPIRED (真正过期): {results['EXPIRED']} ({100*results['EXPIRED']/total_orders:.1f}%)")
    logger.info(f"ERROR (验证错误):   {results['ERROR']} ({100*results['ERROR']/total_orders:.1f}%)")
    logger.info("=" * 70)

    # 验证是否还有未验证的EXPIRED订单
    remaining = get_unvalidated_expired_orders()
    if len(remaining) == 0:
        logger.info("\n✓ 成功: 所有EXPIRED订单都已验证")
    else:
        logger.warning(f"\n⚠ 警告: 仍有 {len(remaining)} 个未验证的EXPIRED订单")

    # 显示新的执行率统计
    logger.info("\n" + "=" * 70)
    logger.info("更新后的执行率统计")
    logger.info("=" * 70)

    db_path = 'data/lending_history.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status='EXPIRED' THEN 1 ELSE 0 END) as expired,
            ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) /
                  NULLIF(SUM(CASE WHEN status IN ('EXECUTED', 'FAILED') THEN 1 ELSE 0 END), 0), 1) as exec_rate
        FROM virtual_orders
        WHERE order_timestamp >= date('now', '-7 days')
          AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
    """)

    stats = cursor.fetchone()
    conn.close()

    logger.info(f"近7天总订单: {stats[0]}")
    logger.info(f"  EXECUTED: {stats[1]} ({100*stats[1]/stats[0]:.1f}%)")
    logger.info(f"  FAILED:   {stats[2]} ({100*stats[2]/stats[0]:.1f}%)")
    logger.info(f"  EXPIRED:  {stats[3]} ({100*stats[3]/stats[0]:.1f}%)")
    logger.info(f"真实执行率 (EXECUTED / (EXECUTED + FAILED)): {stats[4]:.1f}%")
    logger.info("=" * 70)


if __name__ == '__main__':
    try:
        backfill_orders()
    except KeyboardInterrupt:
        logger.warning("\n中断: 用户取消操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n错误: {str(e)}", exc_info=True)
        sys.exit(1)
