#!/usr/bin/env python3
"""
重新验证所有历史订单（使用修复后的验证逻辑）

这个脚本将：
1. 备份当前的验证结果
2. 将所有已验证的订单重置为PENDING状态
3. 重新验证所有订单
4. 生成对比报告
"""

import sys
import os
import sqlite3
from datetime import datetime
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.execution_validator import ExecutionValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'data/lending_history.db'


def backup_validation_results():
    """备份当前的验证结果"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get current statistics
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status='EXPIRED' THEN 1 ELSE 0 END) as expired
        FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED', 'EXPIRED')
    """)

    stats = cursor.fetchone()

    backup = {
        'timestamp': datetime.now().isoformat(),
        'before_fix': {
            'total_validated': stats[0],
            'executed': stats[1],
            'failed': stats[2],
            'expired': stats[3],
            'execution_rate': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
        }
    }

    # Save backup to file
    backup_file = f'data/validation_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(backup_file, 'w') as f:
        json.dump(backup, f, indent=2)

    logger.info(f"Backup saved to {backup_file}")
    logger.info(f"Before fix: {stats[1]} EXECUTED, {stats[2]} FAILED out of {stats[0]} total")
    logger.info(f"Execution rate before fix: {backup['before_fix']['execution_rate']:.2f}%")

    conn.close()
    return backup


def reset_validated_orders():
    """将所有已验证的订单重置为PENDING状态"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count orders to reset
    cursor.execute("""
        SELECT COUNT(*) FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED', 'EXPIRED')
    """)
    count = cursor.fetchone()[0]

    logger.info(f"Resetting {count} validated orders to PENDING...")

    # Reset to PENDING
    cursor.execute("""
        UPDATE virtual_orders
        SET status = 'PENDING',
            executed_at = NULL,
            execution_rate = NULL,
            execution_delay_minutes = NULL,
            max_market_rate = NULL,
            rate_gap = NULL,
            validated_at = NULL
        WHERE status IN ('EXECUTED', 'FAILED', 'EXPIRED')
    """)

    conn.commit()
    conn.close()

    logger.info(f"Reset {count} orders to PENDING")
    return count


def revalidate_orders():
    """重新验证所有订单"""
    logger.info("Starting revalidation with fixed logic...")

    validator = ExecutionValidator()

    # Use the force_validate script for old orders
    from ml_engine.validate_old_pending import get_old_pending_orders, force_validate_order

    # Get all PENDING orders (including those we just reset)
    orders = get_old_pending_orders(days_old=0)  # Get all PENDING orders
    total = len(orders)

    logger.info(f"Found {total} orders to validate")

    results = {
        'EXECUTED': 0,
        'FAILED': 0,
        'EXPIRED': 0,
        'ERROR': 0
    }

    for i, order in enumerate(orders, 1):
        if i % 100 == 0:
            logger.info(f"Progress: [{i}/{total}] {100*i/total:.1f}%")

        try:
            result = force_validate_order(validator, order)
            status = result.get('status', 'ERROR')
            results[status] = results.get(status, 0) + 1
        except Exception as e:
            results['ERROR'] += 1
            logger.error(f"Validation error for order {order['order_id'][:8]}: {e}")

    return results, total


def generate_comparison_report(backup, results, total):
    """生成修复前后对比报告"""
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPARISON REPORT")
    logger.info("=" * 80)

    before = backup['before_fix']

    logger.info("\nBEFORE FIX (Old Logic - HIGH >= PREDICTED):")
    logger.info(f"  Total validated: {before['total_validated']}")
    logger.info(f"  EXECUTED: {before['executed']} ({before['execution_rate']:.2f}%)")
    logger.info(f"  FAILED:   {before['failed']}")
    logger.info(f"  EXPIRED:  {before['expired']}")

    exec_rate_after = (results['EXECUTED'] / total * 100) if total > 0 else 0

    logger.info("\nAFTER FIX (New Logic - PREDICTED <= P25):")
    logger.info(f"  Total validated: {total}")
    logger.info(f"  EXECUTED: {results['EXECUTED']} ({exec_rate_after:.2f}%)")
    logger.info(f"  FAILED:   {results['FAILED']}")
    logger.info(f"  EXPIRED:  {results['EXPIRED']}")
    logger.info(f"  ERROR:    {results['ERROR']}")

    logger.info("\nCHANGES:")
    exec_change = exec_rate_after - before['execution_rate']
    logger.info(f"  Execution rate change: {exec_change:+.2f}% "
                f"({before['execution_rate']:.2f}% → {exec_rate_after:.2f}%)")
    logger.info(f"  EXECUTED count change: {results['EXECUTED'] - before['executed']:+d}")
    logger.info(f"  FAILED count change: {results['FAILED'] - before['failed']:+d}")

    logger.info("=" * 80)

    # Save report to file
    report = {
        'timestamp': datetime.now().isoformat(),
        'before_fix': before,
        'after_fix': {
            'total_validated': total,
            'executed': results['EXECUTED'],
            'failed': results['FAILED'],
            'expired': results['EXPIRED'],
            'execution_rate': exec_rate_after
        },
        'changes': {
            'execution_rate_change': exec_change,
            'executed_count_change': results['EXECUTED'] - before['executed'],
            'failed_count_change': results['FAILED'] - before['failed']
        }
    }

    report_file = f'data/revalidation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nReport saved to {report_file}")


def main():
    logger.info("=" * 80)
    logger.info("REVALIDATING ALL ORDERS WITH FIXED LOGIC")
    logger.info("=" * 80)

    # Step 1: Backup current results
    logger.info("\nStep 1: Backing up current validation results...")
    backup = backup_validation_results()

    # Step 2: Reset all validated orders to PENDING
    logger.info("\nStep 2: Resetting validated orders to PENDING...")
    reset_count = reset_validated_orders()

    # Step 3: Revalidate all orders with fixed logic
    logger.info("\nStep 3: Revalidating all orders with fixed logic...")
    results, total = revalidate_orders()

    # Step 4: Generate comparison report
    logger.info("\nStep 4: Generating comparison report...")
    generate_comparison_report(backup, results, total)

    logger.info("\n" + "=" * 80)
    logger.info("REVALIDATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
