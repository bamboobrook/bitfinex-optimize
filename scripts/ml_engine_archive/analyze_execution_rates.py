#!/usr/bin/env python3
"""
分析修复后的执行率数据

生成详细的分析报告，包括：
1. 总体执行率对比
2. 按币种分析
3. 按期限分析
4. 预测利率 vs 实际执行利率分析
5. 对系统影响的评估
"""

import sqlite3
import json
from datetime import datetime
import numpy as np

DB_PATH = 'data/lending_history.db'


def get_execution_stats():
    """获取执行统计数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Overall stats
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
            SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed,
            ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 2) as exec_rate
        FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED')
    """)
    overall = cursor.fetchone()

    # By currency
    cursor.execute("""
        SELECT
            currency,
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
            ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 2) as exec_rate,
            ROUND(AVG(predicted_rate), 4) as avg_pred_rate,
            ROUND(AVG(CASE WHEN status='EXECUTED' THEN execution_rate END), 4) as avg_exec_rate,
            ROUND(AVG(CASE WHEN status='EXECUTED' THEN execution_rate - predicted_rate END), 4) as avg_spread
        FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED')
        GROUP BY currency
        ORDER BY currency
    """)
    by_currency = cursor.fetchall()

    # By period
    cursor.execute("""
        SELECT
            currency,
            period,
            COUNT(*) as total,
            SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) as executed,
            ROUND(100.0 * SUM(CASE WHEN status='EXECUTED' THEN 1 ELSE 0 END) / COUNT(*), 2) as exec_rate,
            ROUND(AVG(predicted_rate), 4) as avg_pred_rate,
            ROUND(AVG(CASE WHEN status='EXECUTED' THEN execution_rate END), 4) as avg_exec_rate,
            ROUND(AVG(CASE WHEN status='FAILED' THEN rate_gap END), 4) as avg_rate_gap
        FROM virtual_orders
        WHERE status IN ('EXECUTED', 'FAILED')
        GROUP BY currency, period
        ORDER BY currency, period
    """)
    by_period = cursor.fetchall()

    conn.close()

    return {
        'overall': overall,
        'by_currency': by_currency,
        'by_period': by_period
    }


def print_analysis_report(stats):
    """打印详细的分析报告"""
    print("\n" + "=" * 100)
    print("EXECUTION RATE ANALYSIS REPORT (AFTER FIX)")
    print("=" * 100)

    # Overall stats
    overall = stats['overall']
    print(f"\n{'OVERALL STATISTICS':^100}")
    print("-" * 100)
    print(f"Total validated orders:  {overall[0]:>6}")
    print(f"EXECUTED:                {overall[1]:>6} ({overall[3]:>6.2f}%)")
    print(f"FAILED:                  {overall[2]:>6} ({100-overall[3]:>6.2f}%)")

    # By currency
    print(f"\n{'BY CURRENCY':^100}")
    print("-" * 100)
    print(f"{'Currency':<10} {'Total':>8} {'Executed':>10} {'Exec Rate':>12} {'Avg Pred':>12} {'Avg Exec':>12} {'Spread':>10}")
    print("-" * 100)

    for row in stats['by_currency']:
        currency, total, executed, exec_rate, avg_pred, avg_exec, avg_spread = row
        print(f"{currency:<10} {total:>8} {executed:>10} {exec_rate:>11.2f}% {avg_pred:>11.4f}% {avg_exec or 0:>11.4f}% {avg_spread or 0:>9.4f}%")

    # By period - grouped by currency
    print(f"\n{'BY PERIOD (DETAILED)':^100}")
    print("-" * 100)

    current_currency = None
    for row in stats['by_period']:
        currency, period, total, executed, exec_rate, avg_pred, avg_exec, avg_rate_gap = row

        if currency != current_currency:
            print(f"\n{currency} Currency:")
            print(f"{'Period':<10} {'Total':>8} {'Executed':>10} {'Exec Rate':>12} {'Avg Pred':>12} {'Avg Exec':>12} {'Rate Gap':>12}")
            print("-" * 100)
            current_currency = currency

        print(f"{period:<10} {total:>8} {executed:>10} {exec_rate:>11.2f}% {avg_pred:>11.4f}% {avg_exec or 0:>11.4f}% {avg_rate_gap or 0:>11.4f}%")

    print("=" * 100)

    # Key insights
    print(f"\n{'KEY INSIGHTS':^100}")
    print("-" * 100)

    fusd_stats = [r for r in stats['by_currency'] if r[0] == 'fUSD'][0]
    fust_stats = [r for r in stats['by_currency'] if r[0] == 'fUST'][0]

    print(f"\n1. Currency Performance:")
    print(f"   - fUSD: {fusd_stats[3]:.2f}% execution rate ({fusd_stats[2]}/{fusd_stats[1]})")
    print(f"   - fUST: {fust_stats[3]:.2f}% execution rate ({fust_stats[2]}/{fust_stats[1]})")
    print(f"   - fUST performs {fust_stats[3] - fusd_stats[3]:.2f}% better than fUSD")

    # Period performance
    fusd_periods = [r for r in stats['by_period'] if r[0] == 'fUSD']
    fust_periods = [r for r in stats['by_period'] if r[0] == 'fUST']

    fusd_short = [r for r in fusd_periods if r[1] <= 7]
    fusd_long = [r for r in fusd_periods if r[1] >= 60]

    fust_short = [r for r in fust_periods if r[1] <= 7]
    fust_long = [r for r in fust_periods if r[1] >= 60]

    if fusd_short:
        avg_short_rate = np.mean([r[4] for r in fusd_short])
        print(f"\n2. Period Performance (fUSD):")
        print(f"   - Short-term (2-7 days): {avg_short_rate:.2f}% avg execution rate")

    if fusd_long:
        avg_long_rate = np.mean([r[4] for r in fusd_long])
        print(f"   - Long-term (60-120 days): {avg_long_rate:.2f}% avg execution rate")
        print(f"   - Long-term performs {avg_long_rate - avg_short_rate:.2f}% better")

    # Spread analysis
    executed_spreads = []
    for row in stats['by_period']:
        if row[6] is not None:  # avg_exec_rate
            spread = row[6] - row[5]  # avg_exec - avg_pred
            executed_spreads.append(spread)

    if executed_spreads:
        avg_spread = np.mean(executed_spreads)
        print(f"\n3. Spread Analysis (Executed Orders):")
        print(f"   - Average spread: {avg_spread:.4f}% (execution rate - predicted rate)")
        if avg_spread > 0:
            print(f"   - Users are getting {avg_spread:.4f}% HIGHER rates than predicted (GOOD!)")
        else:
            print(f"   - Users are getting {abs(avg_spread):.4f}% LOWER rates than predicted (BAD)")

    print("=" * 100)


def save_detailed_report(stats):
    """保存详细报告到JSON文件"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'total': stats['overall'][0],
            'executed': stats['overall'][1],
            'failed': stats['overall'][2],
            'execution_rate': stats['overall'][3]
        },
        'by_currency': [
            {
                'currency': row[0],
                'total': row[1],
                'executed': row[2],
                'execution_rate': row[3],
                'avg_predicted_rate': row[4],
                'avg_execution_rate': row[5],
                'avg_spread': row[6]
            }
            for row in stats['by_currency']
        ],
        'by_period': [
            {
                'currency': row[0],
                'period': row[1],
                'total': row[2],
                'executed': row[3],
                'execution_rate': row[4],
                'avg_predicted_rate': row[5],
                'avg_execution_rate': row[6],
                'avg_rate_gap': row[7]
            }
            for row in stats['by_period']
        ]
    }

    report_file = f'data/execution_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")


def main():
    print("Analyzing execution rates after fix...")

    stats = get_execution_stats()
    print_analysis_report(stats)
    save_detailed_report(stats)


if __name__ == '__main__':
    main()
