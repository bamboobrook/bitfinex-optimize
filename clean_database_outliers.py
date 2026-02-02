#!/usr/bin/env python3
"""
清理数据库中的异常数据 (>200%的利率)
"""
import sqlite3
import os
from datetime import datetime
from loguru import logger

DB_PATH = "data/lending_history.db"

def backup_outliers(conn):
    """备份异常数据到单独的表"""
    logger.info("Creating backup of outlier records...")

    cursor = conn.cursor()

    # 创建备份表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates_outliers_backup (
            currency TEXT,
            period INTEGER,
            timestamp INTEGER,
            datetime TEXT,
            open_annual REAL,
            close_annual REAL,
            high_annual REAL,
            low_annual REAL,
            volume REAL,
            hour INTEGER,
            day_of_week INTEGER,
            month INTEGER,
            backup_timestamp TEXT,
            PRIMARY KEY (currency, period, timestamp)
        )
    """)

    # 备份异常数据
    cursor.execute("""
        INSERT OR REPLACE INTO funding_rates_outliers_backup
        SELECT
            currency, period, timestamp, datetime,
            open_annual, close_annual, high_annual, low_annual, volume,
            hour, day_of_week, month,
            ? as backup_timestamp
        FROM funding_rates
        WHERE close_annual > 200
           OR high_annual > 200
           OR low_annual > 200
           OR open_annual > 200
    """, (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),))

    backup_count = cursor.rowcount
    conn.commit()
    logger.info(f"Backed up {backup_count} outlier records to funding_rates_outliers_backup")

    return backup_count

def get_outlier_stats(conn):
    """获取异常数据统计"""
    cursor = conn.cursor()

    # 总体统计
    cursor.execute("""
        SELECT
            currency,
            COUNT(*) as count,
            MIN(close_annual) as min_rate,
            MAX(close_annual) as max_rate,
            COUNT(DISTINCT period) as periods
        FROM funding_rates
        WHERE close_annual > 200
           OR high_annual > 200
           OR low_annual > 200
           OR open_annual > 200
        GROUP BY currency
    """)

    stats = cursor.fetchall()
    return stats

def clean_outliers(conn):
    """删除异常数据"""
    logger.info("Removing outlier records from funding_rates...")

    cursor = conn.cursor()

    # 删除异常数据
    cursor.execute("""
        DELETE FROM funding_rates
        WHERE close_annual > 200
           OR high_annual > 200
           OR low_annual > 200
           OR open_annual > 200
    """)

    deleted_count = cursor.rowcount
    conn.commit()
    logger.info(f"Deleted {deleted_count} outlier records from funding_rates")

    return deleted_count

def clean_negative_rates(conn):
    """删除负利率数据"""
    logger.info("Removing negative rate records...")

    cursor = conn.cursor()

    cursor.execute("""
        DELETE FROM funding_rates
        WHERE close_annual < 0
           OR high_annual < 0
           OR low_annual < 0
           OR open_annual < 0
    """)

    deleted_count = cursor.rowcount
    conn.commit()
    logger.info(f"Deleted {deleted_count} negative rate records")

    return deleted_count

def fix_invalid_hlc(conn):
    """修复不合理的high/low/close关系"""
    logger.info("Fixing invalid high/low/close relationships...")

    cursor = conn.cursor()

    # 找出不合理的记录
    cursor.execute("""
        SELECT COUNT(*)
        FROM funding_rates
        WHERE NOT (high_annual >= close_annual AND close_annual >= low_annual)
    """)

    invalid_count = cursor.fetchone()[0]

    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} records with invalid high/low/close relationships")

        # 修复：将high和low设置为close
        cursor.execute("""
            UPDATE funding_rates
            SET high_annual = close_annual,
                low_annual = close_annual
            WHERE NOT (high_annual >= close_annual AND close_annual >= low_annual)
        """)

        conn.commit()
        logger.info(f"Fixed {invalid_count} records")
    else:
        logger.info("No invalid high/low/close relationships found")

    return invalid_count

def vacuum_database(conn):
    """清理并优化数据库"""
    logger.info("Vacuuming database to reclaim space...")
    cursor = conn.cursor()
    cursor.execute("VACUUM")
    logger.info("Database vacuumed successfully")

def main():
    logger.info("=" * 70)
    logger.info("DATABASE CLEANUP SCRIPT")
    logger.info("=" * 70)

    if not os.path.exists(DB_PATH):
        logger.error(f"Database not found: {DB_PATH}")
        return

    # 获取文件大小
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    logger.info(f"Database size before cleanup: {db_size_mb:.2f} MB")

    conn = sqlite3.connect(DB_PATH)

    # 1. 获取异常数据统计
    logger.info("\n>>> Step 1: Analyzing outliers")
    stats = get_outlier_stats(conn)
    total_outliers = 0
    for currency, count, min_rate, max_rate, periods in stats:
        logger.info(f"{currency}: {count} outliers (range: {min_rate:.2f}% - {max_rate:.2f}%, {periods} periods)")
        total_outliers += count

    if total_outliers == 0:
        logger.info("No outliers found. Database is clean.")
        conn.close()
        return

    # 2. 备份异常数据
    logger.info("\n>>> Step 2: Backing up outliers")
    backup_count = backup_outliers(conn)

    # 3. 删除异常数据
    logger.info("\n>>> Step 3: Cleaning outliers (>200%)")
    deleted_outliers = clean_outliers(conn)

    # 4. 删除负利率数据
    logger.info("\n>>> Step 4: Cleaning negative rates")
    deleted_negatives = clean_negative_rates(conn)

    # 5. 修复不合理的high/low/close关系
    logger.info("\n>>> Step 5: Fixing invalid high/low/close relationships")
    fixed_count = fix_invalid_hlc(conn)

    # 6. 清理数据库
    logger.info("\n>>> Step 6: Vacuuming database")
    vacuum_database(conn)

    conn.close()

    # 获取清理后的文件大小
    db_size_after_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    space_saved_mb = db_size_mb - db_size_after_mb

    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Outliers backed up: {backup_count}")
    logger.info(f"Outliers deleted: {deleted_outliers}")
    logger.info(f"Negative rates deleted: {deleted_negatives}")
    logger.info(f"Invalid records fixed: {fixed_count}")
    logger.info(f"Database size: {db_size_mb:.2f} MB -> {db_size_after_mb:.2f} MB")
    logger.info(f"Space saved: {space_saved_mb:.2f} MB")
    logger.info("=" * 70)
    logger.info("✅ Database cleanup completed successfully!")

if __name__ == "__main__":
    main()
