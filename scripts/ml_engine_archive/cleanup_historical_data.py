#!/usr/bin/env python3
"""
Cleanup Historical Data

This script removes historical backtest data from the virtual_orders table
to eliminate look-ahead bias. It deletes orders created using historical
timestamps instead of real-time data.

Run this script ONCE after implementing the look-ahead bias fixes.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

DB_PATH = Path(__file__).parent.parent / "data" / "lending_history.db"


def backup_database(db_path: Path):
    """Create a backup of the database before cleanup"""
    backup_path = db_path.parent / f"lending_history_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

    print(f"Creating backup at: {backup_path}")

    import shutil
    shutil.copy2(db_path, backup_path)

    print(f"✅ Backup created successfully")
    return backup_path


def analyze_current_data(conn):
    """Analyze current data to show what will be deleted"""
    cursor = conn.cursor()

    # Total orders
    cursor.execute("SELECT COUNT(*) FROM virtual_orders")
    total_orders = cursor.fetchone()[0]

    # Orders by status
    cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM virtual_orders
        GROUP BY status
    """)
    status_counts = cursor.fetchall()

    # Time range
    cursor.execute("""
        SELECT
            MIN(order_timestamp) as earliest,
            MAX(order_timestamp) as latest,
            MIN(created_at) as first_created,
            MAX(created_at) as last_created
        FROM virtual_orders
    """)
    time_range = cursor.fetchone()

    # Suspicious orders (order_timestamp >> created_at)
    cursor.execute("""
        SELECT COUNT(*) as suspicious
        FROM virtual_orders
        WHERE julianday(created_at) - julianday(order_timestamp) > 1
    """)
    suspicious = cursor.fetchone()[0]

    print("\n=== Current Database Status ===")
    print(f"Total orders: {total_orders}")
    print(f"\nOrders by status:")
    for status, count in status_counts:
        print(f"  {status}: {count}")

    print(f"\nTime range:")
    print(f"  Order timestamps: {time_range[0]} to {time_range[1]}")
    print(f"  Created timestamps: {time_range[2]} to {time_range[3]}")

    print(f"\n⚠️  Suspicious orders (created >1 day after order_timestamp): {suspicious}")

    return total_orders, suspicious


def cleanup_options():
    """Present cleanup options to user"""
    print("\n=== Cleanup Options ===")
    print("1. Delete ALL virtual orders (complete reset)")
    print("2. Delete orders older than 3 days")
    print("3. Delete only suspicious orders (order_timestamp far from created_at)")
    print("4. Cancel (no changes)")

    choice = input("\nSelect option (1-4): ").strip()
    return choice


def execute_cleanup(conn, option: str):
    """Execute the selected cleanup option"""
    cursor = conn.cursor()

    if option == '1':
        # Complete reset
        print("\n⚠️  This will DELETE ALL virtual orders!")
        confirm = input("Type 'DELETE ALL' to confirm: ").strip()

        if confirm != 'DELETE ALL':
            print("Cancelled.")
            return False

        cursor.execute("DELETE FROM virtual_orders")
        deleted = cursor.rowcount

        # Also clear execution statistics if you want a complete reset
        cursor.execute("DELETE FROM execution_statistics")

        print(f"✅ Deleted {deleted} orders (complete reset)")

    elif option == '2':
        # Delete orders older than 3 days
        cutoff = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute("""
            DELETE FROM virtual_orders
            WHERE order_timestamp < ?
        """, (cutoff,))
        deleted = cursor.rowcount

        print(f"✅ Deleted {deleted} orders older than {cutoff}")

    elif option == '3':
        # Delete suspicious orders only
        cursor.execute("""
            DELETE FROM virtual_orders
            WHERE julianday(created_at) - julianday(order_timestamp) > 1
        """)
        deleted = cursor.rowcount

        print(f"✅ Deleted {deleted} suspicious orders")

    elif option == '4':
        print("Cancelled.")
        return False

    else:
        print("Invalid option.")
        return False

    conn.commit()
    return True


def verify_cleanup(conn):
    """Verify cleanup results"""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM virtual_orders")
    remaining = cursor.fetchone()[0]

    print(f"\n=== Cleanup Complete ===")
    print(f"Remaining orders: {remaining}")

    if remaining > 0:
        cursor.execute("""
            SELECT
                MIN(order_timestamp) as earliest,
                MAX(order_timestamp) as latest
            FROM virtual_orders
        """)
        time_range = cursor.fetchone()
        print(f"Order timestamp range: {time_range[0]} to {time_range[1]}")


def main():
    """Main cleanup workflow"""
    print("=" * 60)
    print("Virtual Orders Cleanup Utility")
    print("Purpose: Remove historical backtest data with look-ahead bias")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Analyze current data
    total, suspicious = analyze_current_data(conn)

    if total == 0:
        print("\n✅ Database is already empty (no cleanup needed)")
        conn.close()
        return

    # Create backup
    backup_path = backup_database(DB_PATH)

    # Get cleanup option
    option = cleanup_options()

    # Execute cleanup
    success = execute_cleanup(conn, option)

    if success:
        # Verify results
        verify_cleanup(conn)

        print(f"\n✅ Cleanup completed successfully")
        print(f"📁 Backup saved at: {backup_path}")
        print(f"\n⚡ Next steps:")
        print(f"   1. Run predictor.py to create new real-time orders")
        print(f"   2. Wait 24-72 hours for validation windows to complete")
        print(f"   3. Run execution_validator.py to validate orders")
        print(f"   4. Monitor execution rates (expect 40-60%, not 90%+)")

    conn.close()


if __name__ == "__main__":
    main()
