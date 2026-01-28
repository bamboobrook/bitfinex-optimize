#!/usr/bin/env python3
"""
Database initialization script for execution feedback system.
Creates tables for virtual orders and execution statistics.
"""

import sqlite3
import sys
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "lending_history.db"


def init_execution_tables():
    """Initialize virtual_orders and execution_statistics tables"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"Initializing execution feedback tables in {DB_PATH}")

    # ===== Table 1: virtual_orders =====
    print("Creating virtual_orders table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS virtual_orders (
            order_id TEXT PRIMARY KEY,
            currency TEXT NOT NULL,
            period INTEGER NOT NULL,
            predicted_rate REAL NOT NULL,
            order_timestamp TEXT NOT NULL,
            validation_window_hours INTEGER NOT NULL,
            status TEXT DEFAULT 'PENDING',

            -- Execution details
            executed_at TEXT,
            execution_rate REAL,
            execution_delay_minutes INTEGER,
            max_market_rate REAL,
            rate_gap REAL,

            -- Auxiliary info
            model_version TEXT,
            prediction_confidence TEXT,
            prediction_strategy TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            validated_at TIMESTAMP,

            CHECK(status IN ('PENDING', 'EXECUTED', 'FAILED', 'EXPIRED'))
        )
    """)

    # Indexes for virtual_orders
    print("Creating indexes for virtual_orders...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_status
        ON virtual_orders(status)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_validation
        ON virtual_orders(status, order_timestamp)
        WHERE status = 'PENDING'
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_currency_period
        ON virtual_orders(currency, period)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_orders_timestamp
        ON virtual_orders(order_timestamp DESC)
    """)

    # ===== Table 2: execution_statistics =====
    print("Creating execution_statistics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_statistics (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            currency TEXT NOT NULL,
            period INTEGER NOT NULL,
            date TEXT NOT NULL,

            -- Statistics
            total_orders INTEGER,
            executed_orders INTEGER,
            execution_rate REAL,
            avg_execution_delay REAL,
            avg_spread REAL,
            avg_rate_gap REAL,

            -- Market state
            market_volatility REAL,
            avg_market_rate REAL,

            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(currency, period, date)
        )
    """)

    # Indexes for execution_statistics
    print("Creating indexes for execution_statistics...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_stats_lookup
        ON execution_statistics(currency, period, date DESC)
    """)

    conn.commit()

    # Verify tables created
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name IN ('virtual_orders', 'execution_statistics')
    """)
    tables = cursor.fetchall()

    print(f"\nTables created: {[t[0] for t in tables]}")

    # Show table schemas
    for table_name in ['virtual_orders', 'execution_statistics']:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"\n{table_name} schema:")
        for col in columns:
            print(f"  {col[1]:30s} {col[2]:15s} {'NOT NULL' if col[3] else ''}")

    conn.close()
    print("\n✓ Database initialization complete!")


if __name__ == "__main__":
    try:
        init_execution_tables()
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
