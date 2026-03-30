#!/usr/bin/env python3
"""
Virtual order management system.
Handles creation, querying, and updating of virtual orders.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "lending_history.db"


def determine_validation_window(period: int) -> int:
    """
    Determine validation window hours based on lending period.

    Args:
        period: Lending period in days (2-120)

    Returns:
        Validation window in hours (24/48/72)
    """
    if period <= 7:
        return 24
    elif period <= 30:
        return 48
    else:
        return 72


class OrderManager:
    """Manager for virtual orders"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._ensure_schema()

    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        """Ensure optional columns used by closed-loop diagnostics exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(virtual_orders)")
            existing = {row[1] for row in cursor.fetchall()}

            required_columns = {
                "market_follow_error": "REAL",
                "direction_match": "INTEGER",
                "step_change_pct": "REAL",
                "step_capped": "INTEGER",
                "policy_step_cap_pct": "REAL",
                "probe_type": "TEXT",
                "gate_reject_reason": "TEXT",
                "follow_error_at_order": "REAL",
                "execution_threshold": "REAL",
                "market_percentile_40": "REAL",
                "path_value_score": "REAL",
                "stage1_fill_probability": "REAL",
                "frr_proxy_rate": "REAL",
                "frr_fallback_value": "REAL",
                "rank6_fallback_penalty": "REAL",
                "fast_liquidity_score": "REAL",
                "currency_regime_state": "TEXT",
                "expected_terminal_mode": "TEXT",
                "path_stage_outcome": "TEXT",
                "stage1_fill_hours": "INTEGER",
                "stage2_frr_proxy_rate": "REAL",
                "terminal_mode": "TEXT",
            }

            for column, col_type in required_columns.items():
                if column not in existing:
                    cursor.execute(f"ALTER TABLE virtual_orders ADD COLUMN {column} {col_type}")

            conn.commit()
        finally:
            conn.close()

    def create_virtual_order(self, prediction: dict) -> str:
        """
        Create a virtual order from a prediction result.

        Args:
            prediction: Dictionary from predictor.py containing:
                - currency: fUSD/fUST
                - period: lending period
                - predicted_rate: predicted lending rate
                - data_timestamp: timestamp of market data used for prediction
                - prediction_timestamp: when prediction was made
                - confidence: Low/Medium/High
                - strategy: strategy description

        Returns:
            order_id: UUID string
        """
        order_id = str(uuid.uuid4())
        validation_window = determine_validation_window(prediction['period'])
        probe_type = str(prediction.get('probe_type', 'normal') or 'normal')
        force_create = bool(prediction.get('force_create', False))

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Use explicit timestamps to avoid timezone issues
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # CRITICAL FIX: Use data_timestamp as order_timestamp to avoid time inconsistencies
            # order_timestamp should reflect when the market data was captured,
            # not when the order was created
            data_timestamp = prediction.get('data_timestamp', current_time)
            # B2 FIX: Ensure consistent timestamp format (strftime, not isoformat)
            if isinstance(data_timestamp, datetime):
                data_timestamp = data_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(data_timestamp, str) and 'T' in data_timestamp:
                # Convert isoformat to strftime format
                data_timestamp = data_timestamp.replace('T', ' ')[:19]

            # 修改5.1: 订单去重 - 检查是否已存在相同 (currency, period, order_timestamp) 的 PENDING 订单
            if not force_create:
                cursor.execute("""
                    SELECT COUNT(*) FROM virtual_orders
                    WHERE currency = ? AND period = ? AND order_timestamp = ? AND status = 'PENDING'
                """, (prediction['currency'], prediction['period'], data_timestamp))
                existing_count = cursor.fetchone()[0]
                if existing_count > 0:
                    logger.warning(
                        f"Duplicate order skipped: {prediction['currency']}-{prediction['period']}d "
                        f"@ {data_timestamp} already has {existing_count} PENDING order(s)"
                    )
                    conn.close()
                    return f"DUPLICATE_SKIPPED_{prediction['currency']}_{prediction['period']}"

            cursor.execute("""
                INSERT INTO virtual_orders
                (order_id, currency, period, predicted_rate, order_timestamp,
                 validation_window_hours, prediction_confidence, prediction_strategy,
                 model_version, status, created_at, market_follow_error,
                 direction_match, step_change_pct, step_capped, policy_step_cap_pct,
                 probe_type, path_value_score, stage1_fill_probability,
                 frr_proxy_rate, frr_fallback_value, rank6_fallback_penalty,
                 fast_liquidity_score, currency_regime_state, expected_terminal_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id,
                prediction['currency'],
                prediction['period'],
                prediction['predicted_rate'],
                data_timestamp,  # Use data_timestamp for order_timestamp
                validation_window,
                prediction.get('confidence', 'Medium'),
                prediction.get('strategy', 'Unknown'),
                'v1.0',  # Model version
                current_time,  # created_at is when order was inserted
                prediction.get('market_follow_error'),
                prediction.get('direction_match'),
                prediction.get('step_change_pct'),
                int(bool(prediction.get('step_capped', False))) if prediction.get('step_capped') is not None else None,
                prediction.get('policy_step_cap_pct'),
                probe_type,
                prediction.get('path_value_score'),
                prediction.get('stage1_fill_probability'),
                prediction.get('frr_proxy_rate'),
                prediction.get('frr_fallback_value'),
                prediction.get('rank6_fallback_penalty'),
                prediction.get('fast_liquidity_score'),
                prediction.get('currency_regime_state'),
                prediction.get('expected_terminal_mode'),
            ))

            conn.commit()
            logger.info(f"Created virtual order {order_id} for {prediction['currency']}-{prediction['period']}d @ {prediction['predicted_rate']:.4f}%")

        except Exception as e:
            logger.error(f"Failed to create virtual order: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

        return order_id

    def get_pending_orders(self, expired_only: bool = False) -> List[Dict]:
        """
        Get all pending orders that need validation.

        Args:
            expired_only: If True, only return orders past their validation window

        Returns:
            List of order dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if expired_only:
            # Only orders past their validation window
            query = """
                SELECT * FROM virtual_orders
                WHERE status = 'PENDING'
                  AND datetime(order_timestamp, '+' || validation_window_hours || ' hours') <= datetime(?)
                ORDER BY order_timestamp ASC
            """
            cursor.execute(query, (now,))
        else:
            # All pending orders
            query = """
                SELECT * FROM virtual_orders
                WHERE status = 'PENDING'
                ORDER BY order_timestamp ASC
            """
            cursor.execute(query)

        orders = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return orders

    def update_order_status(self, order_id: str, execution_details: dict):
        """
        Update order status and execution details.

        Args:
            order_id: Order UUID
            execution_details: Dict containing:
                - status: EXECUTED/FAILED/EXPIRED
                - executed_at (optional): execution timestamp
                - execution_rate (optional): actual execution rate
                - execution_delay_minutes (optional): minutes until execution
                - max_market_rate: highest market rate in validation window
                - rate_gap (optional): predicted - max_market (for failed orders)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE virtual_orders
                SET status = ?,
                    executed_at = ?,
                    execution_rate = ?,
                    execution_delay_minutes = ?,
                    max_market_rate = ?,
                    rate_gap = ?,
                    path_stage_outcome = ?,
                    stage1_fill_hours = ?,
                    stage2_frr_proxy_rate = ?,
                    terminal_mode = ?,
                    validated_at = ?
                WHERE order_id = ?
            """, (
                execution_details['status'],
                execution_details.get('executed_at'),
                execution_details.get('execution_rate'),
                execution_details.get('execution_delay_minutes'),
                execution_details.get('max_market_rate'),
                execution_details.get('rate_gap'),
                execution_details.get('path_stage_outcome'),
                execution_details.get('stage1_fill_hours'),
                execution_details.get('stage2_frr_proxy_rate'),
                execution_details.get('terminal_mode'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                order_id
            ))

            conn.commit()
            logger.info(f"Updated order {order_id} status to {execution_details['status']}")

        except Exception as e:
            logger.error(f"Failed to update order {order_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_execution_stats(self, currency: str, period: int, lookback_days: int) -> Dict:
        """
        Get execution statistics for a currency-period combination.

        Args:
            currency: fUSD/fUST
            period: Lending period
            lookback_days: Days to look back (7/30/90)

        Returns:
            Dict with statistics:
                - total_orders: Total validated orders
                - executed_orders: Successfully executed orders
                - execution_rate: Percentage (0.0-1.0)
                - avg_execution_delay: Average delay in minutes
                - avg_spread: Average spread for executed orders
                - avg_rate_gap: Average rate gap for failed orders
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d %H:%M:%S')

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed,
                AVG(CASE WHEN status = 'EXECUTED' THEN execution_delay_minutes END) as avg_delay,
                AVG(CASE WHEN status = 'EXECUTED' THEN (execution_rate - predicted_rate) END) as avg_spread,
                AVG(CASE WHEN status = 'FAILED' THEN rate_gap END) as avg_gap
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND order_timestamp >= ?
              AND status IN ('EXECUTED', 'FAILED')
        """

        cursor.execute(query, (currency, period, cutoff_date))
        row = cursor.fetchone()
        conn.close()

        total = row[0] or 0
        executed = row[1] or 0
        execution_rate = (executed / total) if total > 0 else 0.7  # Cold start default

        return {
            'total_orders': total,
            'executed_orders': executed,
            'execution_rate': execution_rate,
            'avg_execution_delay': row[2] or 0,
            'avg_spread': row[3] or 0,
            'avg_rate_gap': row[4] or 0
        }

    def get_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Query orders with optional status filter.

        Args:
            status: Optional filter by status (PENDING/EXECUTED/FAILED)
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row

        if status:
            query = """
                SELECT * FROM virtual_orders
                WHERE status = ?
                ORDER BY order_timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (status, limit))
        else:
            query = """
                SELECT * FROM virtual_orders
                ORDER BY order_timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (limit,))

        orders = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return orders

    def get_order_count(self, currency: str, period: int) -> int:
        """
        Get total number of validated orders for a currency-period combination.

        Args:
            currency: fUSD/fUST
            period: Lending period

        Returns:
            Number of validated orders (EXECUTED or FAILED)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT COUNT(*) as total
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status IN ('EXECUTED', 'FAILED')
        """

        cursor.execute(query, (currency, period))
        row = cursor.fetchone()
        conn.close()

        return row[0] or 0

    def get_recent_validation_count(self, currency: str, period: int, lookback_hours: int = 24) -> int:
        """
        Count recently validated orders for refresh-probe decision.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status IN ('EXECUTED', 'FAILED')
              AND validated_at IS NOT NULL
              AND validated_at >= ?
            """,
            (currency, period, cutoff),
        )
        row = cursor.fetchone()
        conn.close()
        return int(row[0] or 0)

    def needs_refresh_probe(
        self,
        currency: str,
        period: int,
        lookback_hours: int = 24,
        min_validations: int = 1
    ) -> bool:
        """
        Decide whether this combo needs a refresh probe order.
        """
        return self.get_recent_validation_count(currency, period, lookback_hours) < int(min_validations)

    def aggregate_execution_statistics(self):
        """
        Aggregate execution statistics from virtual_orders into execution_statistics table.
        Called after order validation to keep execution_statistics up to date.

        This fixes issue #6 where execution_statistics table was always empty.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Ensure the execution_statistics table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    currency TEXT NOT NULL,
                    period INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    total_orders INTEGER DEFAULT 0,
                    executed_orders INTEGER DEFAULT 0,
                    failed_orders INTEGER DEFAULT 0,
                    expired_orders INTEGER DEFAULT 0,
                    execution_rate REAL DEFAULT 0,
                    avg_predicted_rate REAL,
                    avg_execution_rate REAL,
                    avg_rate_gap REAL,
                    avg_execution_delay REAL,
                    updated_at TEXT,
                    UNIQUE(currency, period, date)
                )
            """)

            # Aggregate today's stats from virtual_orders
            today = datetime.now().strftime('%Y-%m-%d')

            cursor.execute("""
                SELECT
                    currency,
                    period,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed,
                    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'EXPIRED' THEN 1 ELSE 0 END) as expired,
                    AVG(predicted_rate) as avg_predicted,
                    AVG(CASE WHEN status = 'EXECUTED' THEN execution_rate END) as avg_exec_rate,
                    AVG(CASE WHEN status = 'FAILED' THEN rate_gap END) as avg_gap,
                    AVG(CASE WHEN status = 'EXECUTED' THEN execution_delay_minutes END) as avg_delay
                FROM virtual_orders
                WHERE validated_at IS NOT NULL
                  AND date(validated_at) = ?
                  AND status IN ('EXECUTED', 'FAILED', 'EXPIRED')
                GROUP BY currency, period
            """, (today,))

            rows = cursor.fetchall()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for row in rows:
                currency, period, total, executed, failed, expired, avg_predicted, avg_exec_rate, avg_gap, avg_delay = row
                exec_rate = (executed / total * 100) if total > 0 else 0

                cursor.execute("""
                    INSERT INTO execution_statistics
                    (currency, period, date, total_orders, executed_orders, failed_orders, expired_orders,
                     execution_rate, avg_predicted_rate, avg_execution_rate, avg_rate_gap,
                     avg_execution_delay, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(currency, period, date) DO UPDATE SET
                        total_orders = excluded.total_orders,
                        executed_orders = excluded.executed_orders,
                        failed_orders = excluded.failed_orders,
                        expired_orders = excluded.expired_orders,
                        execution_rate = excluded.execution_rate,
                        avg_predicted_rate = excluded.avg_predicted_rate,
                        avg_execution_rate = excluded.avg_execution_rate,
                        avg_rate_gap = excluded.avg_rate_gap,
                        avg_execution_delay = excluded.avg_execution_delay,
                        updated_at = excluded.updated_at
                """, (currency, period, today, total, executed, failed, expired,
                      exec_rate, avg_predicted, avg_exec_rate, avg_gap, avg_delay, now))

            conn.commit()
            if rows:
                logger.info(f"Aggregated execution statistics: {len(rows)} currency-period combos for {today}")
            else:
                logger.debug(f"No validated orders today ({today}) to aggregate")

        except Exception as e:
            logger.error(f"Failed to aggregate execution statistics: {e}")
            conn.rollback()
        finally:
            conn.close()


if __name__ == "__main__":
    # Test the order manager
    logging.basicConfig(level=logging.INFO)

    manager = OrderManager()

    # Test create order
    test_prediction = {
        'currency': 'fUSD',
        'period': 30,
        'predicted_rate': 12.5,
        'timestamp': datetime.now().isoformat(),
        'confidence': 'High',
        'strategy': 'Balanced'
    }

    order_id = manager.create_virtual_order(test_prediction)
    print(f"Created order: {order_id}")

    # Test get pending orders
    pending = manager.get_pending_orders()
    print(f"\nPending orders: {len(pending)}")
    if pending:
        print(f"Latest: {pending[-1]}")

    # Test get stats (cold start scenario)
    stats = manager.get_execution_stats('fUSD', 30, 7)
    print(f"\nExecution stats: {stats}")
