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

    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def create_virtual_order(self, prediction: dict) -> str:
        """
        Create a virtual order from a prediction result.

        Args:
            prediction: Dictionary from predictor.py containing:
                - currency: fUSD/fUST
                - period: lending period
                - predicted_rate: predicted lending rate
                - timestamp: prediction timestamp
                - confidence: Low/Medium/High
                - strategy: strategy description

        Returns:
            order_id: UUID string
        """
        order_id = str(uuid.uuid4())
        validation_window = determine_validation_window(prediction['period'])

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Use explicit created_at to avoid timezone issues with SQLite CURRENT_TIMESTAMP (UTC)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute("""
                INSERT INTO virtual_orders
                (order_id, currency, period, predicted_rate, order_timestamp,
                 validation_window_hours, prediction_confidence, prediction_strategy,
                 model_version, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """, (
                order_id,
                prediction['currency'],
                prediction['period'],
                prediction['predicted_rate'],
                prediction.get('timestamp', current_time),
                validation_window,
                prediction.get('confidence', 'Medium'),
                prediction.get('strategy', 'Unknown'),
                'v1.0',  # Model version
                current_time  # Explicit created_at
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

        now = datetime.now().isoformat()

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
                    validated_at = ?
                WHERE order_id = ?
            """, (
                execution_details['status'],
                execution_details.get('executed_at'),
                execution_details.get('execution_rate'),
                execution_details.get('execution_delay_minutes'),
                execution_details.get('max_market_rate'),
                execution_details.get('rate_gap'),
                datetime.now().isoformat(),
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

        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

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
