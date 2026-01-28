"""
Execution Validator

Validates pending virtual orders against actual market rates.
Determines if orders would have executed based on historical data.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import os

# Resolve database path relative to this file
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "lending_history.db")

class ExecutionValidator:
    """Validates virtual orders against market data"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def validate_pending_orders(self) -> Dict:
        """
        Validate all pending orders

        Returns:
            Summary dictionary with validation results
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Get all pending orders
            cursor.execute("""
            SELECT *
            FROM virtual_orders
            WHERE status = 'PENDING'
            ORDER BY order_timestamp ASC
            """)

            pending_orders = [dict(row) for row in cursor.fetchall()]
            conn.close()  # Close before processing

            print(f"Found {len(pending_orders)} pending orders to validate")

            executed_count = 0
            failed_count = 0
            expired_count = 0

            for order in pending_orders:
                result = self.validate_single_order(order)

                if result['status'] == 'EXECUTED':
                    executed_count += 1
                elif result['status'] == 'FAILED':
                    failed_count += 1
                elif result['status'] == 'EXPIRED':
                    expired_count += 1

            return {
                'total': len(pending_orders),
                'executed': executed_count,
                'failed': failed_count,
                'expired': expired_count
            }

        except Exception as e:
            print(f"Error validating orders: {e}")
            import traceback
            traceback.print_exc()
            return {
                'total': 0,
                'executed': 0,
                'failed': 0,
                'expired': 0,
                'error': str(e)
            }

    def validate_single_order(self, order: Dict) -> Dict:
        """
        Validate a single virtual order

        Args:
            order: Order dictionary from database

        Returns:
            Execution details dictionary
        """
        try:
            # Parse order timestamp
            order_time = datetime.fromisoformat(order['order_timestamp'])
            window_hours = order['validation_window_hours']
            window_end = order_time + timedelta(hours=window_hours)

            # Check if validation window has passed
            now = datetime.now()
            if now < window_end:
                # Not ready for validation yet
                return {
                    'order_id': order['order_id'],
                    'status': 'PENDING',
                    'reason': 'Validation window not complete'
                }

            # Query market rates during validation window
            market_data = self.query_market_rates(
                order['currency'],
                order['period'],
                order_time,
                window_end
            )

            if not market_data:
                # Mark as EXPIRED if no market data
                update_data = {
                    'status': 'EXPIRED'
                }
                self.update_order_status(order['order_id'], update_data)
                return {
                    'order_id': order['order_id'],
                    'status': 'EXPIRED',
                    'reason': 'No market data available'
                }

            # Check for execution
            predicted_rate = order['predicted_rate']
            executed = False
            execution_details = {}

            max_rate = 0.0
            for row in market_data:
                timestamp_str, high_annual, close_annual = row
                high_annual = high_annual or 0.0

                # Track maximum rate
                if high_annual > max_rate:
                    max_rate = high_annual

                # Check if order would execute (market reached predicted rate)
                if high_annual >= predicted_rate:
                    executed = True
                    row_time = datetime.fromisoformat(timestamp_str)
                    delay_minutes = int((row_time - order_time).total_seconds() / 60)

                    execution_details = {
                        'status': 'EXECUTED',
                        'executed_at': timestamp_str,
                        'execution_rate': high_annual,
                        'execution_delay_minutes': delay_minutes,
                        'max_market_rate': high_annual
                    }
                    break

            if not executed:
                # Order failed to execute
                execution_details = {
                    'status': 'FAILED',
                    'max_market_rate': max_rate,
                    'rate_gap': predicted_rate - max_rate
                }

            # Update database
            self.update_order_status(order['order_id'], execution_details)

            return {
                'order_id': order['order_id'],
                **execution_details
            }

        except Exception as e:
            print(f"Error validating order {order.get('order_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'order_id': order.get('order_id', 'unknown'),
                'status': 'ERROR',
                'error': str(e)
            }

    def query_market_rates(self, currency: str, period: int,
                          start_time: datetime, end_time: datetime) -> List:
        """
        Query market rates during validation window

        Args:
            currency: Currency (fUSD/fUST)
            period: Lending period
            start_time: Validation window start
            end_time: Validation window end

        Returns:
            List of (timestamp, high_annual, close_annual) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT datetime, high_annual, close_annual
            FROM funding_rates
            WHERE currency = ?
              AND period = ?
              AND datetime >= ?
              AND datetime <= ?
            ORDER BY datetime ASC
            """

            cursor.execute(query, (
                currency,
                period,
                start_time.strftime('%Y-%m-%d %H:%M:%S'),
                end_time.strftime('%Y-%m-%d %H:%M:%S')
            ))

            return cursor.fetchall()

        finally:
            conn.close()

    def update_order_status(self, order_id: str, update_data: Dict) -> bool:
        """
        Update order status in database

        Args:
            order_id: Order UUID
            update_data: Dictionary with update fields

        Returns:
            Success boolean
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Build dynamic update query
            set_clauses = []
            values = []

            for field in ['status', 'executed_at', 'execution_rate',
                         'execution_delay_minutes', 'max_market_rate', 'rate_gap']:
                if field in update_data:
                    set_clauses.append(f"{field} = ?")
                    values.append(update_data[field])

            # Always update validated_at
            set_clauses.append("validated_at = ?")
            values.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Add order_id for WHERE clause
            values.append(order_id)

            query = f"""
            UPDATE virtual_orders
            SET {', '.join(set_clauses)}
            WHERE order_id = ?
            """

            cursor.execute(query, values)
            conn.commit()

            return cursor.rowcount > 0

        except Exception as e:
            print(f"Error updating order status: {e}")
            raise
        finally:
            conn.close()


if __name__ == "__main__":
    # Test the execution validator
    print("Testing ExecutionValidator...")

    validator = ExecutionValidator()

    print("\nValidating pending orders...")
    results = validator.validate_pending_orders()

    print(f"\nValidation Results:")
    print(f"  Total: {results['total']}")
    print(f"  Executed: {results['executed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Expired: {results['expired']}")

    print("\n✅ Execution validator tests completed")
