"""
Execution Validator

Validates pending virtual orders against actual market rates.
Determines if orders would have executed based on historical data.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import os
import numpy as np

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

            # CRITICAL FIX: Add 1-hour buffer to prevent premature validation
            # This ensures we strictly enforce the validation window requirement
            # and avoid look-ahead bias from validating orders too early
            validation_safe_time = window_end + timedelta(hours=1)

            # Check if validation window has passed (with 1-hour safety buffer)
            now = datetime.now()
            if now < validation_safe_time:
                # Not ready for validation yet
                hours_remaining = (validation_safe_time - now).total_seconds() / 3600
                return {
                    'order_id': order['order_id'],
                    'status': 'PENDING',
                    'reason': f'Validation window not complete (wait {hours_remaining:.1f}h more)'
                }

            # CRITICAL: Validation window end must not exceed current time
            # This prevents look-ahead bias from using "future" data
            # Use the original window_end (without buffer) for data query
            actual_window_end = min(window_end, now)

            # Query market rates during validation window
            market_data = self.query_market_rates(
                order['currency'],
                order['period'],
                order_time,
                actual_window_end
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

            # Check for execution - FIXED LOGIC (Plan B: Percentile-based)
            # 在Lending市场中，Lender设置的利率越低，越容易被Borrower选中
            # 因此只有当预测利率 <= 市场25分位数时，才认为订单成交
            predicted_rate = order['predicted_rate']
            executed = False
            execution_details = {}

            # Collect all market rates during validation window
            market_close_rates = []
            market_high_rates = []
            market_data_timestamped = []

            for row in market_data:
                timestamp_str, high_annual, close_annual = row
                close_annual = close_annual or 0.0
                high_annual = high_annual or 0.0

                market_close_rates.append(close_annual)
                market_high_rates.append(high_annual)
                market_data_timestamped.append((timestamp_str, close_annual, high_annual))

            if not market_close_rates:
                # No valid market data
                execution_details = {
                    'status': 'FAILED',
                    'max_market_rate': 0.0,
                    'rate_gap': predicted_rate
                }
            else:
                # Calculate market percentiles (using close_annual as reference)
                percentile_25 = np.percentile(market_close_rates, 25)
                min_rate = np.min(market_close_rates)
                max_rate = np.max(market_close_rates)
                median_rate = np.median(market_close_rates)

                # Execution logic: predicted_rate <= 25th percentile
                # This means your offer is in the lowest 25% of market rates
                # → High chance of execution (competitive pricing)
                if predicted_rate <= percentile_25:
                    executed = True

                    # Find the first timestamp where market rate is close to predicted rate
                    # (simulating when the order would be matched)
                    execution_timestamp = None
                    execution_rate = None

                    for timestamp_str, close_rate, high_rate in market_data_timestamped:
                        # Order executes when market rate is near or above predicted rate
                        # Use close_rate as reference for actual execution price
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
                        'max_market_rate': max_rate,
                        'market_percentile_25': percentile_25,
                        'market_median': median_rate,
                        'market_min': min_rate
                    }
                else:
                    # Order failed - predicted rate too high (not competitive)
                    execution_details = {
                        'status': 'FAILED',
                        'max_market_rate': max_rate,
                        'rate_gap': predicted_rate - percentile_25,  # Gap to 25th percentile
                        'market_percentile_25': percentile_25,
                        'market_median': median_rate,
                        'market_min': min_rate
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
