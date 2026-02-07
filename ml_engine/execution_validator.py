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
                # 使用混合判断策略
                should_execute, confidence, score_details = self._calculate_hybrid_execution_score(
                    predicted_rate, market_close_rates
                )

                if should_execute:
                    executed = True

                    # 找到第一个匹配的执行时间点
                    execution_timestamp = None
                    execution_rate = None

                    for timestamp_str, close_rate, high_rate in market_data_timestamped:
                        if close_rate >= predicted_rate * 0.95:  # Within 5% tolerance
                            execution_timestamp = timestamp_str
                            execution_rate = close_rate
                            break

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
                        'execution_confidence': confidence,
                        **score_details
                    }
                else:
                    # 订单失败
                    execution_details = {
                        'status': 'FAILED',
                        'execution_confidence': confidence,
                        **score_details
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

    def _calculate_hybrid_execution_score(
        self,
        predicted_rate: float,
        market_close_rates: list
    ) -> tuple:
        """
        混合判断: 分位数 + 利率差距 + 市场密度

        Args:
            predicted_rate: 预测利率
            market_close_rates: 市场收盘利率列表

        Returns:
            (should_execute, confidence_score, score_details)
        """
        if not market_close_rates or len(market_close_rates) == 0:
            return False, 0.0, {'error': 'No market data'}

        # 计算市场统计值
        percentile_25 = np.percentile(market_close_rates, 25)
        percentile_30 = np.percentile(market_close_rates, 30)
        percentile_35 = np.percentile(market_close_rates, 35)
        percentile_40 = np.percentile(market_close_rates, 40)
        median = np.median(market_close_rates)
        min_rate = np.min(market_close_rates)
        max_rate = np.max(market_close_rates)

        # 1. 分位数得分 (0-40分)
        # 预测利率在市场分位数中的位置，越低分数越高
        if predicted_rate <= percentile_25:
            percentile_score = 40
        elif predicted_rate <= percentile_30:
            percentile_score = 35
        elif predicted_rate <= percentile_35:
            percentile_score = 28
        elif predicted_rate <= percentile_40:
            percentile_score = 20
        elif predicted_rate <= median:
            percentile_score = 10
        else:
            percentile_score = 0

        # 2. 利率差距得分 (0-30分)
        # 预测利率越接近市场最低值，得分越高
        rate_gap = predicted_rate - min_rate
        median_gap = median - min_rate

        if median_gap > 0:
            gap_ratio = rate_gap / median_gap
            gap_score = max(30 - gap_ratio * 30, 0)
        else:
            gap_score = 30 if rate_gap < 0.001 else 0

        # 3. 市场密度得分 (0-30分)
        # 在预测利率±5%附近有多少市场报价
        if predicted_rate > 0:
            tolerance = 0.05  # 5% tolerance
            nearby_rates = [
                r for r in market_close_rates
                if abs(r - predicted_rate) / predicted_rate <= tolerance
            ]
            density_ratio = len(nearby_rates) / len(market_close_rates)
            density_score = density_ratio * 30
        else:
            nearby_rates = []
            density_score = 0

        # 综合评分 (0-100)
        total_score = percentile_score + gap_score + density_score

        # 执行阈值: 45分 (微调后更接近目标执行率)
        should_execute = total_score >= 45
        confidence = total_score / 100

        score_details = {
            'percentile_score': round(percentile_score, 2),
            'gap_score': round(gap_score, 2),
            'density_score': round(density_score, 2),
            'total_score': round(total_score, 2),
            'market_percentile_25': percentile_25,
            'market_percentile_30': percentile_30,
            'market_percentile_35': percentile_35,
            'market_median': median,
            'market_min': min_rate,
            'market_max': max_rate,
            'rate_gap': rate_gap,
            'nearby_rate_count': len(nearby_rates) if predicted_rate > 0 else 0
        }

        return should_execute, confidence, score_details

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
                         'execution_delay_minutes', 'max_market_rate', 'rate_gap',
                         'execution_confidence', 'percentile_score', 'gap_score',
                         'density_score', 'total_score', 'market_percentile_25',
                         'market_percentile_30', 'market_median', 'market_min',
                         'market_max', 'nearby_rate_count', 'market_percentile_35',
                         'rate_gap']:
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
