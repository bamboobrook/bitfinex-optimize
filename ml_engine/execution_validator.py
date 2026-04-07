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
        self._ensure_schema()

    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        """Ensure optional columns used by enhanced validation are available."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(virtual_orders)")
            existing = {row[1] for row in cursor.fetchall()}
            required_columns = {
                "executed_at": "TEXT",
                "execution_rate": "REAL",
                "execution_delay_minutes": "INTEGER",
                "max_market_rate": "REAL",
                "rate_gap": "REAL",
                "validated_at": "TEXT",
                "execution_confidence": "REAL",
                "percentile_score": "REAL",
                "gap_score": "REAL",
                "density_score": "REAL",
                "total_score": "REAL",
                "gate_reject_reason": "TEXT",
                "follow_error_at_order": "REAL",
                "execution_threshold": "REAL",
                "market_percentile_25": "REAL",
                "market_percentile_30": "REAL",
                "market_percentile_35": "REAL",
                "market_percentile_40": "REAL",
                "market_median": "REAL",
                "market_min": "REAL",
                "market_max": "REAL",
                "nearby_rate_count": "INTEGER",
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

    def _simulate_stage1_fixed_path(self, order_time: datetime, market_rows: List, predicted_rate: float, period: int):
        """
        Replay the first 6 hours of the fixed-rate path, reducing the ask by 1% each hour.

        Returns:
            (filled, fill_hour, effective_rate, fill_timestamp)
        """
        fill_tolerance = self._get_fill_tolerance(period)

        for hour in range(6):
            hour_start = order_time + timedelta(hours=hour)
            hour_end = order_time + timedelta(hours=hour + 1)
            target_rate = predicted_rate * (0.99 ** hour)

            for timestamp_str, close_rate, high_rate in market_rows:
                row_time = datetime.fromisoformat(timestamp_str)
                if row_time <= hour_start or row_time > hour_end:
                    continue

                candidate_rate = float(close_rate or 0.0)
                if candidate_rate >= target_rate * fill_tolerance:
                    return True, hour + 1, target_rate, timestamp_str

        return False, None, predicted_rate * (0.99 ** 5), None

    def _estimate_stage2_frr_proxy(self, currency: str, stage2_start: datetime, stage2_end: datetime) -> float:
        """
        Estimate FRR proxy from same-currency 120d close_annual values in the stage2 window.
        """
        if stage2_end <= stage2_start:
            return 0.0

        rows = self.query_market_rates(currency, 120, stage2_start, stage2_end)
        close_rates = [float(row[2]) for row in rows if row[2] is not None and float(row[2]) > 0.0]

        if not close_rates:
            return 0.0

        return float(np.percentile(close_rates, 40))

    def _get_recent_validation_count(self, period: int, currency: str, hours: int = 4) -> int:
        """
        获取最近 N 小时内指定周期和货币的验证数量，作为市场活跃度指标。

        Args:
            period: 周期天数
            currency: 货币 (fUSD/fUST)
            hours: 回看小时数，默认4小时

        Returns:
            验证数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(DISTINCT o.order_id)
                FROM virtual_orders o
                WHERE o.period = ?
                  AND o.currency = ?
                  AND o.validated_at IS NOT NULL
                  AND o.validated_at >= datetime('now', '-' || ? || ' hours')
            """, (period, currency, hours))
            result = cursor.fetchone()
            return result[0] if result else 0
        finally:
            conn.close()

    def _get_execution_threshold(self, period: int, currency: str = 'fUSD') -> float:
        """
        自适应阈值：市场不活跃时自动放宽。

        Period-aware score threshold: short more sensitive, long more conservative.
        When market is inactive (low validation count), thresholds are relaxed.
        Fix9: 若 7 天内该 pair 无成交（市场已死亡），不放宽阈值，避免误判。
        """
        # 基础阈值
        if period <= 7:
            base = 38.0
        elif period <= 30:
            base = 43.0
        else:
            base = 48.0

        # Fix9: 检查 7 天内真实成交数（EXECUTED+FAILED，排除 EXPIRED）
        # 若近 7 天无成交，市场可能已死亡，保持严格阈值，不放宽
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM virtual_orders
                WHERE currency = ? AND period = ?
                  AND status IN ('EXECUTED', 'FAILED')
                  AND order_timestamp >= datetime('now', '-7 days')
            """, (currency, period))
            recent_7d_count = cursor.fetchone()[0] or 0
        except Exception:
            recent_7d_count = 1  # 查询失败时保守处理，仍允许放宽
        finally:
            conn.close()

        if recent_7d_count < 2:
            # 7 天内几乎无成交，市场可能已死亡，返回严格基础阈值（不放宽）
            return base

        # 获取最近验证数量作为市场活跃度指标
        recent_count = self._get_recent_validation_count(period, currency, hours=4)

        # 市场不活跃时放宽阈值
        if recent_count < 5:
            # 过去4小时验证少于5条：放宽25%
            return base * 0.75
        elif recent_count < 10:
            # 过去4小时验证少于10条：放宽15%
            return base * 0.85
        elif recent_count < 20:
            # 过去4小时验证少于20条：放宽10%
            return base * 0.92

        return base

    @staticmethod
    def _get_fill_tolerance(period: int) -> float:
        """
        Execution candidate tolerance:
        close_rate >= predicted_rate * tolerance
        """
        if period <= 7:
            return 0.94
        if period <= 30:
            return 0.96
        return 0.98

    @staticmethod
    def _get_q40_multiplier(period: int) -> float:
        """Short periods allow slightly higher relative pricing than long periods."""
        if period <= 7:
            return 1.08
        if period <= 30:
            return 1.04
        return 1.00

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

            # ========== 阶段4修复: 长周期/高利率优先 ==========
            # 按周期降序、预测利率降序排序，让长周期高利率订单优先验证
            pending_orders.sort(key=lambda x: (
                -x.get('period', 0),        # 周期越长越优先
                -x.get('predicted_rate', 0) # 利率越高越优先
            ))

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
                    'status': 'EXPIRED',
                    'gate_reject_reason': 'NO_MARKET_DATA'
                }
                self.update_order_status(order['order_id'], update_data)
                return {
                    'order_id': order['order_id'],
                    'status': 'EXPIRED',
                    'reason': 'No market data available'
                }

            predicted_rate = order['predicted_rate']
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
                execution_details = {
                    'status': 'FAILED',
                    'gate_reject_reason': 'NO_VALID_MARKET_DATA',
                    'max_market_rate': 0.0,
                    'rate_gap': predicted_rate,
                    'path_stage_outcome': 'NO_VALID_MARKET_DATA',
                    'stage1_fill_hours': None,
                    'stage2_frr_proxy_rate': 0.0,
                    'terminal_mode': 'RANK6_PROXY',
                    'execution_confidence': 0.0,
                    'percentile_score': 0.0,
                    'gap_score': 0.0,
                    'density_score': 0.0,
                    'total_score': 0.0,
                    'execution_threshold': self._get_execution_threshold(int(order['period']), order['currency']),
                    'market_percentile_25': 0.0,
                    'market_percentile_30': 0.0,
                    'market_percentile_35': 0.0,
                    'market_percentile_40': 0.0,
                    'market_median': 0.0,
                    'market_min': 0.0,
                    'market_max': 0.0,
                    'nearby_rate_count': 0,
                }
            else:
                _, confidence, score_details = self._calculate_hybrid_execution_score(
                    predicted_rate,
                    market_close_rates,
                    int(order['period']),
                    order['currency']
                )
                market_median = float(score_details['market_median'])
                max_market_rate = max(market_high_rates) if market_high_rates else max(market_close_rates)
                follow_error_at_order = predicted_rate - market_median
                compatibility_details = {
                    'execution_confidence': confidence,
                    **score_details,
                }

                stage1_ok, stage1_fill_hours, stage1_effective_rate, stage1_fill_timestamp = self._simulate_stage1_fixed_path(
                    order_time,
                    market_data_timestamped,
                    predicted_rate,
                    int(order['period']),
                )

                stage2_start = order_time + timedelta(hours=6)
                stage2_end = min(order_time + timedelta(hours=12), actual_window_end)
                stage2_frr_proxy_rate = self._estimate_stage2_frr_proxy(
                    order['currency'],
                    stage2_start,
                    stage2_end,
                )

                if stage1_ok:
                    row_time = datetime.fromisoformat(stage1_fill_timestamp)
                    execution_details = {
                        'status': 'EXECUTED',
                        'executed_at': stage1_fill_timestamp,
                        'execution_rate': stage1_effective_rate,
                        'execution_delay_minutes': int((row_time - order_time).total_seconds() / 60),
                        'max_market_rate': max_market_rate,
                        'rate_gap': predicted_rate - max_market_rate,
                        'follow_error_at_order': follow_error_at_order,
                        'path_stage_outcome': 'FIXED_FILLED',
                        'stage1_fill_hours': stage1_fill_hours,
                        'stage2_frr_proxy_rate': stage2_frr_proxy_rate,
                        'terminal_mode': 'FIXED',
                        **compatibility_details,
                    }
                else:
                    execution_details = {
                        'status': 'FAILED',
                        'gate_reject_reason': 'PATH_FIXED_MISS',
                        'max_market_rate': max_market_rate,
                        'rate_gap': predicted_rate - max_market_rate,
                        'follow_error_at_order': follow_error_at_order,
                        'path_stage_outcome': 'FIXED_MISS',
                        'stage1_fill_hours': None,
                        'stage2_frr_proxy_rate': stage2_frr_proxy_rate,
                        'terminal_mode': 'FRR_PROXY' if stage2_frr_proxy_rate > 0 else 'RANK6_PROXY',
                        **compatibility_details,
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
        market_close_rates: list,
        period: int,
        currency: str = 'fUSD'
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

        # 1. 分位数得分 (0-40分) — 线性衰减，比阶梯式更平滑
        # 预测利率在市场分位数中的位置，越低分数越高
        if predicted_rate <= percentile_25:
            percentile_score = 40.0
        elif predicted_rate <= median:
            # 线性从 40 分(在P25) 衰减到 0 分(在median)
            ratio = (median - predicted_rate) / (median - percentile_25 + 1e-8)
            percentile_score = 40.0 * ratio
        else:
            percentile_score = 0.0

        # 2. 利率差距得分 (0-30分)
        # 预测利率越接近市场最低值，得分越高
        rate_gap = predicted_rate - min_rate
        median_gap = median - min_rate

        if median_gap > 0:
            gap_ratio = rate_gap / median_gap
            gap_score = min(max(30 - gap_ratio * 30, 0), 30.0)
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

        threshold = self._get_execution_threshold(period, currency)
        should_execute = total_score >= threshold
        confidence = total_score / 100

        score_details = {
            'percentile_score': round(percentile_score, 2),
            'gap_score': round(gap_score, 2),
            'density_score': round(density_score, 2),
            'total_score': round(total_score, 2),
            'execution_threshold': round(threshold, 2),
            'market_percentile_25': percentile_25,
            'market_percentile_30': percentile_30,
            'market_percentile_35': percentile_35,
            'market_percentile_40': percentile_40,
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

            for field in [
                'status', 'executed_at', 'execution_rate',
                'execution_delay_minutes', 'max_market_rate', 'rate_gap',
                'execution_confidence', 'percentile_score', 'gap_score',
                'density_score', 'total_score', 'execution_threshold',
                'market_percentile_25', 'market_percentile_30',
                'market_percentile_35', 'market_percentile_40',
                'market_median', 'market_min', 'market_max',
                'nearby_rate_count', 'gate_reject_reason',
                'follow_error_at_order', 'path_stage_outcome',
                'stage1_fill_hours', 'stage2_frr_proxy_rate',
                'terminal_mode'
            ]:
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

    # Aggregate execution statistics after validation (fixes issue #6)
    try:
        # 使用相对导入兼容直接运行和模块运行两种方式
        import importlib, sys
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from ml_engine.order_manager import OrderManager
        manager = OrderManager()
        manager.aggregate_execution_statistics()
        print("\n✅ Execution statistics aggregated")
    except Exception as e:
        print(f"\n⚠️  Failed to aggregate execution statistics: {e}")

    print("\n✅ Execution validator tests completed")
