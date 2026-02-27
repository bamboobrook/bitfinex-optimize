"""
Execution Features Calculator

Calculates execution-based features from virtual order history.
Used to enhance model training with execution feedback.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import os

# Resolve database path relative to this file
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "lending_history.db")

class ExecutionFeatures:
    """Calculates execution feedback features"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._cache = {}  # Simple cache to avoid repeated queries

    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def calculate_execution_rate(self, currency: str, period: int,
                                 lookback_days: int,
                                 as_of_date: Optional[datetime] = None) -> float:
        """
        Calculate historical execution rate

        Args:
            currency: fUSD/fUST
            period: Lending period (days)
            lookback_days: Lookback window (7/30/90)
            as_of_date: Cutoff date (None = now)

        Returns:
            Execution rate (0.0-1.0), default 0.7 if no data
        """
        cache_key = f"exec_rate_{currency}_{period}_{lookback_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if as_of_date is None:
            as_of_date = datetime.now()

        start_date = as_of_date - timedelta(days=lookback_days)

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END) as executed
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND order_timestamp >= ?
              AND order_timestamp <= ?
              AND status IN ('EXECUTED', 'FAILED')
            """

            cursor.execute(query, (
                currency,
                period,
                start_date.strftime('%Y-%m-%d %H:%M:%S'),
                as_of_date.strftime('%Y-%m-%d %H:%M:%S')
            ))

            row = cursor.fetchone()
            total = row[0] or 0
            executed = row[1] or 0

            # 长周期(>=60天)订单天然较少,降低冷启动阈值加速脱离默认值
            cold_start_threshold = 5 if period >= 60 else 10

            # 冷启动默认值按周期差异化
            if period <= 7:
                default_rate = 0.55   # 短周期天然执行率更高
            elif period <= 30:
                default_rate = 0.50   # 中周期
            else:
                default_rate = 0.45   # 长周期天然执行率更低

            if total == 0:
                exec_rate = default_rate
            else:
                calculated_rate = executed / total
                # 渐进混合: 避免跨过阈值时从默认值瞬间跳到计算值
                # blend_ceiling = 2x阈值, 在0~ceiling之间线性过渡
                blend_ceiling = cold_start_threshold * 2
                default_weight = max(0.0, 1.0 - total / blend_ceiling)
                exec_rate = default_weight * default_rate + (1.0 - default_weight) * calculated_rate

            self._cache[cache_key] = exec_rate
            return exec_rate

        finally:
            conn.close()

    def calculate_avg_spread(self, currency: str, period: int,
                            lookback_days: int) -> float:
        """
        Calculate average spread for executed orders
        Spread = execution_rate - predicted_rate

        Returns:
            Average spread (%), 0.0 if no data
        """
        cache_key = f"avg_spread_{currency}_{period}_{lookback_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d %H:%M:%S')

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT AVG(execution_rate - predicted_rate) as avg_spread
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status = 'EXECUTED'
              AND order_timestamp >= ?
            """

            cursor.execute(query, (currency, period, start_date))
            row = cursor.fetchone()

            avg_spread = row[0] if row[0] is not None else 0.0
            self._cache[cache_key] = avg_spread
            return avg_spread

        finally:
            conn.close()

    def calculate_avg_rate_gap(self, currency: str, period: int,
                               lookback_days: int) -> float:
        """
        Calculate average rate gap for failed orders
        Rate gap = predicted_rate - max_market_rate

        Returns:
            Average gap (%), 0.0 if no data
        """
        cache_key = f"avg_gap_{currency}_{period}_{lookback_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d %H:%M:%S')

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT AVG(rate_gap) as avg_gap
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status = 'FAILED'
              AND order_timestamp >= ?
              AND rate_gap IS NOT NULL
            """

            cursor.execute(query, (currency, period, start_date))
            row = cursor.fetchone()

            avg_gap = row[0] if row[0] is not None else 0.0
            self._cache[cache_key] = avg_gap
            return avg_gap

        finally:
            conn.close()

    def calculate_execution_delay_percentile(self, currency: str, period: int,
                                             lookback_days: int,
                                             percentile: float = 0.5) -> float:
        """
        Calculate execution delay percentile

        Args:
            currency: fUSD/fUST
            period: Lending period
            lookback_days: Lookback window
            percentile: Percentile to calculate (0.5 = median, 0.9 = p90)

        Returns:
            Delay in minutes, 0.0 if no data
        """
        cache_key = f"delay_p{int(percentile*100)}_{currency}_{period}_{lookback_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d %H:%M:%S')

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT execution_delay_minutes
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status = 'EXECUTED'
              AND order_timestamp >= ?
              AND execution_delay_minutes IS NOT NULL
            ORDER BY execution_delay_minutes
            """

            cursor.execute(query, (currency, period, start_date))
            delays = [row[0] for row in cursor.fetchall()]

            if not delays:
                delay_p = 0.0
            else:
                delay_p = float(np.percentile(delays, percentile * 100))

            self._cache[cache_key] = delay_p
            return delay_p

        finally:
            conn.close()

    def get_all_features(self, currency: str, period: int) -> Dict[str, float]:
        """
        Get all execution features for a currency/period combination

        Returns:
            Dictionary with all execution features
        """
        features = {
            # Execution rates
            'exec_rate_7d': self.calculate_execution_rate(currency, period, 7),
            'exec_rate_30d': self.calculate_execution_rate(currency, period, 30),

            # Spreads (executed orders)
            'avg_spread_7d': self.calculate_avg_spread(currency, period, 7),
            'avg_spread_30d': self.calculate_avg_spread(currency, period, 30),

            # Rate gaps (failed orders)
            'avg_rate_gap_failed_7d': self.calculate_avg_rate_gap(currency, period, 7),
            'avg_rate_gap_failed_30d': self.calculate_avg_rate_gap(currency, period, 30),

            # Execution delays
            'exec_delay_p50': self.calculate_execution_delay_percentile(currency, period, 7, 0.5),
            'exec_delay_p90': self.calculate_execution_delay_percentile(currency, period, 7, 0.9),
        }

        # Derived features
        features['exec_rate_trend'] = features['exec_rate_7d'] / (features['exec_rate_30d'] + 1e-8)
        features['rate_gap_trend'] = features['avg_rate_gap_failed_7d']

        # Risk adjustment factor — 对称化: 3级下调 + 3级上调, [0.5, 0.6] 为中性区间
        exec_r = features['exec_rate_7d']
        if exec_r < 0.3:
            features['risk_adjustment_factor'] = 0.90
        elif exec_r < 0.4:
            features['risk_adjustment_factor'] = 0.94
        elif exec_r < 0.5:
            features['risk_adjustment_factor'] = 0.97
        elif exec_r <= 0.6:
            features['risk_adjustment_factor'] = 1.0    # 目标区间
        elif exec_r <= 0.7:
            features['risk_adjustment_factor'] = 1.03
        elif exec_r <= 0.85:
            features['risk_adjustment_factor'] = 1.06
        else:
            features['risk_adjustment_factor'] = 1.10

        return features

    def is_cold_start(self, currency: str, period: int, threshold: int = 10) -> bool:
        """
        Check if a currency-period combination is in cold start phase

        Args:
            currency: fUSD/fUST
            period: Lending period
            threshold: Minimum number of validated orders to exit cold start

        Returns:
            True if cold start (< threshold orders), False otherwise
        """
        cache_key = f"cold_start_{currency}_{period}_{threshold}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT COUNT(*) as total
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status IN ('EXECUTED', 'FAILED')
            """

            cursor.execute(query, (currency, period))
            row = cursor.fetchone()
            total = row[0] or 0

            is_cold = total < threshold
            self._cache[cache_key] = is_cold
            return is_cold

        finally:
            conn.close()

    def get_order_count(self, currency: str, period: int) -> int:
        """
        Get total number of validated orders for a currency-period combination

        Args:
            currency: fUSD/fUST
            period: Lending period

        Returns:
            Number of validated orders (EXECUTED or FAILED)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = """
            SELECT COUNT(*) as total
            FROM virtual_orders
            WHERE currency = ?
              AND period = ?
              AND status IN ('EXECUTED', 'FAILED')
            """

            cursor.execute(query, (currency, period))
            row = cursor.fetchone()
            return row[0] or 0

        finally:
            conn.close()

    def clear_cache(self):
        """Clear the feature cache"""
        self._cache = {}


# Standalone functions for use in data_processor.py

def calculate_execution_rate(currency: str, period: int, lookback_days: int,
                            as_of_date: Optional[datetime] = None) -> float:
    """Standalone function to calculate execution rate"""
    calc = ExecutionFeatures()
    return calc.calculate_execution_rate(currency, period, lookback_days, as_of_date)


def calculate_avg_spread(currency: str, period: int, lookback_days: int) -> float:
    """Standalone function to calculate average spread"""
    calc = ExecutionFeatures()
    return calc.calculate_avg_spread(currency, period, lookback_days)


def calculate_avg_rate_gap(currency: str, period: int, lookback_days: int) -> float:
    """Standalone function to calculate average rate gap"""
    calc = ExecutionFeatures()
    return calc.calculate_avg_rate_gap(currency, period, lookback_days)


def calculate_execution_delay_percentile(currency: str, period: int,
                                        lookback_days: int, percentile: float) -> float:
    """Standalone function to calculate execution delay percentile"""
    calc = ExecutionFeatures()
    return calc.calculate_execution_delay_percentile(currency, period, lookback_days, percentile)


if __name__ == "__main__":
    # Test execution features
    print("Testing ExecutionFeatures...")

    calc = ExecutionFeatures()

    # Test for fUSD period 30
    print("\nCalculating features for fUSD period 30...")
    features = calc.get_all_features('fUSD', 30)

    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

    print("\n✅ Execution features tests completed")
