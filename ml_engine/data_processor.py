import pandas as pd
import numpy as np
import sqlite3
from loguru import logger
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from functools import partial

class DataProcessor:
    def __init__(self, db_path: str = "../data/lending_history.db"):
        self.db_path = db_path
        self.conn = None

    def connect_db(self):
        self.conn = sqlite3.connect(self.db_path)

    def close_db(self):
        if self.conn:
            self.conn.close()

    def load_data(self, currency: str, with_month: bool = True) -> pd.DataFrame:
        """
        加载指定币种的所有历史数据。
        """
        logger.info(f"Loading data for {currency}...")
        self.connect_db()

        # 添加month列到查询中
        month_col = ", month" if with_month else ""

        query = f"""
        SELECT
            currency, period, timestamp, datetime,
            open_annual, close_annual, high_annual, low_annual, volume,
            hour, day_of_week{month_col}
        FROM funding_rates
        WHERE currency = '{currency}'
        ORDER BY period, timestamp
        """
        try:
            df = pd.read_sql(query, self.conn)
            logger.info(f"Loaded {len(df)} rows for {currency}.")

            # 简单的数据清洗
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values(['period', 'timestamp'])

            # 填充缺失值（如有）
            df = df.fillna(method='ffill').fillna(method='bfill')

            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
        finally:
            self.close_db()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        计算相对强弱指标 (RSI)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算MACD指标
        返回: (MACD线, 信号线, MACD柱)
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        计算布林带
        返回: (上轨, 下轨, 带宽, %B位置)
        """
        ma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        bandwidth = (upper_band - lower_band) / (ma + 1e-8)
        percent_b = (series - lower_band) / (upper_band - lower_band + 1e-8)
        return upper_band, lower_band, bandwidth, percent_b

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算真实波动幅度 (ATR)
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def add_technical_indicators(self, group_df):
        """
        对单个 (currency, period) 的分组数据进行特征工程 - 增强版
        """
        # 避免 SettingWithCopyWarning
        df = group_df.copy()

        # ============ 1. 基础滞后特征 (保留原有) ============
        lags = [15, 30, 60, 120, 240, 1440] # 15m, 30m, 1h, 2h, 4h, 24h
        for lag in lags:
            df[f'rate_lag_{lag}'] = df['close_annual'].shift(lag)
            df[f'rate_chg_{lag}'] = df['close_annual'] - df[f'rate_lag_{lag}']

        # ============ 2. 滚动统计特征 (保留原有) ============
        windows = [60, 120, 1440] # 1h, 2h, 24h
        for window in windows:
            df[f'ma_{window}'] = df['close_annual'].rolling(window=window).mean()
            df[f'std_{window}'] = df['close_annual'].rolling(window=window).std()
            df[f'zscore_{window}'] = (df['close_annual'] - df[f'ma_{window}']) / (df[f'std_{window}'] + 1e-8)
            df[f'vol_ma_{window}'] = df['volume'].rolling(window=window).mean()

        # ============ 3. 新增技术指标 ============
        # RSI指标 (14期和28期)
        df['rsi_14'] = self.calculate_rsi(df['close_annual'], 14)
        df['rsi_28'] = self.calculate_rsi(df['close_annual'], 28)

        # MACD指标
        macd_line, signal_line, macd_hist = self.calculate_macd(df['close_annual'])
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = macd_hist

        # 布林带 (20期)
        bb_upper, bb_lower, bb_bandwidth, bb_percent_b = self.calculate_bollinger_bands(df['close_annual'], 20)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_bandwidth'] = bb_bandwidth
        df['bb_percent_b'] = bb_percent_b

        # ATR波动率 (14期)
        df['atr_14'] = self.calculate_atr(df['high_annual'], df['low_annual'], df['close_annual'], 14)

        # ============ 4. 市场微观结构特征 ============
        # 价差特征
        df['price_spread'] = df['high_annual'] - df['low_annual']
        df['price_spread_ratio'] = df['price_spread'] / (df['close_annual'] + 1e-8)

        # 波动率变化率 (短期波动/长期波动)
        df['volatility_ratio'] = df['std_60'] / (df['std_1440'] + 1e-8)

        # 成交量异常检测
        df['volume_ratio'] = df['volume'] / (df['vol_ma_60'] + 1e-8)

        # 利率加速度 (二阶差分)
        df['rate_acceleration'] = df['rate_chg_60'].diff()

        # ============ 5. 改进时间特征 (周期性编码) ============
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'day_of_week' in df.columns:
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # ============ 6. 成交概率相关特征 ============
        # 当前利率与历史分位数位置
        for window in [60, 1440]:
            df[f'rate_percentile_{window}'] = df['close_annual'].rolling(window=window).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=False
            )

        # 利率偏离MA的程度
        df['rate_deviation_ma60'] = (df['close_annual'] - df['ma_60']) / (df['ma_60'] + 1e-8)
        df['rate_deviation_ma1440'] = (df['close_annual'] - df['ma_1440']) / (df['ma_1440'] + 1e-8)

        # ============ 7. 改进目标定义 (4个目标) ============
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=120)

        # Target 1: 保守利率 (从20%提升到30%分位数)
        df['future_conservative'] = df['low_annual'].rolling(window=indexer).quantile(0.3)

        # Target 2: 激进利率 (从均值改为60%分位数)
        df['future_aggressive'] = df['close_annual'].rolling(window=indexer).quantile(0.6)

        # Target 3: 平衡利率 (新增 - 70%分位数)
        df['future_balanced'] = df['close_annual'].rolling(window=indexer).quantile(0.7)

        # Target 4: 成交概率 (新增 - 二分类标签)
        # 定义：如果当前利率 <= 未来80%分位数，则标记为高成交概率
        future_80pct = df['close_annual'].rolling(window=indexer).quantile(0.8)
        df['future_execution_prob'] = (df['close_annual'] <= future_80pct).astype(int)

        # 清理 NaN
        df = df.dropna()

        return df

    def process_currency(self, currency: str, output_dir: str = "../data/processed", max_workers: int = 8):
        """
        处理单个币种的全流程：加载 -> 特征工程 -> 保存
        使用并行化加速处理

        Args:
            currency: 币种
            output_dir: 输出目录
            max_workers: 最大并行worker数量
        """
        df = self.load_data(currency)
        if df.empty:
            return None

        logger.info(f"Processing features for {currency} with {max_workers} workers...")

        # 按 Period 分组
        unique_periods = df['period'].unique()
        logger.info(f"Found {len(unique_periods)} periods to process")

        # 并行处理每个period
        processed_groups = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 为每个period提交处理任务
            future_to_period = {}
            for period in unique_periods:
                period_df = df[df['period'] == period].copy()
                future = executor.submit(self._process_single_period, period_df)
                future_to_period[future] = period

            # 收集结果
            for future in as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    result_df = future.result()
                    processed_groups.append(result_df)
                    logger.info(f"Completed processing period {period}")
                except Exception as e:
                    logger.error(f"Failed to process period {period}: {e}")

        # 合并所有结果
        if not processed_groups:
            logger.error("No groups were successfully processed")
            return None

        df_processed = pd.concat(processed_groups, ignore_index=True)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{currency}_features.parquet")
        df_processed.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}. Shape: {df_processed.shape}")

        return output_path

    @staticmethod
    def _process_single_period(period_df: pd.DataFrame) -> pd.DataFrame:
        """
        处理单个period的数据（静态方法用于并行化）

        Args:
            period_df: 单个period的DataFrame
        Returns:
            处理后的DataFrame
        """
        processor = DataProcessor()
        return processor.add_technical_indicators(period_df)

if __name__ == "__main__":
    processor = DataProcessor()
    # 示例运行
    for curr in ['fUSD', 'fUST']:
        processor.process_currency(curr)
