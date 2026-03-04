import pandas as pd
import numpy as np
import sqlite3
from loguru import logger
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from functools import partial

class DataProcessor:
    def __init__(self, db_path: str = None):
        # Use absolute path for database
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            db_path = os.path.join(base_dir, "data", "lending_history.db")
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
        WHERE currency = ?
        ORDER BY period, timestamp
        """
        try:
            df = pd.read_sql(query, self.conn, params=(currency,))
            logger.info(f"Loaded {len(df)} rows for {currency}.")

            # 简单的数据清洗
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values(['period', 'timestamp'])

            # Remove extreme outliers with period-aware thresholds
            # Long periods should be cleaner because they heavily affect slow anchors.
            outlier_threshold = np.where(
                df['period'] >= 60, 80.0,
                np.where(df['period'] >= 20, 60.0, 45.0)
            )
            outlier_mask = (
                (df['close_annual'] > outlier_threshold) |
                (df['high_annual'] > outlier_threshold) |
                (df['low_annual'] > outlier_threshold) |
                (df['open_annual'] > outlier_threshold)
            )
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                logger.warning(f"Found {outlier_count} period-aware outlier records for {currency}. Removing.")
                outlier_examples = df[outlier_mask][['period', 'datetime', 'close_annual']].head(5)
                logger.warning(f"Outlier examples:\n{outlier_examples}")
                df = df[~outlier_mask].copy()
                logger.info(f"After outlier removal: {len(df)} rows")

            # Cap extreme values at 99th percentile per period
            for rate_col in ['open_annual', 'close_annual', 'high_annual', 'low_annual']:
                p99_by_period = df.groupby('period')[rate_col].transform(lambda x: x.quantile(0.99))
                capped_count = (df[rate_col] > p99_by_period).sum()
                if capped_count > 0:
                    logger.info(f"Capping {capped_count} extreme {rate_col} values to 99th percentile")
                    df[rate_col] = df[rate_col].clip(upper=p99_by_period)

            # Remove negative rates
            negative_mask = (
                (df['close_annual'] < 0) |
                (df['high_annual'] < 0) |
                (df['low_annual'] < 0) |
                (df['open_annual'] < 0)
            )
            negative_count = negative_mask.sum()
            if negative_count > 0:
                logger.warning(f"Removing {negative_count} records with negative rates")
                df = df[~negative_mask].copy()

            # Sanity check: high >= close >= low
            invalid_hlc = ~((df['high_annual'] >= df['close_annual']) &
                           (df['close_annual'] >= df['low_annual']))
            invalid_count = invalid_hlc.sum()
            if invalid_count > 0:
                logger.warning(f"Fixing {invalid_count} records with invalid high/low/close relationships")
                df.loc[invalid_hlc, 'high_annual'] = df.loc[invalid_hlc, 'close_annual']
                df.loc[invalid_hlc, 'low_annual'] = df.loc[invalid_hlc, 'close_annual']

            logger.info(f"Data quality checks complete for {currency}")

            # 填充缺失值（如有）
            df = df.ffill().bfill()

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

        # 长周期bounds用长期均线,减少短期波动影响
        df['ma_4320'] = df['close_annual'].rolling(window=4320).mean()   # 3天MA
        df['ma_10080'] = df['close_annual'].rolling(window=10080).mean() # 7天MA
        df['ma_720'] = df['close_annual'].rolling(window=720).mean()      # 12h MA

        # 稳健锚点：rolling median 对尖峰值不敏感
        df['robust_ma_720'] = df['close_annual'].rolling(window=720).median()
        df['robust_ma_1440'] = df['close_annual'].rolling(window=1440).median()
        df['robust_ma_4320'] = df['close_annual'].rolling(window=4320).median()
        df['robust_ma_10080'] = df['close_annual'].rolling(window=10080).median()

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

        # ============ 7. 执行反馈特征 (Execution Feedback Features) ============
        # 从虚拟订单历史中提取成交统计，动态调整预测策略
        # v5: 按时间窗口滚动计算,而非所有样本共享同一值
        try:
            from ml_engine.execution_features import ExecutionFeatures, get_period_window_profile

            # 获取当前币种和周期
            currency = df['currency'].iloc[0] if 'currency' in df.columns else 'fUSD'
            period = df['period'].iloc[0] if 'period' in df.columns else 30

            exec_calc = ExecutionFeatures()
            profile = get_period_window_profile(int(period))
            fast_days = profile['fast_days']
            slow_days = profile['slow_days']
            gap_days = profile['gap_days']

            # 按时间采样计算执行特征 (每周一个采样点)
            # 对于每个时间点,使用 as_of_date 参数计算当时的执行率
            if 'datetime' in df.columns and len(df) > 0:
                # 获取数据的时间范围
                min_date = df['datetime'].min()
                max_date = df['datetime'].max()

                # 分层窗口采用不同采样频率：短周期更敏感，长周期更平滑
                if fast_days <= 3:
                    sample_freq = '1D'
                elif fast_days <= 7:
                    sample_freq = '2D'
                elif fast_days <= 14:
                    sample_freq = '3D'
                else:
                    sample_freq = '7D'
                sample_dates = pd.date_range(start=min_date, end=max_date, freq=sample_freq)

                if len(sample_dates) >= 2:
                    # 为每个采样日期计算执行率
                    sample_exec_fast = []
                    sample_exec_slow = []
                    sample_timestamps = []

                    for sd in sample_dates:
                        try:
                            exec_calc.clear_cache()
                            er_fast = exec_calc.calculate_execution_rate(
                                currency, period, fast_days, as_of_date=sd.to_pydatetime()
                            )
                            er_slow = exec_calc.calculate_execution_rate(
                                currency, period, slow_days, as_of_date=sd.to_pydatetime()
                            )
                            sample_exec_fast.append(er_fast)
                            sample_exec_slow.append(er_slow)
                            sample_timestamps.append(sd)
                        except Exception:
                            pass

                    if len(sample_timestamps) >= 2:
                        # 创建采样 Series 并对齐到数据时间线
                        exec_fast_series = pd.Series(sample_exec_fast, index=sample_timestamps)
                        exec_slow_series = pd.Series(sample_exec_slow, index=sample_timestamps)

                        # 用最近邻插值对齐到原始数据时间
                        df['exec_rate_fast'] = exec_fast_series.reindex(
                            df['datetime'], method='ffill'
                        ).fillna(exec_fast_series.iloc[-1]).values
                        df['exec_rate_slow'] = exec_slow_series.reindex(
                            df['datetime'], method='ffill'
                        ).fillna(exec_slow_series.iloc[-1]).values
                    else:
                        # 采样点不够,回退到当前快照值
                        exec_features = exec_calc.get_all_features(currency, period)
                        df['exec_rate_fast'] = exec_features['exec_rate_fast']
                        df['exec_rate_slow'] = exec_features['exec_rate_slow']
                else:
                    # 时间范围太短,回退
                    exec_features = exec_calc.get_all_features(currency, period)
                    df['exec_rate_fast'] = exec_features['exec_rate_fast']
                    df['exec_rate_slow'] = exec_features['exec_rate_slow']
            else:
                exec_features = exec_calc.get_all_features(currency, period)
                df['exec_rate_fast'] = exec_features['exec_rate_fast']
                df['exec_rate_slow'] = exec_features['exec_rate_slow']

            # 兼容旧字段命名
            df['exec_rate_7d'] = df['exec_rate_fast']
            df['exec_rate_30d'] = df['exec_rate_slow']

            # 以下特征仍用当前快照(非时间敏感或数据量不足以滚动)
            exec_calc.clear_cache()
            exec_features_snapshot = exec_calc.get_all_features(currency, period)

            # 7.2 利差统计 (成交订单的平均利差)
            df['avg_spread_profile'] = exec_features_snapshot['avg_spread_profile']
            df['avg_spread_7d'] = exec_features_snapshot['avg_spread_7d']
            df['avg_spread_30d'] = exec_features_snapshot['avg_spread_30d']

            # 7.3 失败订单利率差距
            df['avg_rate_gap_failed_profile'] = exec_features_snapshot['avg_rate_gap_failed_profile']
            df['avg_rate_gap_failed_7d'] = exec_features_snapshot['avg_rate_gap_failed_7d']

            # 7.4 成交延迟分布
            df['exec_delay_p50'] = exec_features_snapshot['exec_delay_p50']
            df['exec_delay_p90'] = exec_features_snapshot['exec_delay_p90']

            # 7.5 市场竞争力评分 (当前利率相对MA的位置)
            df['market_competitiveness'] = df['close_annual'] / (df['ma_1440'] + 1e-8)

            # 7.6 成交可能性综合评分
            # 结合成交率、利率分位数、市场竞争力
            df['exec_likelihood_score'] = (
                df['exec_rate_fast'] * 0.4 +
                (1.0 - df['rate_percentile_1440']) * 0.3 +
                (df['market_competitiveness'] - 1.0).clip(lower=0, upper=0.2) * 1.5
            )

            # 7.7 动态风险调整因子 — 对称化, 与 execution_features.py 保持一致
            df['risk_adjustment_factor'] = np.select(
                [
                    df['exec_rate_fast'] < 0.3,
                    df['exec_rate_fast'] < 0.4,
                    df['exec_rate_fast'] < 0.5,
                    df['exec_rate_fast'] <= 0.6,
                    df['exec_rate_fast'] <= 0.7,
                    df['exec_rate_fast'] <= 0.85,
                    df['exec_rate_fast'] > 0.85,
                ],
                [0.90, 0.94, 0.97, 1.0, 1.03, 1.06, 1.10],
                default=1.0
            )

            # 7.8 成交率趋势 (短期vs长期)
            df['exec_rate_trend'] = df['exec_rate_fast'] / (df['exec_rate_slow'] + 1e-8)

            # 7.9 利率差距趋势 (滚动平均)
            df['rate_gap_trend'] = df['avg_rate_gap_failed_profile']

            # 7.10-7.12 预留特征位 (用于未来扩展)
            df['exec_feature_reserved_1'] = 0.0
            df['exec_feature_reserved_2'] = 0.0
            df['exec_feature_reserved_3'] = 0.0

        except ImportError as e:
            logger.warning(
                f"execution_features module not available ({e}), using default values. "
                f"This is expected if execution tracking is not enabled."
            )
            df = self._apply_default_exec_features(df)
        except Exception as e:
            logger.error(
                f"Error calculating execution features: {e}",
                exc_info=True
            )
            df = self._apply_default_exec_features(df)

        # ============ 8. 改进目标定义 (4个目标) - v3 收益率优先版 ============
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=120)

        # Target 1: 保守利率 (28%分位数 - 介于原始30%和v2的25%之间)
        df['future_conservative'] = df['low_annual'].rolling(window=indexer).quantile(0.28)

        # Target 2: 激进利率 (58%分位数 - 接近原始60%,保留高利率追求)
        df['future_aggressive'] = df['close_annual'].rolling(window=indexer).quantile(0.58)

        # Target 3: 平衡利率 (68%分位数 - 接近原始70%,温和下调)
        df['future_balanced'] = df['close_annual'].rolling(window=indexer).quantile(0.68)

        # Target 4: 成交概率 (78%分位数 - 接近原始80%)
        future_78pct = df['close_annual'].rolling(window=indexer).quantile(0.78)
        df['future_execution_prob'] = (df['close_annual'] <= future_78pct).astype(int)

        # 清理 NaN
        df = df.dropna()

        return df

    def _apply_default_exec_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """当计算失败时应用默认执行特征值 — 按周期差异化,与 execution_features.py 保持一致"""
        # exec_rate 默认值按周期区分
        if 'period' in df.columns:
            df['exec_rate_fast'] = np.where(
                df['period'] <= 7, 0.55,
                np.where(df['period'] <= 30, 0.50, 0.45)
            )
            df['exec_rate_slow'] = df['exec_rate_fast']  # 同步
            df['exec_rate_7d'] = df['exec_rate_fast']
            df['exec_rate_30d'] = df['exec_rate_slow']
        else:
            df['exec_rate_fast'] = 0.50
            df['exec_rate_slow'] = 0.50
            df['exec_rate_7d'] = 0.50
            df['exec_rate_30d'] = 0.50

        # 其余特征使用中性默认值
        other_defaults = {
            'avg_spread_profile': 0.0,
            'avg_spread_7d': 0.0,
            'avg_spread_30d': 0.0,
            'avg_rate_gap_failed_profile': 0.0,
            'avg_rate_gap_failed_7d': 0.0,
            'exec_delay_p50': 0.0,
            'exec_delay_p90': 0.0,
            'market_competitiveness': 1.0,
            'exec_likelihood_score': 0.6,
            'risk_adjustment_factor': 1.0,
            'exec_rate_trend': 1.0,
            'rate_gap_trend': 0.0,
            'exec_feature_reserved_1': 0.0,
            'exec_feature_reserved_2': 0.0,
            'exec_feature_reserved_3': 0.0,
        }
        for col, val in other_defaults.items():
            df[col] = val
        return df

    def process_currency(self, currency: str, output_dir: str = None, max_workers: int = 8):
        """
        处理单个币种的全流程：加载 -> 特征工程 -> 保存
        使用并行化加速处理

        Args:
            currency: 币种
            output_dir: 输出目录
            max_workers: 最大并行worker数量
        """
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            output_dir = os.path.join(base_dir, "data", "processed")

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
    import sys

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("log/ml_optimizer.log", rotation="10 MB", retention="7 days")

    logger.info("Starting data processing pipeline")

    processor = DataProcessor()
    success_count = 0
    failed_currencies = []

    for curr in ['fUSD', 'fUST']:
        try:
            logger.info(f"Processing {curr}")
            output_path = processor.process_currency(curr)

            if output_path and os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✓ Successfully processed {curr}: {output_path} ({file_size_mb:.2f} MB)")
                success_count += 1
            else:
                logger.error(f"✗ Failed: Output file not created for {curr}")
                failed_currencies.append(curr)
        except Exception as e:
            logger.error(f"✗ Failed to process {curr}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_currencies.append(curr)

    logger.info(f"Summary: {success_count}/2 successful")
    if failed_currencies:
        logger.error(f"Failed: {', '.join(failed_currencies)}")
        sys.exit(1)
    else:
        logger.info("All currencies processed successfully")
        sys.exit(0)
