import pandas as pd
import numpy as np
import sqlite3
from loguru import logger
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

class DataProcessor:
    def __init__(self, db_path: str = "../data/lending_history.db"):
        self.db_path = db_path
        self.conn = None

    def connect_db(self):
        self.conn = sqlite3.connect(self.db_path)

    def close_db(self):
        if self.conn:
            self.conn.close()

    def load_data(self, currency: str) -> pd.DataFrame:
        """
        加载指定币种的所有历史数据。
        """
        logger.info(f"Loading data for {currency}...")
        self.connect_db()
        query = f"""
        SELECT 
            currency, period, timestamp, datetime, 
            open_annual, close_annual, high_annual, low_annual, volume,
            hour, day_of_week
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

    def add_technical_indicators(self, group_df):
        """
        对单个 (currency, period) 的分组数据进行特征工程
        """
        # 避免 SettingWithCopyWarning
        df = group_df.copy()
        
        # 1. 基础特征：过去一段时间的利率变化
        # 使用 close_annual 作为主要分析对象
        
        # 滞后特征 (Lags): 告诉模型现在的利率对比过去是如何变化的
        lags = [15, 30, 60, 120, 240, 1440] # 15m, 30m, 1h, 2h, 4h, 24h
        for lag in lags:
            df[f'rate_lag_{lag}'] = df['close_annual'].shift(lag)
            df[f'rate_chg_{lag}'] = df['close_annual'] - df[f'rate_lag_{lag}']
        
        # 2. 滚动统计特征 (Rolling Stats): 捕捉趋势和波动率
        windows = [60, 120, 1440] # 1h, 2h, 24h
        for window in windows:
            # 移动平均
            df[f'ma_{window}'] = df['close_annual'].rolling(window=window).mean()
            # 波动率 (标准差)
            df[f'std_{window}'] = df['close_annual'].rolling(window=window).std()
            # 相对位置 (Z-Score concept): 当前利率偏离均值多少个标准差
            df[f'zscore_{window}'] = (df['close_annual'] - df[f'ma_{window}']) / (df[f'std_{window}'] + 1e-8)
            # 成交量移动平均
            df[f'vol_ma_{window}'] = df['volume'].rolling(window=window).mean()

        # 3. 目标构建 (Target Construction)
        # 我们要预测未来 2小时 (120分钟) 内，最容易成交且收益最高的利率。
        # 策略：取未来 120 分钟内的 low_annual 的 20% 分位数。
        # 逻辑：如果设定为最低价，太保守；设定为均价，可能成不了。20% 分位数是一个兼顾成交率和收益的平衡点。
        
        # 使用 reverse window 计算未来的统计值需要技巧，通常使用 shift(-window) 配合 rolling
        # 这里为了高效，我们直接用 shift 后的 rolling
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=120)
        df['future_min_120'] = df['low_annual'].rolling(window=indexer).min()
        
        # Target 1: 保守成交价 (High Certainty) - 对应 "确保成交"
        df['future_conservative'] = df['low_annual'].rolling(window=indexer).quantile(0.2)
        
        # Target 2: 激进成交价 (High Yield) - 对应 "收益足够高" (未来均价)
        df['future_aggressive'] = df['close_annual'].rolling(window=indexer).mean()
        
        # 清理 NaN (因为 rolling 和 shift 会产生 NaN)
        df = df.dropna()
        
        return df

    def process_currency(self, currency: str, output_dir: str = "../data/processed"):
        """
        处理单个币种的全流程：加载 -> 特征工程 -> 保存
        """
        df = self.load_data(currency)
        if df.empty:
            return None

        logger.info(f"Processing features for {currency}...")
        
        # 按 Period 分组并行处理特征
        unique_periods = df['period'].unique()
        
        # 定义一个包装函数给 apply 使用
        def process_group(group):
            return self.add_technical_indicators(group)

        # 使用 joblib 或直接 groupby apply
        df_processed = df.groupby('period', group_keys=False).apply(process_group)
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{currency}_features.parquet")
        df_processed.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}. Shape: {df_processed.shape}")
        
        return output_path

if __name__ == "__main__":
    processor = DataProcessor()
    # 示例运行
    for curr in ['fUSD', 'fUST']:
        processor.process_currency(curr)
