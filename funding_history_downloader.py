import time
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import json
import warnings
from typing import List, Dict, Any, Optional
import sys
from loguru import logger

# 忽略警告
warnings.filterwarnings('ignore')

class BitfinexDataDownloader:
    def __init__(self, db_path: str = '/home/bumblebee/Project/optimize/data/lending_history.db', max_retries: int = 3, rate_limit_delay: float = 2.5):
        """
        初始化Bitfinex数据下载器
        
        Args:
            db_path: SQLite数据库路径
            max_retries: 最大重试次数
            rate_limit_delay: 基础请求延迟（秒）
        """
        self.base_url = "https://api-pub.bitfinex.com/v2"
        self.max_retries = max_retries
        self.base_delay = rate_limit_delay
        self.db_path = db_path
        
        # 初始化数据库连接
        self.conn = None
        self.cursor = None
        self.init_database()
        
        # 设置请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        })
        
        logger.info("✅ Data downloader initialized")
        logger.info(f"   Database: {db_path}")
        logger.info(f"   Max retries: {max_retries}")
        logger.info(f"   Request interval: {rate_limit_delay} seconds")

    def get_latest_timestamp(self, currency: str, period: int) -> Optional[int]:
        """获取库中该组合的最新时间戳(ms)。"""
        self.cursor.execute(
            "SELECT MAX(timestamp) FROM funding_rates WHERE currency = ? AND period = ?",
            (currency, period)
        )
        row = self.cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def get_latest_age_minutes(self, currency: str, period: int) -> Optional[float]:
        """返回该组合最新数据距离当前的分钟数。"""
        latest_ts = self.get_latest_timestamp(currency, period)
        if latest_ts is None:
            return None
        now_ms = int(datetime.now().timestamp() * 1000)
        return max(0.0, (now_ms - latest_ts) / 60000.0)

    def freshness_target_minutes(self, currency: str) -> int:
        """
        与 predictor 的硬阈值对齐:
        - fUSD: 300 分钟
        - fUST: 900 分钟
        """
        return 900 if currency == 'fUST' else 300
    
    def init_database(self):
        """初始化数据库和表结构"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # 创建主表
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS funding_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT NOT NULL,
                period INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                datetime TEXT NOT NULL,
                open_rate REAL,
                close_rate REAL,
                high_rate REAL,
                low_rate REAL,
                volume REAL,
                open_annual REAL,
                close_annual REAL,
                high_annual REAL,
                low_annual REAL,
                high_rate_flag INTEGER DEFAULT 0,
                hour INTEGER,
                minute INTEGER,
                day_of_week INTEGER,
                month INTEGER,
                year_month TEXT,
                candle_size TEXT DEFAULT '1m',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(currency, period, timestamp)
            )
            ''')
            
            # 创建索引以提高查询性能
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_currency_period ON funding_rates(currency, period)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON funding_rates(timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_datetime ON funding_rates(datetime)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_currency_period_datetime ON funding_rates(currency, period, datetime)')
            
            self.conn.commit()
            logger.info(f"   Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.info(f"   Database initialization failed: {e}")
            sys.exit(1)
    
    def close_database(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("   Database connection closed")
    
    def check_existing_data(self, currency: str, period: int, start_ts: int, end_ts: int) -> tuple:
        """
        检查数据库中已有的数据范围

        Returns:
            (start_ts, end_ts): 需要下载的时间范围
        """
        now_ms = int(datetime.now().timestamp() * 1000)

        try:
            # 首先检查最新一条数据的完整信息（不受查询范围限制）
            self.cursor.execute(
                "SELECT MAX(timestamp) FROM funding_rates WHERE currency = ? AND period = ?",
                (currency, period)
            )
            result = self.cursor.fetchone()
            latest_existing_ts = result[0] if result and result[0] else None

            # 诊断日志：显示当前最新数据时间
            if latest_existing_ts:
                latest_dt = datetime.fromtimestamp(latest_existing_ts / 1000)
                age_minutes = (now_ms - latest_existing_ts) / 60000
                logger.info(f"    🔍 DB latest: {latest_dt.strftime('%Y-%m-%d %H:%M')} (age: {age_minutes:.0f} min)")
            else:
                logger.info(f"    🔍 No data in DB for {currency} period={period}")

            # 计算查询范围内已有的数据
            query = '''
            SELECT MIN(timestamp), MAX(timestamp)
            FROM funding_rates
            WHERE currency = ? AND period = ?
            AND timestamp >= ? AND timestamp <= ?
            '''
            self.cursor.execute(query, (currency, period, start_ts, end_ts))
            range_result = self.cursor.fetchone()

            # 修复方案B: 简化逻辑 - 如果最新数据过期(>2小时)，直接强制刷新最近7天
            freshness_threshold_ms = 2 * 3600 * 1000  # 2小时

            if latest_existing_ts and (now_ms - latest_existing_ts) <= freshness_threshold_ms:
                # 数据是新鲜的，检查范围覆盖
                if range_result and range_result[0] and range_result[1]:
                    existing_start, existing_end = range_result[0], range_result[1]
                    logger.info(f"    Existing data in range: {datetime.fromtimestamp(existing_start/1000)} to {datetime.fromtimestamp(existing_end/1000)}")

                    if existing_start <= start_ts and existing_end >= end_ts:
                        logger.info(f"    ✅ All data already exists and is fresh")
                        return None

                    # 计算缺失范围
                    missing_ranges = []
                    if existing_start > start_ts:
                        missing_ranges.append((start_ts, existing_start - 60000))
                    if existing_end < end_ts:
                        missing_ranges.append((existing_end + 60000, end_ts))

                    return missing_ranges if missing_ranges else None
                else:
                    # 范围外有新数据，但范围内没有
                    return [(start_ts, end_ts)]
            else:
                # 数据过期或不存在 - 修复3: 优先尝试最近24小时窗口，避免长周期无效全量扫描
                logger.warning(f"    ⚠️ Data is stale (age > 2h), trying last 24h window first")
                recent_start = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
                adjusted_start = max(start_ts, recent_start)
                return [(adjusted_start, end_ts)]

        except Exception as e:
            logger.info(f"    Error checking existing data: {e}")
            return [(start_ts, end_ts)]
    
    def insert_data_batch(self, data: List[Dict]):
        """批量插入数据到数据库"""
        if not data:
            return
        
        try:
            # 准备插入语句
            insert_sql = '''
            INSERT OR REPLACE INTO funding_rates 
            (currency, period, timestamp, datetime, open_rate, close_rate, high_rate, low_rate, volume,
             open_annual, close_annual, high_annual, low_annual, high_rate_flag,
             hour, minute, day_of_week, month, year_month, candle_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            # 批量插入
            self.cursor.executemany(insert_sql, data)
            self.conn.commit()
            
            return True
        except Exception as e:
            logger.info(f"    Error inserting data batch: {e}")
            self.conn.rollback()
            return False
    
    def calculate_annualized_rate(self, daily_rate: float) -> float:
        """将日利率转换为年化利率"""
        if pd.isna(daily_rate) or daily_rate == 0:
            return 0.0
        return float(daily_rate) * 365 * 100
    
    def rate_limited_request(self, url: str, params: Dict) -> Optional[List]:
        """
        带有速率限制和自动重试的请求方法
        单线程下载，控制请求频率
        """
        for attempt in range(self.max_retries):
            try:
                # 固定延迟控制（避免过于频繁）
                time.sleep(self.base_delay)

                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    # 修复2: 检测 Bitfinex 错误响应格式 ["error", 10xxx, "msg"]
                    if isinstance(data, list) and len(data) >= 2 and data[0] == 'error':
                        logger.warning(f"    Bitfinex API error response: {data}")
                        return None
                    return data
                elif response.status_code == 429:
                    # 速率限制，等待更长时间
                    wait_time = 60
                    logger.info(f"    Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # B6 FIX: Non-429 errors should also retry instead of silently failing
                    wait_time = min(10 * (2 ** attempt), 120)
                    logger.info(f"    Request failed (status {response.status_code}), retry {attempt+1}/{self.max_retries}, backoff {wait_time}s")
                    time.sleep(wait_time)
                    continue

            except requests.exceptions.Timeout:
                wait_time = min(10 * (2 ** attempt), 120)
                logger.info(f"    Request timeout, retry {attempt+1}/{self.max_retries}, backoff {wait_time}s")
                time.sleep(wait_time)
            except Exception as e:
                wait_time = min(5 * (2 ** attempt), 60)
                logger.info(f"    Request exception: {e}, backoff {wait_time}s")
                time.sleep(wait_time)

        logger.info(f"    Request failed after {self.max_retries} retries")
        return None
    
    def fetch_candles_for_currency(self, currency: str, period: int, 
                                   start_ts: int, end_ts: int) -> List[List[Any]]:
        """
        获取指定币种和period的蜡烛图数据
        
        Args:
            currency: 币种
            period: period参数
            start_ts: 开始时间戳
            end_ts: 结束时间戳
            
        Returns:
            蜡烛图数据
        """
        all_candles = []
        current_start = start_ts
        limit = 10000
        
        logger.info(f"    Downloading {currency} period={period} data...")
        logger.info(f"    Time range: {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}")
        
        while current_start < end_ts:
            # 构建URL
            candle_key = f"trade:1m:{currency}:p{period}"
            url = f"{self.base_url}/candles/{candle_key}/hist"
            
            params = {
                'sort': 1,  # 升序
                'start': current_start,
                'end': end_ts,
                'limit': limit
            }
            
            data = self.rate_limited_request(url, params)
            
            if not data:
                logger.info(f"    No data returned for this time range")
                break
            
            all_candles.extend(data)
            
            if data:
                last_ts = data[-1][0]
                current_start = last_ts + 60000  # 下一分钟
                
                # 显示进度
                if len(data) < limit:
                    break
                
                progress = min(100, (last_ts - start_ts) / (end_ts - start_ts) * 100)
                logger.info(f"      Progress: {progress:.1f}% ({len(all_candles)} records)")
            else:
                break

        # 修复1: hist 返回空时降级调用 /last 端点（流动性枯竭时的 fallback）
        if not all_candles:
            fallback_candle_key = f"trade:1m:{currency}:p{period}"
            last_url = f"{self.base_url}/candles/{fallback_candle_key}/last"
            last_data = self.rate_limited_request(last_url, {})
            if isinstance(last_data, list) and len(last_data) >= 6 and isinstance(last_data[0], (int, float)):
                all_candles = [last_data]
                logger.warning(f"    📍 hist returned empty, /last fallback: 1 candle retrieved")
            elif last_data is not None:
                logger.warning(f"    ⚠️ /last returned incomplete candle for {currency} p{period}: {last_data}")
            else:
                logger.warning(f"    ⚠️ /last fallback also returned no data for {currency} p{period}")

        return all_candles
    
    def process_and_store_candle_data(self, candles: List[List[Any]], 
                                      currency: str, period: int) -> int:
        """
        处理蜡烛图数据并存储到数据库
        
        Returns:
            成功插入的记录数
        """
        if not candles:
            logger.info(f"    No candles data to process")
            return 0
        
        logger.info(f"    Processing {len(candles)} records...")
        
        processed_count = 0
        batch_size = 1000
        batch_data = []
        
        for candle in candles:
            try:
                timestamp = int(candle[0])
                dt = datetime.fromtimestamp(timestamp / 1000)
                datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 提取数据
                open_rate = float(candle[1]) if candle[1] is not None else 0.0
                close_rate = float(candle[2]) if candle[2] is not None else 0.0
                high_rate = float(candle[3]) if candle[3] is not None else 0.0
                low_rate = float(candle[4]) if candle[4] is not None else 0.0
                volume = float(candle[5]) if candle[5] is not None else 0.0
                
                # 计算年化利率
                open_annual = self.calculate_annualized_rate(open_rate)
                close_annual = self.calculate_annualized_rate(close_rate)
                high_annual = self.calculate_annualized_rate(high_rate)
                low_annual = self.calculate_annualized_rate(low_rate)
                
                # 标记高利率
                high_rate_flag = 1 if high_annual > 13 else 0
                
                # 提取时间信息
                hour = dt.hour
                minute = dt.minute
                day_of_week = dt.weekday()
                month = dt.month
                year_month = dt.strftime('%Y-%m')
                
                # 准备数据记录
                record = (
                    currency, period, timestamp, datetime_str,
                    open_rate, close_rate, high_rate, low_rate, volume,
                    open_annual, close_annual, high_annual, low_annual, high_rate_flag,
                    hour, minute, day_of_week, month, year_month, '1m'
                )
                
                batch_data.append(record)
                
                # 批量插入
                if len(batch_data) >= batch_size:
                    if self.insert_data_batch(batch_data):
                        processed_count += len(batch_data)
                        logger.info(f"      Inserted {len(batch_data)} records (Total: {processed_count})")
                    batch_data = []
                    
            except Exception as e:
                logger.info(f"      Error processing candle: {e}")
                continue
        
        # 插入剩余数据
        if batch_data:
            if self.insert_data_batch(batch_data):
                processed_count += len(batch_data)
                logger.info(f"      Inserted {len(batch_data)} records (Total: {processed_count})")
        
        return processed_count
    
    def download_data(self, currency: str, period: int, days: int = 7) -> bool:
        """
        下载指定币种和周期的数据
        
        Args:
            currency: 币种
            period: period参数
            days: 查询天数
            
        Returns:
            是否成功
        """
        logger.info(f"\n📥 Processing {currency} period={period}")
        logger.info("  " + "-" * 40)
        
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            freshness_target = self.freshness_target_minutes(currency)
            before_latest_ts = self.get_latest_timestamp(currency, period)

            logger.info(f"    Requested time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            if before_latest_ts:
                before_dt = datetime.fromtimestamp(before_latest_ts / 1000)
                before_age = self.get_latest_age_minutes(currency, period)
                logger.info(
                    f"    Baseline latest: {before_dt.strftime('%Y-%m-%d %H:%M')} "
                    f"(age: {before_age:.0f} min, target <= {freshness_target} min)"
                )

            # 检查数据库中已有的数据
            missing_ranges = self.check_existing_data(currency, period, start_ts, end_ts)

            if missing_ranges is None:
                logger.info(f"    ✅ All data already exists in database")
                return True

            total_processed = 0
            attempted_ranges = set()

            def run_range(range_start: int, range_end: int, reason: str) -> int:
                if range_start >= range_end:
                    return 0
                cache_key = (range_start, range_end)
                if cache_key in attempted_ranges:
                    return 0
                attempted_ranges.add(cache_key)
                logger.info(
                    f"    Downloading {reason}: "
                    f"{datetime.fromtimestamp(range_start/1000)} to {datetime.fromtimestamp(range_end/1000)}"
                )
                candles = self.fetch_candles_for_currency(currency, period, range_start, range_end)
                if not candles:
                    logger.info(f"    ⚠️ No data retrieved for {reason}")
                    return 0
                return self.process_and_store_candle_data(candles, currency, period)

            for range_start, range_end in missing_ranges:
                processed = run_range(range_start, range_end, "primary refresh window")
                total_processed += processed

            def needs_expanded_refresh() -> bool:
                latest_age = self.get_latest_age_minutes(currency, period)
                latest_ts = self.get_latest_timestamp(currency, period)
                if latest_age is None or latest_ts is None:
                    return True
                latest_advanced = before_latest_ts is None or latest_ts > before_latest_ts
                return (latest_age > freshness_target) or (not latest_advanced)

            if needs_expanded_refresh():
                logger.warning(
                    f"    ⚠️ Primary refresh did not reach freshness target for {currency}-{period}d; "
                    f"expanding lookback windows"
                )
                expansion_windows = [
                    ("72h fallback", end_time - timedelta(hours=72)),
                    ("7d fallback", end_time - timedelta(days=7)),
                    ("30d fallback", end_time - timedelta(days=30)),
                ]
                for label, window_start in expansion_windows:
                    range_start = max(start_ts, int(window_start.timestamp() * 1000))
                    processed = run_range(range_start, end_ts, label)
                    total_processed += processed
                    if not needs_expanded_refresh():
                        break

            final_latest_ts = self.get_latest_timestamp(currency, period)
            final_age = self.get_latest_age_minutes(currency, period)
            latest_advanced = (
                final_latest_ts is not None and
                (before_latest_ts is None or final_latest_ts > before_latest_ts)
            )
            freshness_ok = final_age is not None and final_age <= freshness_target

            # 统计最终结果
            if freshness_ok:
                logger.info(f"    ✅ {currency} period={period} completed")
                logger.info(f"        Processed records: {total_processed:,}")
                logger.info(f"        Latest age after refresh: {final_age:.0f} min")

                # 获取总记录数
                self.cursor.execute(
                    "SELECT COUNT(*) FROM funding_rates WHERE currency = ? AND period = ?",
                    (currency, period)
                )
                total_count = self.cursor.fetchone()[0]
                logger.info(f"        Total records in database: {total_count:,}")
                return True

            if final_latest_ts:
                final_dt = datetime.fromtimestamp(final_latest_ts / 1000)
                # 诊断：区分"Bitfinex 无新数据"和"下载代码问题"
                if not latest_advanced and total_processed == 0:
                    diag = "Bitfinex API returned no new candles (market liquidity exhausted)"
                elif not latest_advanced:
                    diag = f"downloaded {total_processed} records but none newer than existing data"
                else:
                    diag = f"timestamp advanced but still exceeds freshness target"
                logger.warning(
                    f"    ⚠️ Refresh incomplete for {currency}-{period}d: "
                    f"latest={final_dt.strftime('%Y-%m-%d %H:%M')} age={final_age:.0f} min "
                    f"(target <= {freshness_target} min) — {diag}"
                )
            else:
                logger.warning(f"    ⚠️ Refresh incomplete for {currency}-{period}d: still no data in DB")
            return False

        except Exception as e:
            logger.info(f"    ❌ Processing failed: {e}")
            logger.exception("Unexpected error during download")
            return False

    def download_multiple(self, currencies: List[str], periods: List[int], days: int = 7):
        """
        下载多个币种和周期的数据（单线程）
        
        Args:
            currencies: 币种列表
            periods: period参数列表
            days: 查询天数
        """
        logger.info("=" * 60)
        logger.info("🚀 Bitfinex Data Downloader")
        logger.info("=" * 60)
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Currencies: {currencies}")
        logger.info(f"Periods: {periods}")
        logger.info(f"Days to query: {days}")
        logger.info("=" * 60)
        logger.info("💡 Features:")
        logger.info("  • Stores data in SQLite database")
        logger.info("  • Checks existing data before downloading")
        logger.info("  • Avoids duplicate downloads")
        logger.info("  • Single-threaded with rate limiting")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_tasks = len(currencies) * len(periods)
        completed_tasks = 0
        successful_tasks = 0
        failed_items = []

        # 单线程顺序下载
        for currency in currencies:
            for period in periods:
                completed_tasks += 1
                logger.info(f"\n[{completed_tasks}/{total_tasks}] Processing {currency} period={period}")

                success = self.download_data(currency, period, days)
                if success:
                    successful_tasks += 1
                else:
                    failed_items.append((currency, period))

        # P7B: 对失败项重试一次
        if failed_items:
            logger.info(f"\n🔄 Retrying {len(failed_items)} failed downloads...")
            retry_failures = []
            for currency, period in failed_items:
                logger.info(f"  Retry: {currency} period={period}")
                success = self.download_data(currency, period, days)
                if success:
                    successful_tasks += 1
                    logger.info(f"  ✅ Retry succeeded: {currency} period={period}")
                else:
                    retry_failures.append((currency, period))
                    logger.warning(f"  ❌ Retry failed: {currency} period={period}")
            failed_items = retry_failures
        
        # 生成报告
        end_time = datetime.now()
        total_seconds = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 60)
        logger.info("📋 Download Summary Report")
        logger.info("=" * 60)
        logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time: {total_seconds:.1f} seconds ({total_seconds/60:.1f} minutes)")
        logger.info(f"Total tasks: {total_tasks}")
        logger.info(f"Successful tasks: {successful_tasks}")
        logger.info(f"Failed tasks: {len(failed_items)}")
        
        # 数据库统计
        try:
            self.cursor.execute("SELECT COUNT(*) FROM funding_rates")
            total_records = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(DISTINCT currency) FROM funding_rates")
            distinct_currencies = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(DISTINCT period) FROM funding_rates")
            distinct_periods = self.cursor.fetchone()[0]
            
            logger.info(f"\n📊 Database Statistics:")
            logger.info(f"  Total records: {total_records:,}")
            logger.info(f"  Distinct currencies: {distinct_currencies}")
            logger.info(f"  Distinct periods: {distinct_periods}")
            
            # 按币种统计
            logger.info(f"\n📈 Records by currency:")
            self.cursor.execute('''
            SELECT currency, COUNT(*) as count 
            FROM funding_rates 
            GROUP BY currency 
            ORDER BY count DESC
            ''')
            for row in self.cursor.fetchall():
                logger.info(f"  {row[0]}: {row[1]:,} records")
            
        except Exception as e:
            logger.info(f"  Error getting database statistics: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Download completed!")
        logger.info("=" * 60)
        logger.info(f"📁 Database file: {self.db_path}")
        logger.info("  Use SQLite browser or the analysis program to view data")

        # 修改6.2: Log data freshness summary after download
        try:
            logger.info(f"\n📊 Data Freshness Check:")
            stale_count = 0
            for currency in currencies:
                for period in periods:
                    self.cursor.execute(
                        "SELECT MAX(datetime) FROM funding_rates WHERE currency = ? AND period = ?",
                        (currency, period)
                    )
                    result = self.cursor.fetchone()
                    if result and result[0]:
                        latest_dt = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
                        age_hours = (datetime.now() - latest_dt).total_seconds() / 3600
                        if age_hours > 4:
                            logger.warning(f"  STALE: {currency}-{period}d latest data is {age_hours:.1f}h old ({result[0]})")
                            stale_count += 1
                        else:
                            logger.info(f"  OK: {currency}-{period}d latest: {result[0]} ({age_hours:.1f}h ago)")
                    else:
                        logger.warning(f"  MISSING: {currency}-{period}d has no data")
                        stale_count += 1
            if stale_count > 0:
                logger.warning(f"\n  ⚠️  {stale_count} currency-period combinations have stale or missing data")
            else:
                logger.info(f"\n  ✅ All data is fresh (< 4h old)")
        except Exception as e:
            logger.info(f"  Error checking freshness: {e}")

        logger.info("=" * 60)

        # 关闭数据库连接
        self.close_database()
        return len(failed_items) == 0

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Bitfinex Data Downloader')
    parser.add_argument('--days', type=int, default=1200, help='Number of days of history to download')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("📊 Bitfinex Data Downloader")
    logger.info("=" * 60)

    downloader = BitfinexDataDownloader(
        db_path='/home/bumblebee/Project/optimize/data/lending_history.db',
        max_retries=3,
        rate_limit_delay=2.5
    )

    # 完整配置
    currencies = ['fUST', 'fUSD']
    periods = [2,3,4,5,6,7,10,14,15,20,30,60,90,120]

    all_ok = downloader.download_multiple(currencies, periods, args.days)
    if not all_ok:
        logger.error("❌ Download finished with stale/failed combinations")
        sys.exit(1)
    
def check_database():
    try:
        conn = sqlite3.connect('/home/bumblebee/Project/optimize/data/lending_history.db')
        cursor = conn.cursor()
        
        # 获取表信息
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"\nTables in database:")
        for table in tables:
            logger.info(f"  {table[0]}")
        
        # 获取 funding_rates 表统计
        if ('funding_rates',) in tables:
            cursor.execute("SELECT COUNT(*) FROM funding_rates")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT currency) FROM funding_rates")
            distinct_currencies = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT period) FROM funding_rates")
            distinct_periods = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(datetime), MAX(datetime) FROM funding_rates")
            time_range = cursor.fetchone()
            
            logger.info(f"\n📊 Funding Rates Table Statistics:")
            logger.info(f"  Total records: {total_records:,}")
            logger.info(f"  Distinct currencies: {distinct_currencies}")
            logger.info(f"  Distinct periods: {distinct_periods}")
            if time_range[0] and time_range[1]:
                logger.info(f"  Time range: {time_range[0]} to {time_range[1]}")
            
            # 按币种统计
            logger.info(f"\n📈 Records by currency:")
            cursor.execute('''
            SELECT currency, period, COUNT(*) as count, 
                    MIN(datetime), MAX(datetime)
            FROM funding_rates 
            GROUP BY currency, period
            ORDER BY currency, period
            ''')
            for row in cursor.fetchall():
                logger.info(f"  {row[0]} period={row[1]}: {row[2]:,} records ({row[3]} to {row[4]})")
        
        conn.close()
        
    except Exception as e:
        logger.info(f"Error accessing database: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nProgram interrupted by user")
    except Exception as e:
        logger.exception(f"\nProgram error: {e}")
    # check_database()
