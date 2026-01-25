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

# 忽略警告
warnings.filterwarnings('ignore')

class BitfinexDataDownloader:
    def __init__(self, db_path: str = 'data/lending_history.db', max_retries: int = 3, rate_limit_delay: float = 2.5):
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
        
        print("✅ Data downloader initialized")
        print(f"   Database: {db_path}")
        print(f"   Max retries: {max_retries}")
        print(f"   Request interval: {rate_limit_delay} seconds")
    
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
            print(f"   Database initialized: {self.db_path}")
            
        except Exception as e:
            print(f"   Database initialization failed: {e}")
            sys.exit(1)
    
    def close_database(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            print("   Database connection closed")
    
    def check_existing_data(self, currency: str, period: int, start_ts: int, end_ts: int) -> tuple:
        """
        检查数据库中已有的数据范围
        
        Returns:
            (start_ts, end_ts): 需要下载的时间范围
        """
        try:
            query = '''
            SELECT MIN(timestamp), MAX(timestamp) 
            FROM funding_rates 
            WHERE currency = ? AND period = ?
            AND timestamp >= ? AND timestamp <= ?
            '''
            self.cursor.execute(query, (currency, period, start_ts, end_ts))
            result = self.cursor.fetchone()
            
            if result and result[0] and result[1]:
                existing_start, existing_end = result[0], result[1]
                print(f"    Existing data: {datetime.fromtimestamp(existing_start/1000)} to {datetime.fromtimestamp(existing_end/1000)}")
                
                # 如果整个时间段都有数据，返回None表示不需要下载
                if existing_start <= start_ts and existing_end >= end_ts:
                    print(f"    All data already exists in database")
                    return None
                
                # 计算缺失的时间段
                missing_ranges = []
                if existing_start > start_ts:
                    missing_ranges.append((start_ts, existing_start - 60000))  # 减去1分钟避免重叠
                if existing_end < end_ts:
                    missing_ranges.append((existing_end + 60000, end_ts))  # 加上1分钟避免重叠
                
                return missing_ranges
            else:
                print(f"    No existing data found")
                return [(start_ts, end_ts)]
                
        except Exception as e:
            print(f"    Error checking existing data: {e}")
            return [(start_ts, end_ts)]
    
    def insert_data_batch(self, data: List[Dict]):
        """批量插入数据到数据库"""
        if not data:
            return
        
        try:
            # 准备插入语句
            insert_sql = '''
            INSERT OR IGNORE INTO funding_rates 
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
            print(f"    Error inserting data batch: {e}")
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
                    return response.json()
                elif response.status_code == 429:
                    # 速率限制，等待更长时间
                    wait_time = 60
                    print(f"    Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"    Request failed, status code: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"    Request timeout, retry {attempt+1}/{self.max_retries}")
                time.sleep(10)
            except Exception as e:
                print(f"    Request exception: {e}")
                time.sleep(5)
        
        print(f"    Request failed after {self.max_retries} retries")
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
        
        print(f"    Downloading {currency} period={period} data...")
        print(f"    Time range: {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}")
        
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
                print(f"    No data returned for this time range")
                break
            
            all_candles.extend(data)
            
            if data:
                last_ts = data[-1][0]
                current_start = last_ts + 60000  # 下一分钟
                
                # 显示进度
                if len(data) < limit:
                    break
                
                progress = min(100, (last_ts - start_ts) / (end_ts - start_ts) * 100)
                print(f"      Progress: {progress:.1f}% ({len(all_candles)} records)")
            else:
                break
        
        return all_candles
    
    def process_and_store_candle_data(self, candles: List[List[Any]], 
                                      currency: str, period: int) -> int:
        """
        处理蜡烛图数据并存储到数据库
        
        Returns:
            成功插入的记录数
        """
        if not candles:
            print(f"    No candles data to process")
            return 0
        
        print(f"    Processing {len(candles)} records...")
        
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
                        print(f"      Inserted {len(batch_data)} records (Total: {processed_count})")
                    batch_data = []
                    
            except Exception as e:
                print(f"      Error processing candle: {e}")
                continue
        
        # 插入剩余数据
        if batch_data:
            if self.insert_data_batch(batch_data):
                processed_count += len(batch_data)
                print(f"      Inserted {len(batch_data)} records (Total: {processed_count})")
        
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
        print(f"\n📥 Processing {currency} period={period}")
        print("  " + "-" * 40)
        
        try:
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            print(f"    Requested time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            
            # 检查数据库中已有的数据
            missing_ranges = self.check_existing_data(currency, period, start_ts, end_ts)
            
            if missing_ranges is None:
                print(f"    ✅ All data already exists in database")
                return True
            
            total_processed = 0
            for range_start, range_end in missing_ranges:
                print(f"    Downloading missing range: {datetime.fromtimestamp(range_start/1000)} to {datetime.fromtimestamp(range_end/1000)}")
                
                # 下载数据
                candles = self.fetch_candles_for_currency(currency, period, range_start, range_end)
                
                if not candles:
                    print(f"    ⚠️ No data retrieved for this range")
                    continue
                
                # 处理并存储数据
                processed = self.process_and_store_candle_data(candles, currency, period)
                total_processed += processed
            
            # 统计最终结果
            if total_processed > 0:
                print(f"    ✅ {currency} period={period} completed")
                print(f"        New records added: {total_processed:,}")
                
                # 获取总记录数
                self.cursor.execute(
                    "SELECT COUNT(*) FROM funding_rates WHERE currency = ? AND period = ?",
                    (currency, period)
                )
                total_count = self.cursor.fetchone()[0]
                print(f"        Total records in database: {total_count:,}")
                return True
            else:
                print(f"    ⚠️ No new data downloaded")
                return False
            
        except Exception as e:
            print(f"    ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def download_multiple(self, currencies: List[str], periods: List[int], days: int = 7):
        """
        下载多个币种和周期的数据（单线程）
        
        Args:
            currencies: 币种列表
            periods: period参数列表
            days: 查询天数
        """
        print("=" * 60)
        print("🚀 Bitfinex Data Downloader")
        print("=" * 60)
        print(f"Database: {self.db_path}")
        print(f"Currencies: {currencies}")
        print(f"Periods: {periods}")
        print(f"Days to query: {days}")
        print("=" * 60)
        print("💡 Features:")
        print("  • Stores data in SQLite database")
        print("  • Checks existing data before downloading")
        print("  • Avoids duplicate downloads")
        print("  • Single-threaded with rate limiting")
        print("=" * 60)
        
        start_time = datetime.now()
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_tasks = len(currencies) * len(periods)
        completed_tasks = 0
        successful_tasks = 0
        
        # 单线程顺序下载
        for currency in currencies:
            for period in periods:
                completed_tasks += 1
                print(f"\n[{completed_tasks}/{total_tasks}] Processing {currency} period={period}")
                
                success = self.download_data(currency, period, days)
                if success:
                    successful_tasks += 1
        
        # 生成报告
        end_time = datetime.now()
        total_seconds = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("📋 Download Summary Report")
        print("=" * 60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_seconds:.1f} seconds ({total_seconds/60:.1f} minutes)")
        print(f"Total tasks: {total_tasks}")
        print(f"Successful tasks: {successful_tasks}")
        print(f"Failed tasks: {total_tasks - successful_tasks}")
        
        # 数据库统计
        try:
            self.cursor.execute("SELECT COUNT(*) FROM funding_rates")
            total_records = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(DISTINCT currency) FROM funding_rates")
            distinct_currencies = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(DISTINCT period) FROM funding_rates")
            distinct_periods = self.cursor.fetchone()[0]
            
            print(f"\n📊 Database Statistics:")
            print(f"  Total records: {total_records:,}")
            print(f"  Distinct currencies: {distinct_currencies}")
            print(f"  Distinct periods: {distinct_periods}")
            
            # 按币种统计
            print(f"\n📈 Records by currency:")
            self.cursor.execute('''
            SELECT currency, COUNT(*) as count 
            FROM funding_rates 
            GROUP BY currency 
            ORDER BY count DESC
            ''')
            for row in self.cursor.fetchall():
                print(f"  {row[0]}: {row[1]:,} records")
            
        except Exception as e:
            print(f"  Error getting database statistics: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 Download completed!")
        print("=" * 60)
        print(f"📁 Database file: {self.db_path}")
        print("  Use SQLite browser or the analysis program to view data")
        print("=" * 60)
        
        # 关闭数据库连接
        self.close_database()

def main():
    """主函数"""
    print("=" * 60)
    print("📊 Bitfinex Data Downloader")
    print("=" * 60)
    
    downloader = BitfinexDataDownloader(
        db_path='data/lending_history.db',
        max_retries=3,
        rate_limit_delay=2.5
    )
    
    # 完整配置
    currencies = ['fUST', 'fUSD']
    periods = [2,3,4,5,6,7,10,14,15,20,30,60,90,120]
    days = 1200
    
    downloader.download_multiple(currencies, periods, days)
    
def check_database():
    try:
        conn = sqlite3.connect('data/lending_history.db')
        cursor = conn.cursor()
        
        # 获取表信息
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\nTables in database:")
        for table in tables:
            print(f"  {table[0]}")
        
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
            
            print(f"\n📊 Funding Rates Table Statistics:")
            print(f"  Total records: {total_records:,}")
            print(f"  Distinct currencies: {distinct_currencies}")
            print(f"  Distinct periods: {distinct_periods}")
            if time_range[0] and time_range[1]:
                print(f"  Time range: {time_range[0]} to {time_range[1]}")
            
            # 按币种统计
            print(f"\n📈 Records by currency:")
            cursor.execute('''
            SELECT currency, period, COUNT(*) as count, 
                    MIN(datetime), MAX(datetime)
            FROM funding_rates 
            GROUP BY currency, period
            ORDER BY currency, period
            ''')
            for row in cursor.fetchall():
                print(f"  {row[0]} period={row[1]}: {row[2]:,} records ({row[3]} to {row[4]})")
        
        conn.close()
        
    except Exception as e:
        print(f"Error accessing database: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()
    # check_database()
