"""
Historical Data Store for Atlas v4.0
Stores and retrieves historical market data for backtesting
"""

import asyncio
import aiohttp
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


@dataclass
class Candle:
    """Historical candle data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    source: str


@dataclass
class FundingRateRecord:
    """Historical funding rate"""
    timestamp: int
    rate: float
    exchange: str


@dataclass
class OpenInterestRecord:
    """Historical open interest"""
    timestamp: int
    value: float
    exchange: str


class DataFetcher:
    """
    Fetches historical data from multiple sources.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()
    
    async def fetch_binance_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1000
    ) -> List[Candle]:
        """Fetch historical klines from Binance"""
        
        url = "https://api.binance.com/api/v3/klines"
        
        all_candles = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start * 1000,
                "endTime": end_time * 1000,
                "limit": limit
            }
            
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data:
                            candle = Candle(
                                timestamp=item[0] // 1000,
                                open=float(item[1]),
                                high=float(item[2]),
                                low=float(item[3]),
                                close=float(item[4]),
                                volume=float(item[5]),
                                timeframe=interval,
                                source="binance"
                            )
                            all_candles.append(candle)
                        
                        if len(data) < limit:
                            break
                        
                        # Move to next batch
                        current_start = data[-1][0] // 1000 + 1
                    else:
                        break
            except Exception as e:
                print(f"Error fetching klines: {e}")
                break
            
            await asyncio.sleep(0.1)  # Rate limiting
        
        return all_candles
    
    async def fetch_funding_rates(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        limit: int = 1000
    ) -> List[FundingRateRecord]:
        """Fetch historical funding rates from Binance"""
        
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        
        all_rates = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "startTime": current_start * 1000,
                "endTime": end_time * 1000,
                "limit": limit
            }
            
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data:
                            rate = FundingRateRecord(
                                timestamp=item["fundingTime"] // 1000,
                                rate=float(item["fundingRate"]),
                                exchange="binance"
                            )
                            all_rates.append(rate)
                        
                        if len(data) < limit:
                            break
                        
                        current_start = data[-1]["fundingTime"] // 1000 + 1
                    else:
                        break
            except Exception as e:
                print(f"Error fetching funding rates: {e}")
                break
            
            await asyncio.sleep(0.1)
        
        return all_rates
    
    async def fetch_open_interest(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        limit: int = 1000
    ) -> List[OpenInterestRecord]:
        """Fetch historical open interest from Binance"""
        
        url = "https://fapi.binance.com/fapi/v1/openInterestHist"
        
        all_oi = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "period": "5m",
                "startTime": current_start * 1000,
                "endTime": end_time * 1000,
                "limit": limit
            }
            
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data:
                            oi = OpenInterestRecord(
                                timestamp=item["timestamp"] // 1000,
                                value=float(item["openInterest"]),
                                exchange="binance"
                            )
                            all_oi.append(oi)
                        
                        if len(data) < limit:
                            break
                        
                        current_start = data[-1]["timestamp"] // 1000 + 1
                    else:
                        break
            except Exception as e:
                print(f"Error fetching open interest: {e}")
                break
            
            await asyncio.sleep(0.1)
        
        return all_oi


class HistoricalDataStore:
    """
    Stores and retrieves historical market data.
    
    Storage:
    - SQLite for structured data
    - Memory cache for frequently accessed data
    """
    
    def __init__(self, db_path: str = "data/backtest/historical.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._init_db()
        self._cache: Dict[str, Any] = {}
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Candles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timeframe TEXT,
                source TEXT,
                PRIMARY KEY (timestamp, timeframe, source)
            )
        """)
        
        # Funding rates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                timestamp INTEGER PRIMARY KEY,
                rate REAL,
                exchange TEXT
            )
        """)
        
        # Open interest table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                timestamp INTEGER PRIMARY KEY,
                value REAL,
                exchange TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candles_tf ON candles(timeframe)")
        
        conn.commit()
        conn.close()
    
    def store_candles(self, candles: List[Candle]):
        """Store candles in database"""
        if not candles:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for c in candles:
            cursor.execute("""
                INSERT OR REPLACE INTO candles 
                (timestamp, open, high, low, close, volume, timeframe, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (c.timestamp, c.open, c.high, c.low, c.close, c.volume, c.timeframe, c.source))
        
        conn.commit()
        conn.close()
        
        # Invalidate cache
        self._cache.pop("candles", None)
    
    def store_funding_rates(self, rates: List[FundingRateRecord]):
        """Store funding rates in database"""
        if not rates:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for r in rates:
            cursor.execute("""
                INSERT OR REPLACE INTO funding_rates (timestamp, rate, exchange)
                VALUES (?, ?, ?)
            """, (r.timestamp, r.rate, r.exchange))
        
        conn.commit()
        conn.close()
    
    def store_open_interest(self, oi_records: List[OpenInterestRecord]):
        """Store open interest records"""
        if not oi_records:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for oi in oi_records:
            cursor.execute("""
                INSERT OR REPLACE INTO open_interest (timestamp, value, exchange)
                VALUES (?, ?, ?)
            """, (oi.timestamp, oi.value, oi.exchange))
        
        conn.commit()
        conn.close()
    
    def get_candles(
        self,
        timeframe: str,
        start_time: int,
        end_time: int,
        source: str = "binance"
    ) -> List[Dict]:
        """Get candles from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE timeframe = ? AND source = ?
            AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (timeframe, source, start_time, end_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": row[4],
                "volume": row[5]
            }
            for row in rows
        ]
    
    def get_funding_rates(
        self,
        start_time: int,
        end_time: int
    ) -> List[Dict]:
        """Get funding rates from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, rate, exchange
            FROM funding_rates
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (start_time, end_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "rate": row[1],
                "exchange": row[2]
            }
            for row in rows
        ]
    
    def get_open_interest(
        self,
        start_time: int,
        end_time: int
    ) -> List[Dict]:
        """Get open interest from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, value, exchange
            FROM open_interest
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (start_time, end_time))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "value": row[1],
                "exchange": row[2]
            }
            for row in rows
        ]
    
    def get_data_range(self, timeframe: str) -> Tuple[Optional[int], Optional[int]]:
        """Get available data range for a timeframe"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM candles
            WHERE timeframe = ?
        """, (timeframe,))
        
        row = cursor.fetchone()
        conn.close()
        
        return (row[0], row[1]) if row else (None, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data store statistics"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count candles
        cursor.execute("SELECT COUNT(*) FROM candles")
        stats["candle_count"] = cursor.fetchone()[0]
        
        # Count funding rates
        cursor.execute("SELECT COUNT(*) FROM funding_rates")
        stats["funding_rate_count"] = cursor.fetchone()[0]
        
        # Count open interest
        cursor.execute("SELECT COUNT(*) FROM open_interest")
        stats["open_interest_count"] = cursor.fetchone()[0]
        
        # Get time ranges
        cursor.execute("""
            SELECT timeframe, MIN(timestamp), MAX(timestamp)
            FROM candles
            GROUP BY timeframe
        """)
        
        stats["time_ranges"] = {}
        for row in cursor.fetchall():
            stats["time_ranges"][row[0]] = {
                "start": datetime.fromtimestamp(row[1]).isoformat() if row[1] else None,
                "end": datetime.fromtimestamp(row[2]).isoformat() if row[2] else None
            }
        
        conn.close()
        
        return stats
    
    async def fetch_and_store(
        self,
        start_date: str,
        end_date: str,
        symbol: str = "BTCUSDT"
    ):
        """
        Fetch and store historical data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Trading symbol
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        async with DataFetcher() as fetcher:
            # Fetch candles for each timeframe
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            
            for tf in timeframes:
                print(f"Fetching {tf} candles...")
                candles = await fetcher.fetch_binance_klines(symbol, tf, start_ts, end_ts)
                self.store_candles(candles)
                print(f"  Stored {len(candles)} {tf} candles")
            
            # Fetch funding rates
            print("Fetching funding rates...")
            rates = await fetcher.fetch_funding_rates(symbol, start_ts, end_ts)
            self.store_funding_rates(rates)
            print(f"  Stored {len(rates)} funding rate records")
            
            # Fetch open interest
            print("Fetching open interest...")
            oi = await fetcher.fetch_open_interest(symbol, start_ts, end_ts)
            self.store_open_interest(oi)
            print(f"  Stored {len(oi)} open interest records")


async def fetch_historical_data(start_date: str, end_date: str):
    """CLI helper to fetch historical data"""
    store = HistoricalDataStore()
    await store.fetch_and_store(start_date, end_date)
    print("\nData store stats:")
    print(json.dumps(store.get_stats(), indent=2))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        start = sys.argv[1]
        end = sys.argv[2]
        asyncio.run(fetch_historical_data(start, end))
    else:
        print("Usage: python data_store.py START_DATE END_DATE")
        print("  Dates in YYYY-MM-DD format")
