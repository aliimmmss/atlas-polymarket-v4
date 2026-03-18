"""
Polymarket Market Synchronization for Atlas v4.0
Enhanced with better PTB fetching and market timing
"""

import os
import sys
import time
import asyncio
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
import json


# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CRYPTO_PRICE_API = "https://polymarket.com/api/crypto/crypto-price"


class PolymarketMarket:
    """
    Represents a Polymarket 15-minute BTC market.
    """
    
    def __init__(
        self,
        slug: str,
        start_time: int,
        end_time: int,
        ptb: Optional[float] = None,
        up_token: Optional[str] = None,
        down_token: Optional[str] = None,
        up_price: Optional[float] = None,
        down_price: Optional[float] = None
    ):
        self.slug = slug
        self.start_time = start_time
        self.end_time = end_time
        self.ptb = ptb  # Price to Beat
        self.up_token = up_token
        self.down_token = down_token
        self.up_price = up_price
        self.down_price = down_price
    
    @property
    def remaining_seconds(self) -> int:
        """Seconds until market closes"""
        now = int(time.time())
        return max(0, self.end_time - now)
    
    @property
    def elapsed_seconds(self) -> int:
        """Seconds since market opened"""
        now = int(time.time())
        return max(0, now - self.start_time)
    
    @property
    def is_active(self) -> bool:
        """Check if market is currently active"""
        now = int(time.time())
        return self.start_time <= now < self.end_time
    
    @property
    def is_closed(self) -> bool:
        """Check if market has closed"""
        return time.time() >= self.end_time
    
    @property
    def start_datetime(self) -> datetime:
        """Start time as datetime"""
        return datetime.fromtimestamp(self.start_time, tz=timezone.utc)
    
    @property
    def end_datetime(self) -> datetime:
        """End time as datetime"""
        return datetime.fromtimestamp(self.end_time, tz=timezone.utc)


class PolymarketSync:
    """
    Synchronizes predictions with Polymarket 15-minute BTC markets.
    """
    
    INTERVAL = 900  # 15 minutes in seconds
    
    def __init__(self, proxies: Optional[Dict] = None):
        self.proxies = proxies
    
    @staticmethod
    def get_market_slug(unix_timestamp: Optional[int] = None) -> str:
        """Get the market slug for a given timestamp."""
        ts = unix_timestamp or int(time.time())
        market_ts = (ts // PolymarketSync.INTERVAL) * PolymarketSync.INTERVAL
        return f"btc-updown-15m-{market_ts}"
    
    @staticmethod
    def get_next_market_slug() -> str:
        """Get the slug for the NEXT 15-minute market"""
        ts = int(time.time())
        next_ts = ((ts // PolymarketSync.INTERVAL) + 1) * PolymarketSync.INTERVAL
        return f"btc-updown-15m-{next_ts}"
    
    @staticmethod
    def get_current_market_times() -> Tuple[int, int]:
        """Get the start and end timestamps for the current market window."""
        now = int(time.time())
        start = (now // PolymarketSync.INTERVAL) * PolymarketSync.INTERVAL
        end = start + PolymarketSync.INTERVAL
        return start, end
    
    @staticmethod
    def get_next_market_times() -> Tuple[int, int]:
        """Get the start and end timestamps for the NEXT market window."""
        now = int(time.time())
        start = ((now // PolymarketSync.INTERVAL) + 1) * PolymarketSync.INTERVAL
        end = start + PolymarketSync.INTERVAL
        return start, end
    
    @staticmethod
    def seconds_until_next_market() -> int:
        """Seconds until the next market starts"""
        now = int(time.time())
        next_start = ((now // PolymarketSync.INTERVAL) + 1) * PolymarketSync.INTERVAL
        return next_start - now
    
    @staticmethod
    def seconds_remaining_in_current() -> int:
        """Seconds remaining in the current market window"""
        now = int(time.time())
        current_end = ((now // PolymarketSync.INTERVAL) * PolymarketSync.INTERVAL) + PolymarketSync.INTERVAL
        return max(0, current_end - now)
    
    def fetch_market_by_slug(self, slug: str) -> Optional[PolymarketMarket]:
        """Fetch market data from Polymarket Gamma API."""
        # Extract start time from slug
        try:
            slug_parts = slug.split("-")
            if len(slug_parts) >= 4:
                start_ts = int(slug_parts[-1])
            else:
                now = int(time.time())
                start_ts = (now // self.INTERVAL) * self.INTERVAL
        except (ValueError, IndexError):
            now = int(time.time())
            start_ts = (now // self.INTERVAL) * self.INTERVAL
        
        end_ts = start_ts + self.INTERVAL
        
        up_token = None
        down_token = None
        up_price = None
        down_price = None
        
        try:
            response = requests.get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                proxies=self.proxies,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    event = data[0]
                    
                    if event.get("closed", False):
                        return None
                    
                    markets = event.get("markets", [])
                    if markets:
                        m = markets[0]
                        
                        prices_raw = m.get("outcomePrices", "[]")
                        if isinstance(prices_raw, str):
                            prices = json.loads(prices_raw)
                        else:
                            prices = prices_raw
                        
                        tokens_raw = m.get("clobTokenIds", "[]")
                        if isinstance(tokens_raw, str):
                            tokens = json.loads(tokens_raw)
                        else:
                            tokens = tokens_raw
                        
                        up_price = float(prices[0]) if len(prices) > 0 and prices[0] else None
                        down_price = float(prices[1]) if len(prices) > 1 and prices[1] else None
                        up_token = tokens[0] if len(tokens) > 0 else None
                        down_token = tokens[1] if len(tokens) > 1 else None
                        
        except Exception as e:
            print(f"Warning: Could not fetch market details: {e}")
        
        return PolymarketMarket(
            slug=slug,
            start_time=start_ts,
            end_time=end_ts,
            up_token=up_token,
            down_token=down_token,
            up_price=up_price,
            down_price=down_price
        )
    
    def fetch_ptb(self, start_time: int, end_time: int) -> Optional[float]:
        """Fetch the Price to Beat (PTB) from Polymarket's crypto-price API."""
        try:
            start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
            
            start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            params = {
                "symbol": "BTC",
                "eventStartTime": start_str,
                "variant": "fifteen",
                "endDate": end_str
            }
            
            response = requests.get(
                CRYPTO_PRICE_API,
                params=params,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                    "Referer": "https://polymarket.com/"
                },
                proxies=self.proxies,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                ptb = data.get("openPrice")
                if ptb:
                    return float(ptb)
            
        except Exception as e:
            print(f"Error fetching PTB: {e}")
        
        return None
    
    def get_current_market(self, fetch_ptb: bool = True) -> Optional[PolymarketMarket]:
        """Get the current active 15-minute BTC market."""
        slug = self.get_market_slug()
        market = self.fetch_market_by_slug(slug)
        
        if market and market.is_active and fetch_ptb:
            ptb = self.fetch_ptb(market.start_time, market.end_time)
            market.ptb = ptb
        
        return market
    
    async def wait_for_market_start(self) -> PolymarketMarket:
        """Wait until the next market starts, then return it with PTB."""
        seconds = self.seconds_until_next_market()
        
        if seconds > 0:
            await asyncio.sleep(seconds)
        
        # Wait for PTB to be available
        await asyncio.sleep(3)
        
        market = self.get_current_market(fetch_ptb=True)
        
        # Retry PTB fetch
        retries = 0
        while market and not market.ptb and retries < 5:
            await asyncio.sleep(2)
            market.ptb = self.fetch_ptb(market.start_time, market.end_time)
            retries += 1
        
        return market


def format_countdown(seconds: int) -> str:
    """Format seconds as MM:SS countdown"""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def format_timestamp(ts: int) -> str:
    """Format Unix timestamp as readable string"""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
