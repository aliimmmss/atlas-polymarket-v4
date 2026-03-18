"""
Binance Real-Time Price Feed
Free API - No authentication required for public data
Enhanced for Atlas v4.0 with multi-timeframe support
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Candle:
    """Represents a price candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_binance(cls, data: List) -> "Candle":
        """Create Candle from Binance API response"""
        return cls(
            timestamp=datetime.fromtimestamp(data[0] / 1000),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5])
        )


@dataclass
class Ticker:
    """Real-time ticker data"""
    symbol: str
    price: float
    price_change: float
    price_change_percent: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime


class BinanceClient:
    """
    Binance API client for real-time Bitcoin data.
    Uses free public endpoints - no API key required.
    """
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _create_session(self) -> aiohttp.ClientSession:
        """Create session with aiodns disabled for Python 3.13 Windows compatibility"""
        import sys
        if sys.platform == 'win32':
            # Disable aiodns which causes issues on Windows + Python 3.13
            connector = aiohttp.TCPConnector(
                force_close=True,
                enable_cleanup_closed=True,
                use_dns_cache=False
            )
            return aiohttp.ClientSession(connector=connector)
        return aiohttp.ClientSession()
    
    async def __aenter__(self):
        self._session = self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = self._create_session()
        return self._session
    
    async def get_current_price(self) -> float:
        """Get current Bitcoin price"""
        url = f"{self.BASE_URL}/api/v3/ticker/price"
        params = {"symbol": self.symbol}
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return float(data["price"])
    
    async def get_ticker(self) -> Ticker:
        """Get 24-hour ticker data"""
        url = f"{self.BASE_URL}/api/v3/ticker/24hr"
        params = {"symbol": self.symbol}
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return Ticker(
                symbol=data["symbol"],
                price=float(data["lastPrice"]),
                price_change=float(data["priceChange"]),
                price_change_percent=float(data["priceChangePercent"]),
                high_24h=float(data["highPrice"]),
                low_24h=float(data["lowPrice"]),
                volume_24h=float(data["volume"]),
                timestamp=datetime.now()
            )
    
    async def get_klines(
        self, 
        interval: str = "1m", 
        limit: int = 100
    ) -> List[Candle]:
        """
        Get candlestick data.
        
        Intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return [Candle.from_binance(c) for c in data]
    
    async def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        url = f"{self.BASE_URL}/api/v3/trades"
        params = {
            "symbol": self.symbol,
            "limit": limit
        }
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_order_book(self, limit: int = 20) -> Dict:
        """Get order book depth"""
        url = f"{self.BASE_URL}/api/v3/depth"
        params = {
            "symbol": self.symbol,
            "limit": limit
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return {
                "bids": [(float(b[0]), float(b[1])) for b in data.get("bids", [])],
                "asks": [(float(a[0]), float(a[1])) for a in data.get("asks", [])]
            }
    
    async def get_funding_rate(self) -> Dict:
        """Get current funding rate from Binance Futures"""
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": self.symbol, "limit": 1}
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                if data:
                    return {
                        "funding_rate": float(data[0]["fundingRate"]),
                        "timestamp": datetime.fromtimestamp(data[0]["fundingTime"] / 1000)
                    }
        except:
            pass
        return {"funding_rate": 0.0, "timestamp": datetime.now()}
    
    async def get_open_interest(self) -> Dict:
        """Get open interest from Binance Futures"""
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {"symbol": self.symbol}
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                return {
                    "open_interest": float(data.get("openInterest", 0)),
                    "symbol": data.get("symbol", self.symbol),
                    "timestamp": datetime.now()
                }
        except:
            pass
        return {"open_interest": 0, "timestamp": datetime.now()}


class BitcoinPriceMonitor:
    """
    Enhanced price monitoring for Atlas v4.0.
    Multi-timeframe analysis with derivatives data.
    """
    
    def __init__(self):
        self.client = BinanceClient("BTCUSDT")
        self._resolution_source = "chainlink"
        self._price_source = "binance"
    
    async def get_market_context(self) -> Dict[str, Any]:
        """
        Get complete market context for analysis.
        Returns all data needed for multi-timeframe prediction.
        """
        async with self.client as c:
            # Get multi-timeframe candles
            candles_1m = await c.get_klines(interval="1m", limit=60)
            candles_5m = await c.get_klines(interval="5m", limit=48)
            candles_15m = await c.get_klines(interval="15m", limit=32)
            candles_1h = await c.get_klines(interval="1h", limit=24)
            candles_4h = await c.get_klines(interval="4h", limit=18)
            candles_1d = await c.get_klines(interval="1d", limit=7)
            
            # Get ticker
            ticker = await c.get_ticker()
            
            # Get order book
            order_book = await c.get_order_book(limit=20)
            
            # Get recent trades
            trades = await c.get_recent_trades(limit=50)
            
            # Get derivatives data
            funding = await c.get_funding_rate()
            oi = await c.get_open_interest()
        
        # Calculate price stats
        current_price = ticker.price
        prices_1m = [c.close for c in candles_1m]
        prices_5m = [c.close for c in candles_5m]
        prices_15m = [c.close for c in candles_15m]
        prices_1h = [c.close for c in candles_1h]
        prices_4h = [c.close for c in candles_4h]
        prices_1d = [c.close for c in candles_1d]
        
        # Calculate volatility for each timeframe
        volatility = {
            "1m": self._calculate_volatility(prices_1m),
            "5m": self._calculate_volatility(prices_5m),
            "15m": self._calculate_volatility(prices_15m),
            "1h": self._calculate_volatility(prices_1h),
            "4h": self._calculate_volatility(prices_4h),
        }
        
        # Calculate momentum for each timeframe
        momentum = {
            "1m": self._calculate_momentum(prices_1m),
            "5m": self._calculate_momentum(prices_5m),
            "15m": self._calculate_momentum(prices_15m),
            "1h": self._calculate_momentum(prices_1h),
            "4h": self._calculate_momentum(prices_4h),
        }
        
        # Order book imbalance
        bid_volume = sum(b[1] for b in order_book["bids"])
        ask_volume = sum(a[1] for a in order_book["asks"])
        order_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Recent trade analysis
        buy_trades = sum(1 for t in trades if not t.get("isBuyerMaker", True))
        sell_trades = len(trades) - buy_trades
        trade_ratio = buy_trades / len(trades) if trades else 0.5
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "price_change_24h": ticker.price_change_percent,
            "high_24h": ticker.high_24h,
            "low_24h": ticker.low_24h,
            "volume_24h": ticker.volume_24h,
            
            # Resolution info
            "resolution_source": self._resolution_source,
            "price_source": self._price_source,
            
            # Multi-timeframe candles
            "candles": {
                "1m": self._candles_to_dict(candles_1m),
                "5m": self._candles_to_dict(candles_5m),
                "15m": self._candles_to_dict(candles_15m),
                "1h": self._candles_to_dict(candles_1h),
                "4h": self._candles_to_dict(candles_4h),
                "1d": self._candles_to_dict(candles_1d),
            },
            
            # Price arrays for indicators
            "prices": {
                "1m": prices_1m,
                "5m": prices_5m,
                "15m": prices_15m,
                "1h": prices_1h,
                "4h": prices_4h,
                "1d": prices_1d,
            },
            
            # Calculated metrics
            "volatility": volatility,
            "momentum": momentum,
            
            # Order book
            "order_imbalance": order_imbalance,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "order_book": order_book,
            
            # Trade flow
            "buy_trade_ratio": trade_ratio,
            "recent_trade_count": len(trades),
            
            # Derivatives data
            "funding_rate": funding["funding_rate"],
            "open_interest": oi["open_interest"],
        }
    
    def _candles_to_dict(self, candles: List[Candle]) -> List[Dict]:
        """Convert candles to list of dicts"""
        return [
            {
                "time": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in candles
        ]
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                   for i in range(1, len(prices))]
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return (variance ** 0.5) * 100
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum (rate of change)"""
        if len(prices) < 10:
            return 0.0
        
        recent = prices[-5:]
        earlier = prices[-10:-5]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        return ((recent_avg - earlier_avg) / earlier_avg) * 100


async def test_connection():
    """Test Binance connection"""
    async with BinanceClient() as client:
        price = await client.get_current_price()
        print(f"Current BTC Price: ${price:,.2f}")
        
        ticker = await client.get_ticker()
        print(f"24h Change: {ticker.price_change_percent:+.2f}%")
        
        funding = await client.get_funding_rate()
        print(f"Funding Rate: {funding['funding_rate']*100:.4f}%")


if __name__ == "__main__":
    asyncio.run(test_connection())
