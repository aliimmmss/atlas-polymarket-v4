"""
Chainlink Data Streams Client for Atlas v4.0
Real-time price data from Chainlink (same as Polymarket resolution source)

Supports:
- Real-time BTC/USD price via REST API
- Liquidity-Weighted Bid/Ask (LWBA) prices
- High-frequency, low-latency data

Documentation: https://docs.chain.link/data-streams
"""

import os
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Chainlink Data Streams Feed IDs
# See: https://data.chain.link/streams
FEED_IDS = {
    "BTC_USD": "0x0000000000000000000000000000000000000000000000000000000000000044",
    "ETH_USD": "0x0000000000000000000000000000000000000000000000000000000000000002",
}

# Data Streams REST API endpoint
REST_API_URL = "https://api.chain.link/datastreams/v1/reports"


@dataclass
class ChainlinkPriceReport:
    """Chainlink price report structure"""
    feed_id: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    timestamp: datetime
    observations: int
    source: str = "chainlink"


class ChainlinkClient:
    """
    Chainlink Data Streams client for real-time price data.
    
    Features:
    - Low-latency price feeds (< 1 second)
    - Liquidity-Weighted Bid/Ask (LWBA) prices
    - High availability with multiple data providers
    
    Requirements:
    - Chainlink Data Streams API key and secret
    - Get access at: https://chain.link/data-streams
    """
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        secret: Optional[str] = None,
        feed_id: str = "BTC_USD"
    ):
        # Try to get credentials from parameters, environment, or config
        self.access_key = access_key or os.getenv("CHAINLINK_ACCESS_KEY")
        self.secret = secret or os.getenv("CHAINLINK_SECRET")
        
        # If still not found, try config module
        if not self.access_key or not self.secret:
            try:
                import sys
                config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if config_dir not in sys.path:
                    sys.path.insert(0, config_dir)
                from config import config as app_config
                if not self.access_key:
                    self.access_key = app_config.chainlink.access_key
                if not self.secret:
                    self.secret = app_config.chainlink.secret
            except Exception as e:
                logger.debug(f"Could not load config: {e}")
        
        self.feed_id = FEED_IDS.get(feed_id, FEED_IDS["BTC_USD"])
        self.feed_name = feed_id
        
        self._available = bool(self.access_key and self.secret)
        self._session: Optional[aiohttp.ClientSession] = None
        
        if self._available:
            logger.info(f"Chainlink Data Streams client initialized for {feed_id}")
        else:
            logger.warning("Chainlink credentials not found, will use fallback sources")
    
    def _create_session(self) -> aiohttp.ClientSession:
        """Create session with aiodns disabled for Python 3.13 Windows compatibility"""
        import sys
        if sys.platform == 'win32':
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
    
    def is_available(self) -> bool:
        """Check if Chainlink client is available"""
        return self._available
    
    async def get_latest_report(self) -> Optional[ChainlinkPriceReport]:
        """Get the latest price report from Chainlink Data Streams."""
        if not self._available:
            return None
        
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        try:
            url = f"{REST_API_URL}/{self.feed_id}"
            headers = {
                "Authorization": f"Bearer {self.access_key}",
                "Content-Type": "application/json"
            }
            
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_report(data)
                elif resp.status == 401:
                    logger.error("Chainlink API authentication failed")
                    return None
                else:
                    logger.warning(f"Chainlink API error: {resp.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Chainlink request failed: {e}")
            return None
    
    def _parse_report(self, data: Dict) -> Optional[ChainlinkPriceReport]:
        """Parse Chainlink report response"""
        try:
            report_data = data.get("report", data)
            
            # Price is typically in 8 decimals
            price_raw = report_data.get("price", report_data.get("midPrice", 0))
            price = float(price_raw) / 1e8 if price_raw else 0
            
            # LWBA prices
            bid_raw = report_data.get("bid", report_data.get("bidPrice"))
            ask_raw = report_data.get("ask", report_data.get("askPrice"))
            
            bid = float(bid_raw) / 1e8 if bid_raw else None
            ask = float(ask_raw) / 1e8 if ask_raw else None
            
            # Timestamp
            timestamp_raw = report_data.get("timestamp", 0)
            timestamp = datetime.fromtimestamp(timestamp_raw / 1000) if timestamp_raw else datetime.now()
            
            observations = report_data.get("numObservations", 1)
            
            return ChainlinkPriceReport(
                feed_id=self.feed_id,
                price=price,
                bid=bid,
                ask=ask,
                timestamp=timestamp,
                observations=observations
            )
            
        except Exception as e:
            logger.error(f"Error parsing Chainlink report: {e}")
            return None
    
    async def get_btc_price(self) -> Optional[float]:
        """Get current BTC/USD price from Chainlink"""
        report = await self.get_latest_report()
        return report.price if report else None
    
    async def get_price_with_context(self) -> Dict[str, Any]:
        """Get BTC price with full market context"""
        report = await self.get_latest_report()
        
        if not report:
            return {
                "price": None,
                "source": "chainlink",
                "available": False
            }
        
        spread = None
        spread_percent = None
        
        if report.bid and report.ask:
            spread = report.ask - report.bid
            spread_percent = (spread / report.price) * 100 if report.price else 0
        
        return {
            "price": report.price,
            "bid": report.bid,
            "ask": report.ask,
            "spread": spread,
            "spread_percent": spread_percent,
            "timestamp": report.timestamp.isoformat(),
            "observations": report.observations,
            "source": "chainlink",
            "feed_id": self.feed_id,
            "available": True
        }


class FreePriceFallback:
    """
    Free price data fallback when Chainlink is not available.
    Uses CoinGecko and Binance public APIs.
    """
    
    BINANCE_API = "https://api.binance.com/api/v3"
    COINGECKO_API = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _create_session(self) -> aiohttp.ClientSession:
        """Create session with aiodns disabled for Python 3.13 Windows compatibility"""
        import sys
        if sys.platform == 'win32':
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
    
    async def get_btc_price_binance(self) -> Optional[float]:
        """Get BTC price from Binance (free, no auth)"""
        try:
            url = f"{self.BINANCE_API}/ticker/price?symbol=BTCUSDT"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("price", 0))
        except Exception as e:
            logger.error(f"Binance price fetch failed: {e}")
        return None
    
    async def get_btc_price_coingecko(self) -> Optional[float]:
        """Get BTC price from CoinGecko (free)"""
        try:
            url = f"{self.COINGECKO_API}/simple/price?ids=bitcoin&vs_currencies=usd"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("bitcoin", {}).get("usd", 0)
        except Exception as e:
            logger.error(f"CoinGecko price fetch failed: {e}")
        return None
    
    async def get_best_price(self) -> Optional[float]:
        """Get price from best available source"""
        price = await self.get_btc_price_binance()
        if price:
            return price
        return await self.get_btc_price_coingecko()


async def test_chainlink():
    """Test Chainlink Data Streams client"""
    print("\n" + "="*60)
    print("CHAINLINK DATA STREAMS TEST")
    print("="*60)
    
    async with ChainlinkClient() as client:
        if client.is_available():
            print("✓ Chainlink client initialized")
            price_data = await client.get_price_with_context()
            
            if price_data.get("available"):
                print(f"\n📊 BTC/USD Price: ${price_data['price']:,.2f}")
                if price_data.get("bid"):
                    print(f"   Bid: ${price_data['bid']:,.2f}")
                if price_data.get("ask"):
                    print(f"   Ask: ${price_data['ask']:,.2f}")
            else:
                print("⚠ Failed to get price data")
        else:
            print("⚠ Chainlink credentials not configured")
            print("\nTesting free fallback...")
            async with FreePriceFallback() as fallback:
                price = await fallback.get_best_price()
                if price:
                    print(f"✓ Fallback price: ${price:,.2f}")
    
    print("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_chainlink())
