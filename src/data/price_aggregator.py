"""
Multi-Exchange Price Aggregator for Atlas v4.0
Aggregates BTC prices from multiple exchanges for more accurate pricing
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class ExchangePrice:
    """Price from a single exchange"""
    exchange: str
    price: float
    volume_24h: float
    timestamp: datetime
    latency_ms: int = 0


class PriceAggregator:
    """
    Aggregates BTC prices from multiple exchanges for more accurate pricing.
    
    Exchanges:
    - Binance (largest volume)
    - Coinbase (US institutional)
    - Kraken (reliable)
    - Bitstamp (oldest, regulated)
    - Bitfinex (large traders)
    
    Outputs:
    - Volume-weighted average price
    - Price deviation alerts
    - Arbitrage signals
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._prices: Dict[str, ExchangePrice] = {}
    
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
    
    async def fetch_all_prices(self) -> Dict[str, ExchangePrice]:
        """Fetch prices from all exchanges concurrently"""
        tasks = {
            "binance": self._fetch_binance(),
            "coinbase": self._fetch_coinbase(),
            "kraken": self._fetch_kraken(),
            "bitstamp": self._fetch_bitstamp(),
            "bitfinex": self._fetch_bitfinex(),
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for (exchange, _), result in zip(tasks.items(), results):
            if isinstance(result, ExchangePrice):
                self._prices[exchange] = result
        
        return self._prices
    
    async def _fetch_binance(self) -> Optional[ExchangePrice]:
        """Fetch from Binance"""
        start = datetime.now()
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {"symbol": "BTCUSDT"}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                latency = (datetime.now() - start).microseconds // 1000
                
                return ExchangePrice(
                    exchange="binance",
                    price=float(data["lastPrice"]),
                    volume_24h=float(data["volume"]),
                    timestamp=datetime.now(),
                    latency_ms=latency
                )
        except Exception as e:
            print(f"Binance fetch error: {e}")
        return None
    
    async def _fetch_coinbase(self) -> Optional[ExchangePrice]:
        """Fetch from Coinbase Pro"""
        start = datetime.now()
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
            
            async with self._session.get(url) as resp:
                data = await resp.json()
                latency = (datetime.now() - start).microseconds // 1000
                
                return ExchangePrice(
                    exchange="coinbase",
                    price=float(data["price"]),
                    volume_24h=float(data.get("volume_24h", 0)),
                    timestamp=datetime.now(),
                    latency_ms=latency
                )
        except Exception as e:
            print(f"Coinbase fetch error: {e}")
        return None
    
    async def _fetch_kraken(self) -> Optional[ExchangePrice]:
        """Fetch from Kraken"""
        start = datetime.now()
        try:
            url = "https://api.kraken.com/0/public/Ticker"
            params = {"pair": "XBTUSD"}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                latency = (datetime.now() - start).microseconds // 1000
                
                if data.get("result"):
                    pair_data = list(data["result"].values())[0]
                    return ExchangePrice(
                        exchange="kraken",
                        price=float(pair_data["c"][0]),  # Last trade close
                        volume_24h=float(pair_data["v"][1]),  # 24h volume
                        timestamp=datetime.now(),
                        latency_ms=latency
                    )
        except Exception as e:
            print(f"Kraken fetch error: {e}")
        return None
    
    async def _fetch_bitstamp(self) -> Optional[ExchangePrice]:
        """Fetch from Bitstamp"""
        start = datetime.now()
        try:
            url = "https://www.bitstamp.net/api/v2/ticker/btcusd/"
            
            async with self._session.get(url) as resp:
                data = await resp.json()
                latency = (datetime.now() - start).microseconds // 1000
                
                return ExchangePrice(
                    exchange="bitstamp",
                    price=float(data["last"]),
                    volume_24h=float(data["volume"]),
                    timestamp=datetime.now(),
                    latency_ms=latency
                )
        except Exception as e:
            print(f"Bitstamp fetch error: {e}")
        return None
    
    async def _fetch_bitfinex(self) -> Optional[ExchangePrice]:
        """Fetch from Bitfinex"""
        start = datetime.now()
        try:
            url = "https://api-pub.bitfinex.com/v2/tickers"
            params = {"symbols": "tBTCUSD"}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                latency = (datetime.now() - start).microseconds // 1000
                
                if data and len(data) > 0:
                    ticker = data[0]
                    return ExchangePrice(
                        exchange="bitfinex",
                        price=float(ticker[7]),  # Last price
                        volume_24h=float(ticker[8]),  # Volume
                        timestamp=datetime.now(),
                        latency_ms=latency
                    )
        except Exception as e:
            print(f"Bitfinex fetch error: {e}")
        return None
    
    async def get_aggregated_price(self) -> Dict[str, Any]:
        """
        Get aggregated price with volume weighting.
        
        Returns VWAP across all exchanges.
        """
        await self.fetch_all_prices()
        
        if not self._prices:
            return {
                "vwap": 0,
                "median": 0,
                "prices": {},
                "deviation": 0,
                "arbitrage_opportunity": None
            }
        
        prices = [p.price for p in self._prices.values()]
        volumes = [p.volume_24h for p in self._prices.values()]
        
        # Volume-weighted average price
        total_volume = sum(volumes)
        if total_volume > 0:
            vwap = sum(p.price * p.volume_24h for p in self._prices.values()) / total_volume
        else:
            vwap = statistics.mean(prices)
        
        # Median price
        median_price = statistics.median(prices)
        
        # Price deviation
        if len(prices) > 1:
            stdev = statistics.stdev(prices)
            deviation_percent = (stdev / median_price) * 100
        else:
            deviation_percent = 0
        
        # Arbitrage opportunities
        min_price = min(self._prices.values(), key=lambda p: p.price)
        max_price = max(self._prices.values(), key=lambda p: p.price)
        
        arb_spread = ((max_price.price - min_price.price) / min_price.price) * 100
        
        arbitrage = None
        if arb_spread > 0.1:  # > 0.1% spread
            arbitrage = {
                "buy_exchange": min_price.exchange,
                "buy_price": min_price.price,
                "sell_exchange": max_price.exchange,
                "sell_price": max_price.price,
                "spread_percent": arb_spread
            }
        
        return {
            "vwap": vwap,
            "median": median_price,
            "min": min(prices),
            "max": max(prices),
            "deviation_percent": deviation_percent,
            "arbitrage_opportunity": arbitrage,
            "prices": {
                exchange: {
                    "price": p.price,
                    "volume_24h": p.volume_24h,
                    "latency_ms": p.latency_ms
                }
                for exchange, p in self._prices.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_price_deviation(self) -> float:
        """Calculate cross-exchange price deviation"""
        if len(self._prices) < 2:
            return 0.0
        
        prices = [p.price for p in self._prices.values()]
        mean = statistics.mean(prices)
        stdev = statistics.stdev(prices)
        
        return (stdev / mean) * 100
    
    async def get_chainlink_equivalent(self) -> float:
        """
        Get a price equivalent to Chainlink's BTC/USD.
        
        Chainlink aggregates from multiple sources - we approximate
        using volume-weighted average from major exchanges.
        """
        agg = await self.get_aggregated_price()
        return agg["vwap"]


async def test_aggregator():
    """Test price aggregator"""
    async with PriceAggregator() as aggregator:
        agg = await aggregator.get_aggregated_price()
        
        print(f"VWAP: ${agg['vwap']:,.2f}")
        print(f"Median: ${agg['median']:,.2f}")
        print(f"Deviation: {agg['deviation_percent']:.4f}%")
        
        print("\nPrices by Exchange:")
        for exchange, data in agg["prices"].items():
            print(f"  {exchange}: ${data['price']:,.2f} (vol: ${data['volume_24h']:,.0f})")
        
        if agg["arbitrage_opportunity"]:
            arb = agg["arbitrage_opportunity"]
            print(f"\n⚠️ Arbitrage: Buy on {arb['buy_exchange']} @ ${arb['buy_price']:,.2f}, "
                  f"Sell on {arb['sell_exchange']} @ ${arb['sell_price']:,.2f} "
                  f"(+{arb['spread_percent']:.2f}%)")


if __name__ == "__main__":
    asyncio.run(test_aggregator())
