"""
Derivatives Data Feed for Atlas v4.0
Fetches funding rates, open interest, liquidation data, and basis
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class FundingRate:
    """Funding rate data"""
    exchange: str
    symbol: str
    rate: float
    timestamp: datetime
    next_funding_time: Optional[datetime] = None


@dataclass
class OpenInterest:
    """Open interest data"""
    exchange: str
    symbol: str
    value: float
    currency: str
    timestamp: datetime


@dataclass
class LiquidationLevel:
    """Liquidation level data"""
    price: float
    amount: float
    side: str  # "long" or "short"


class DerivativesFeed:
    """
    Aggregates derivatives data from multiple exchanges.
    
    Key Metrics:
    - Funding Rate: Positive = longs pay shorts (bearish sentiment)
                   Negative = shorts pay longs (bullish sentiment)
    - Open Interest: Increasing = new positions, Decreasing = closing
    - Liquidation Levels: Where cascades might occur
    - Basis: Futures price vs spot
    """
    
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
    
    async def get_all_funding_rates(self) -> List[FundingRate]:
        """Get funding rates from all major exchanges"""
        tasks = [
            self._get_binance_funding(),
            self._get_bybit_funding(),
            self._get_dydx_funding(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        funding_rates = []
        for r in results:
            if isinstance(r, FundingRate):
                funding_rates.append(r)
        
        return funding_rates
    
    async def _get_binance_funding(self) -> Optional[FundingRate]:
        """Get Binance futures funding rate"""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {"symbol": "BTCUSDT", "limit": 1}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                if data:
                    return FundingRate(
                        exchange="binance",
                        symbol="BTCUSDT",
                        rate=float(data[0]["fundingRate"]),
                        timestamp=datetime.fromtimestamp(data[0]["fundingTime"] / 1000)
                    )
        except Exception as e:
            print(f"Binance funding error: {e}")
        return None
    
    async def _get_bybit_funding(self) -> Optional[FundingRate]:
        """Get Bybit funding rate"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "linear", "symbol": "BTCUSDT"}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("result", {}).get("list"):
                    item = data["result"]["list"][0]
                    return FundingRate(
                        exchange="bybit",
                        symbol="BTCUSDT",
                        rate=float(item.get("fundingRate", 0)),
                        timestamp=datetime.now()
                    )
        except Exception as e:
            print(f"Bybit funding error: {e}")
        return None
    
    async def _get_dydx_funding(self) -> Optional[FundingRate]:
        """Get dYdX funding rate (approximate from API)"""
        try:
            # dYdX v4 API
            url = "https://api.dydx.exchange/v3/markets/BTC-USD"
            
            async with self._session.get(url) as resp:
                data = await resp.json()
                if data.get("market"):
                    # dYdX uses different funding mechanism
                    next_funding = data["market"].get("nextFundingRate", "0")
                    return FundingRate(
                        exchange="dydx",
                        symbol="BTC-USD",
                        rate=float(next_funding),
                        timestamp=datetime.now()
                    )
        except Exception as e:
            print(f"dYdX funding error: {e}")
        return None
    
    async def get_aggregated_funding(self) -> Dict[str, Any]:
        """
        Get aggregated funding rate across exchanges.
        
        Returns weighted average funding rate and sentiment signal.
        """
        funding_rates = await self.get_all_funding_rates()
        
        if not funding_rates:
            return {
                "average_rate": 0.0,
                "sentiment": "neutral",
                "signal": "neutral",
                "exchanges": []
            }
        
        # Calculate average
        rates = [f.rate for f in funding_rates]
        avg_rate = sum(rates) / len(rates)
        
        # Determine sentiment
        if avg_rate > 0.0005:  # > 0.05%
            sentiment = "very_bullish"  # Shorts paying longs heavily
            signal = "up"
        elif avg_rate > 0.0001:
            sentiment = "bullish"
            signal = "up"
        elif avg_rate < -0.0005:
            sentiment = "very_bearish"  # Longs paying shorts heavily
            signal = "down"
        elif avg_rate < -0.0001:
            sentiment = "bearish"
            signal = "down"
        else:
            sentiment = "neutral"
            signal = "neutral"
        
        return {
            "average_rate": avg_rate,
            "rate_percent": avg_rate * 100,
            "sentiment": sentiment,
            "signal": signal,
            "exchanges": [
                {"name": f.exchange, "rate": f.rate, "rate_percent": f.rate * 100}
                for f in funding_rates
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_open_interest_data(self) -> Dict[str, Any]:
        """
        Get open interest from multiple exchanges.
        
        High OI + Rising = Strong trend
        High OI + Falling = Trend exhaustion
        Low OI + Rising = New trend forming
        Low OI + Falling = No interest
        """
        oi_data = []
        
        # Binance
        try:
            url = "https://fapi.binance.com/fapi/v1/openInterest"
            params = {"symbol": "BTCUSDT"}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                oi_data.append({
                    "exchange": "binance",
                    "open_interest": float(data.get("openInterest", 0)),
                    "symbol": "BTCUSDT"
                })
        except:
            pass
        
        # Bybit
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "linear", "symbol": "BTCUSDT"}
            
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("result", {}).get("list"):
                    item = data["result"]["list"][0]
                    oi_data.append({
                        "exchange": "bybit",
                        "open_interest": float(item.get("openInterest", 0)),
                        "symbol": "BTCUSDT"
                    })
        except:
            pass
        
        total_oi = sum(d["open_interest"] for d in oi_data)
        
        return {
            "total_open_interest": total_oi,
            "exchanges": oi_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_liquidation_levels(self, current_price: float) -> Dict[str, Any]:
        """
        Estimate liquidation levels based on price.
        
        This is approximate - actual liquidations depend on leverage and position sizes.
        """
        # Estimate liquidation clusters at common leverage levels
        levels = []
        
        # Long liquidations (below current price)
        for leverage in [10, 20, 25, 50, 100]:
            liq_price = current_price * (1 - 1/leverage * 0.9)  # 90% of liquidation distance
            levels.append({
                "price": liq_price,
                "distance_percent": ((liq_price - current_price) / current_price) * 100,
                "side": "long",
                "estimated_leverage": leverage
            })
        
        # Short liquidations (above current price)
        for leverage in [10, 20, 25, 50, 100]:
            liq_price = current_price * (1 + 1/leverage * 0.9)
            levels.append({
                "price": liq_price,
                "distance_percent": ((liq_price - current_price) / current_price) * 100,
                "side": "short",
                "estimated_leverage": leverage
            })
        
        # Sort by distance
        levels.sort(key=lambda x: abs(x["distance_percent"]))
        
        return {
            "current_price": current_price,
            "nearest_long_liq": next((l for l in levels if l["side"] == "long"), None),
            "nearest_short_liq": next((l for l in levels if l["side"] == "short"), None),
            "all_levels": levels[:10],  # Top 10 nearest
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_futures_basis(self, spot_price: float) -> Dict[str, Any]:
        """
        Calculate futures basis (contango/backwardation).
        
        Contango (futures > spot) = Bullish sentiment
        Backwardation (futures < spot) = Bearish sentiment
        """
        basis_data = []
        
        try:
            # Get futures prices for different expiries
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            
            async with self._session.get(url) as resp:
                data = await resp.json()
                
                for item in data:
                    symbol = item.get("symbol", "")
                    if symbol == "BTCUSDT":
                        futures_price = float(item.get("price", 0))
                        basis = futures_price - spot_price
                        basis_percent = (basis / spot_price) * 100
                        
                        basis_data.append({
                            "symbol": symbol,
                            "futures_price": futures_price,
                            "spot_price": spot_price,
                            "basis": basis,
                            "basis_percent": basis_percent,
                            "type": "perpetual"
                        })
        except:
            pass
        
        # Determine market structure
        if basis_data:
            perpetual_basis = basis_data[0]["basis_percent"]
            
            if perpetual_basis > 0.1:
                structure = "contango"
                signal = "bullish"
            elif perpetual_basis < -0.1:
                structure = "backwardation"
                signal = "bearish"
            else:
                structure = "neutral"
                signal = "neutral"
        else:
            structure = "unknown"
            signal = "neutral"
        
        return {
            "spot_price": spot_price,
            "basis_data": basis_data,
            "market_structure": structure,
            "signal": signal,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_full_derivatives_context(self, spot_price: float) -> Dict[str, Any]:
        """Get complete derivatives market context"""
        funding = await self.get_aggregated_funding()
        oi = await self.get_open_interest_data()
        liq_levels = await self.get_liquidation_levels(spot_price)
        basis = await self.get_futures_basis(spot_price)
        
        # Aggregate signals
        signals = []
        
        if funding["signal"] == "up":
            signals.append({"source": "funding", "direction": "up", "strength": abs(funding["average_rate"]) * 10000})
        elif funding["signal"] == "down":
            signals.append({"source": "funding", "direction": "down", "strength": abs(funding["average_rate"]) * 10000})
        
        if basis["signal"] == "bullish":
            signals.append({"source": "basis", "direction": "up", "strength": 1})
        elif basis["signal"] == "bearish":
            signals.append({"source": "basis", "direction": "down", "strength": 1})
        
        # Combined signal
        up_strength = sum(s["strength"] for s in signals if s["direction"] == "up")
        down_strength = sum(s["strength"] for s in signals if s["direction"] == "down")
        
        if up_strength > down_strength:
            combined_signal = "up"
        elif down_strength > up_strength:
            combined_signal = "down"
        else:
            combined_signal = "neutral"
        
        return {
            "funding": funding,
            "open_interest": oi,
            "liquidation_levels": liq_levels,
            "basis": basis,
            "signals": signals,
            "combined_signal": combined_signal,
            "timestamp": datetime.now().isoformat()
        }


async def test_derivatives():
    """Test derivatives feed"""
    async with DerivativesFeed() as feed:
        funding = await feed.get_aggregated_funding()
        print(f"Average Funding Rate: {funding['rate_percent']:.4f}%")
        print(f"Sentiment: {funding['sentiment']}")
        print(f"Signal: {funding['signal']}")
        
        oi = await feed.get_open_interest_data()
        print(f"\nTotal Open Interest: ${oi['total_open_interest']:,.0f}")
        
        context = await feed.get_full_derivatives_context(70000.0)
        print(f"\nCombined Signal: {context['combined_signal']}")


if __name__ == "__main__":
    asyncio.run(test_derivatives())
