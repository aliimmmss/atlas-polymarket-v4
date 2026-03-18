"""
On-Chain Data Feed for Atlas v4.0
FREE APIs only - No paid subscriptions required

Data Sources:
- Mempool.space (FREE) - Transaction data, mempool stats
- Blockchain.com (FREE) - Network stats, hashrate, difficulty
- Alternative.me (FREE) - Fear & Greed Index
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WhaleAlert:
    """Large transaction alert"""
    tx_hash: str
    amount_btc: float
    amount_usd: float
    timestamp: datetime
    transaction_type: str


class MempoolSpaceClient:
    """
    Mempool.space API Client - FREE, no authentication required.
    """
    
    BASE_URL = "https://mempool.space/api"
    
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
    
    async def get_mempool_stats(self) -> Dict[str, Any]:
        """Get mempool statistics"""
        try:
            url = f"{self.BASE_URL}/mempool"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "count": data.get("count", 0),
                        "vsize": data.get("vsize", 0),
                        "total_fee": data.get("total_fee", 0) / 100000000,
                        "fee_histogram": data.get("fee_histogram", []),
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Mempool stats request failed: {e}")
        return {}
    
    async def get_recent_transactions(self, limit: int = 50) -> List[Dict]:
        """Get recent transactions from mempool"""
        try:
            url = f"{self.BASE_URL}/mempool/recent"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data[:limit]
        except Exception as e:
            logger.error(f"Recent transactions request failed: {e}")
        return []
    
    async def get_large_transactions(self, min_btc: float = 50) -> List[Dict]:
        """Get large transactions from mempool"""
        txs = await self.get_recent_transactions(100)
        large_txs = []
        
        for tx in txs:
            value_btc = tx.get("value", 0) / 100000000
            if value_btc >= min_btc:
                large_txs.append({
                    "tx_id": tx.get("txid", ""),
                    "value_btc": value_btc,
                    "fee": tx.get("fee", 0) / 100000000,
                    "timestamp": datetime.now().isoformat()
                })
        
        return large_txs
    
    async def get_fee_estimates(self) -> Dict[str, int]:
        """Get fee estimates for different confirmation times"""
        try:
            url = f"{self.BASE_URL}/v1/fees/recommended"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Fee estimates request failed: {e}")
        return {"fastestFee": 0, "halfHourFee": 0, "hourFee": 0}


class BlockchainInfoClient:
    """
    Blockchain.com API Client - FREE, no authentication required.
    """
    
    BASE_URL = "https://api.blockchain.info"
    
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
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        try:
            url = f"{self.BASE_URL}/stats"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "btc_mined": data.get("n_btc_mined", 0),
                        "blocks_mined": data.get("n_blocks_mined", 0),
                        "tx_count": data.get("n_tx", 0),
                        "total_btc": data.get("totalbc", 0),
                        "difficulty": data.get("difficulty", 0),
                        "hashrate": data.get("hash_rate", 0) / 1e18,  # to EH/s
                        "block_height": data.get("n_blocks_total", 0),
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Blockchain.info request failed: {e}")
        return {}


class FearGreedIndexClient:
    """
    Alternative.me Fear & Greed Index Client - FREE.
    """
    
    BASE_URL = "https://api.alternative.me/fng"
    
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
    
    async def get_index(self, days: int = 7) -> Dict[str, Any]:
        """Get Fear & Greed Index data"""
        try:
            url = f"{self.BASE_URL}?limit={days}"
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if data.get("data"):
                        current = int(data["data"][0]["value"])
                        classification = data["data"][0]["value_classification"]
                        
                        if len(data["data"]) >= 7:
                            week_ago = int(data["data"][-1]["value"])
                            trend = "rising" if current > week_ago else "falling" if current < week_ago else "stable"
                            change = current - week_ago
                        else:
                            trend = "stable"
                            change = 0
                        
                        if current <= 25:
                            signal = "strong_buy"
                        elif current <= 40:
                            signal = "buy"
                        elif current <= 60:
                            signal = "neutral"
                        elif current <= 75:
                            signal = "sell"
                        else:
                            signal = "strong_sell"
                        
                        return {
                            "value": current,
                            "classification": classification,
                            "trend": trend,
                            "weekly_change": change,
                            "signal": signal,
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            logger.error(f"Fear & Greed API error: {e}")
        
        return {
            "value": 50,
            "classification": "Neutral",
            "trend": "stable",
            "weekly_change": 0,
            "signal": "neutral",
            "timestamp": datetime.now().isoformat()
        }


class OnChainFeed:
    """
    FREE On-Chain Data Feed for Bitcoin prediction.
    
    Uses only free APIs:
    - Mempool.space: Transaction data, mempool stats
    - Blockchain.com: Network stats, hashrate, difficulty
    - Alternative.me: Fear & Greed Index
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._mempool = MempoolSpaceClient()
        self._blockchain = BlockchainInfoClient()
        self._fear_greed = FearGreedIndexClient()
    
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
        await self._mempool.__aenter__()
        await self._blockchain.__aenter__()
        await self._fear_greed.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
        await self._mempool.__aexit__(exc_type, exc_val, exc_tb)
        await self._blockchain.__aexit__(exc_type, exc_val, exc_tb)
        await self._fear_greed.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_exchange_flows(self, current_price: float = 70000.0) -> Dict[str, Any]:
        """Estimate exchange flows from mempool data"""
        mempool = await self._mempool.get_mempool_stats()
        fees = await self._mempool.get_fee_estimates()
        
        mempool_congestion = mempool.get("count", 0) / 10000
        fee_pressure = fees.get("fastestFee", 0) / 100
        pressure_score = (mempool_congestion + fee_pressure) / 2
        
        if pressure_score > 0.7:
            signal = "bearish"
        elif pressure_score < 0.3:
            signal = "bullish"
        else:
            signal = "neutral"
        
        return {
            "netflow": 0,
            "netflow_btc": 0,
            "inflow_btc": 0,
            "outflow_btc": 0,
            "pressure_score": pressure_score,
            "signal": signal,
            "sources": [],
            "data_quality": "estimated",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_whale_alerts(self, min_btc: float = 100, current_price: float = 70000.0) -> Dict[str, Any]:
        """Get whale activity from mempool data"""
        large_txs = await self._mempool.get_large_transactions(min_btc=min_btc)
        
        total_btc = sum(tx["value_btc"] for tx in large_txs)
        
        if len(large_txs) > 10:
            activity = "very_high"
        elif len(large_txs) > 5:
            activity = "high"
        elif len(large_txs) > 2:
            activity = "normal"
        else:
            activity = "low"
        
        return {
            "alerts": large_txs[:10],
            "count": len(large_txs),
            "total_btc_moved": total_btc,
            "total_usd_moved": total_btc * current_price,
            "whale_activity": activity,
            "data_source": "mempool.space",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_miner_flows(self) -> Dict[str, Any]:
        """Estimate miner activity from network data"""
        stats = await self._blockchain.get_stats()
        hashrate = stats.get("hashrate", 0)
        
        if hashrate > 600:
            miner_status = "very_active"
            signal = "bullish"
        elif hashrate > 500:
            miner_status = "active"
            signal = "neutral"
        else:
            miner_status = "reduced"
            signal = "cautious"
        
        return {
            "miner_outflow_btc": 0,
            "miner_outflow_change": 0,
            "hashrate_eh": hashrate,
            "miner_status": miner_status,
            "signal": signal,
            "data_quality": "estimated",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index"""
        return await self._fear_greed.get_index()
    
    async def get_market_indicators(self) -> Dict[str, Any]:
        """Estimate market indicators from available data"""
        fgi = await self._fear_greed.get_index()
        
        estimated_nupl = fgi["value"] / 100
        
        if estimated_nupl < 0.25:
            nupl_zone = "capitulation"
            nupl_signal = "bullish"
        elif estimated_nupl < 0.5:
            nupl_zone = "belief"
            nupl_signal = "neutral"
        elif estimated_nupl < 0.75:
            nupl_zone = "greed"
            nupl_signal = "cautious"
        else:
            nupl_zone = "euphoria"
            nupl_signal = "bearish"
        
        estimated_sopr = 1.0 + (estimated_nupl - 0.5) * 0.1
        
        return {
            "fear_greed_index": fgi,
            "estimated_nupl": {
                "value": estimated_nupl,
                "zone": nupl_zone,
                "signal": nupl_signal
            },
            "estimated_sopr": {
                "value": estimated_sopr,
                "signal": "profit_taking" if estimated_sopr > 1.05 else "loss_taking" if estimated_sopr < 0.95 else "neutral"
            },
            "data_quality": "estimated",
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        stats = await self._blockchain.get_stats()
        mempool = await self._mempool.get_mempool_stats()
        stats["mempool"] = mempool
        return stats
    
    async def get_full_onchain_context(self, current_price: float = 70000.0) -> Dict[str, Any]:
        """Get complete on-chain market context"""
        
        # Run all queries in parallel
        flows_task = self.get_exchange_flows(current_price)
        whales_task = self.get_whale_alerts(100, current_price)
        miners_task = self.get_miner_flows()
        indicators_task = self.get_market_indicators()
        stats_task = self.get_blockchain_stats()
        
        flows, whales, miners, indicators, stats = await asyncio.gather(
            flows_task, whales_task, miners_task, indicators_task, stats_task
        )
        
        # Aggregate signals
        signals = []
        
        if flows.get("signal") == "bullish":
            signals.append({"source": "exchange_flow", "direction": "up", "strength": 1})
        elif flows.get("signal") == "bearish":
            signals.append({"source": "exchange_flow", "direction": "down", "strength": 1})
        
        if whales.get("whale_activity") == "very_high":
            signals.append({"source": "whale_activity", "direction": "volatile", "strength": 1.5})
        
        if miners.get("signal") == "bullish":
            signals.append({"source": "miner_activity", "direction": "up", "strength": 1})
        elif miners.get("signal") == "cautious":
            signals.append({"source": "miner_activity", "direction": "down", "strength": 0.5})
        
        fgi = indicators.get("fear_greed_index", {})
        if fgi.get("signal") in ["strong_buy", "buy"]:
            signals.append({"source": "fear_greed", "direction": "up", "strength": 2})
        elif fgi.get("signal") in ["strong_sell", "sell"]:
            signals.append({"source": "fear_greed", "direction": "down", "strength": 2})
        
        up_strength = sum(s["strength"] for s in signals if s["direction"] == "up")
        down_strength = sum(s["strength"] for s in signals if s["direction"] == "down")
        
        if up_strength > down_strength + 1:
            combined = "up"
        elif down_strength > up_strength + 1:
            combined = "down"
        else:
            combined = "neutral"
        
        return {
            "exchange_flows": flows,
            "whale_alerts": whales,
            "miner_flows": miners,
            "market_indicators": indicators,
            "fear_greed_index": indicators.get("fear_greed_index", {}),
            "blockchain_stats": stats,
            "signals": signals,
            "combined_signal": combined,
            "signal_strength": {
                "up": up_strength,
                "down": down_strength,
                "net": up_strength - down_strength
            },
            "data_quality": "estimated",
            "timestamp": datetime.now().isoformat()
        }


async def test_onchain():
    """Test on-chain feed"""
    async with OnChainFeed() as feed:
        print("\n" + "="*60)
        print("ON-CHAIN DATA TEST (FREE APIs)")
        print("="*60)
        
        context = await feed.get_full_onchain_context(current_price=70000.0)
        
        print(f"\n📊 Network Stats:")
        stats = context.get('blockchain_stats', {})
        print(f"   Hashrate: {stats.get('hashrate', 0):.1f} EH/s")
        print(f"   Difficulty: {stats.get('difficulty', 0):,.0f}")
        
        print(f"\n🐋 Whale Activity:")
        whales = context['whale_alerts']
        print(f"   Large TXs: {whales.get('count', 0)}")
        print(f"   Activity: {whales.get('whale_activity', 'unknown')}")
        
        print(f"\n😱 Fear & Greed Index:")
        fgi = context.get('fear_greed_index', {})
        print(f"   Value: {fgi.get('value', 50)} ({fgi.get('classification', 'N/A')})")
        print(f"   Signal: {fgi.get('signal', 'neutral')}")
        
        print(f"\n📈 Combined Signal: {context['combined_signal'].upper()}")
        print("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_onchain())
