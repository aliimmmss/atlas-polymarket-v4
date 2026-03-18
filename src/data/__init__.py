"""
Atlas v4.0 Data Pipeline
Enhanced multi-source data aggregation with FREE APIs
"""

from .binance_feed import BinanceClient, BitcoinPriceMonitor
from .derivatives_feed import DerivativesFeed
from .onchain_feed import OnChainFeed
from .price_aggregator import PriceAggregator
from .sentiment_feed import SentimentFeed
from .market_sync import PolymarketSync, PolymarketMarket, format_countdown, format_timestamp
from .chainlink_streams import ChainlinkClient, FreePriceFallback

__all__ = [
    'BinanceClient',
    'BitcoinPriceMonitor',
    'DerivativesFeed',
    'OnChainFeed',
    'PriceAggregator',
    'SentimentFeed',
    'PolymarketSync',
    'PolymarketMarket',
    'format_countdown',
    'format_timestamp',
    'ChainlinkClient',
    'FreePriceFallback'
]
