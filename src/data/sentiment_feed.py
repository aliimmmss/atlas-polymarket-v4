"""
Sentiment & News Feed for Atlas v4.0
Aggregates sentiment from multiple sources including social media and news
"""

import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class NewsItem:
    """News item with sentiment"""
    title: str
    source: str
    url: str
    timestamp: datetime
    sentiment: str  # "bullish", "bearish", "neutral"
    relevance: float  # 0-1


class SentimentFeed:
    """
    Aggregates sentiment from multiple sources.
    
    Sources:
    - Fear & Greed Index
    - Reddit sentiment (r/Bitcoin)
    - News headlines
    - Social trends
    
    Outputs:
    - Sentiment score (-1 to 1)
    - Trending topics
    - Breaking news alerts
    """
    
    # Bullish keywords
    BULLISH_KEYWORDS = [
        "bullish", "moon", "rally", "surge", "breakout", "all-time high", "ath",
        "adoption", "institutional", "buy", "accumulate", "support", "bounce",
        "positive", "growth", "upgrade", "approval", "etf", "spot etf"
    ]
    
    # Bearish keywords
    BEARISH_KEYWORDS = [
        "bearish", "crash", "dump", "sell-off", "breakdown", "resistance",
        "rejection", "fraud", "hack", "ban", "regulation", "sec", "lawsuit",
        "negative", "concern", "warning", "bubble", "overbought"
    ]
    
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
    
    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/?limit=7"
            
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if data.get("data"):
                        current = int(data["data"][0]["value"])
                        classification = data["data"][0]["value_classification"]
                        
                        # Get trend
                        if len(data["data"]) >= 7:
                            week_ago = int(data["data"][-1]["value"])
                            trend = "rising" if current > week_ago else "falling" if current < week_ago else "stable"
                            change = current - week_ago
                        else:
                            trend = "stable"
                            change = 0
                        
                        return {
                            "value": current,
                            "classification": classification,
                            "trend": trend,
                            "weekly_change": change,
                            "signal": self._fgi_to_signal(current),
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"FGI error: {e}")
        
        return {
            "value": 50,
            "classification": "Neutral",
            "trend": "stable",
            "weekly_change": 0,
            "signal": "neutral",
            "timestamp": datetime.now().isoformat()
        }
    
    def _fgi_to_signal(self, value: int) -> str:
        """Convert FGI to trading signal (contrarian)"""
        if value <= 20:
            return "strong_buy"  # Extreme fear
        elif value <= 40:
            return "buy"
        elif value <= 60:
            return "neutral"
        elif value <= 80:
            return "sell"
        else:
            return "strong_sell"  # Extreme greed
    
    async def get_reddit_sentiment(self) -> Dict[str, Any]:
        """
        Get sentiment from r/Bitcoin.
        
        Note: Reddit API requires OAuth, using public RSS/JSON fallback.
        """
        posts = []
        sentiment_scores = []
        
        try:
            # Try to get hot posts via public JSON
            url = "https://www.reddit.com/r/Bitcoin/hot.json?limit=25"
            headers = {"User-Agent": "Atlas Sentiment Bot 1.0"}
            
            async with self._session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for post in data.get("data", {}).get("children", []):
                        title = post["data"].get("title", "").lower()
                        score = post["data"].get("score", 0)
                        
                        sentiment = self._analyze_text_sentiment(title)
                        sentiment_scores.append(sentiment["score"])
                        
                        posts.append({
                            "title": post["data"].get("title", ""),
                            "score": score,
                            "sentiment": sentiment["label"],
                            "url": f"https://reddit.com{post['data'].get('permalink', '')}"
                        })
        except Exception as e:
            print(f"Reddit error: {e}")
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            "average_sentiment": avg_sentiment,
            "sentiment_label": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
            "post_count": len(posts),
            "top_posts": posts[:5],
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_crypto_news(self) -> Dict[str, Any]:
        """Get latest crypto news headlines"""
        news_items = []
        
        try:
            # CryptoCompare news API (free tier)
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&limit=15"
            
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for item in data.get("Data", []):
                        title = item.get("title", "")
                        sentiment = self._analyze_text_sentiment(title)
                        
                        news_items.append({
                            "title": title,
                            "source": item.get("source", "unknown"),
                            "url": item.get("url", ""),
                            "sentiment": sentiment["label"],
                            "sentiment_score": sentiment["score"],
                            "timestamp": datetime.now().isoformat()
                        })
        except Exception as e:
            print(f"News error: {e}")
        
        # Analyze overall news sentiment
        if news_items:
            scores = [n["sentiment_score"] for n in news_items]
            avg_sentiment = sum(scores) / len(scores)
            
            bullish_count = sum(1 for n in news_items if n["sentiment"] == "bullish")
            bearish_count = sum(1 for n in news_items if n["sentiment"] == "bearish")
        else:
            avg_sentiment = 0
            bullish_count = 0
            bearish_count = 0
        
        return {
            "average_sentiment": avg_sentiment,
            "sentiment_label": "bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": len(news_items) - bullish_count - bearish_count,
            "news": news_items[:10],
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text for bullish/bearish sentiment"""
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        
        if total == 0:
            return {"label": "neutral", "score": 0}
        
        score = (bullish_count - bearish_count) / total
        
        if score > 0.2:
            label = "bullish"
        elif score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        
        return {"label": label, "score": score}
    
    async def get_trending_topics(self) -> List[str]:
        """Get currently trending crypto topics"""
        trending = []
        
        try:
            # Get trending from CryptoCompare
            url = "https://min-api.cryptocompare.com/data/news/categories"
            
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    trending = [item.get("category", "") for item in data[:10]]
        except:
            pass
        
        return trending
    
    async def get_aggregate_sentiment(self) -> Dict[str, Any]:
        """
        Get aggregated sentiment score from all sources.
        
        Returns:
        - Combined sentiment score (-1 to 1)
        - Signal strength
        - Source breakdown
        """
        # Fetch all sentiment sources
        fgi = await self.get_fear_greed_index()
        reddit = await self.get_reddit_sentiment()
        news = await self.get_crypto_news()
        
        # Weight factors
        weights = {
            "fear_greed": 0.4,  # Contrarian indicator
            "reddit": 0.25,
            "news": 0.35
        }
        
        # Convert to -1 to 1 scale
        fgi_score = (50 - fgi["value"]) / 50  # Contrarian: low FGI = bullish
        reddit_score = reddit["average_sentiment"]
        news_score = news["average_sentiment"]
        
        # Weighted average
        combined_score = (
            fgi_score * weights["fear_greed"] +
            reddit_score * weights["reddit"] +
            news_score * weights["news"]
        )
        
        # Determine signal
        if combined_score > 0.3:
            signal = "strong_bullish"
        elif combined_score > 0.1:
            signal = "bullish"
        elif combined_score < -0.3:
            signal = "strong_bearish"
        elif combined_score < -0.1:
            signal = "bearish"
        else:
            signal = "neutral"
        
        return {
            "combined_score": combined_score,
            "signal": signal,
            "confidence": abs(combined_score),
            "sources": {
                "fear_greed": {
                    "value": fgi["value"],
                    "classification": fgi["classification"],
                    "score": fgi_score,
                    "weight": weights["fear_greed"]
                },
                "reddit": {
                    "sentiment": reddit["sentiment_label"],
                    "score": reddit_score,
                    "weight": weights["reddit"]
                },
                "news": {
                    "sentiment": news["sentiment_label"],
                    "score": news_score,
                    "bullish_count": news["bullish_count"],
                    "bearish_count": news["bearish_count"],
                    "weight": weights["news"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }


async def test_sentiment():
    """Test sentiment feed"""
    async with SentimentFeed() as feed:
        sentiment = await feed.get_aggregate_sentiment()
        
        print(f"Combined Sentiment: {sentiment['combined_score']:.3f}")
        print(f"Signal: {sentiment['signal']}")
        print(f"Confidence: {sentiment['confidence']:.1%}")
        
        print("\nSource Breakdown:")
        for source, data in sentiment["sources"].items():
            print(f"  {source}: score={data['score']:.3f}, weight={data['weight']}")


if __name__ == "__main__":
    asyncio.run(test_sentiment())
