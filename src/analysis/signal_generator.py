"""
Signal Generator for Atlas v4.0
Generates comprehensive trading signals from all data sources
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SignalConfidence:
    """Signal confidence with breakdown"""
    overall: float  # 0-1
    technical: float
    derivatives: float
    onchain: float
    sentiment: float
    confluence: float


class SignalGenerator:
    """
    Generates trading signals from all available data.
    
    Combines:
    - Technical analysis signals
    - Derivatives signals
    - On-chain signals
    - Sentiment signals
    - Multi-timeframe confluence
    """
    
    def __init__(self):
        from .technical_indicators import TechnicalIndicators, AdvancedIndicators
        from .regime_detector import RegimeDetector
        from .multi_timeframe import MultiTimeframeAnalyzer
        
        self.tech = TechnicalIndicators()
        self.advanced = AdvancedIndicators()
        self.regime_detector = RegimeDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
    
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all trading signals from market data.
        
        Args:
            market_data: Complete market context from data feeds
        
        Returns:
            Dictionary of all signals for agent consumption
        """
        signals = {
            "timestamp": datetime.now().isoformat(),
            "current_price": market_data.get("current_price", 0),
            "price_change_24h": market_data.get("price_change_24h", 0),
        }
        
        # Get price data
        prices_5m = market_data.get("prices", {}).get("5m", [])
        prices_15m = market_data.get("prices", {}).get("15m", [])
        candles_5m = market_data.get("candles", {}).get("5m", [])
        
        # Technical signals
        if prices_5m:
            tech_signals = self._generate_technical_signals(
                prices_5m, 
                candles_5m
            )
            signals["technical"] = tech_signals
        
        # Derivatives signals
        derivatives = market_data.get("derivatives", {})
        if derivatives:
            signals["derivatives"] = self._generate_derivatives_signals(derivatives)
        
        # On-chain signals
        onchain = market_data.get("onchain", {})
        if onchain:
            signals["onchain"] = self._generate_onchain_signals(onchain)
        
        # Sentiment signals
        sentiment = market_data.get("sentiment", {})
        if sentiment:
            signals["sentiment"] = self._generate_sentiment_signals(sentiment)
        
        # Regime detection
        if candles_5m:
            regime = self._detect_regime(prices_5m, candles_5m)
            signals["regime"] = regime
        
        # Multi-timeframe analysis
        candles_by_tf = market_data.get("candles", {})
        if candles_by_tf:
            mtf = self._analyze_multi_timeframe(candles_by_tf)
            signals["multi_timeframe"] = mtf
        
        # Combined signal
        combined = self._combine_signals(signals)
        signals["combined"] = combined
        
        # Quick vote for agents
        signals["quick_vote"] = self._generate_quick_vote(signals)
        
        return signals
    
    def _generate_technical_signals(
        self,
        prices: List[float],
        candles: List[Dict]
    ) -> Dict[str, Any]:
        """Generate technical analysis signals"""
        
        signals = {}
        
        if len(prices) < 30:
            return signals
        
        # Extract OHLCV
        highs = [c.get("high", prices[-1]) for c in candles] if candles else prices
        lows = [c.get("low", prices[-1]) for c in candles] if candles else prices
        closes = prices
        volumes = [c.get("volume", 0) for c in candles] if candles else []
        
        # RSI
        rsi = self.tech.rsi(prices, 14)
        signals["rsi"] = {
            "value": rsi,
            "signal": "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        }
        
        # MACD
        macd = self.tech.macd(prices)
        signals["macd"] = macd
        
        # Bollinger Bands
        bb = self.tech.bollinger_bands(prices, 20, 2)
        signals["bollinger"] = bb
        
        # Stochastic
        stoch = self.tech.stochastic(prices, 14, 3)
        signals["stochastic"] = stoch
        
        # Support/Resistance
        sr = self.tech.support_resistance(prices, 20)
        signals["support_resistance"] = sr
        
        # Momentum
        momentum = self.tech.momentum(prices, 10)
        signals["momentum"] = {
            "value": momentum,
            "signal": "bullish" if momentum > 0 else "bearish"
        }
        
        # ATR (volatility)
        if len(highs) > 14:
            atr = self.tech.atr(highs, lows, closes, 14)
            signals["atr"] = {
                "value": atr,
                "percent": (atr / closes[-1]) * 100 if closes[-1] > 0 else 0
            }
        
        # Advanced indicators
        if len(candles) >= 20:
            # VWAP
            if volumes:
                vwap = self.advanced.vwap(highs, lows, closes, volumes)
                signals["vwap"] = vwap
            
            # Supertrend
            supertrend = self.advanced.supertrend(highs, lows, closes)
            signals["supertrend"] = supertrend
        
        # Technical vote
        votes = self._calculate_technical_vote(signals)
        signals["technical_vote"] = votes
        
        return signals
    
    def _calculate_technical_vote(self, tech_signals: Dict) -> Dict[str, Any]:
        """Calculate aggregate technical vote"""
        
        up_votes = 0
        down_votes = 0
        total_weight = 0
        
        # Weighted voting
        weights = {
            "rsi": 1.5,
            "macd": 2.0,
            "bollinger": 1.0,
            "stochastic": 1.0,
            "momentum": 2.0,
            "supertrend": 1.5,
            "vwap": 1.5
        }
        
        for indicator, weight in weights.items():
            if indicator not in tech_signals:
                continue
            
            signal = tech_signals[indicator]
            
            # Handle different signal formats
            if isinstance(signal, dict):
                sig_value = signal.get("signal", signal.get("trend", "neutral"))
            else:
                sig_value = "neutral"
            
            if sig_value in ["oversold", "bullish", "up", "weak_bullish", "strong_up", "trending_up"]:
                up_votes += weight
            elif sig_value in ["overbought", "bearish", "down", "weak_bearish", "strong_down", "trending_down"]:
                down_votes += weight
            
            total_weight += weight
        
        # Calculate percentage
        if total_weight > 0:
            up_percent = up_votes / total_weight
        else:
            up_percent = 0.5
        
        return {
            "up_votes": up_votes,
            "down_votes": down_votes,
            "up_percent": up_percent,
            "direction": "UP" if up_votes > down_votes else "DOWN" if down_votes > up_votes else "NEUTRAL"
        }
    
    def _generate_derivatives_signals(self, derivatives: Dict) -> Dict[str, Any]:
        """Generate derivatives-based signals"""
        
        signals = {}
        
        # Funding rate
        funding = derivatives.get("funding", {})
        if funding:
            rate = funding.get("average_rate", 0)
            
            if rate > 0.0005:
                signal = "bullish"  # Shorts paying longs
            elif rate < -0.0005:
                signal = "bearish"  # Longs paying shorts
            else:
                signal = "neutral"
            
            signals["funding_rate"] = {
                "value": rate,
                "percent": rate * 100,
                "signal": signal
            }
        
        # Open interest
        oi = derivatives.get("open_interest", {})
        if oi:
            signals["open_interest"] = {
                "total": oi.get("total_open_interest", 0),
                "signal": "neutral"  # Need historical for trend
            }
        
        # Basis
        basis = derivatives.get("basis", {})
        if basis:
            signals["basis"] = {
                "structure": basis.get("market_structure", "neutral"),
                "signal": "bullish" if basis.get("signal") == "bullish" else 
                         "bearish" if basis.get("signal") == "bearish" else "neutral"
            }
        
        # Liquidation levels
        liq = derivatives.get("liquidation_levels", {})
        if liq:
            nearest_long = liq.get("nearest_long_liq", {})
            nearest_short = liq.get("nearest_short_liq", {})
            
            # If close to liquidation, expect cascade
            long_dist = nearest_long.get("distance_percent", -10) if nearest_long else -10
            short_dist = nearest_short.get("distance_percent", 10) if nearest_short else 10
            
            if abs(long_dist) < 1:
                signals["liquidation"] = {"signal": "bearish", "reason": "long_liquidation_risk"}
            elif abs(short_dist) < 1:
                signals["liquidation"] = {"signal": "bullish", "reason": "short_squeeze_risk"}
        
        # Combined derivatives signal
        combined = derivatives.get("combined_signal", "neutral")
        signals["combined"] = combined
        
        return signals
    
    def _generate_onchain_signals(self, onchain: Dict) -> Dict[str, Any]:
        """Generate on-chain based signals"""
        
        signals = {}
        
        # Exchange flows
        flows = onchain.get("exchange_flows", {})
        if flows:
            netflow = flows.get("netflow_btc", 0)
            signal = "bullish" if netflow < 0 else "bearish" if netflow > 0 else "neutral"
            signals["exchange_flows"] = {
                "netflow": netflow,
                "signal": signal
            }
        
        # Whale alerts
        whales = onchain.get("whale_alerts", {})
        if whales:
            activity = whales.get("whale_activity", "normal")
            signals["whale_activity"] = {
                "level": activity,
                "count": whales.get("count", 0)
            }
        
        # Fear & Greed
        fgi = onchain.get("fear_greed_index", {})
        if fgi:
            value = fgi.get("value", 50)
            
            # Contrarian signal
            if value <= 25:
                signal = "bullish"  # Extreme fear = buy
            elif value >= 75:
                signal = "bearish"  # Extreme greed = sell
            else:
                signal = "neutral"
            
            signals["fear_greed"] = {
                "value": value,
                "classification": fgi.get("classification", "Neutral"),
                "signal": signal
            }
        
        # Combined on-chain signal
        combined = onchain.get("combined_signal", "neutral")
        signals["combined"] = combined
        
        return signals
    
    def _generate_sentiment_signals(self, sentiment: Dict) -> Dict[str, Any]:
        """Generate sentiment-based signals"""
        
        signals = {}
        
        combined_score = sentiment.get("combined_score", 0)
        
        if combined_score > 0.2:
            signal = "bullish"
        elif combined_score < -0.2:
            signal = "bearish"
        else:
            signal = "neutral"
        
        signals["combined_score"] = combined_score
        signals["signal"] = signal
        
        # Source breakdown
        sources = sentiment.get("sources", {})
        if sources:
            signals["sources"] = sources
        
        return signals
    
    def _detect_regime(
        self,
        prices: List[float],
        candles: List[Dict]
    ) -> Dict[str, Any]:
        """Detect market regime"""
        
        highs = [c.get("high", prices[-1]) for c in candles]
        lows = [c.get("low", prices[-1]) for c in candles]
        closes = prices
        volumes = [c.get("volume", 0) for c in candles]
        
        regime = self.regime_detector.detect_regime(
            prices, highs, lows, closes, volumes
        )
        
        return {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "risk_level": regime.risk_level,
            "recommended_agents": regime.recommended_agents,
            "characteristics": regime.characteristics
        }
    
    def _analyze_multi_timeframe(
        self,
        candles_by_tf: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Analyze multiple timeframes"""
        
        confluence = self.mtf_analyzer.analyze_all_timeframes(candles_by_tf)
        
        return {
            "overall_trend": confluence.overall_trend.value,
            "confluence_score": confluence.confluence_score,
            "trading_signal": confluence.trading_signal,
            "confidence": confluence.confidence,
            "aligned_signals": confluence.aligned_signals,
            "conflicting_signals": confluence.conflicting_signals,
            "higher_tf_direction": self.mtf_analyzer.get_higher_timeframe_direction()
        }
    
    def _combine_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all signals into final decision"""
        
        up_score = 0
        down_score = 0
        total_weight = 0
        
        # Signal weights
        weights = {
            "technical": 0.35,
            "derivatives": 0.20,
            "onchain": 0.15,
            "sentiment": 0.10,
            "multi_timeframe": 0.20
        }
        
        for category, weight in weights.items():
            if category not in signals:
                continue
            
            cat_signals = signals[category]
            
            # Get direction from category
            if category == "technical":
                vote = cat_signals.get("technical_vote", {})
                direction = vote.get("direction", "NEUTRAL")
            elif category == "derivatives":
                direction = cat_signals.get("combined", "neutral")
            elif category == "onchain":
                direction = cat_signals.get("combined", "neutral")
            elif category == "sentiment":
                direction = cat_signals.get("signal", "neutral")
            elif category == "multi_timeframe":
                direction = cat_signals.get("trading_signal", "neutral")
            else:
                direction = "neutral"
            
            # Convert to score
            if direction.upper() in ["UP", "BULLISH"]:
                up_score += weight
            elif direction.upper() in ["DOWN", "BEARISH"]:
                down_score += weight
            
            total_weight += weight
        
        # Calculate final probability
        if total_weight > 0:
            up_prob = up_score / total_weight
        else:
            up_prob = 0.5
        
        return {
            "up_probability": up_prob,
            "down_probability": 1 - up_prob,
            "direction": "UP" if up_prob > 0.53 else "DOWN" if up_prob < 0.47 else "NEUTRAL",
            "confidence": abs(up_prob - 0.5) * 2
        }
    
    def _generate_quick_vote(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quick vote summary for agents"""
        
        combined = signals.get("combined", {})
        regime = signals.get("regime", {})
        mtf = signals.get("multi_timeframe", {})
        
        return {
            "direction": combined.get("direction", "NEUTRAL"),
            "probability": combined.get("up_probability", 0.5),
            "confidence": combined.get("confidence", 0.5),
            "regime": regime.get("regime", "ranging"),
            "higher_tf_trend": mtf.get("higher_tf_direction", "neutral"),
            "confluence": mtf.get("confluence_score", 50)
        }
