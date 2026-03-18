"""
Multi-Timeframe Analysis for Atlas v4.0
Analyzes multiple timeframes for confluence and trend alignment
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class TrendDirection(Enum):
    """Trend direction classification"""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe"""
    timeframe: str
    trend: TrendDirection
    strength: float  # 0-1
    momentum: float  # -1 to 1
    support: float
    resistance: float
    signal: str  # "bullish", "bearish", "neutral"
    indicators: Dict[str, Any]


@dataclass
class TimeframeConfluence:
    """Multi-timeframe confluence result"""
    overall_trend: TrendDirection
    confluence_score: float  # 0-100
    aligned_signals: List[str]
    conflicting_signals: List[str]
    timeframe_analyses: Dict[str, TimeframeAnalysis]
    trading_signal: str
    confidence: float


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes for confluence scoring.
    
    Timeframes:
    - 1m: Entry timing
    - 5m: Primary analysis
    - 15m: Market context
    - 1h: Short-term trend
    - 4h: Medium-term trend
    - 1d: Daily direction
    
    Philosophy: Trade in the direction of higher timeframes.
    Use lower timeframes for entry timing.
    """
    
    # Timeframe weights for confluence calculation
    TIMEFRAME_WEIGHTS = {
        "1m": 0.05,    # Entry timing only
        "5m": 0.20,    # Primary
        "15m": 0.25,   # Context (same as prediction window)
        "1h": 0.20,    # Short-term trend
        "4h": 0.15,    # Medium-term trend
        "1d": 0.15,    # Daily direction
    }
    
    def __init__(self):
        self.analyses: Dict[str, TimeframeAnalysis] = {}
    
    def analyze_all_timeframes(
        self,
        candles_by_tf: Dict[str, List[Dict[str, float]]]
    ) -> TimeframeConfluence:
        """
        Analyze all timeframes and calculate confluence.
        
        Args:
            candles_by_tf: Dict mapping timeframe to list of candle dicts
        
        Returns:
            TimeframeConfluence with overall analysis
        """
        self.analyses = {}
        
        for tf, candles in candles_by_tf.items():
            if candles:
                self.analyses[tf] = self._analyze_timeframe(tf, candles)
        
        return self._calculate_confluence()
    
    def _analyze_timeframe(
        self,
        timeframe: str,
        candles: List[Dict[str, float]]
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c.get("volume", 0) for c in candles]
        
        if len(closes) < 20:
            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=TrendDirection.NEUTRAL,
                strength=0.5,
                momentum=0,
                support=closes[-1] if closes else 0,
                resistance=closes[-1] if closes else 0,
                signal="neutral",
                indicators={}
            )
        
        # Calculate indicators
        ema_9 = self._ema(closes, 9)
        ema_21 = self._ema(closes, 21)
        ema_50 = self._ema(closes, 50) if len(closes) >= 50 else None
        
        rsi = self._rsi(closes, 14)
        macd = self._macd(closes)
        
        momentum = ((closes[-1] - closes[-10]) / closes[-10]) * 100 if len(closes) >= 10 else 0
        
        # Support/Resistance
        support = min(lows[-20:])
        resistance = max(highs[-20:])
        
        # Determine trend
        trend = self._determine_trend(closes, ema_9, ema_21, ema_50)
        
        # Determine strength
        strength = self._calculate_strength(closes, highs, lows)
        
        # Determine signal
        signal = self._determine_signal(rsi, macd, momentum, trend)
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            strength=strength,
            momentum=momentum,
            support=support,
            resistance=resistance,
            signal=signal,
            indicators={
                "ema_9": ema_9[-1] if ema_9 else closes[-1],
                "ema_21": ema_21[-1] if ema_21 else closes[-1],
                "ema_50": ema_50[-1] if ema_50 else closes[-1],
                "rsi": rsi,
                "macd": macd.get("histogram", 0),
                "momentum": momentum
            }
        )
    
    def _calculate_confluence(self) -> TimeframeConfluence:
        """Calculate confluence across all timeframes"""
        
        if not self.analyses:
            return TimeframeConfluence(
                overall_trend=TrendDirection.NEUTRAL,
                confluence_score=50,
                aligned_signals=[],
                conflicting_signals=[],
                timeframe_analyses={},
                trading_signal="neutral",
                confidence=0.5
            )
        
        # Weight trend scores
        trend_scores = {
            TrendDirection.STRONG_UP: 2,
            TrendDirection.UP: 1,
            TrendDirection.NEUTRAL: 0,
            TrendDirection.DOWN: -1,
            TrendDirection.STRONG_DOWN: -2
        }
        
        weighted_trend = 0
        total_weight = 0
        
        for tf, analysis in self.analyses.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            weighted_trend += trend_scores[analysis.trend] * weight
            total_weight += weight
        
        # Normalize to -2 to 2 range
        if total_weight > 0:
            normalized_trend = weighted_trend / total_weight
        else:
            normalized_trend = 0
        
        # Determine overall trend
        if normalized_trend >= 1.5:
            overall_trend = TrendDirection.STRONG_UP
        elif normalized_trend >= 0.5:
            overall_trend = TrendDirection.UP
        elif normalized_trend <= -1.5:
            overall_trend = TrendDirection.STRONG_DOWN
        elif normalized_trend <= -0.5:
            overall_trend = TrendDirection.DOWN
        else:
            overall_trend = TrendDirection.NEUTRAL
        
        # Find aligned and conflicting signals
        signals = [a.signal for a in self.analyses.values()]
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        
        aligned_signals = []
        conflicting_signals = []
        
        if bullish_count > len(signals) * 0.6:
            aligned_signals.append("bullish_alignment")
            trading_signal = "bullish"
        elif bearish_count > len(signals) * 0.6:
            aligned_signals.append("bearish_alignment")
            trading_signal = "bearish"
        else:
            conflicting_signals.append("mixed_signals")
            trading_signal = "neutral"
        
        # Check higher timeframe alignment
        higher_tf_signals = []
        for tf in ["1h", "4h", "1d"]:
            if tf in self.analyses:
                higher_tf_signals.append(self.analyses[tf].signal)
        
        if len(set(higher_tf_signals)) == 1 and higher_tf_signals[0] != "neutral":
            aligned_signals.append(f"higher_tf_{higher_tf_signals[0]}")
        elif len(set(higher_tf_signals)) > 1:
            conflicting_signals.append("higher_tf_divergence")
        
        # Calculate confluence score (0-100)
        confluence_score = self._calculate_confluence_score()
        
        # Calculate confidence
        confidence = confluence_score / 100
        
        return TimeframeConfluence(
            overall_trend=overall_trend,
            confluence_score=confluence_score,
            aligned_signals=aligned_signals,
            conflicting_signals=conflicting_signals,
            timeframe_analyses=self.analyses,
            trading_signal=trading_signal,
            confidence=confidence
        )
    
    def _calculate_confluence_score(self) -> float:
        """Calculate confluence score 0-100"""
        
        if not self.analyses:
            return 50
        
        # Count agreeing signals
        signals = [a.signal for a in self.analyses.values()]
        
        if not signals:
            return 50
        
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        neutral_count = signals.count("neutral")
        
        total = len(signals)
        
        # Max agreement = 100, mixed = 50
        max_agreement = max(bullish_count, bearish_count, neutral_count)
        
        # Weight by timeframe importance
        weighted_agreement = 0
        weighted_total = 0
        
        for tf, analysis in self.analyses.items():
            weight = self.TIMEFRAME_WEIGHTS.get(tf, 0.1)
            weighted_total += weight
            
            if analysis.signal != "neutral":
                weighted_agreement += weight
        
        # Confluence = agreement percentage
        if weighted_total > 0:
            agreement_ratio = max_agreement / total
            confluence = agreement_ratio * 100
        else:
            confluence = 50
        
        # Boost if higher timeframes agree
        higher_tf_agree = False
        higher_tf_signals = []
        for tf in ["1h", "4h", "1d"]:
            if tf in self.analyses:
                higher_tf_signals.append(self.analyses[tf].signal)
        
        if len(set(higher_tf_signals)) == 1 and higher_tf_signals[0] != "neutral":
            confluence = min(100, confluence * 1.2)  # 20% boost
            higher_tf_agree = True
        
        return confluence
    
    def _determine_trend(
        self,
        closes: List[float],
        ema_9: List[float],
        ema_21: List[float],
        ema_50: Optional[List[float]]
    ) -> TrendDirection:
        """Determine trend direction from price and EMAs"""
        
        if not ema_9 or not ema_21:
            return TrendDirection.NEUTRAL
        
        current_price = closes[-1]
        ema9 = ema_9[-1]
        ema21 = ema_21[-1]
        ema50 = ema_50[-1] if ema_50 else None
        
        # Count trend indicators
        bullish_count = 0
        bearish_count = 0
        
        # Price above/below EMAs
        if current_price > ema9:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if current_price > ema21:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if ema50:
            if current_price > ema50:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # EMA alignment
        if ema9 > ema21:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if ema50:
            if ema21 > ema50:
                bullish_count += 1
            elif ema21 < ema50:
                bearish_count += 1
        
        # Determine trend
        if bullish_count >= 4:
            return TrendDirection.STRONG_UP
        elif bullish_count >= 3:
            return TrendDirection.UP
        elif bearish_count >= 4:
            return TrendDirection.STRONG_DOWN
        elif bearish_count >= 3:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL
    
    def _calculate_strength(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float]
    ) -> float:
        """Calculate trend strength (0-1)"""
        
        if len(closes) < 20:
            return 0.5
        
        # ADX-like calculation
        tr_list = []
        dm_plus_list = []
        dm_minus_list = []
        
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
            
            dm_plus = highs[i] - highs[i-1] if (highs[i] - highs[i-1]) > 0 else 0
            dm_minus = lows[i-1] - lows[i] if (lows[i-1] - lows[i]) > 0 else 0
            
            dm_plus_list.append(dm_plus)
            dm_minus_list.append(dm_minus)
        
        period = min(14, len(tr_list))
        
        if period == 0:
            return 0.5
        
        tr_avg = sum(tr_list[-period:]) / period
        dm_plus_avg = sum(dm_plus_list[-period:]) / period
        dm_minus_avg = sum(dm_minus_list[-period:]) / period
        
        if tr_avg == 0:
            return 0.5
        
        di_plus = (dm_plus_avg / tr_avg) * 100
        di_minus = (dm_minus_avg / tr_avg) * 100
        
        if (di_plus + di_minus) == 0:
            return 0.5
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        
        # Normalize to 0-1
        return min(1, dx / 100)
    
    def _determine_signal(
        self,
        rsi: float,
        macd: Dict,
        momentum: float,
        trend: TrendDirection
    ) -> str:
        """Determine trading signal"""
        
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI
        if rsi < 30:
            bullish_signals += 2
        elif rsi < 40:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 2
        elif rsi > 60:
            bearish_signals += 1
        
        # MACD
        macd_hist = macd.get("histogram", 0)
        if macd_hist > 0:
            bullish_signals += 1
        elif macd_hist < 0:
            bearish_signals += 1
        
        # Momentum
        if momentum > 0.5:
            bullish_signals += 1
        elif momentum < -0.5:
            bearish_signals += 1
        
        # Trend
        if trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
            bullish_signals += 1
        elif trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
            bearish_signals += 1
        
        # Final decision
        if bullish_signals > bearish_signals + 1:
            return "bullish"
        elif bearish_signals > bullish_signals + 1:
            return "bearish"
        else:
            return "neutral"
    
    # Technical indicator helpers
    
    def _ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate EMA"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        result = [sum(prices[:period]) / period]
        
        for price in prices[period:]:
            ema = (price - result[-1]) * multiplier + result[-1]
            result.append(ema)
        
        return result
    
    def _rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _macd(
        self,
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, float]:
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        if not ema_fast or not ema_slow:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        macd_line = [f - s for f, s in zip(ema_fast[-(len(ema_slow)):], ema_slow)]
        
        if len(macd_line) < signal:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        signal_line = sum(macd_line[-signal:]) / signal
        histogram = macd_line[-1] - signal_line
        
        return {
            "macd": macd_line[-1],
            "signal": signal_line,
            "histogram": histogram
        }
    
    def get_higher_timeframe_direction(self) -> str:
        """Get direction from higher timeframes (1h, 4h, 1d)"""
        if not self.analyses:
            return "neutral"
        
        higher_tf_signals = []
        for tf in ["1h", "4h", "1d"]:
            if tf in self.analyses:
                higher_tf_signals.append(self.analyses[tf].signal)
        
        if not higher_tf_signals:
            return "neutral"
        
        bullish = higher_tf_signals.count("bullish")
        bearish = higher_tf_signals.count("bearish")
        
        if bullish > bearish:
            return "bullish"
        elif bearish > bullish:
            return "bearish"
        else:
            return "neutral"
