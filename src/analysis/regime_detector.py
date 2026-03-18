"""
Market Regime Detection for Atlas v4.0
Detects current market conditions to select appropriate strategies
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    EXHAUSTION = "exhaustion"


@dataclass
class RegimeResult:
    """Regime detection result"""
    regime: MarketRegime
    confidence: float
    duration_bars: int
    characteristics: Dict[str, Any]
    recommended_agents: List[str]
    risk_level: str  # "low", "medium", "high"


class RegimeDetector:
    """
    Detects current market regime for strategy selection.
    
    Uses multiple indicators to classify market conditions:
    - ADX for trend strength
    - ATR for volatility regime
    - Bollinger Band width for compression
    - Volume profile for range detection
    - Price action patterns
    """
    
    # Agent recommendations by regime
    REGIME_AGENT_MAP = {
        MarketRegime.TRENDING_UP: ["momentum_hawk", "macd_trend", "volume_whale", "trend_rider"],
        MarketRegime.TRENDING_DOWN: ["momentum_hawk", "macd_trend", "support_resist", "trend_fader"],
        MarketRegime.RANGING: ["support_resist", "rsi_master", "range_trader", "breakout_hunter"],
        MarketRegime.VOLATILE: ["volatility_harvester", "order_flow", "risk_guard", "volatility"],
        MarketRegime.BREAKOUT: ["volume_whale", "momentum_hawk", "breakout_hunter", "order_flow"],
        MarketRegime.REVERSAL: ["rsi_master", "support_resist", "mean_reverter", "sentiment"],
        MarketRegime.CONSOLIDATION: ["support_resist", "volatility", "breakout_hunter"],
        MarketRegime.EXHAUSTION: ["mean_reverter", "rsi_master", "sentiment", "risk_guard"],
    }
    
    def __init__(self):
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[Tuple[float, MarketRegime]] = []
        self.regime_start_bar: int = 0
    
    def detect_regime(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: Optional[List[float]] = None
    ) -> RegimeResult:
        """
        Detect current market regime using multiple factors.
        
        Args:
            prices: List of close prices
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            volumes: Optional list of volumes
        
        Returns:
            RegimeResult with classification and recommendations
        """
        if len(prices) < 50:
            return self._default_result()
        
        # Calculate all indicators
        indicators = self._calculate_indicators(prices, highs, lows, closes, volumes)
        
        # Score each regime
        scores = self._score_regimes(indicators, prices)
        
        # Select best regime
        best_regime = max(scores.keys(), key=lambda r: scores[r])
        confidence = scores[best_regime]
        
        # Calculate duration
        duration = self._calculate_regime_duration(best_regime)
        
        # Get characteristics
        characteristics = self._get_characteristics(indicators)
        
        # Determine risk level
        risk = self._determine_risk_level(best_regime, indicators)
        
        # Update state
        if self.current_regime != best_regime:
            self.current_regime = best_regime
            self.regime_start_bar = len(prices)
        
        return RegimeResult(
            regime=best_regime,
            confidence=confidence,
            duration_bars=duration,
            characteristics=characteristics,
            recommended_agents=self.REGIME_AGENT_MAP.get(best_regime, []),
            risk_level=risk
        )
    
    def _calculate_indicators(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: Optional[List[float]]
    ) -> Dict[str, Any]:
        """Calculate all indicators for regime detection"""
        
        indicators = {}
        
        # ADX for trend strength
        indicators["adx"] = self._calculate_adx(highs, lows, closes)
        
        # ATR for volatility
        indicators["atr"] = self._calculate_atr(highs, lows, closes)
        indicators["atr_percent"] = (indicators["atr"] / closes[-1]) * 100
        
        # Bollinger Band width
        indicators["bb_width"] = self._calculate_bb_width(prices)
        
        # Price momentum
        indicators["momentum_5"] = ((prices[-1] - prices[-5]) / prices[-5]) * 100 if len(prices) >= 5 else 0
        indicators["momentum_10"] = ((prices[-1] - prices[-10]) / prices[-10]) * 100 if len(prices) >= 10 else 0
        indicators["momentum_20"] = ((prices[-1] - prices[-20]) / prices[-20]) * 100 if len(prices) >= 20 else 0
        
        # Higher highs / higher lows count
        indicators["hh_count"] = self._count_higher_highs(prices)
        indicators["hl_count"] = self._count_higher_lows(lows)
        indicators["lh_count"] = self._count_lower_highs(highs)
        indicators["ll_count"] = self._count_lower_lows(lows)
        
        # RSI
        indicators["rsi"] = self._calculate_rsi(prices)
        
        # Price position in range
        if len(prices) >= 20:
            recent_high = max(highs[-20:])
            recent_low = min(lows[-20:])
            if recent_high != recent_low:
                indicators["range_position"] = (closes[-1] - recent_low) / (recent_high - recent_low)
            else:
                indicators["range_position"] = 0.5
        else:
            indicators["range_position"] = 0.5
        
        # Volume trend (if available)
        if volumes and len(volumes) >= 10:
            avg_vol = sum(volumes[-10:]) / 10
            recent_vol = sum(volumes[-3:]) / 3
            indicators["volume_trend"] = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0
        else:
            indicators["volume_trend"] = 0
        
        # Volatility of volatility
        if len(prices) >= 20:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            vol_series = [abs(r) for r in returns[-20:]]
            indicators["vol_of_vol"] = self._std(vol_series) if len(vol_series) > 1 else 0
        else:
            indicators["vol_of_vol"] = 0
        
        return indicators
    
    def _score_regimes(
        self,
        indicators: Dict[str, Any],
        prices: List[float]
    ) -> Dict[MarketRegime, float]:
        """Score each regime based on indicators"""
        
        scores = {}
        
        adx = indicators.get("adx", 25)
        atr_pct = indicators.get("atr_percent", 0.5)
        momentum = indicators.get("momentum_10", 0)
        hh_count = indicators.get("hh_count", 0)
        ll_count = indicators.get("ll_count", 0)
        rsi = indicators.get("rsi", 50)
        range_pos = indicators.get("range_position", 0.5)
        bb_width = indicators.get("bb_width", 0.02)
        vol_trend = indicators.get("volume_trend", 0)
        
        # TRENDING_UP: Strong ADX, positive momentum, higher highs
        trending_up_score = 0
        if adx > 25:
            trending_up_score += 0.3
        if momentum > 0.5:
            trending_up_score += 0.3
        if hh_count > ll_count + 2:
            trending_up_score += 0.2
        if rsi > 50:
            trending_up_score += 0.1
        if range_pos > 0.6:
            trending_up_score += 0.1
        scores[MarketRegime.TRENDING_UP] = trending_up_score
        
        # TRENDING_DOWN: Strong ADX, negative momentum, lower lows
        trending_down_score = 0
        if adx > 25:
            trending_down_score += 0.3
        if momentum < -0.5:
            trending_down_score += 0.3
        if ll_count > hh_count + 2:
            trending_down_score += 0.2
        if rsi < 50:
            trending_down_score += 0.1
        if range_pos < 0.4:
            trending_down_score += 0.1
        scores[MarketRegime.TRENDING_DOWN] = trending_down_score
        
        # RANGING: Low ADX, tight Bollinger bands, oscillating RSI
        ranging_score = 0
        if adx < 20:
            ranging_score += 0.4
        if bb_width < 0.02:
            ranging_score += 0.3
        if 40 < rsi < 60:
            ranging_score += 0.2
        if 0.3 < range_pos < 0.7:
            ranging_score += 0.1
        scores[MarketRegime.RANGING] = ranging_score
        
        # VOLATILE: High ATR, high volatility of volatility
        volatile_score = 0
        if atr_pct > 0.3:
            volatile_score += 0.4
        if indicators.get("vol_of_vol", 0) > 0.01:
            volatile_score += 0.3
        if abs(momentum) > 1:
            volatile_score += 0.2
        if bb_width > 0.03:
            volatile_score += 0.1
        scores[MarketRegime.VOLATILE] = volatile_score
        
        # BREAKOUT: Low ADX turning higher, Bollinger squeeze, volume spike
        breakout_score = 0
        if adx < 25 and adx > 15:
            breakout_score += 0.2
        if bb_width < 0.015:  # Squeeze
            breakout_score += 0.4
        if vol_trend > 0.5:  # Volume spike
            breakout_score += 0.3
        if 0.45 < range_pos < 0.55:  # Near middle of range
            breakout_score += 0.1
        scores[MarketRegime.BREAKOUT] = breakout_score
        
        # REVERSAL: Overbought/oversold RSI, extreme momentum
        reversal_score = 0
        if rsi > 70 or rsi < 30:
            reversal_score += 0.4
        if abs(momentum) > 1.5:
            reversal_score += 0.2
        if (rsi > 70 and momentum < 0) or (rsi < 30 and momentum > 0):  # Divergence hint
            reversal_score += 0.3
        scores[MarketRegime.REVERSAL] = reversal_score
        
        # CONSOLIDATION: Very low volatility, tight range
        consolidation_score = 0
        if atr_pct < 0.2:
            consolidation_score += 0.4
        if bb_width < 0.01:
            consolidation_score += 0.3
        if adx < 15:
            consolidation_score += 0.2
        if abs(momentum) < 0.2:
            consolidation_score += 0.1
        scores[MarketRegime.CONSOLIDATION] = consolidation_score
        
        # EXHAUSTION: Strong prior trend weakening, momentum divergence
        exhaustion_score = 0
        if adx > 20 and adx < 30:  # Trend weakening
            exhaustion_score += 0.2
        if (momentum > 1 and rsi > 70) or (momentum < -1 and rsi < 30):
            exhaustion_score += 0.4
        if abs(indicators.get("momentum_5", 0)) < abs(indicators.get("momentum_20", 0)) * 0.5:
            exhaustion_score += 0.3  # Momentum fading
        scores[MarketRegime.EXHAUSTION] = exhaustion_score
        
        return scores
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long we've been in current regime"""
        return len(self.regime_history) - self.regime_start_bar if self.regime_history else 0
    
    def _get_characteristics(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Get market characteristics summary"""
        return {
            "trend_strength": indicators.get("adx", 25),
            "volatility": indicators.get("atr_percent", 0),
            "momentum": indicators.get("momentum_10", 0),
            "rsi": indicators.get("rsi", 50),
            "range_position": indicators.get("range_position", 0.5),
            "bb_compression": indicators.get("bb_width", 0.02) < 0.015,
            "volume_trend": indicators.get("volume_trend", 0)
        }
    
    def _determine_risk_level(
        self,
        regime: MarketRegime,
        indicators: Dict[str, Any]
    ) -> str:
        """Determine risk level for current regime"""
        
        high_risk_regimes = [MarketRegime.VOLATILE, MarketRegime.BREAKOUT, MarketRegime.EXHAUSTION]
        low_risk_regimes = [MarketRegime.RANGING, MarketRegime.CONSOLIDATION]
        
        if regime in high_risk_regimes:
            return "high"
        elif regime in low_risk_regimes:
            return "low"
        else:
            return "medium"
    
    def _default_result(self) -> RegimeResult:
        """Return default regime when insufficient data"""
        return RegimeResult(
            regime=MarketRegime.RANGING,
            confidence=0.5,
            duration_bars=0,
            characteristics={},
            recommended_agents=["rsi_master", "support_resist"],
            risk_level="medium"
        )
    
    # Helper calculation methods
    
    def _calculate_adx(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> float:
        """Calculate ADX"""
        if len(closes) < period * 2:
            return 25.0
        
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
            
            dm_plus = highs[i] - highs[i-1] if (highs[i] - highs[i-1]) > max(0, lows[i-1] - lows[i]) else 0
            dm_minus = lows[i-1] - lows[i] if (lows[i-1] - lows[i]) > max(0, highs[i] - highs[i-1]) else 0
            
            dm_plus_list.append(dm_plus)
            dm_minus_list.append(dm_minus)
        
        # Smooth
        tr_smooth = sum(tr_list[-period:]) / period
        dm_plus_smooth = sum(dm_plus_list[-period:]) / period
        dm_minus_smooth = sum(dm_minus_list[-period:]) / period
        
        # DI
        di_plus = (dm_plus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
        di_minus = (dm_minus_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
        
        # DX
        if (di_plus + di_minus) > 0:
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        else:
            dx = 0
        
        return dx
    
    def _calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> float:
        """Calculate ATR"""
        if len(closes) < period + 1:
            return 0.0
        
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
        
        return sum(tr_list[-period:]) / period
    
    def _calculate_bb_width(self, prices: List[float], period: int = 20) -> float:
        """Calculate Bollinger Band width"""
        if len(prices) < period:
            return 0.02
        
        recent = prices[-period:]
        sma = sum(recent) / period
        
        variance = sum((p - sma) ** 2 for p in recent) / period
        std = math.sqrt(variance)
        
        return (2 * std) / sma if sma > 0 else 0
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
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
    
    def _count_higher_highs(self, prices: List[float], window: int = 20) -> int:
        """Count higher highs in window"""
        if len(prices) < window:
            return 0
        
        count = 0
        for i in range(len(prices) - window, len(prices) - 1):
            if prices[i + 1] > prices[i]:
                count += 1
        return count
    
    def _count_higher_lows(self, lows: List[float], window: int = 20) -> int:
        """Count higher lows in window"""
        if len(lows) < window:
            return 0
        
        count = 0
        for i in range(len(lows) - window, len(lows) - 1):
            if lows[i + 1] > lows[i]:
                count += 1
        return count
    
    def _count_lower_highs(self, highs: List[float], window: int = 20) -> int:
        """Count lower highs in window"""
        if len(highs) < window:
            return 0
        
        count = 0
        for i in range(len(highs) - window, len(highs) - 1):
            if highs[i + 1] < highs[i]:
                count += 1
        return count
    
    def _count_lower_lows(self, lows: List[float], window: int = 20) -> int:
        """Count lower lows in window"""
        if len(lows) < window:
            return 0
        
        count = 0
        for i in range(len(lows) - window, len(lows) - 1):
            if lows[i + 1] < lows[i]:
                count += 1
        return count
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
