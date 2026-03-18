"""
Confidence Scorer for Atlas v4.0
Scores signal confidence based on multiple factors
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConfidenceBreakdown:
    """Detailed confidence breakdown"""
    overall: float
    signal_strength: float
    historical_accuracy: float
    confluence: float
    data_quality: float
    regime_alignment: float
    factors: Dict[str, Any]


class ConfidenceScorer:
    """
    Scores signal confidence based on multiple factors.
    
    Factors:
    - Signal strength (how extreme the reading)
    - Historical accuracy (past performance of this signal)
    - Confluence (agreement across indicators)
    - Data quality (freshness and completeness)
    - Regime alignment (signal matches current regime)
    """
    
    def __init__(self):
        self.signal_history: Dict[str, List[Dict]] = {}
        self.accuracy_by_signal: Dict[str, float] = {}
    
    def score_signal(
        self,
        signal_type: str,
        value: float,
        context: Dict[str, Any]
    ) -> ConfidenceBreakdown:
        """
        Score a single signal's confidence.
        
        Args:
            signal_type: Type of signal (e.g., "rsi", "macd")
            value: Signal value
            context: Market context
        
        Returns:
            ConfidenceBreakdown with detailed scoring
        """
        factors = {}
        
        # 1. Signal strength (how extreme)
        strength = self._calculate_signal_strength(signal_type, value)
        factors["strength"] = strength
        
        # 2. Historical accuracy
        historical = self.accuracy_by_signal.get(signal_type, 0.5)
        factors["historical_accuracy"] = historical
        
        # 3. Confluence with other signals
        confluence = self._calculate_confluence(signal_type, context)
        factors["confluence"] = confluence
        
        # 4. Data quality
        data_quality = self._calculate_data_quality(context)
        factors["data_quality"] = data_quality
        
        # 5. Regime alignment
        regime_alignment = self._calculate_regime_alignment(signal_type, value, context)
        factors["regime_alignment"] = regime_alignment
        
        # Calculate weighted overall confidence
        weights = {
            "strength": 0.25,
            "historical_accuracy": 0.20,
            "confluence": 0.25,
            "data_quality": 0.15,
            "regime_alignment": 0.15
        }
        
        overall = sum(
            factors[k] * weights[k] 
            for k in weights
        )
        
        return ConfidenceBreakdown(
            overall=overall,
            signal_strength=strength,
            historical_accuracy=historical,
            confluence=confluence,
            data_quality=data_quality,
            regime_alignment=regime_alignment,
            factors=factors
        )
    
    def get_overall_confidence(
        self,
        signals: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence across all signals.
        
        Args:
            signals: All generated signals
            market_context: Market context data
        
        Returns:
            Overall confidence score (0-1)
        """
        if not signals:
            return 0.5
        
        confidence_scores = []
        
        # Score each signal category
        categories = ["technical", "derivatives", "onchain", "sentiment", "multi_timeframe"]
        
        for category in categories:
            if category not in signals:
                continue
            
            cat_signals = signals[category]
            
            if isinstance(cat_signals, dict):
                # Get category confidence
                if category == "technical":
                    vote = cat_signals.get("technical_vote", {})
                    direction = vote.get("direction", "NEUTRAL")
                    up_pct = vote.get("up_percent", 0.5)
                    strength = abs(up_pct - 0.5) * 2
                    confidence_scores.append(strength)
                
                elif category == "multi_timeframe":
                    conf = cat_signals.get("confidence", 0.5)
                    confluence = cat_signals.get("confluence_score", 50) / 100
                    confidence_scores.append((conf + confluence) / 2)
                
                else:
                    # Check for confidence indicators
                    if "confidence" in cat_signals:
                        confidence_scores.append(cat_signals["confidence"])
        
        # Adjust for regime alignment
        regime = signals.get("regime", {})
        regime_confidence = regime.get("confidence", 0.5)
        
        # Adjust for data freshness
        timestamp = signals.get("timestamp")
        if timestamp:
            data_age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
            if data_age > 60:  # More than 1 minute old
                age_penalty = min(0.2, data_age / 300)  # Up to 20% penalty
            else:
                age_penalty = 0
        else:
            age_penalty = 0
        
        # Calculate overall
        if confidence_scores:
            base_confidence = sum(confidence_scores) / len(confidence_scores)
            adjusted_confidence = base_confidence * (1 - age_penalty)
            
            # Boost if regime confidence is high
            if regime_confidence > 0.7:
                adjusted_confidence = min(1, adjusted_confidence * 1.1)
        else:
            adjusted_confidence = 0.5
        
        return adjusted_confidence
    
    def _calculate_signal_strength(self, signal_type: str, value: float) -> float:
        """Calculate signal strength based on value extremity"""
        
        # Define signal strength thresholds
        thresholds = {
            "rsi": {
                "neutral_range": (40, 60),
                "strong_range": (20, 80),
                "extreme_range": (0, 100)
            },
            "stochastic": {
                "neutral_range": (40, 60),
                "strong_range": (20, 80),
                "extreme_range": (0, 100)
            },
            "macd": {
                "neutral_range": (-0.001, 0.001),
                "strong_range": (-0.01, 0.01),
                "extreme_range": (-0.1, 0.1)
            },
            "funding_rate": {
                "neutral_range": (-0.0001, 0.0001),
                "strong_range": (-0.0005, 0.0005),
                "extreme_range": (-0.001, 0.001)
            }
        }
        
        if signal_type not in thresholds:
            return 0.5
        
        thresh = thresholds[signal_type]
        neutral_min, neutral_max = thresh["neutral_range"]
        strong_min, strong_max = thresh["strong_range"]
        
        # In neutral range
        if neutral_min <= value <= neutral_max:
            return 0.3
        
        # In strong range
        if strong_min <= value <= strong_max:
            # Calculate position within strong range
            if value < neutral_min:
                position = (value - strong_min) / (neutral_min - strong_min)
            else:
                position = (strong_max - value) / (strong_max - neutral_max)
            return 0.3 + (0.4 * position)
        
        # Extreme values
        return 0.7 + (0.3 * min(1, abs(value) / abs(strong_max if value > 0 else strong_min)))
    
    def _calculate_confluence(
        self,
        signal_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confluence with other signals"""
        
        # Get signal direction
        signal_direction = context.get("signals", {}).get(signal_type, {}).get("signal", "neutral")
        
        # Count agreeing signals
        agreeing = 0
        total = 0
        
        for other_type, other_signal in context.get("signals", {}).items():
            if other_type == signal_type:
                continue
            
            if isinstance(other_signal, dict):
                other_direction = other_signal.get("signal", other_signal.get("trend", "neutral"))
            else:
                other_direction = "neutral"
            
            total += 1
            
            if signal_direction.lower() == other_direction.lower():
                agreeing += 1
        
        if total == 0:
            return 0.5
        
        return agreeing / total
    
    def _calculate_data_quality(self, context: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        
        quality = 1.0
        
        # Check data freshness
        timestamp = context.get("timestamp")
        if timestamp:
            age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
            if age > 30:
                quality *= 0.9
            if age > 60:
                quality *= 0.8
            if age > 120:
                quality *= 0.6
        
        # Check data completeness
        required_fields = ["prices", "highs", "lows", "closes"]
        missing = sum(1 for f in required_fields if f not in context)
        quality *= (1 - missing * 0.1)
        
        # Check data length
        prices = context.get("prices", [])
        if len(prices) < 30:
            quality *= 0.7
        elif len(prices) < 50:
            quality *= 0.9
        
        return max(0.3, quality)
    
    def _calculate_regime_alignment(
        self,
        signal_type: str,
        value: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate alignment with current market regime"""
        
        regime = context.get("regime", {}).get("regime", "ranging")
        
        # Signal effectiveness by regime
        regime_signal_effectiveness = {
            "trending_up": {
                "momentum": 0.9,
                "macd": 0.85,
                "rsi": 0.5,  # Less reliable in trends
                "bollinger": 0.4,
                "stochastic": 0.4
            },
            "trending_down": {
                "momentum": 0.9,
                "macd": 0.85,
                "rsi": 0.5,
                "bollinger": 0.4,
                "stochastic": 0.4
            },
            "ranging": {
                "rsi": 0.9,
                "stochastic": 0.85,
                "bollinger": 0.85,
                "macd": 0.4,  # Less reliable in ranges
                "momentum": 0.3
            },
            "volatile": {
                "bollinger": 0.7,
                "atr": 0.8,
                "rsi": 0.5,
                "momentum": 0.6
            },
            "breakout": {
                "volume": 0.9,
                "momentum": 0.8,
                "bollinger": 0.7,
                "rsi": 0.5
            },
            "reversal": {
                "rsi": 0.85,
                "stochastic": 0.8,
                "bollinger": 0.75,
                "macd": 0.6
            }
        }
        
        # Get effectiveness for this signal in current regime
        regime_effects = regime_signal_effectiveness.get(regime, {})
        effectiveness = regime_effects.get(signal_type, 0.5)
        
        return effectiveness
    
    def record_signal_outcome(
        self,
        signal_type: str,
        predicted_direction: str,
        actual_direction: str
    ):
        """Record signal outcome for accuracy tracking"""
        
        if signal_type not in self.signal_history:
            self.signal_history[signal_type] = []
        
        correct = predicted_direction.lower() == actual_direction.lower()
        
        self.signal_history[signal_type].append({
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": correct,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update accuracy
        recent = self.signal_history[signal_type][-20:]  # Last 20 signals
        if recent:
            self.accuracy_by_signal[signal_type] = sum(1 for s in recent if s["correct"]) / len(recent)
    
    def get_signal_stats(self, signal_type: str) -> Dict[str, Any]:
        """Get statistics for a signal type"""
        
        history = self.signal_history.get(signal_type, [])
        
        if not history:
            return {"accuracy": 0.5, "count": 0}
        
        correct = sum(1 for h in history if h["correct"])
        
        return {
            "accuracy": correct / len(history),
            "count": len(history),
            "recent_accuracy": self.accuracy_by_signal.get(signal_type, 0.5)
        }
