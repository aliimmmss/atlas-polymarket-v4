"""
Risk-Adjusted Confidence for Atlas v4.0
Adjusts confidence based on risk factors
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskFactors:
    """Risk factors that affect confidence"""
    volatility_risk: float  # 0-1
    regime_uncertainty: float  # 0-1
    agent_disagreement: float  # 0-1
    data_quality: float  # 0-1 (1 = high quality)
    time_risk: float  # 0-1 (higher near market close)
    liquidity_risk: float  # 0-1
    news_risk: float  # 0-1 (breaking news uncertainty)


@dataclass
class AdjustedConfidence:
    """Risk-adjusted confidence result"""
    original_confidence: float
    adjusted_confidence: float
    adjustment_factor: float
    risk_breakdown: RiskFactors
    recommendation: str
    should_trade: bool


class RiskAdjustedConfidence:
    """
    Adjusts confidence score for risk factors.
    
    Risk Factors:
    - Volatility risk: Higher volatility = lower confidence
    - Regime uncertainty: Transitioning regimes = lower confidence
    - Agent disagreement: High disagreement = lower confidence
    - Data quality: Missing/stale data = lower confidence
    - Time risk: Near market close = higher uncertainty
    - Liquidity risk: Low liquidity = higher slippage
    - News risk: Breaking news = unpredictable moves
    """
    
    # Risk weights (how much each factor affects confidence)
    RISK_WEIGHTS = {
        "volatility": 0.20,
        "regime": 0.20,
        "agent_disagreement": 0.15,
        "data_quality": 0.15,
        "time": 0.10,
        "liquidity": 0.10,
        "news": 0.10
    }
    
    def __init__(
        self,
        min_confidence_to_trade: float = 0.5,
        max_risk_tolerance: float = 0.7
    ):
        self.min_confidence = min_confidence_to_trade
        self.max_risk_tolerance = max_risk_tolerance
        
        # History
        self.confidence_history: List[Dict] = []
    
    def calculate_risk_adjusted_confidence(
        self,
        base_confidence: float,
        market_context: Dict[str, Any],
        agent_predictions: List[Dict]
    ) -> AdjustedConfidence:
        """
        Calculate risk-adjusted confidence.
        
        Args:
            base_confidence: Original confidence (0-1)
            market_context: Market data and regime info
            agent_predictions: List of agent predictions
        
        Returns:
            AdjustedConfidence with risk breakdown
        """
        # Calculate all risk factors
        risk_factors = RiskFactors(
            volatility_risk=self._calculate_volatility_risk(market_context),
            regime_uncertainty=self._calculate_regime_uncertainty(market_context),
            agent_disagreement=self._calculate_agent_disagreement(agent_predictions),
            data_quality=self._calculate_data_quality(market_context),
            time_risk=self._calculate_time_risk(market_context),
            liquidity_risk=self._calculate_liquidity_risk(market_context),
            news_risk=self._calculate_news_risk(market_context)
        )
        
        # Calculate overall risk score (0 = no risk, 1 = max risk)
        overall_risk = (
            risk_factors.volatility_risk * self.RISK_WEIGHTS["volatility"] +
            risk_factors.regime_uncertainty * self.RISK_WEIGHTS["regime"] +
            risk_factors.agent_disagreement * self.RISK_WEIGHTS["agent_disagreement"] +
            (1 - risk_factors.data_quality) * self.RISK_WEIGHTS["data_quality"] +
            risk_factors.time_risk * self.RISK_WEIGHTS["time"] +
            risk_factors.liquidity_risk * self.RISK_WEIGHTS["liquidity"] +
            risk_factors.news_risk * self.RISK_WEIGHTS["news"]
        )
        
        # Calculate adjustment factor
        adjustment_factor = 1 - overall_risk
        
        # Apply adjustment
        adjusted_confidence = base_confidence * adjustment_factor
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            adjusted_confidence, risk_factors, overall_risk
        )
        
        # Determine if should trade
        should_trade = (
            adjusted_confidence >= self.min_confidence and
            overall_risk <= self.max_risk_tolerance
        )
        
        return AdjustedConfidence(
            original_confidence=base_confidence,
            adjusted_confidence=adjusted_confidence,
            adjustment_factor=adjustment_factor,
            risk_breakdown=risk_factors,
            recommendation=recommendation,
            should_trade=should_trade
        )
    
    def _calculate_volatility_risk(self, context: Dict) -> float:
        """
        Calculate volatility risk.
        
        Higher volatility = higher risk.
        """
        volatility = context.get("volatility", context.get("volatility_5m", 0))
        
        # Normalize volatility (typical range 0-1%)
        # 0% = 0 risk, 0.5% = 0.5 risk, 1%+ = 1 risk
        risk = min(1.0, volatility / 0.5) if volatility > 0 else 0
        
        return risk
    
    def _calculate_regime_uncertainty(self, context: Dict) -> float:
        """
        Calculate regime uncertainty.
        
        Low confidence in regime = high uncertainty.
        """
        regime_info = context.get("regime", {})
        
        if not regime_info:
            return 0.5  # Unknown regime
        
        regime_confidence = regime_info.get("confidence", 0.5)
        
        # Low confidence = high uncertainty
        uncertainty = 1 - regime_confidence
        
        return uncertainty
    
    def _calculate_agent_disagreement(self, predictions: List[Dict]) -> float:
        """
        Calculate agent disagreement.
        
        More disagreement = higher risk.
        """
        if not predictions:
            return 0.5
        
        directions = [p.get("direction", "NEUTRAL") for p in predictions]
        
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")
        neutral_count = directions.count("NEUTRAL")
        total = len(directions)
        
        if total == 0:
            return 0.5
        
        # Calculate disagreement ratio
        max_agreement = max(up_count, down_count, neutral_count)
        disagreement = 1 - (max_agreement / total)
        
        return disagreement
    
    def _calculate_data_quality(self, context: Dict) -> float:
        """
        Calculate data quality.
        
        More complete/fresher data = higher quality.
        """
        quality = 1.0
        
        # Check for missing data
        required_keys = ["current_price", "volume", "volatility"]
        missing = sum(1 for k in required_keys if k not in context)
        quality -= missing * 0.2
        
        # Check data freshness
        timestamp = context.get("timestamp")
        if timestamp:
            try:
                data_time = datetime.fromisoformat(timestamp)
                age_seconds = (datetime.now() - data_time).total_seconds()
                
                if age_seconds > 60:
                    quality -= 0.1
                if age_seconds > 120:
                    quality -= 0.2
                if age_seconds > 300:
                    quality -= 0.3
            except:
                pass
        
        return max(0.1, quality)
    
    def _calculate_time_risk(self, context: Dict) -> float:
        """
        Calculate time risk.
        
        Near market window boundaries = higher risk.
        """
        market_info = context.get("market", {})
        
        if not market_info:
            return 0.3
        
        remaining_seconds = market_info.get("remaining_seconds", 900)
        
        # Risk is higher at start (first 30 seconds) and end (last 30 seconds)
        if remaining_seconds > 870:  # First 30 seconds
            return 0.3
        elif remaining_seconds < 30:  # Last 30 seconds
            return 0.5
        else:
            return 0.1  # Normal
    
    def _calculate_liquidity_risk(self, context: Dict) -> float:
        """
        Calculate liquidity risk.
        
        Lower liquidity = higher risk (more slippage).
        """
        volume = context.get("volume", context.get("volume_24h", 0))
        
        # Normalize based on typical BTC volume
        # Low volume = high risk
        if volume < 100:
            return 0.8
        elif volume < 500:
            return 0.5
        elif volume < 1000:
            return 0.3
        else:
            return 0.1
    
    def _calculate_news_risk(self, context: Dict) -> float:
        """
        Calculate news risk.
        
        Breaking news = higher uncertainty.
        """
        sentiment = context.get("sentiment", {})
        
        # Check for news events
        news_count = sentiment.get("news_count", 0)
        
        if news_count > 5:
            return 0.7  # Lots of news activity
        elif news_count > 2:
            return 0.4
        else:
            return 0.1
    
    def _generate_recommendation(
        self,
        adjusted_confidence: float,
        risk_factors: RiskFactors,
        overall_risk: float
    ) -> str:
        """Generate trading recommendation"""
        
        # High confidence, low risk
        if adjusted_confidence >= 0.7 and overall_risk <= 0.3:
            return "strong_trade"
        
        # Good confidence, moderate risk
        if adjusted_confidence >= 0.6 and overall_risk <= 0.5:
            return "trade"
        
        # Marginal
        if adjusted_confidence >= 0.5 and overall_risk <= 0.6:
            return "small_trade"
        
        # High risk situations
        if risk_factors.volatility_risk > 0.8:
            return "skip_high_volatility"
        
        if risk_factors.agent_disagreement > 0.7:
            return "skip_agent_conflict"
        
        if risk_factors.regime_uncertainty > 0.7:
            return "skip_regime_uncertain"
        
        if risk_factors.news_risk > 0.6:
            return "skip_news_uncertainty"
        
        return "skip"
    
    def record_confidence(
        self,
        original: float,
        adjusted: float,
        actual_outcome: bool
    ):
        """Record confidence for analysis"""
        
        self.confidence_history.append({
            "timestamp": datetime.now().isoformat(),
            "original_confidence": original,
            "adjusted_confidence": adjusted,
            "outcome": actual_outcome
        })
    
    def get_calibration_analysis(self) -> Dict[str, Any]:
        """
        Analyze confidence calibration.
        
        Well-calibrated predictions should have outcomes that match
        confidence levels (e.g., 70% confidence = 70% win rate).
        """
        if not self.confidence_history:
            return {"error": "No history"}
        
        # Group by confidence buckets
        buckets = {
            "0-20": [],
            "20-40": [],
            "40-60": [],
            "60-80": [],
            "80-100": []
        }
        
        for entry in self.confidence_history:
            conf = entry["adjusted_confidence"] * 100
            
            if conf < 20:
                buckets["0-20"].append(entry)
            elif conf < 40:
                buckets["20-40"].append(entry)
            elif conf < 60:
                buckets["40-60"].append(entry)
            elif conf < 80:
                buckets["60-80"].append(entry)
            else:
                buckets["80-100"].append(entry)
        
        # Calculate actual win rates
        calibration = {}
        for bucket_name, entries in buckets.items():
            if entries:
                wins = sum(1 for e in entries if e["outcome"])
                calibration[bucket_name] = {
                    "count": len(entries),
                    "expected_rate": sum(e["adjusted_confidence"] for e in entries) / len(entries),
                    "actual_rate": wins / len(entries)
                }
        
        return {
            "calibration": calibration,
            "total_predictions": len(self.confidence_history)
        }
