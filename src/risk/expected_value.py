"""
Expected Value Calculator for Atlas v4.0
Calculates expected value of predictions
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExpectedValueResult:
    """Expected value calculation result"""
    expected_value: float
    expected_return: float
    risk_reward_ratio: float
    edge: float
    is_positive_ev: bool
    recommendation: str
    details: Dict[str, Any]


# Alias for backwards compatibility
EVResult = ExpectedValueResult


class ExpectedValueCalculator:
    """
    Calculates expected value of predictions.
    
    Factors:
    - Prediction probability
    - Market odds (from Polymarket)
    - Slippage estimate
    - Fee impact
    
    Outputs:
    - Expected value (positive = good bet)
    - Risk-reward ratio
    - Confidence interval
    """
    
    # Typical Polymarket fees
    TRADING_FEE = 0.02  # 2% (approximate)
    SLIPPAGE_ESTIMATE = 0.005  # 0.5% average slippage
    
    def __init__(
        self,
        trading_fee: float = 0.02,
        slippage: float = 0.005,
        min_ev: float = 0.02,  # Minimum 2% EV to recommend
        min_edge: float = 0.03  # Minimum 3% edge to recommend
    ):
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.min_ev = min_ev
        self.min_edge = min_edge
        
        # History tracking
        self.ev_history: List[Dict] = []
        self.actual_results: List[Dict] = []
    
    def calculate_ev(
        self,
        predicted_prob: float,
        market_odds: float,
        confidence: float = 1.0,
        direction: str = "UP"
    ) -> EVResult:
        """
        Calculate expected value of a bet.
        
        Args:
            predicted_prob: Model's predicted probability
            market_odds: Current market price (e.g., 0.55 for 55%)
            confidence: Confidence in prediction (0-1)
            direction: "UP" or "DOWN"
        
        Returns:
            EVResult with EV calculation and recommendation
        """
        # Determine actual buy price
        if direction == "UP":
            buy_price = market_odds
            win_payout = 1.0
        else:
            buy_price = 1.0 - market_odds
            win_payout = 1.0
        
        # Calculate costs
        total_cost = buy_price * (1 + self.trading_fee + self.slippage)
        
        # Net payout if win
        net_win = win_payout - total_cost
        
        # Net loss if lose
        net_loss = -total_cost
        
        # Expected value calculation
        # EV = P(win) * net_win + P(lose) * net_loss
        ev = predicted_prob * net_win + (1 - predicted_prob) * net_loss
        
        # Expected return as percentage
        expected_return = ev / total_cost if total_cost > 0 else 0
        
        # Edge calculation
        if direction == "UP":
            edge = predicted_prob - market_odds
        else:
            edge = (1 - predicted_prob) - (1 - market_odds)
        
        # Risk-reward ratio
        if net_loss != 0:
            risk_reward = abs(net_win / net_loss)
        else:
            risk_reward = float('inf')
        
        # Determine if positive EV
        is_positive_ev = ev > 0
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            ev, edge, confidence, risk_reward
        )
        
        return EVResult(
            expected_value=ev,
            expected_return=expected_return,
            risk_reward_ratio=risk_reward,
            edge=edge,
            is_positive_ev=is_positive_ev,
            recommendation=recommendation,
            details={
                "buy_price": buy_price,
                "total_cost": total_cost,
                "net_win": net_win,
                "net_loss": net_loss,
                "predicted_prob": predicted_prob,
                "market_odds": market_odds,
                "direction": direction,
                "confidence": confidence
            }
        )
    
    def _generate_recommendation(
        self,
        ev: float,
        edge: float,
        confidence: float,
        risk_reward: float
    ) -> str:
        """Generate betting recommendation"""
        
        if ev <= 0:
            return "skip"  # Negative EV
        
        if edge < self.min_edge:
            return "skip"  # Not enough edge
        
        if ev < self.min_ev:
            return "skip"  # EV too small
        
        if confidence < 0.5:
            return "skip"  # Low confidence
        
        # Determine strength
        if ev > 0.10 and edge > 0.10 and confidence > 0.8:
            return "strong_buy"
        elif ev > 0.05 and edge > 0.05 and confidence > 0.6:
            return "buy"
        else:
            return "small_bet"
    
    def should_bet(
        self,
        ev_result: EVResult,
        min_ev: float = None,
        min_edge: float = None
    ) -> bool:
        """
        Decision: should we make this bet?
        
        Args:
            ev_result: EV calculation result
            min_ev: Minimum EV threshold (uses default if not specified)
            min_edge: Minimum edge threshold (uses default if not specified)
        
        Returns:
            True if bet is recommended
        """
        min_ev = min_ev or self.min_ev
        min_edge = min_edge or self.min_edge
        
        return (
            ev_result.is_positive_ev and
            ev_result.expected_value >= min_ev and
            ev_result.edge >= min_edge
        )
    
    def calculate_multi_outcome_ev(
        self,
        predictions: Dict[str, float],
        market_prices: Dict[str, float]
    ) -> Dict[str, EVResult]:
        """
        Calculate EV for multiple outcomes.
        
        Useful when comparing UP vs DOWN bets.
        """
        results = {}
        
        for outcome, prob in predictions.items():
            market_price = market_prices.get(outcome, 0.5)
            
            ev_result = self.calculate_ev(
                predicted_prob=prob,
                market_odds=market_price,
                direction=outcome
            )
            
            results[outcome] = ev_result
        
        return results
    
    def get_best_bet(
        self,
        predictions: Dict[str, float],
        market_prices: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best bet option.
        
        Returns the outcome with highest positive EV.
        """
        results = self.calculate_multi_outcome_ev(predictions, market_prices)
        
        # Filter positive EV
        positive_ev = {
            k: v for k, v in results.items()
            if v.is_positive_ev
        }
        
        if not positive_ev:
            return None
        
        # Get highest EV
        best = max(positive_ev.items(), key=lambda x: x[1].expected_value)
        
        return {
            "direction": best[0],
            "ev": best[1].expected_value,
            "recommendation": best[1].recommendation,
            "details": best[1].details
        }
    
    def record_ev_prediction(
        self,
        prediction: Dict[str, Any],
        ev_result: EVResult
    ):
        """Record EV prediction for later analysis"""
        
        self.ev_history.append({
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "ev": ev_result.expected_value,
            "edge": ev_result.edge,
            "recommendation": ev_result.recommendation,
            "outcome": None  # To be updated later
        })
    
    def record_actual_outcome(
        self,
        prediction_id: str,
        actual_outcome: bool
    ):
        """Record actual outcome of a prediction"""
        
        for entry in self.ev_history:
            if entry["prediction"].get("id") == prediction_id:
                entry["outcome"] = actual_outcome
                break
    
    def get_ev_accuracy(self) -> Dict[str, Any]:
        """
        Analyze how well EV predictions correlated with outcomes.
        
        Higher EV should correlate with higher win rate.
        """
        if not self.ev_history:
            return {"error": "No history"}
        
        # Group by EV ranges
        ev_ranges = {
            "negative": [],
            "0_to_2": [],
            "2_to_5": [],
            "5_to_10": [],
            "above_10": []
        }
        
        for entry in self.ev_history:
            if entry["outcome"] is None:
                continue
            
            ev = entry["ev"] * 100  # Convert to percentage
            
            if ev < 0:
                ev_ranges["negative"].append(entry)
            elif ev < 2:
                ev_ranges["0_to_2"].append(entry)
            elif ev < 5:
                ev_ranges["2_to_5"].append(entry)
            elif ev < 10:
                ev_ranges["5_to_10"].append(entry)
            else:
                ev_ranges["above_10"].append(entry)
        
        # Calculate win rates
        range_stats = {}
        for range_name, entries in ev_ranges.items():
            if entries:
                wins = sum(1 for e in entries if e["outcome"])
                range_stats[range_name] = {
                    "count": len(entries),
                    "win_rate": wins / len(entries),
                    "avg_ev": sum(e["ev"] for e in entries) / len(entries)
                }
        
        # Overall correlation
        all_with_outcomes = [
            e for e in self.ev_history
            if e["outcome"] is not None
        ]
        
        if all_with_outcomes:
            outcomes = [1.0 if e["outcome"] else 0.0 for e in all_with_outcomes]
            evs = [e["ev"] for e in all_with_outcomes]
            
            # Simple correlation (would use numpy for real implementation)
            mean_outcome = sum(outcomes) / len(outcomes)
            mean_ev = sum(evs) / len(evs)
            
            numerator = sum(
                (o - mean_outcome) * (e - mean_ev)
                for o, e in zip(outcomes, evs)
            )
            
            denom_outcome = sum((o - mean_outcome) ** 2 for o in outcomes) ** 0.5
            denom_ev = sum((e - mean_ev) ** 2 for e in evs) ** 0.5
            
            if denom_outcome > 0 and denom_ev > 0:
                correlation = numerator / (denom_outcome * denom_ev)
            else:
                correlation = 0
        else:
            correlation = 0
        
        return {
            "range_stats": range_stats,
            "ev_outcome_correlation": correlation,
            "total_predictions": len(self.ev_history),
            "resolved_predictions": len(all_with_outcomes)
        }
