"""
Position Sizing for Atlas v4.0
Kelly Criterion and other position sizing methods

IMPLEMENTATION NOTES:
Based on academic paper "Application of the Kelly Criterion to Prediction Markets"
by Bernhard K. Meister (Dec 2024), the correct Kelly fraction for prediction markets is:

    f* = (Q - P) / (1 + Q)

Where:
    Q = q/(1-q) = belief odds (our predicted probability converted to odds)
    P = p/(1-p) = market odds (market price converted to odds)
    f* = optimal fraction of bankroll to bet

This differs from the standard Kelly formula because prediction markets have
binary outcomes with bounded payoffs (0 or 1).
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import math


@dataclass
class PositionSize:
    """Position sizing result"""
    size: float
    size_percent: float
    method: str
    kelly_fraction: float
    risk_amount: float
    confidence_adjusted: float
    # New fields for academic implementation
    belief_odds: float = 0.0
    market_odds: float = 0.0
    price_probability_gap: float = 0.0


class PositionSizer:
    """Base position sizer"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_percent: float = 10.0,
        min_position_percent: float = 0.5
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_percent = max_position_percent
        self.min_position_percent = min_position_percent
        
        # Track drawdown
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
    
    def update_capital(self, new_capital: float):
        """Update current capital and track drawdown"""
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)
        self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
    
    def calculate_size(self, probability: float, confidence: float) -> PositionSize:
        """Calculate position size"""
        raise NotImplementedError


class KellyPositionSizer(PositionSizer):
    """
    Kelly Criterion for optimal position sizing in Prediction Markets.
    
    IMPLEMENTATION BASED ON ACADEMIC PAPER:
    "Application of the Kelly Criterion to Prediction Markets" - Meister (2024)
    
    The correct Kelly formula for all-or-nothing contracts (like Polymarket) is:
    
        f* = (Q - P) / (1 + Q)
    
    Where:
        - Q = q/(1-q) is the belief odds (our probability converted to odds)
        - P = p/(1-p) is the market odds (market price converted to odds)
        - f* is the optimal fraction to invest
    
    Key insights from the paper:
    1. Market prices systematically diverge from true probabilities
    2. The gap depends on risk aversion and leverage asymmetry
    3. KL divergence measures the cost of prediction errors
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_percent: float = 10.0,
        min_position_percent: float = 0.5,
        kelly_fraction: float = 0.5,  # Half Kelly by default (paper recommends conservative)
        max_drawdown_percent: float = 20.0,
        min_edge: float = 0.02,
        # New parameters from academic paper
        price_probability_adjustment: float = 0.02,  # Systematic bias correction
        use_academic_formula: bool = True  # Use the academically correct formula
    ):
        super().__init__(initial_capital, max_position_percent, min_position_percent)
        self.kelly_fraction = kelly_fraction
        self.max_drawdown_percent = max_drawdown_percent
        self.min_edge = min_edge
        self.price_probability_adjustment = price_probability_adjustment
        self.use_academic_formula = use_academic_formula
        
        # Performance tracking
        self.trades_history: List[Dict] = []
        self.win_rate = 0.5
        self.avg_win = 1.0
        self.avg_loss = 1.0
        
        # Track price-probability relationships for calibration
        self.price_outcome_history: List[Dict] = []
        self.calibrated_adjustment = price_probability_adjustment
    
    def _prob_to_odds(self, prob: float) -> float:
        """
        Convert probability to odds.
        odds = p / (1 - p)
        
        From paper: If p=0.6, odds = 0.6/0.4 = 1.5
        """
        if prob <= 0.001:
            return 0.001  # Prevent division issues
        if prob >= 0.999:
            return 999.0  # Cap extreme odds
        return prob / (1 - prob)
    
    def _odds_to_prob(self, odds: float) -> float:
        """
        Convert odds back to probability.
        p = odds / (1 + odds)
        """
        return odds / (1 + odds)
    
    def _adjust_probability_for_bias(self, market_price: float, our_prob: float) -> float:
        """
        Adjust probability for systematic market bias.
        
        From academic paper: Market prices systematically diverge from
        true probabilities due to:
        1. Risk aversion of marginal investor
        2. Leverage asymmetry (UP vs DOWN have different payout structures)
        3. Information aggregation noise
        
        The paper shows the gap can range from 0 to 0.5 depending on conditions.
        We use a calibrated adjustment based on historical data.
        """
        # Apply calibrated adjustment
        # If market price is higher than 0.5, true probability tends to be lower
        # If market price is lower than 0.5, true probability tends to be higher
        bias_direction = -1 if market_price > 0.5 else 1
        bias_magnitude = abs(market_price - 0.5) * self.calibrated_adjustment * 2
        
        adjusted_prob = our_prob + (bias_direction * bias_magnitude)
        return max(0.05, min(0.95, adjusted_prob))
    
    def calculate_size(
        self,
        probability: float,
        confidence: float,
        market_odds: float = 0.5,
        direction: str = "UP"
    ) -> PositionSize:
        """
        Calculate optimal position size using academically correct Kelly Criterion.
        
        IMPLEMENTATION BASED ON MEISTER (2024):
        
        For all-or-nothing contracts:
        - If you pay price p for a UP contract and event occurs, you get $1
        - If event doesn't occur, you lose your investment
        
        The optimal Kelly fraction is:
            f* = (Q - P) / (1 + Q)
        
        Where Q = q/(1-q) and P = p/(1-p)
        
        Args:
            probability: Our predicted probability (0-1)
            confidence: Confidence in the prediction (0-1)
            market_odds: Current market price for UP (0-1)
            direction: "UP" or "DOWN"
        
        Returns:
            PositionSize with sizing details
        """
        # Get market price for the direction we're betting
        if direction == "UP":
            market_price = market_odds
            our_probability = probability
        else:
            market_price = 1.0 - market_odds  # DOWN price
            our_probability = 1.0 - probability  # Our prob of DOWN occurring
        
        # Adjust probability for systematic bias (from academic paper)
        adjusted_prob = self._adjust_probability_for_bias(market_price, our_probability)
        
        if self.use_academic_formula:
            # ACADEMICALLY CORRECT FORMULA from Meister (2024)
            # Q = belief odds, P = market odds
            Q = self._prob_to_odds(adjusted_prob)
            P = self._prob_to_odds(market_price)
            
            # Kelly fraction: f* = (Q - P) / (1 + Q)
            if Q > 0:
                kelly_raw = (Q - P) / (1 + Q)
            else:
                kelly_raw = 0
            
            # Store for diagnostics
            belief_odds = Q
            market_odds_converted = P
            price_prob_gap = abs(adjusted_prob - market_price)
        else:
            # Fallback to standard Kelly for comparison
            # Standard: f* = (bp - q) / b where b = odds
            win_payout = 1.0 - market_price
            loss_amount = market_price
            b = win_payout / loss_amount if loss_amount > 0 else 1.0
            
            p = adjusted_prob
            q = 1 - p
            kelly_raw = (p * b - q) / b if b > 0 else 0
            
            belief_odds = self._prob_to_odds(adjusted_prob)
            market_odds_converted = self._prob_to_odds(market_price)
            price_prob_gap = abs(adjusted_prob - market_price)
        
        # Only bet if we have positive expected value
        if kelly_raw <= 0:
            return PositionSize(
                size=0,
                size_percent=0,
                method="kelly_academic",
                kelly_fraction=0,
                risk_amount=0,
                confidence_adjusted=confidence,
                belief_odds=belief_odds,
                market_odds=market_odds_converted,
                price_probability_gap=price_prob_gap
            )
        
        # Apply fractional Kelly (paper recommends half-Kelly for safety)
        kelly_adjusted = kelly_raw * self.kelly_fraction
        
        # Apply confidence scaling
        kelly_confidence = kelly_adjusted * confidence
        
        # Check minimum edge (from paper: need meaningful probability gap)
        edge = abs(adjusted_prob - market_price)
        if edge < self.min_edge:
            kelly_confidence *= 0.5  # Reduce size for low edge
        
        # Drawdown adjustment
        drawdown_factor = self._calculate_drawdown_factor()
        kelly_final = kelly_confidence * drawdown_factor
        
        # Calculate actual position size
        position_size = kelly_final * self.current_capital
        
        # Apply limits
        max_size = self.current_capital * self.max_position_percent / 100
        min_size = self.current_capital * self.min_position_percent / 100
        
        if position_size > max_size:
            position_size = max_size
        elif position_size < min_size and kelly_final > 0:
            position_size = min_size
        elif kelly_final <= 0:
            position_size = 0
        
        size_percent = (position_size / self.current_capital) * 100
        risk_amount = position_size * market_price  # Max loss on this trade
        
        return PositionSize(
            size=position_size,
            size_percent=size_percent,
            method="kelly_academic",
            kelly_fraction=kelly_final,
            risk_amount=risk_amount,
            confidence_adjusted=confidence,
            belief_odds=belief_odds,
            market_odds=market_odds_converted,
            price_probability_gap=price_prob_gap
        )
    
    def calculate_size_with_belief_volatility(
        self,
        probability: float,
        confidence: float,
        market_odds: float,
        direction: str,
        belief_volatility: float = 0.0
    ) -> PositionSize:
        """
        Calculate position size with belief volatility adjustment.
        
        From "Toward Black-Scholes for Prediction Markets" (Dalen, 2025):
        Belief volatility measures how fast log-odds move over time.
        High belief volatility = unstable predictions = reduce position size.
        
        Args:
            probability: Predicted probability
            confidence: Prediction confidence
            market_odds: Current market price
            direction: "UP" or "DOWN"
            belief_volatility: Volatility of log-odds (0-1 scale)
        """
        # Get base position size
        base_size = self.calculate_size(probability, confidence, market_odds, direction)
        
        # Apply belief volatility adjustment
        if belief_volatility > 0:
            # High volatility = reduce position
            # From paper: sigma_b (belief volatility) should reduce confidence
            vol_adjustment = 1.0 - (belief_volatility * 0.5)  # Up to 50% reduction
            vol_adjustment = max(0.5, vol_adjustment)  # Floor at 50%
            
            adjusted_size = base_size.size * vol_adjustment
            adjusted_fraction = base_size.kelly_fraction * vol_adjustment
            
            return PositionSize(
                size=adjusted_size,
                size_percent=(adjusted_size / self.current_capital) * 100,
                method="kelly_academic_vol_adjusted",
                kelly_fraction=adjusted_fraction,
                risk_amount=adjusted_size * (market_odds if direction == "UP" else 1 - market_odds),
                confidence_adjusted=confidence * vol_adjustment,
                belief_odds=base_size.belief_odds,
                market_odds=base_size.market_odds,
                price_probability_gap=base_size.price_probability_gap
            )
        
        return base_size
    
    def calculate_size_with_history(
        self,
        probability: float,
        confidence: float,
        market_odds: float = 0.5,
        direction: str = "UP"
    ) -> PositionSize:
        """
        Calculate size using historical performance data.
        
        Uses actual win rate and average win/loss from history.
        Also updates the price-probability calibration.
        """
        # Update calibration from history
        self._update_price_probability_calibration()
        
        # Use historical data if available
        if len(self.trades_history) >= 10:
            actual_p = self.win_rate
            
            # Blend predicted and historical (paper suggests being conservative)
            blended_p = 0.6 * probability + 0.4 * actual_p
        else:
            blended_p = probability
        
        return self.calculate_size(blended_p, confidence, market_odds, direction)
    
    def _update_price_probability_calibration(self):
        """
        Update the price-probability adjustment based on historical data.
        
        From academic paper: The gap between market prices and actual
        outcomes can be measured and used for calibration.
        """
        if len(self.price_outcome_history) < 10:
            return
        
        # Calculate average gap between market prices and outcomes
        gaps = []
        for record in self.price_outcome_history[-50:]:  # Last 50 records
            market_price = record["market_price"]
            outcome = record["outcome"]  # 1 if UP happened, 0 if DOWN
            
            # The "correct" price would have been the outcome
            # Gap = how much the market was off
            gap = abs(market_price - outcome)
            gaps.append(gap)
        
        if gaps:
            # Update calibrated adjustment
            avg_gap = sum(gaps) / len(gaps)
            # Smooth update
            self.calibrated_adjustment = 0.8 * self.calibrated_adjustment + 0.2 * avg_gap
    
    def _calculate_drawdown_factor(self) -> float:
        """Calculate position size reduction based on drawdown"""
        
        if self.current_drawdown <= 0:
            return 1.0
        
        if self.current_drawdown >= self.max_drawdown_percent / 100:
            return 0.0  # Stop trading at max drawdown
        
        # Linear reduction from 1.0 to 0.5 as drawdown increases
        factor = 1.0 - (self.current_drawdown / (self.max_drawdown_percent / 100)) * 0.5
        
        return max(0.5, factor)
    
    def record_trade(
        self,
        outcome: bool,
        pnl: float,
        probability: float,
        direction: str,
        market_price: float = 0.5
    ):
        """Record trade outcome for historical tracking and calibration"""
        
        self.trades_history.append({
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "pnl": pnl,
            "probability": probability,
            "direction": direction,
            "market_price": market_price
        })
        
        # Record for price-probability calibration
        self.price_outcome_history.append({
            "timestamp": datetime.now().isoformat(),
            "market_price": market_price if direction == "UP" else 1 - market_price,
            "outcome": 1 if outcome else 0
        })
        
        # Update capital
        self.update_capital(self.current_capital + pnl)
        
        # Recalculate stats
        self._update_stats()
    
    def _update_stats(self):
        """Update historical statistics"""
        
        if not self.trades_history:
            return
        
        wins = [t for t in self.trades_history if t["outcome"]]
        losses = [t for t in self.trades_history if not t["outcome"]]
        
        self.win_rate = len(wins) / len(self.trades_history)
        
        if wins:
            self.avg_win = sum(t["pnl"] for t in wins) / len(wins)
        else:
            self.avg_win = 1.0
        
        if losses:
            self.avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
        else:
            self.avg_loss = 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get position sizer statistics"""
        
        return {
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": self.current_drawdown * 100,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "kelly_fraction_setting": self.kelly_fraction,
            "total_trades": len(self.trades_history),
            "calibrated_adjustment": self.calibrated_adjustment,
            "use_academic_formula": self.use_academic_formula
        }


class FixedPositionSizer(PositionSizer):
    """Fixed percentage position sizer"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_percent: float = 2.0
    ):
        super().__init__(initial_capital)
        self.position_percent = position_percent
    
    def calculate_size(
        self,
        probability: float,
        confidence: float
    ) -> PositionSize:
        """Calculate fixed percentage position size"""
        
        size = self.current_capital * self.position_percent / 100
        
        return PositionSize(
            size=size,
            size_percent=self.position_percent,
            method="fixed",
            kelly_fraction=self.position_percent / 100,
            risk_amount=size,
            confidence_adjusted=confidence
        )


class VolatilityAdjustedSizer(PositionSizer):
    """
    Volatility-adjusted position sizer.
    
    Reduces position size in high volatility conditions.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        base_percent: float = 2.0,
        target_volatility: float = 0.5,  # Target daily volatility %
        max_volatility: float = 2.0  # Max allowed volatility
    ):
        super().__init__(initial_capital)
        self.base_percent = base_percent
        self.target_volatility = target_volatility
        self.max_volatility = max_volatility
    
    def calculate_size(
        self,
        probability: float,
        confidence: float,
        current_volatility: float = 0.5
    ) -> PositionSize:
        """Calculate volatility-adjusted position size"""
        
        # Volatility scaling factor
        if current_volatility > 0:
            vol_factor = self.target_volatility / current_volatility
        else:
            vol_factor = 1.0
        
        # Cap the factor
        vol_factor = max(0.25, min(2.0, vol_factor))
        
        # If volatility too high, reduce further
        if current_volatility > self.max_volatility:
            vol_factor *= 0.5
        
        # Calculate adjusted size
        adjusted_percent = self.base_percent * vol_factor * confidence
        
        # Apply limits
        adjusted_percent = max(0.5, min(10.0, adjusted_percent))
        
        size = self.current_capital * adjusted_percent / 100
        
        return PositionSize(
            size=size,
            size_percent=adjusted_percent,
            method="volatility_adjusted",
            kelly_fraction=adjusted_percent / 100,
            risk_amount=size,
            confidence_adjusted=confidence
        )
