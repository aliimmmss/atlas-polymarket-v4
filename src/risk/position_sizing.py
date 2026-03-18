"""
Position Sizing for Atlas v4.0
Kelly Criterion and other position sizing methods
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
    Kelly Criterion for optimal position sizing.
    
    Kelly formula: f* = (p * b - q) / b
    where:
    - f* = fraction of bankroll to bet
    - p = probability of winning
    - q = probability of losing (1 - p)
    - b = odds received (win amount / lose amount)
    
    Features:
    - Half Kelly for reduced variance
    - Fractional Kelly for risk adjustment
    - Maximum drawdown constraints
    - Edge requirements
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_percent: float = 10.0,
        min_position_percent: float = 0.5,
        kelly_fraction: float = 0.5,  # Half Kelly by default
        max_drawdown_percent: float = 20.0,
        min_edge: float = 0.02
    ):
        super().__init__(initial_capital, max_position_percent, min_position_percent)
        self.kelly_fraction = kelly_fraction
        self.max_drawdown_percent = max_drawdown_percent
        self.min_edge = min_edge
        
        # Performance tracking
        self.trades_history: List[Dict] = []
        self.win_rate = 0.5
        self.avg_win = 1.0
        self.avg_loss = 1.0
    
    def calculate_size(
        self,
        probability: float,
        confidence: float,
        market_odds: float = 0.5,
        direction: str = "UP"
    ) -> PositionSize:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            probability: Predicted probability of winning
            confidence: Confidence in the prediction
            market_odds: Current market odds (e.g., 0.55 for UP at 55%)
            direction: "UP" or "DOWN"
        
        Returns:
            PositionSize with sizing details
        """
        # Calculate actual odds from market
        if direction == "UP":
            buy_price = market_odds
            win_payout = 1.0 - buy_price  # If win, get 1.0, paid buy_price
            loss_amount = buy_price  # If lose, lose buy_price
        else:
            buy_price = 1.0 - market_odds  # DOWN price
            win_payout = market_odds
            loss_amount = buy_price
        
        # Odds (b) = win_payout / loss_amount
        b = win_payout / loss_amount if loss_amount > 0 else 1.0
        
        # Kelly formula
        p = probability
        q = 1 - p
        
        kelly_raw = (p * b - q) / b if b > 0 else 0
        
        # Apply Kelly fraction (e.g., half Kelly)
        kelly_adjusted = kelly_raw * self.kelly_fraction
        
        # Apply confidence scaling
        kelly_confidence = kelly_adjusted * confidence
        
        # Check minimum edge
        edge = abs(probability - market_odds) if direction == "UP" else abs((1 - probability) - market_odds)
        
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
        risk_amount = position_size * loss_amount  # Max loss on this trade
        
        return PositionSize(
            size=position_size,
            size_percent=size_percent,
            method="kelly",
            kelly_fraction=kelly_final,
            risk_amount=risk_amount,
            confidence_adjusted=confidence
        )
    
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
        """
        # Use historical data if available
        if len(self.trades_history) >= 10:
            actual_p = self.win_rate
            actual_b = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0
            
            # Blend predicted and historical
            blended_p = 0.7 * probability + 0.3 * actual_p
        else:
            blended_p = probability
            actual_b = 1.0
        
        return self.calculate_size(blended_p, confidence, market_odds, direction)
    
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
        direction: str
    ):
        """Record trade outcome for historical tracking"""
        
        self.trades_history.append({
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "pnl": pnl,
            "probability": probability,
            "direction": direction
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
            "total_trades": len(self.trades_history)
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
