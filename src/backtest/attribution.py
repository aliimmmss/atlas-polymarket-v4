"""
Performance Attribution for Atlas v4.0
Analyzes what drives performance
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class AttributionReport:
    """Performance attribution report"""
    total_return: float
    agent_contributions: Dict[str, float]
    signal_contributions: Dict[str, float]
    regime_contributions: Dict[str, float]
    time_contributions: Dict[str, float]
    improvement_suggestions: List[str]
    detailed_analysis: Dict[str, Any]


class PerformanceAttribution:
    """
    Attributes performance to specific factors.
    
    Attribution Levels:
    - Agent level: Which agents contributed most?
    - Signal level: Which signals were predictive?
    - Regime level: Performance by market regime
    - Time level: Performance by time of day/week
    """
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.agent_history: Dict[str, List[Dict]] = {}
        self.signal_history: Dict[str, List[Dict]] = {}
    
    def add_trades(self, trades: List[Any]):
        """Add trades for analysis"""
        for trade in trades:
            trade_dict = self._trade_to_dict(trade)
            self.trades.append(trade_dict)
            
            # Track by agent
            for agent_id, vote in trade_dict.get("agent_votes", {}).items():
                if agent_id not in self.agent_history:
                    self.agent_history[agent_id] = []
                self.agent_history[agent_id].append({
                    "timestamp": trade_dict["entry_time"],
                    "direction": vote,
                    "correct": trade_dict["outcome"]
                })
    
    def _trade_to_dict(self, trade) -> Dict:
        """Convert trade to dictionary"""
        if hasattr(trade, '__dict__'):
            return {
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "direction": trade.direction,
                "outcome": trade.outcome,
                "pnl": trade.pnl,
                "pnl_percent": trade.pnl_percent,
                "confidence": trade.confidence,
                "agent_votes": trade.agent_votes if hasattr(trade, 'agent_votes') else {}
            }
        return trade
    
    def attribute_by_agent(self) -> Dict[str, Dict]:
        """
        Attribute returns by agent.
        
        Returns contribution of each agent to overall performance.
        """
        if not self.trades:
            return {}
        
        agent_stats = {}
        
        for agent_id, history in self.agent_history.items():
            if not history:
                continue
            
            correct_votes = sum(1 for h in history if h["correct"])
            total_votes = len(history)
            
            # Calculate win rate when agent's vote was followed
            followed_correct = 0
            followed_total = 0
            
            for trade in self.trades:
                if agent_id in trade.get("agent_votes", {}):
                    vote = trade["agent_votes"][agent_id]
                    if vote == trade["direction"]:
                        followed_total += 1
                        if trade["outcome"]:
                            followed_correct += 1
            
            # Contribution score
            # Higher weight for agents whose votes were followed and correct
            if followed_total > 0:
                contribution = followed_correct / followed_total
            else:
                contribution = 0.5
            
            agent_stats[agent_id] = {
                "total_votes": total_votes,
                "correct_votes": correct_votes,
                "vote_accuracy": correct_votes / total_votes if total_votes > 0 else 0,
                "followed_votes": followed_total,
                "followed_correct": followed_correct,
                "follow_accuracy": followed_correct / followed_total if followed_total > 0 else 0,
                "contribution_score": contribution
            }
        
        return agent_stats
    
    def attribute_by_signal(self, trades: List[Dict] = None) -> Dict[str, Any]:
        """
        Attribute returns by signal type.
        
        Analyzes which technical/fundamental signals were most predictive.
        """
        trades = trades or self.trades
        
        if not trades:
            return {}
        
        signal_stats = {
            "rsi": {"correct": 0, "total": 0},
            "macd": {"correct": 0, "total": 0},
            "momentum": {"correct": 0, "total": 0},
            "volatility": {"correct": 0, "total": 0},
            "support_resistance": {"correct": 0, "total": 0},
            "funding_rate": {"correct": 0, "total": 0},
            "open_interest": {"correct": 0, "total": 0},
            "sentiment": {"correct": 0, "total": 0}
        }
        
        # This is a simplified version
        # In reality, would track signal direction at trade time
        
        return {
            "signal_stats": signal_stats,
            "best_signals": [],
            "worst_signals": []
        }
    
    def attribute_by_regime(self, trades: List[Dict] = None) -> Dict[str, Dict]:
        """
        Attribute returns by market regime.
        
        Analyzes performance in different market conditions.
        """
        trades = trades or self.trades
        
        regime_stats = {
            "TRENDING_UP": {"trades": 0, "correct": 0, "pnl": 0},
            "TRENDING_DOWN": {"trades": 0, "correct": 0, "pnl": 0},
            "RANGING": {"trades": 0, "correct": 0, "pnl": 0},
            "VOLATILE": {"trades": 0, "correct": 0, "pnl": 0},
            "BREAKOUT": {"trades": 0, "correct": 0, "pnl": 0},
            "REVERSAL": {"trades": 0, "correct": 0, "pnl": 0},
            "UNKNOWN": {"trades": 0, "correct": 0, "pnl": 0}
        }
        
        for trade in trades:
            # Would need regime label on each trade
            regime = trade.get("regime", "UNKNOWN")
            if regime not in regime_stats:
                regime = "UNKNOWN"
            
            regime_stats[regime]["trades"] += 1
            if trade.get("outcome"):
                regime_stats[regime]["correct"] += 1
            regime_stats[regime]["pnl"] += trade.get("pnl", 0)
        
        # Calculate win rates
        for regime, stats in regime_stats.items():
            if stats["trades"] > 0:
                stats["win_rate"] = stats["correct"] / stats["trades"]
                stats["avg_pnl"] = stats["pnl"] / stats["trades"]
            else:
                stats["win_rate"] = 0
                stats["avg_pnl"] = 0
        
        return regime_stats
    
    def attribute_by_time(self, trades: List[Dict] = None) -> Dict[str, Dict]:
        """
        Attribute returns by time.
        
        Analyzes performance by:
        - Hour of day
        - Day of week
        - Market session (Asian, European, US)
        """
        trades = trades or self.trades
        
        if not trades:
            return {}
        
        hour_stats = {str(h): {"trades": 0, "correct": 0} for h in range(24)}
        day_stats = {str(d): {"trades": 0, "correct": 0} for d in range(7)}
        session_stats = {
            "asian": {"trades": 0, "correct": 0},  # 00:00-08:00 UTC
            "european": {"trades": 0, "correct": 0},  # 08:00-16:00 UTC
            "us": {"trades": 0, "correct": 0}  # 16:00-00:00 UTC
        }
        
        for trade in trades:
            ts = trade.get("entry_time", 0)
            if ts == 0:
                continue
            
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            day = dt.weekday()
            
            # Hour stats
            hour_stats[str(hour)]["trades"] += 1
            if trade.get("outcome"):
                hour_stats[str(hour)]["correct"] += 1
            
            # Day stats
            day_stats[str(day)]["trades"] += 1
            if trade.get("outcome"):
                day_stats[str(day)]["correct"] += 1
            
            # Session stats
            if 0 <= hour < 8:
                session = "asian"
            elif 8 <= hour < 16:
                session = "european"
            else:
                session = "us"
            
            session_stats[session]["trades"] += 1
            if trade.get("outcome"):
                session_stats[session]["correct"] += 1
        
        # Calculate win rates
        for stats in [hour_stats, day_stats, session_stats]:
            for key, s in stats.items():
                if s["trades"] > 0:
                    s["win_rate"] = s["correct"] / s["trades"]
                else:
                    s["win_rate"] = 0
        
        return {
            "by_hour": hour_stats,
            "by_day": day_stats,
            "by_session": session_stats
        }
    
    def generate_report(self, trades: List[Any] = None) -> AttributionReport:
        """
        Generate comprehensive attribution report.
        """
        if trades:
            self.add_trades(trades)
        
        # Calculate total return
        total_pnl = sum(t.get("pnl", 0) for t in self.trades)
        
        # Get all attributions
        agent_attr = self.attribute_by_agent()
        signal_attr = self.attribute_by_signal()
        regime_attr = self.attribute_by_regime()
        time_attr = self.attribute_by_time()
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            agent_attr, signal_attr, regime_attr, time_attr
        )
        
        return AttributionReport(
            total_return=total_pnl,
            agent_contributions=agent_attr,
            signal_contributions=signal_attr,
            regime_contributions=regime_attr,
            time_contributions=time_attr,
            improvement_suggestions=suggestions,
            detailed_analysis={
                "trade_count": len(self.trades),
                "agent_count": len(self.agent_history)
            }
        )
    
    def _generate_suggestions(
        self,
        agent_attr: Dict,
        signal_attr: Dict,
        regime_attr: Dict,
        time_attr: Dict
    ) -> List[str]:
        """Generate improvement suggestions"""
        
        suggestions = []
        
        # Agent suggestions
        if agent_attr:
            # Find underperforming agents
            for agent_id, stats in agent_attr.items():
                if stats.get("follow_accuracy", 0.5) < 0.4 and stats.get("followed_votes", 0) > 5:
                    suggestions.append(
                        f"Consider reducing weight of {agent_id} - accuracy below 40%"
                    )
            
            # Find overperforming agents
            for agent_id, stats in agent_attr.items():
                if stats.get("follow_accuracy", 0.5) > 0.6 and stats.get("followed_votes", 0) > 5:
                    suggestions.append(
                        f"Increase weight of {agent_id} - accuracy above 60%"
                    )
        
        # Regime suggestions
        if regime_attr:
            poor_regimes = [
                regime for regime, stats in regime_attr.items()
                if stats.get("win_rate", 0) < 0.4 and stats.get("trades", 0) > 5
            ]
            
            if poor_regimes:
                suggestions.append(
                    f"Consider avoiding trades in: {', '.join(poor_regimes)} regimes"
                )
        
        # Time suggestions
        if time_attr:
            by_hour = time_attr.get("by_hour", {})
            
            bad_hours = [
                hour for hour, stats in by_hour.items()
                if stats.get("win_rate", 0) < 0.4 and stats.get("trades", 0) > 3
            ]
            
            if bad_hours:
                suggestions.append(
                    f"Poor performance during hours: {', '.join(bad_hours)} UTC"
                )
        
        if not suggestions:
            suggestions.append("Performance is balanced across all factors")
        
        return suggestions
    
    def get_agent_ranking(self) -> List[Dict]:
        """Get agents ranked by contribution"""
        
        agent_attr = self.attribute_by_agent()
        
        ranking = []
        for agent_id, stats in agent_attr.items():
            ranking.append({
                "agent_id": agent_id,
                "accuracy": stats.get("follow_accuracy", 0),
                "trades": stats.get("followed_votes", 0),
                "score": stats.get("contribution_score", 0)
            })
        
        ranking.sort(key=lambda x: x["score"], reverse=True)
        
        return ranking
    
    def get_optimal_weights(self) -> Dict[str, float]:
        """
        Calculate optimal agent weights based on historical performance.
        
        Uses win rate and trade count to determine weights.
        """
        ranking = self.get_agent_ranking()
        
        if not ranking:
            return {}
        
        weights = {}
        total_score = sum(r["score"] for r in ranking)
        
        for agent in ranking:
            if total_score > 0:
                weight = agent["score"] / total_score * 2  # Scale to ~1.0 average
            else:
                weight = 1.0
            
            # Clamp weights
            weights[agent["agent_id"]] = max(0.1, min(3.0, weight))
        
        return weights
