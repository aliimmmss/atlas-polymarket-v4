"""
Meta-Agent for Atlas v4.0
Selects and weights agents based on market conditions
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AgentSelection:
    """Result of meta-agent selection"""
    active_agents: List[str]
    weights: Dict[str, float]
    voting_method: str
    reasoning: str
    confidence: float


class MetaAgent:
    """
    Selects and weights agents based on market conditions.
    
    Inputs:
    - Current market regime
    - Recent agent performance
    - Market volatility
    - Time of day
    - Signal agreement
    
    Outputs:
    - Active agent list
    - Agent weights
    - Voting method selection
    """
    
    # Performance tracking
    REGIME_PERFORMANCE = {
        "TRENDING_UP": {"trend_rider": 1.2, "momentum_hawk": 1.1, "trend_fader": 0.8},
        "TRENDING_DOWN": {"trend_fader": 1.2, "momentum_hawk": 1.1, "trend_rider": 0.8},
        "RANGING": {"range_trader": 1.3, "breakout_hunter": 1.0, "mean_reverter": 1.1},
        "VOLATILE": {"risk_guard": 1.4, "volatility_harvester": 1.1, "mean_reverter": 0.9},
        "BREAKOUT": {"breakout_hunter": 1.3, "momentum_hawk": 1.1, "volatility_harvester": 1.0},
        "REVERSAL": {"mean_reverter": 1.3, "rsi_master": 1.2, "sentiment_surfer": 1.0},
        "CONSOLIDATION": {"breakout_hunter": 1.2, "range_trader": 1.1},
        "EXHAUSTION": {"mean_reverter": 1.3, "trend_fader": 1.2, "risk_guard": 1.1},
    }
    
    # Voting method by regime
    VOTING_METHOD_MAP = {
        "TRENDING_UP": "weighted_average",
        "TRENDING_DOWN": "weighted_average",
        "RANGING": "confidence_weighted",
        "VOLATILE": "consensus_requiring",
        "BREAKOUT": "confidence_weighted",
        "REVERSAL": "bayesian_averaging",
        "CONSOLIDATION": "median_aggregation",
        "EXHAUSTION": "consensus_requiring",
    }
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.selection_history: List[Dict] = []
        self.agent_performance_history: Dict[str, List[Dict]] = {}
    
    async def select_agents(
        self,
        market_context: Dict[str, Any],
        agent_history: Dict[str, Any],
        all_agents: List[Any]
    ) -> AgentSelection:
        """
        Select which agents should be active.
        
        Args:
            market_context: Current market data
            agent_history: Historical agent performance
            all_agents: All available agents
        
        Returns:
            AgentSelection with active agents and weights
        """
        # Get current regime
        regime = market_context.get("regime", {}).get("regime", "RANGING")
        
        # Get regime-specific performance adjustments
        regime_weights = self.REGIME_PERFORMANCE.get(regime, {})
        
        # Get all agent IDs
        all_agent_ids = [a.agent_id for a in all_agents]
        
        # Select active agents based on regime
        active_agents = []
        weights = {}
        
        for agent in all_agents:
            agent_id = agent.agent_id
            agent_regime = getattr(agent, 'regime', 'ALL')
            
            # Include if regime matches or agent is universal
            if agent_regime == "ALL" or agent_regime.upper() == regime.upper():
                active_agents.append(agent_id)
                
                # Calculate weight
                base_weight = agent.weight
                
                # Adjust for regime performance
                regime_adj = regime_weights.get(agent_id, 1.0)
                
                # Adjust for recent performance
                recent_perf = self._get_recent_performance(agent_id, agent_history)
                
                # Final weight
                final_weight = base_weight * regime_adj * recent_perf
                weights[agent_id] = max(0.1, min(5.0, final_weight))
        
        # Select voting method
        voting_method = self.VOTING_METHOD_MAP.get(regime, "weighted_average")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(regime, active_agents, weights)
        
        # Calculate confidence
        confidence = self._calculate_selection_confidence(
            regime, active_agents, agent_history
        )
        
        selection = AgentSelection(
            active_agents=active_agents,
            weights=weights,
            voting_method=voting_method,
            reasoning=reasoning,
            confidence=confidence
        )
        
        # Record selection
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "active_agents": active_agents,
            "voting_method": voting_method
        })
        
        return selection
    
    def _get_recent_performance(
        self,
        agent_id: str,
        agent_history: Dict[str, Any]
    ) -> float:
        """Get recent performance multiplier for an agent"""
        
        if agent_id not in agent_history:
            return 1.0
        
        history = agent_history[agent_id]
        
        if not history:
            return 1.0
        
        # Calculate recent win rate
        recent = history[-20:]  # Last 20 predictions
        correct = sum(1 for h in recent if h.get("correct", False))
        
        if len(recent) < 3:
            return 1.0
        
        win_rate = correct / len(recent)
        
        # Convert to performance multiplier
        # 50% win rate = 1.0, 60% = 1.2, 70% = 1.4, etc.
        if win_rate >= 0.6:
            return 1.0 + (win_rate - 0.5) * 2
        elif win_rate <= 0.4:
            return 0.6 + win_rate
        else:
            return 1.0
    
    def _generate_reasoning(
        self,
        regime: str,
        active_agents: List[str],
        weights: Dict[str, float]
    ) -> str:
        """Generate reasoning for agent selection"""
        
        top_agents = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_names = [a[0].replace("_", " ").title() for a in top_agents]
        
        reasoning = f"Market regime: {regime}. "
        reasoning += f"Selected {len(active_agents)} agents for this regime. "
        reasoning += f"Top weighted: {', '.join(top_names)}."
        
        return reasoning
    
    def _calculate_selection_confidence(
        self,
        regime: str,
        active_agents: List[str],
        agent_history: Dict[str, Any]
    ) -> float:
        """Calculate confidence in agent selection"""
        
        if not active_agents:
            return 0.3
        
        # Count agents with history
        agents_with_history = sum(
            1 for a in active_agents 
            if a in agent_history and len(agent_history[a]) >= 5
        )
        
        # Base confidence on historical data availability
        data_confidence = agents_with_history / len(active_agents)
        
        # Boost for regime-specific agents
        regime_specific = sum(
            1 for a in active_agents 
            if a in self.REGIME_PERFORMANCE.get(regime, {})
        )
        regime_confidence = regime_specific / len(active_agents) if active_agents else 0
        
        return (data_confidence * 0.6 + regime_confidence * 0.4)
    
    def adjust_weights_dynamically(
        self,
        base_weights: Dict[str, float],
        recent_performance: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Dynamically adjust weights based on recent performance.
        
        Args:
            base_weights: Base weights from selection
            recent_performance: Recent performance by agent
        
        Returns:
            Adjusted weights
        """
        adjusted = {}
        
        for agent_id, weight in base_weights.items():
            perf = recent_performance.get(agent_id, {})
            
            # Get recent Brier score
            recent_brier = perf.get("recent_brier", 0.25)
            
            # Adjust weight based on Brier score
            # Lower Brier = better, so invert
            brier_factor = 1.0 - (recent_brier - 0.1)  # 0.1 is excellent
            
            # Get recent win rate
            win_rate = perf.get("win_rate", 0.5)
            win_factor = 0.5 + win_rate
            
            # Combine factors
            adjusted_weight = weight * brier_factor * win_factor
            
            # Clamp
            adjusted[agent_id] = max(0.1, min(5.0, adjusted_weight))
        
        return adjusted
    
    def select_voting_method(
        self,
        agent_predictions: List[Dict],
        regime: str
    ) -> str:
        """
        Choose best voting method for current situation.
        
        Args:
            agent_predictions: List of agent predictions
            regime: Current market regime
        
        Returns:
            Name of voting method to use
        """
        # Start with regime default
        default_method = self.VOTING_METHOD_MAP.get(regime, "weighted_average")
        
        # Check agreement level
        if not agent_predictions:
            return default_method
        
        directions = [p["direction"] for p in agent_predictions]
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")
        total = len(directions)
        
        agreement = max(up_count, down_count) / total if total > 0 else 0.5
        
        # Override based on agreement
        if agreement < 0.55:
            # Low agreement - use consensus requiring
            return "consensus_requiring"
        elif agreement > 0.75:
            # High agreement - use weighted average
            return "weighted_average"
        
        # Check confidence variance
        confidences = [p.get("confidence", 0.5) for p in agent_predictions]
        if confidences:
            conf_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            
            if conf_variance > 0.1:
                return "confidence_weighted"
        
        return default_method
    
    def get_stats(self) -> Dict[str, Any]:
        """Get meta-agent statistics"""
        
        if not self.selection_history:
            return {"selections": 0}
        
        # Count regime occurrences
        regime_counts = {}
        for sel in self.selection_history:
            regime = sel["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            "selections": len(self.selection_history),
            "regime_distribution": regime_counts,
            "last_selection": self.selection_history[-1] if self.selection_history else None
        }
