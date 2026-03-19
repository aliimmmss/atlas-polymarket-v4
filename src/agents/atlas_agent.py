"""
Core Agent System for Atlas v4.0
Self-improving agents with Darwinian weight adjustment

ACADEMIC PAPER IMPLEMENTATIONS:
1. Log-Odds Aggregation - From "Toward Black-Scholes for Prediction Markets" (Dalen, 2025)
   - Transform probabilities to log-odds space before aggregation
   - Better mathematical properties, especially near boundaries (0 and 1)
   - x_t = log(p_t / (1 - p_t)) where x_t is log-odds

2. Belief Volatility Tracking - From Dalen (2025)
   - Track how fast log-odds move over time
   - High belief volatility = unstable predictions = smaller positions
   - sigma_b (belief volatility) is analogous to implied volatility

3. Bayesian Inverse Problem - From "Prediction Markets as Bayesian Inverse Problems" (Madrigal-Cianci, 2026)
   - Treat predictions as observations with noise
   - Use KL divergence for information gain
   - Compute posterior probabilities with uncertainty bounds
"""

import json
import random
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import os
import math


# ============================================================================
# LOG-ODDS UTILITIES (From Academic Paper: Dalen 2025)
# ============================================================================

def prob_to_logit(p: float) -> float:
    """
    Convert probability to log-odds (logit space).
    
    From "Toward Black-Scholes for Prediction Markets":
    x_t = log(p_t / (1 - p_t))
    
    Logit space advantages:
    - Unbounded (no 0/1 boundaries)
    - Symmetric movements
    - Better for aggregation and modeling
    """
    if p <= 0.001:
        return -6.9  # Approx log(0.001/0.999)
    if p >= 0.999:
        return 6.9   # Approx log(0.999/0.001)
    return math.log(p / (1 - p))


def logit_to_prob(x: float) -> float:
    """
    Convert log-odds back to probability.
    
    p = 1 / (1 + exp(-x))
    """
    if x <= -6.9:
        return 0.001
    if x >= 6.9:
        return 0.999
    return 1.0 / (1.0 + math.exp(-x))


def compute_belief_volatility(log_odds_history: List[float], window: int = 10) -> float:
    """
    Compute belief volatility - how fast log-odds move.
    
    From Dalen (2025): Belief volatility sigma_b measures the
    standard deviation of log-odds changes over time.
    
    High sigma_b = unstable predictions
    Low sigma_b = stable predictions
    """
    if len(log_odds_history) < 3:
        return 0.0
    
    # Use recent history
    recent = log_odds_history[-window:] if len(log_odds_history) >= window else log_odds_history
    
    # Compute differences (returns in log-odds space)
    diffs = [recent[i+1] - recent[i] for i in range(len(recent) - 1)]
    
    if not diffs:
        return 0.0
    
    # Standard deviation of log-odds changes
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff)**2 for d in diffs) / len(diffs)
    sigma_b = math.sqrt(variance)
    
    return sigma_b


def kl_divergence(p: float, q: float) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(p || q).
    
    From "Bayesian Inverse Problems" paper:
    KL divergence measures information gain and is used for:
    - Identifiability criteria
    - Information gain metrics
    - Stability analysis
    """
    if p <= 0 or p >= 1 or q <= 0 or q >= 1:
        return 0.0
    
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# ============================================================================
# AGENT CLASSES
# ============================================================================

@dataclass
class AgentPrompt:
    """Agent's prompt that can be modified"""
    system_prompt: str
    analysis_template: str
    version: int = 1
    history: List[Dict] = field(default_factory=list)
    
    def record_modification(self, old_prompt: str, new_prompt: str, improvement: float):
        """Record a prompt modification"""
        self.history.append({
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "old_prompt": old_prompt[:200],
            "new_prompt": new_prompt[:200],
            "improvement": improvement
        })
        self.version += 1


@dataclass 
class AgentPerformance:
    """
    Track agent's prediction performance.
    
    Enhanced with log-odds tracking for belief volatility.
    """
    predictions: List[Dict] = field(default_factory=list)
    total_predictions: int = 0
    correct_predictions: int = 0
    total_brier_score: float = 0.0
    recent_brier_scores: List[float] = field(default_factory=list)
    
    # NEW: Track log-odds history for belief volatility
    log_odds_history: List[float] = field(default_factory=list)
    belief_volatility: float = 0.0
    
    def add_prediction(self, predicted_prob: float, actual_outcome: bool, direction: str):
        """Add a prediction result"""
        # For DOWN predictions, the probability should be inverted
        # predicted_prob is the probability of UP
        # actual_outcome is whether UP actually happened
        brier = (predicted_prob - (1.0 if actual_outcome else 0.0)) ** 2
        
        # NEW: Store log-odds for belief volatility tracking
        log_odds = prob_to_logit(predicted_prob)
        self.log_odds_history.append(log_odds)
        
        # Keep only last 50 log-odds
        if len(self.log_odds_history) > 50:
            self.log_odds_history = self.log_odds_history[-50:]
        
        # Update belief volatility
        self.belief_volatility = compute_belief_volatility(self.log_odds_history)
        
        self.predictions.append({
            "timestamp": datetime.now().isoformat(),
            "predicted_prob": predicted_prob,
            "log_odds": log_odds,
            "actual_outcome": actual_outcome,
            "direction": direction,
            "brier_score": brier
        })
        
        self.total_predictions += 1
        self.total_brier_score += brier
        self.recent_brier_scores.append(brier)
        
        # Keep only last 20 scores
        if len(self.recent_brier_scores) > 20:
            self.recent_brier_scores = self.recent_brier_scores[-20:]
        
        # Direction correctness: predicted direction matches actual
        predicted_up = predicted_prob > 0.5
        if (predicted_up and actual_outcome) or (not predicted_up and not actual_outcome):
            self.correct_predictions += 1
    
    @property
    def average_brier_score(self) -> float:
        """Average Brier score (lower is better)"""
        if self.total_predictions == 0:
            return 0.25  # Neutral
        return self.total_brier_score / self.total_predictions
    
    @property
    def recent_average_brier(self) -> float:
        """Recent average Brier score"""
        if not self.recent_brier_scores:
            return 0.25
        return sum(self.recent_brier_scores) / len(self.recent_brier_scores)
    
    @property
    def win_rate(self) -> float:
        """Percentage of correct direction predictions"""
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions


class Agent:
    """
    Self-improving agent with Darwinian weight adjustment.
    Each agent has a unique focus and can modify its own prompts.
    
    ENHANCED: Now includes belief volatility tracking and
    log-odds based predictions.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        focus: str,
        prompt_config: AgentPrompt,
        llm_client=None,
        initial_weight: float = 1.0,
        regime: str = "ALL"
    ):
        self.agent_id = agent_id
        self.name = name
        self.focus = focus
        self.prompt_config = prompt_config
        self.performance = AgentPerformance()
        self.llm_client = llm_client
        self.regime = regime  # Market regime this agent specializes in
        
        # Darwinian weight (adjusted based on performance)
        self.weight = initial_weight
        self.min_weight = 0.1
        self.max_weight = 5.0
    
    async def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and produce a prediction.
        """
        # Build analysis prompt
        analysis_prompt = self._build_prompt(context)
        
        # Get prediction from LLM
        if self.llm_client:
            response = await self._call_llm(analysis_prompt)
        else:
            response = self._fallback_analysis(context)
        
        return response
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build the analysis prompt with context"""
        try:
            return self.prompt_config.analysis_template.format(
                focus=self.focus,
                current_price=context.get("current_price", 0),
                price_change_24h=context.get("price_change_24h", 0),
                momentum_5m=context.get("momentum_5m", context.get("momentum", {}).get("5m", 0)),
                volatility_5m=context.get("volatility_5m", context.get("volatility", {}).get("5m", 0)),
                rsi=context.get("rsi", context.get("technical", {}).get("rsi", {}).get("value", 50)),
                macd_trend=context.get("macd_trend", context.get("technical", {}).get("macd", {}).get("trend", "neutral")),
                order_imbalance=context.get("order_imbalance", 0),
                support=context.get("support", context.get("technical", {}).get("support_resistance", {}).get("support", 0)),
                resistance=context.get("resistance", context.get("technical", {}).get("support_resistance", {}).get("resistance", 0)),
                volume_ratio=context.get("volume_ratio", 1),
                timestamp=datetime.now().isoformat()
            )
        except KeyError:
            # Fallback for missing keys
            return f"Analyze BTC market for 15-min prediction. Focus: {self.focus}. Current price: ${context.get('current_price', 0):,.2f}"
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the LLM and parse response"""
        try:
            messages = [
                {"role": "system", "content": self.prompt_config.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_client.messages.create(
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            text = response["content"][0]["text"]
            return self._parse_llm_response(text)
        except Exception as e:
            return {"direction": "NEUTRAL", "probability": 0.5, "confidence": 0.3, "reasoning": f"Error: {str(e)}"}
    
    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM response for structured output"""
        direction = "NEUTRAL"
        probability = 0.5
        confidence = 0.5
        
        text_lower = text.lower()
        
        # Parse direction
        if "direction: up" in text_lower or "direction: buy up" in text_lower:
            direction = "UP"
        elif "direction: down" in text_lower or "direction: buy down" in text_lower:
            direction = "DOWN"
        
        # Parse probability
        for line in text_lower.split("\n"):
            if "probability:" in line:
                try:
                    prob_str = line.split(":")[-1].strip().replace("%", "")
                    probability = float(prob_str) / 100
                    probability = max(0.1, min(0.9, probability))
                except:
                    pass
            
            if "confidence:" in line:
                try:
                    conf_str = line.split(":")[-1].strip().replace("%", "")
                    confidence = float(conf_str) / 100
                    confidence = max(0.1, min(1.0, confidence))
                except:
                    pass
        
        return {
            "direction": direction,
            "probability": probability,
            "confidence": confidence,
            "reasoning": text[:300]
        }
    
    def _fallback_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        # Simple rule-based analysis based on focus
        return {"direction": "NEUTRAL", "probability": 0.5, "confidence": 0.3, "reasoning": "No LLM available"}
    
    def update_weight(self, alpha: float = 0.1):
        """
        Update Darwinian weight based on recent performance.
        Good performers get louder, bad performers get quieter.
        
        FIXED: Now uses absolute performance AND improvement trend.
        - Agents with LOW Brier scores (good) get increased weight
        - Agents with HIGH Brier scores (bad) get decreased weight
        - Bonus for improving performance
        - Penalty for high belief volatility
        """
        if self.performance.total_predictions < 3:
            return  # Not enough data
        
        recent_brier = self.performance.recent_average_brier
        overall_brier = self.performance.average_brier_score
        win_rate = self.performance.win_rate
        
        # Base adjustment on absolute performance (Brier score)
        # Lower Brier = better, 0.25 = random guessing
        brier_threshold = 0.22  # Good performance threshold
        
        if overall_brier < brier_threshold:
            # Good performer - increase weight based on how good
            performance_bonus = (brier_threshold - overall_brier) / brier_threshold
            adjustment = alpha * (1 + performance_bonus)
            self.weight = min(self.weight * (1 + adjustment), self.max_weight)
        else:
            # Poor performer - decrease weight based on how bad
            performance_penalty = (overall_brier - brier_threshold) / brier_threshold
            adjustment = alpha * (1 + min(performance_penalty, 1.0))
            self.weight = max(self.weight * (1 - adjustment), self.min_weight)
        
        # Additional bonus/penalty for trend
        if recent_brier < overall_brier * 0.95:  # Improving (>5% better recent)
            self.weight = min(self.weight * 1.05, self.max_weight)
        elif recent_brier > overall_brier * 1.05:  # Worsening (>5% worse recent)
            self.weight = max(self.weight * 0.95, self.min_weight)
        
        # NEW: Penalty for high belief volatility (from academic paper)
        # High volatility = unstable predictions = should have lower weight
        if self.performance.belief_volatility > 0.5:
            vol_penalty = 0.95  # 5% reduction
            self.weight = max(self.weight * vol_penalty, self.min_weight)
    
    def record_result(self, predicted_prob: float, actual_outcome: bool, direction: str):
        """Record a prediction result"""
        self.performance.add_prediction(predicted_prob, actual_outcome, direction)
        self.update_weight()
    
    async def improve_prompt(self) -> bool:
        """
        Attempt to improve the agent's prompt based on performance.
        Returns True if prompt was modified.
        """
        if self.performance.total_predictions < 5:
            return False
        
        # Only modify if recent performance is worse than overall
        if self.performance.recent_average_brier >= self.performance.average_brier_score:
            return False
        
        if not self.llm_client:
            return False
        
        # Generate prompt improvement
        improvement_prompt = f"""You are helping improve an AI agent's system prompt for Bitcoin 15-minute price prediction.

CURRENT PROMPT:
---
{self.prompt_config.system_prompt}
---

PERFORMANCE:
- Total Predictions: {self.performance.total_predictions}
- Win Rate: {self.performance.win_rate:.1%}
- Average Brier Score: {self.performance.average_brier_score:.4f} (lower is better, 0=perfect)
- Recent Brier Score: {self.performance.recent_average_brier:.4f}
- Belief Volatility: {self.performance.belief_volatility:.4f} (lower is more stable)

FOCUS: {self.focus}

The agent is underperforming. Modify the system prompt to improve prediction accuracy.
Make it more specific and actionable for {self.focus} analysis.

Return ONLY the new system prompt (2-3 paragraphs max). No explanation."""

        try:
            response = await self.llm_client.messages.create(
                messages=[{"role": "user", "content": improvement_prompt}],
                max_tokens=600,
                temperature=0.7
            )
            
            new_prompt = response["content"][0]["text"]
            
            # Record the modification
            old_prompt = self.prompt_config.system_prompt
            self.prompt_config.record_modification(
                old_prompt, 
                new_prompt, 
                self.performance.recent_average_brier - self.performance.average_brier_score
            )
            
            # Apply new prompt
            self.prompt_config.system_prompt = new_prompt
            
            return True
        except:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "focus": self.focus,
            "weight": self.weight,
            "regime": self.regime,
            "prompt_config": {
                "system_prompt": self.prompt_config.system_prompt,
                "analysis_template": self.prompt_config.analysis_template,
                "version": self.prompt_config.version
            },
            "performance": {
                "total_predictions": self.performance.total_predictions,
                "correct_predictions": self.performance.correct_predictions,
                "total_brier_score": self.performance.total_brier_score,
                "recent_brier_scores": self.performance.recent_brier_scores[-10:],
                "log_odds_history": self.performance.log_odds_history[-20:],
                "belief_volatility": self.performance.belief_volatility
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm_client=None) -> "Agent":
        """Deserialize agent state"""
        prompt_config = AgentPrompt(
            system_prompt=data["prompt_config"]["system_prompt"],
            analysis_template=data["prompt_config"]["analysis_template"],
            version=data["prompt_config"]["version"]
        )
        
        agent = cls(
            agent_id=data["agent_id"],
            name=data["name"],
            focus=data["focus"],
            prompt_config=prompt_config,
            llm_client=llm_client,
            initial_weight=data["weight"],
            regime=data.get("regime", "ALL")
        )
        
        # Restore performance
        agent.performance.total_predictions = data["performance"]["total_predictions"]
        agent.performance.correct_predictions = data["performance"]["correct_predictions"]
        agent.performance.total_brier_score = data["performance"]["total_brier_score"]
        agent.performance.recent_brier_scores = data["performance"]["recent_brier_scores"]
        
        # Restore log-odds history if available
        if "log_odds_history" in data["performance"]:
            agent.performance.log_odds_history = data["performance"]["log_odds_history"]
        if "belief_volatility" in data["performance"]:
            agent.performance.belief_volatility = data["performance"]["belief_volatility"]
        
        return agent


class AgentTeam:
    """
    Team of self-improving agents with collective prediction.
    Implements ATLAS-GIC style weight adjustment and prompt optimization.
    
    ENHANCED: Uses log-odds aggregation (from Dalen 2025) and
    Bayesian confidence scoring (from Madrigal-Cianci 2026).
    """
    
    DEFAULT_AGENT_CONFIGS = [
        {"id": "rsi_master", "name": "RSI Master", "focus": "RSI and overbought/oversold conditions", "regime": "ALL"},
        {"id": "macd_trend", "name": "MACD Trend", "focus": "MACD crossovers and trend direction", "regime": "ALL"},
        {"id": "momentum_hawk", "name": "Momentum Hawk", "focus": "Price momentum and velocity", "regime": "TRENDING"},
        {"id": "volume_whale", "name": "Volume Whale", "focus": "Volume patterns and whale activity", "regime": "ALL"},
        {"id": "support_resist", "name": "Support/Resistance", "focus": "Key price levels and breakouts", "regime": "ALL"},
        {"id": "order_flow", "name": "Order Flow", "focus": "Buy/sell pressure and order book", "regime": "ALL"},
        {"id": "volatility", "name": "Volatility Watcher", "focus": "Volatility and price swings", "regime": "ALL"},
        {"id": "sentiment", "name": "Sentiment Reader", "focus": "Market sentiment and positioning", "regime": "ALL"},
    ]
    
    def __init__(self, llm_client=None, agent_configs: List[Dict] = None):
        self.llm_client = llm_client
        self.agents: List[Agent] = []
        self.prediction_history: List[Dict] = []
        self.total_brier_score: float = 0.0
        self.total_predictions: int = 0
        
        # NEW: Track team-level belief volatility
        self.team_log_odds_history: List[float] = []
        self.team_belief_volatility: float = 0.0
        
        # NEW: Track price-probability calibration
        self.price_outcome_calibration: List[Dict] = []
        self.price_probability_adjustment: float = 0.02  # Default adjustment
        
        # Initialize agents
        configs = agent_configs or self.DEFAULT_AGENT_CONFIGS
        for config in configs:
            agent = self._create_agent(config)
            self.agents.append(agent)
    
    def _create_agent(self, config: Dict) -> Agent:
        """Create an agent with default prompts"""
        focus = config["focus"]
        
        prompt_config = AgentPrompt(
            system_prompt=f"""You are an expert Bitcoin analyst specializing in {focus}.
Your role is to analyze market data and predict Bitcoin's price movement over the next 15 minutes.
Be objective, data-driven, and concise. Focus on {focus} signals.

IMPORTANT: You must be EQUALLY willing to predict UP or DOWN based on the data.
- Do NOT have a bullish bias
- Do NOT have a bearish bias  
- Let the data dictate the direction
- If signals are mixed or weak, predict NEUTRAL with 50% probability

Avoid emotional decisions. Stick to what the data shows.""",
            
            analysis_template=f"""Analyze the following Bitcoin market data focusing on {focus}:

Current Price: ${{current_price:,.2f}}
24h Change: {{price_change_24h:+.2f}}%
5m Momentum: {{momentum_5m:+.3f}}%
5m Volatility: {{volatility_5m:.3f}}
RSI: {{rsi:.1f}}
MACD Trend: {{macd_trend}}
Order Imbalance: {{order_imbalance:+.2f}}
Support: ${{support:,.2f}}
Resistance: ${{resistance:,.2f}}
Volume Ratio: {{volume_ratio:.2f}}x

Focus on {focus} and provide your 15-minute prediction.

CRITICAL: Be balanced. Predict UP only if clear bullish signals exist. Predict DOWN only if clear bearish signals exist.

DIRECTION: UP/DOWN/NEUTRAL
PROBABILITY: XX%
CONFIDENCE: High/Medium/Low
REASONING: [1-2 sentences explaining the KEY factor]"""
        )
        
        return Agent(
            agent_id=config["id"],
            name=config["name"],
            focus=focus,
            prompt_config=prompt_config,
            llm_client=self.llm_client,
            regime=config.get("regime", "ALL")
        )
    
    async def predict(self, market_context: Dict[str, Any], current_regime: str = None) -> Dict[str, Any]:
        """
        Generate collective prediction from all agents.
        
        FIXED: All agents now participate, but regime-matching agents get weight bonus.
        ENHANCED: Uses log-odds aggregation from academic paper.
        """
        # All agents participate now (no filtering)
        active_agents = self.agents
        
        # Get predictions from all agents in parallel
        tasks = [agent.analyze(market_context) for agent in active_agents]
        results = await asyncio.gather(*tasks)
        
        # Collect predictions
        agent_predictions = []
        for agent, result in zip(active_agents, results):
            # Calculate effective weight with regime bonus
            effective_weight = agent.weight
            
            # Regime-matching agents get a weight bonus
            if current_regime and agent.regime != "ALL":
                if agent.regime.upper() in (current_regime or "").upper():
                    effective_weight *= 1.5  # 50% bonus for regime match
            
            # NEW: Reduce weight for high belief volatility
            if agent.performance.belief_volatility > 0.5:
                vol_factor = 1.0 - (agent.performance.belief_volatility * 0.3)
                effective_weight *= max(0.5, vol_factor)
            
            agent_predictions.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "focus": agent.focus,
                "weight": effective_weight,
                "base_weight": agent.weight,
                "regime": agent.regime,
                "direction": result["direction"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                # NEW: Include log-odds
                "log_odds": prob_to_logit(result["probability"]) if result["direction"] == "UP" 
                           else prob_to_logit(1 - result["probability"]),
                "belief_volatility": agent.performance.belief_volatility
            })
        
        # Aggregate predictions with LOG-ODDS method (from academic paper)
        final_prediction = self._aggregate_predictions_log_odds(agent_predictions)
        
        # Update team belief volatility
        team_log_odds = prob_to_logit(final_prediction["probability"])
        self.team_log_odds_history.append(team_log_odds)
        if len(self.team_log_odds_history) > 50:
            self.team_log_odds_history = self.team_log_odds_history[-50:]
        self.team_belief_volatility = compute_belief_volatility(self.team_log_odds_history)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_price": market_context.get("current_price", 0),
            "current_regime": current_regime,
            "agent_predictions": agent_predictions,
            "final": final_prediction,
            # NEW: Include team belief volatility
            "team_belief_volatility": self.team_belief_volatility
        }
    
    def _aggregate_predictions_log_odds(self, predictions: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate predictions using LOG-ODDS space.
        
        From "Toward Black-Scholes for Prediction Markets" (Dalen, 2025):
        - Transform probabilities to log-odds (logit space)
        - Aggregate in log-odds space (better mathematical properties)
        - Transform back to probability
        
        This is superior to aggregating raw probabilities because:
        1. Log-odds are unbounded (no 0/1 boundaries)
        2. Movements are symmetric
        3. Better handling of extreme probabilities
        """
        
        # Aggregate in LOG-ODDS space
        total_weight = 0
        weighted_log_odds = 0
        
        for pred in predictions:
            weight = pred["weight"]
            direction = pred["direction"]
            confidence = pred["confidence"]
            
            # Weight by both agent weight and confidence
            effective_weight = weight * confidence
            
            # Get log-odds for UP probability
            if direction == "UP":
                log_odds = pred["log_odds"]
            elif direction == "DOWN":
                # DOWN prediction: log-odds of DOWN = -log-odds of UP
                log_odds = -pred["log_odds"]
            else:
                log_odds = 0.0  # Neutral (log-odds of 0.5 = 0)
            
            weighted_log_odds += log_odds * effective_weight
            total_weight += effective_weight
        
        # Compute final log-odds
        final_log_odds = weighted_log_odds / total_weight if total_weight > 0 else 0.0
        
        # Convert back to probability
        final_prob = logit_to_prob(final_log_odds)
        
        # Determine direction (simple majority in probability space)
        if final_prob > 0.50:
            direction = "UP"
        elif final_prob < 0.50:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        # Calculate agreement
        up_votes = sum(1 for p in predictions if p["direction"] == "UP")
        down_votes = sum(1 for p in predictions if p["direction"] == "DOWN")
        neutral_votes = sum(1 for p in predictions if p["direction"] == "NEUTRAL")
        total_votes = up_votes + down_votes + neutral_votes
        
        agreement = max(up_votes, down_votes) / total_votes if total_votes > 0 else 0.5
        
        # Compute Bayesian-style confidence (from academic paper)
        confidence = self._compute_bayesian_confidence(
            final_prob, final_log_odds, agreement, predictions
        )
        
        return {
            "direction": direction,
            "probability": final_prob,
            "log_odds": final_log_odds,
            "confidence": confidence,
            "agreement": agreement,
            "up_votes": up_votes,
            "down_votes": down_votes,
            "neutral_votes": neutral_votes,
            "total_weight": total_weight
        }
    
    def _compute_bayesian_confidence(
        self,
        final_prob: float,
        final_log_odds: float,
        agreement: float,
        predictions: List[Dict]
    ) -> float:
        """
        Compute Bayesian-style confidence score.
        
        From "Prediction Markets as Bayesian Inverse Problems" (Madrigal-Cianci, 2026):
        
        Confidence should reflect:
        1. Distance from 0.5 (signal strength)
        2. Agreement among agents (consensus)
        3. KL divergence between predictions (information quality)
        4. Belief volatility (stability)
        """
        
        # 1. Distance confidence (from academic paper)
        # How far from neutral (0.5)
        distance_confidence = abs(final_log_odds) / 3.0  # Normalized by ~3 std devs
        distance_confidence = min(1.0, distance_confidence)
        
        # 2. Agreement confidence
        agreement_confidence = agreement
        
        # 3. Information quality via KL divergence
        # Higher KL divergence = more informative predictions
        kl_scores = []
        for pred in predictions:
            if pred["direction"] != "NEUTRAL":
                kl = kl_divergence(pred["probability"], 0.5)
                kl_scores.append(kl * pred["weight"])
        
        info_quality = sum(kl_scores) / len(kl_scores) if kl_scores else 0.0
        info_quality = min(1.0, info_quality * 2)  # Normalize
        
        # 4. Belief volatility penalty
        avg_belief_vol = sum(p.get("belief_volatility", 0) for p in predictions) / len(predictions) if predictions else 0
        vol_penalty = 1.0 - min(0.3, avg_belief_vol * 0.5)
        
        # Combined confidence (weighted average)
        weights = {
            "distance": 0.35,
            "agreement": 0.30,
            "info_quality": 0.20,
            "vol_penalty": 0.15
        }
        
        confidence = (
            weights["distance"] * distance_confidence +
            weights["agreement"] * agreement_confidence +
            weights["info_quality"] * info_quality +
            weights["vol_penalty"] * vol_penalty
        )
        
        # Boost confidence when most agents agree
        if agreement > 0.7:
            confidence = min(confidence * 1.15, 1.0)
        
        return confidence
    
    def adjust_probability_for_market_bias(
        self,
        probability: float,
        market_price: float
    ) -> float:
        """
        Adjust probability for market price bias.
        
        From academic papers:
        1. "Application of Kelly Criterion" (Meister 2024): Market prices systematically
           diverge from true probabilities due to risk aversion.
        2. The gap tends to be toward 0.5 (prices are "conservatively" priced).
        
        Args:
            probability: Our predicted probability
            market_price: Current market price for UP
        
        Returns:
            Adjusted probability accounting for market bias
        """
        # Use historical calibration if available
        if len(self.price_outcome_calibration) >= 10:
            # Calculate average deviation from prices
            deviations = []
            for record in self.price_outcome_calibration[-30:]:
                market_p = record["market_price"]
                outcome = record["outcome"]  # 1 if UP, 0 if DOWN
                # If market was at 60% UP, but UP only happened 55% of the time at that price
                # then there's a 5% "optimism bias"
                deviations.append(market_p - outcome)
            
            avg_bias = sum(deviations) / len(deviations)
            adjustment = avg_bias * 0.5  # Use half the historical bias
        else:
            # Default adjustment from academic paper findings
            # Market prices tend to be slightly biased toward 50%
            adjustment = self.price_probability_adjustment
        
        # Apply adjustment (subtract because market tends to overprice slightly)
        adjusted = probability - adjustment
        
        # Ensure bounds
        return max(0.05, min(0.95, adjusted))
    
    def record_outcome(self, prediction_id: str, actual_outcome: bool, actual_price_change: float):
        """
        Record the outcome of a prediction and update all agents.
        """
        # Find the prediction
        prediction = None
        for p in self.prediction_history:
            if p.get("prediction_id") == prediction_id:
                prediction = p
                break
        
        if not prediction:
            return
        
        # Calculate Brier score for the final prediction
        predicted_prob = prediction["final"]["probability"]
        brier_score = (predicted_prob - (1.0 if actual_outcome else 0.0)) ** 2
        
        # Update team stats
        self.total_predictions += 1
        self.total_brier_score += brier_score
        
        # Record for price-probability calibration
        market_price = prediction.get("market_price", 0.5)
        self.price_outcome_calibration.append({
            "market_price": market_price,
            "outcome": 1 if actual_outcome else 0,
            "predicted_prob": predicted_prob,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.price_outcome_calibration) > 100:
            self.price_outcome_calibration = self.price_outcome_calibration[-100:]
        
        # Update each agent
        for agent_pred in prediction["agent_predictions"]:
            agent = next((a for a in self.agents if a.agent_id == agent_pred["agent_id"]), None)
            if agent:
                # The agent's predicted probability for UP
                agent_up_prob = agent_pred["probability"]
                if agent_pred["direction"] == "DOWN":
                    agent_up_prob = 1 - agent_pred["probability"]
                
                agent.record_result(
                    predicted_prob=agent_up_prob,
                    actual_outcome=actual_outcome,
                    direction=agent_pred["direction"]
                )
        
        # Store result
        prediction["outcome"] = {
            "actual_outcome": actual_outcome,
            "actual_price_change": actual_price_change,
            "brier_score": brier_score,
            "correct": (predicted_prob > 0.5 and actual_outcome) or 
                       (predicted_prob < 0.5 and not actual_outcome)
        }
    
    async def improve_agents(self) -> Dict[str, Any]:
        """
        Attempt to improve underperforming agents.
        """
        improvements = []
        
        for agent in self.agents:
            # Only improve agents with enough predictions and poor performance
            if agent.performance.total_predictions >= 5:
                if agent.performance.recent_average_brier > agent.performance.average_brier_score:
                    improved = await agent.improve_prompt()
                    if improved:
                        improvements.append({
                            "agent_id": agent.agent_id,
                            "agent_name": agent.name,
                            "new_version": agent.prompt_config.version
                        })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "improvements_made": improvements,
            "total_agents_checked": len(self.agents)
        }
    
    def save_state(self, filepath: str):
        """Save team state to file"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": self.total_predictions,
            "total_brier_score": self.total_brier_score,
            "average_brier": self.total_brier_score / self.total_predictions if self.total_predictions > 0 else 0.25,
            "agents": [agent.to_dict() for agent in self.agents],
            "prediction_history": self.prediction_history[-100:],  # Keep last 100
            # NEW: Save calibration data
            "team_log_odds_history": self.team_log_odds_history[-50:],
            "team_belief_volatility": self.team_belief_volatility,
            "price_outcome_calibration": self.price_outcome_calibration[-50:],
            "price_probability_adjustment": self.price_probability_adjustment
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> bool:
        """Load team state from file"""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, "r") as f:
            state = json.load(f)
        
        self.total_predictions = state.get("total_predictions", 0)
        self.total_brier_score = state.get("total_brier_score", 0)
        self.prediction_history = state.get("prediction_history", [])
        
        # NEW: Load calibration data
        self.team_log_odds_history = state.get("team_log_odds_history", [])
        self.team_belief_volatility = state.get("team_belief_volatility", 0.0)
        self.price_outcome_calibration = state.get("price_outcome_calibration", [])
        self.price_probability_adjustment = state.get("price_probability_adjustment", 0.02)
        
        # Restore agents
        for agent_data in state.get("agents", []):
            agent = next((a for a in self.agents if a.agent_id == agent_data["agent_id"]), None)
            if agent:
                agent.weight = agent_data["weight"]
                agent.prompt_config.system_prompt = agent_data["prompt_config"]["system_prompt"]
                agent.prompt_config.version = agent_data["prompt_config"]["version"]
                agent.performance.total_predictions = agent_data["performance"]["total_predictions"]
                agent.performance.correct_predictions = agent_data["performance"]["correct_predictions"]
                agent.performance.total_brier_score = agent_data["performance"]["total_brier_score"]
                agent.performance.recent_brier_scores = agent_data["performance"]["recent_brier_scores"]
                
                # NEW: Restore log-odds history
                if "log_odds_history" in agent_data["performance"]:
                    agent.performance.log_odds_history = agent_data["performance"]["log_odds_history"]
                if "belief_volatility" in agent_data["performance"]:
                    agent.performance.belief_volatility = agent_data["performance"]["belief_volatility"]
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get team statistics"""
        return {
            "total_predictions": self.total_predictions,
            "average_brier_score": self.total_brier_score / self.total_predictions if self.total_predictions > 0 else 0.25,
            "agent_weights": {a.name: a.weight for a in self.agents},
            "agent_performance": {
                a.name: {
                    "predictions": a.performance.total_predictions,
                    "win_rate": a.performance.win_rate,
                    "brier": a.performance.average_brier_score,
                    "belief_volatility": a.performance.belief_volatility
                }
                for a in self.agents
            },
            # NEW: Include team-level metrics
            "team_belief_volatility": self.team_belief_volatility,
            "price_probability_adjustment": self.price_probability_adjustment,
            "calibration_samples": len(self.price_outcome_calibration)
        }
