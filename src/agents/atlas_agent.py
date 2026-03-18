"""
Core Agent System for Atlas v4.0
Self-improving agents with Darwinian weight adjustment
"""

import json
import random
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import os


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
    """Track agent's prediction performance"""
    predictions: List[Dict] = field(default_factory=list)
    total_predictions: int = 0
    correct_predictions: int = 0
    total_brier_score: float = 0.0
    recent_brier_scores: List[float] = field(default_factory=list)
    
    def add_prediction(self, predicted_prob: float, actual_outcome: bool, direction: str):
        """Add a prediction result"""
        # For DOWN predictions, the probability should be inverted
        # predicted_prob is the probability of UP
        # actual_outcome is whether UP actually happened
        brier = (predicted_prob - (1.0 if actual_outcome else 0.0)) ** 2
        
        self.predictions.append({
            "timestamp": datetime.now().isoformat(),
            "predicted_prob": predicted_prob,
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
        """
        if self.performance.total_predictions < 3:
            return  # Not enough data
        
        recent_brier = self.performance.recent_average_brier
        overall_brier = self.performance.average_brier_score
        
        # Lower Brier = better performance
        if recent_brier < overall_brier:
            # Improving - increase weight
            self.weight = min(self.weight * (1 + alpha), self.max_weight)
        else:
            # Worsening - decrease weight
            self.weight = max(self.weight * (1 - alpha * 0.5), self.min_weight)
    
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
                "recent_brier_scores": self.performance.recent_brier_scores[-10:]
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
        
        return agent


class AgentTeam:
    """
    Team of self-improving agents with collective prediction.
    Implements ATLAS-GIC style weight adjustment and prompt optimization.
    """
    
    DEFAULT_AGENT_CONFIGS = [
        {"id": "rsi_master", "name": "RSI Master", "focus": "RSI and overbought/oversold conditions", "regime": "ALL"},
        {"id": "macd_trend", "name": "MACD Trend", "focus": "MACD crossovers and trend direction", "regime": "ALL"},
        {"id": "momentum_hawk", "name": "Momentum Hawk", "focus": "Price momentum and velocity", "regime": "TRENDING_UP"},
        {"id": "volume_whale", "name": "Volume Whale", "focus": "Volume patterns and whale activity", "regime": "BREAKOUT"},
        {"id": "support_resist", "name": "Support/Resistance", "focus": "Key price levels and breakouts", "regime": "RANGING"},
        {"id": "order_flow", "name": "Order Flow", "focus": "Buy/sell pressure and order book", "regime": "ALL"},
        {"id": "volatility", "name": "Volatility Watcher", "focus": "Volatility and price swings", "regime": "VOLATILE"},
        {"id": "sentiment", "name": "Sentiment Reader", "focus": "Market sentiment and positioning", "regime": "REVERSAL"},
    ]
    
    def __init__(self, llm_client=None, agent_configs: List[Dict] = None):
        self.llm_client = llm_client
        self.agents: List[Agent] = []
        self.prediction_history: List[Dict] = []
        self.total_brier_score: float = 0.0
        self.total_predictions: int = 0
        
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

DIRECTION: UP/DOWN
PROBABILITY: XX%
CONFIDENCE: High/Medium/Low
REASONING: [1-2 sentences]"""
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
        Optionally filter by regime.
        """
        # Select agents based on regime
        if current_regime:
            active_agents = [
                a for a in self.agents 
                if a.regime == "ALL" or a.regime.upper() == current_regime.upper()
            ]
            if not active_agents:
                active_agents = self.agents
        else:
            active_agents = self.agents
        
        # Get predictions from all agents in parallel
        tasks = [agent.analyze(market_context) for agent in active_agents]
        results = await asyncio.gather(*tasks)
        
        # Collect predictions
        agent_predictions = []
        for agent, result in zip(active_agents, results):
            agent_predictions.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "focus": agent.focus,
                "weight": agent.weight,
                "regime": agent.regime,
                "direction": result["direction"],
                "probability": result["probability"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"]
            })
        
        # Aggregate predictions with Darwinian weights
        final_prediction = self._aggregate_predictions(agent_predictions)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_price": market_context.get("current_price", 0),
            "current_regime": current_regime,
            "agent_predictions": agent_predictions,
            "final": final_prediction
        }
    
    def _aggregate_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Aggregate predictions using weighted voting"""
        
        # Calculate weighted probabilities for UP
        total_weight = 0
        weighted_up_prob = 0
        
        for pred in predictions:
            weight = pred["weight"]
            direction = pred["direction"]
            confidence = pred["confidence"]
            
            # Weight by both agent weight and confidence
            effective_weight = weight * confidence
            
            # Convert direction to UP probability
            if direction == "UP":
                up_prob = pred["probability"]
            elif direction == "DOWN":
                # If agent says DOWN with X% probability, that means
                # (1-X)% probability of UP
                up_prob = 1 - pred["probability"]
            else:
                up_prob = 0.5  # Neutral
            
            weighted_up_prob += up_prob * effective_weight
            total_weight += effective_weight
        
        final_prob = weighted_up_prob / total_weight if total_weight > 0 else 0.5
        
        # Determine direction
        if final_prob > 0.53:
            direction = "UP"
        elif final_prob < 0.47:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        # Calculate agreement
        up_votes = sum(1 for p in predictions if p["direction"] == "UP")
        down_votes = sum(1 for p in predictions if p["direction"] == "DOWN")
        total_votes = up_votes + down_votes
        
        agreement = max(up_votes, down_votes) / total_votes if total_votes > 0 else 0.5
        
        # Calculate confidence
        confidence = agreement * (0.5 + 0.5 * abs(final_prob - 0.5) * 2)
        
        return {
            "direction": direction,
            "probability": final_prob,
            "confidence": confidence,
            "agreement": agreement,
            "up_votes": up_votes,
            "down_votes": down_votes,
            "total_weight": total_weight
        }
    
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
            "prediction_history": self.prediction_history[-100:]  # Keep last 100
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
                    "brier": a.performance.average_brier_score
                }
                for a in self.agents
            }
        }
