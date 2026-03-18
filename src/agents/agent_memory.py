"""
Agent Memory System for Atlas v4.0
Contextual learning from past predictions
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import os
import math


@dataclass
class MemoryEntry:
    """A single memory entry"""
    timestamp: str
    context: Dict[str, Any]
    prediction: Dict[str, Any]
    outcome: Dict[str, Any]
    similarity_key: str


class AgentMemory:
    """
    Stores and retrieves contextual memories for agents.
    
    Memory Types:
    - Success memories: What worked in similar situations
    - Failure memories: What didn't work
    - Pattern memories: Recurring patterns
    
    Context Keys:
    - Market regime
    - Volatility level
    - Time of day
    - Recent price action
    - Signal agreement
    """
    
    def __init__(self, agent_id: str, storage_path: str = "data/agent_memories"):
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.memories: List[MemoryEntry] = []
        self.success_patterns: Dict[str, List[Dict]] = {}
        self.failure_patterns: Dict[str, List[Dict]] = {}
        self.context_embeddings: Dict[str, List[float]] = {}
        
        # Load existing memories
        self._load_memories()
    
    def store_outcome(
        self,
        context: Dict[str, Any],
        prediction: Dict[str, Any],
        outcome: Dict[str, Any]
    ):
        """
        Store a prediction outcome with context.
        
        Args:
            context: Market context at prediction time
            prediction: Agent's prediction
            outcome: Actual outcome
        """
        # Create similarity key
        similarity_key = self._create_similarity_key(context)
        
        # Create memory entry
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            context=context,
            prediction=prediction,
            outcome=outcome,
            similarity_key=similarity_key
        )
        
        # Store in main memory
        self.memories.append(entry)
        
        # Store in pattern memory
        correct = outcome.get("correct", False)
        
        if correct:
            if similarity_key not in self.success_patterns:
                self.success_patterns[similarity_key] = []
            self.success_patterns[similarity_key].append({
                "prediction": prediction,
                "context": context,
                "outcome": outcome
            })
        else:
            if similarity_key not in self.failure_patterns:
                self.failure_patterns[similarity_key] = []
            self.failure_patterns[similarity_key].append({
                "prediction": prediction,
                "context": context,
                "outcome": outcome
            })
        
        # Keep only recent memories
        if len(self.memories) > 500:
            self.memories = self.memories[-500:]
        
        # Save to disk
        self._save_memories()
    
    def retrieve_similar(
        self,
        current_context: Dict[str, Any],
        limit: int = 5
    ) -> List[Dict]:
        """
        Retrieve similar past situations.
        
        Args:
            current_context: Current market context
            limit: Maximum number of memories to return
        
        Returns:
            List of similar past situations
        """
        current_key = self._create_similarity_key(current_context)
        
        # Score memories by similarity
        scored_memories = []
        
        for memory in self.memories:
            similarity = self._calculate_similarity(current_key, memory.similarity_key)
            scored_memories.append((similarity, memory))
        
        # Sort by similarity
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Return top matches
        return [
            {
                "similarity": score,
                "context": memory.context,
                "prediction": memory.prediction,
                "outcome": memory.outcome,
                "timestamp": memory.timestamp
            }
            for score, memory in scored_memories[:limit]
        ]
    
    def get_adjusted_confidence(
        self,
        context: Dict[str, Any],
        base_confidence: float
    ) -> float:
        """
        Adjust confidence based on similar past outcomes.
        
        Args:
            context: Current market context
            base_confidence: Agent's base confidence
        
        Returns:
            Adjusted confidence
        """
        similar = self.retrieve_similar(context, limit=10)
        
        if not similar:
            return base_confidence
        
        # Calculate historical accuracy in similar situations
        correct_count = sum(1 for m in similar if m["outcome"].get("correct", False))
        accuracy = correct_count / len(similar)
        
        # Average similarity
        avg_similarity = sum(m["similarity"] for m in similar) / len(similar)
        
        # Adjust confidence
        if accuracy > 0.6:
            # Good history in similar situations
            adjustment = (accuracy - 0.5) * avg_similarity
            adjusted = base_confidence + adjustment * 0.2
        elif accuracy < 0.4:
            # Poor history in similar situations
            adjustment = (0.5 - accuracy) * avg_similarity
            adjusted = base_confidence - adjustment * 0.2
        else:
            adjusted = base_confidence
        
        return max(0.1, min(1.0, adjusted))
    
    def get_pattern_insight(
        self,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get insight from pattern memory.
        
        Args:
            context: Current market context
        
        Returns:
            Pattern insight or None
        """
        current_key = self._create_similarity_key(context)
        
        # Find matching patterns
        matching_success = []
        matching_failure = []
        
        for pattern_key, patterns in self.success_patterns.items():
            similarity = self._calculate_similarity(current_key, pattern_key)
            if similarity > 0.7:
                matching_success.extend(patterns)
        
        for pattern_key, patterns in self.failure_patterns.items():
            similarity = self._calculate_similarity(current_key, pattern_key)
            if similarity > 0.7:
                matching_failure.extend(patterns)
        
        # Analyze patterns
        if len(matching_success) + len(matching_failure) < 3:
            return None
        
        # Calculate success rate
        success_rate = len(matching_success) / (len(matching_success) + len(matching_failure))
        
        # Get common characteristics
        success_predictions = [p["prediction"] for p in matching_success]
        
        if success_predictions:
            # Find most common direction
            directions = [p.get("direction", "NEUTRAL") for p in success_predictions]
            most_common_direction = max(set(directions), key=directions.count)
            
            # Average probability
            avg_prob = sum(p.get("probability", 0.5) for p in success_predictions) / len(success_predictions)
        else:
            most_common_direction = "NEUTRAL"
            avg_prob = 0.5
        
        return {
            "pattern_match": True,
            "historical_success_rate": success_rate,
            "sample_size": len(matching_success) + len(matching_failure),
            "recommended_direction": most_common_direction,
            "recommended_probability": avg_prob,
            "success_count": len(matching_success),
            "failure_count": len(matching_failure)
        }
    
    def learn_from_mistake(
        self,
        context: Dict[str, Any],
        prediction: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate learning from a failed prediction.
        
        Args:
            context: Market context
            prediction: Failed prediction
            outcome: Actual outcome
        
        Returns:
            Learning insight or None
        """
        if outcome.get("correct", True):
            return None
        
        # Get similar past failures
        similar_failures = []
        current_key = self._create_similarity_key(context)
        
        for pattern_key, patterns in self.failure_patterns.items():
            similarity = self._calculate_similarity(current_key, pattern_key)
            if similarity > 0.6:
                similar_failures.extend(patterns)
        
        if len(similar_failures) < 2:
            return None
        
        # Analyze common failure patterns
        common_contexts = []
        for failure in similar_failures:
            common_contexts.append(failure["context"])
        
        # Find common factors
        insights = []
        
        # Check regime pattern
        regimes = [c.get("regime", {}).get("regime", "") for c in common_contexts]
        if regimes.count(regimes[0]) > len(regimes) * 0.7:
            insights.append(f"Often fails in {regimes[0]} regime")
        
        # Check volatility pattern
        volatilities = [c.get("volatility", 0) for c in common_contexts]
        avg_vol = sum(volatilities) / len(volatilities)
        if avg_vol > 0.3:
            insights.append("High volatility leads to errors")
        
        # Check momentum pattern
        momenta = [c.get("momentum", 0) for c in common_contexts]
        avg_mom = sum(momenta) / len(momenta)
        if abs(avg_mom) > 0.5:
            insights.append(f"Strong momentum ({avg_mom:+.1f}%) causes misreads")
        
        if insights:
            return "Learning: " + "; ".join(insights)
        
        return None
    
    def _create_similarity_key(self, context: Dict[str, Any]) -> str:
        """
        Create a similarity key from context.
        
        This creates a simplified representation for fast comparison.
        """
        # Extract key features
        regime = context.get("regime", {}).get("regime", "UNKNOWN")
        volatility_bucket = self._bucketize(context.get("volatility", 0.1), [0.05, 0.1, 0.2, 0.3])
        momentum_bucket = self._bucketize(context.get("momentum", 0), [-1, -0.5, 0, 0.5, 1])
        rsi_bucket = self._bucketize(context.get("rsi", 50), [30, 40, 50, 60, 70])
        time_bucket = datetime.now().hour // 6  # 4 time buckets
        
        return f"{regime}_{volatility_bucket}_{momentum_bucket}_{rsi_bucket}_{time_bucket}"
    
    def _bucketize(self, value: float, thresholds: List[float]) -> str:
        """Bucketize a continuous value"""
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return f"<{threshold}"
        return f">={thresholds[-1]}"
    
    def _calculate_similarity(self, key1: str, key2: str) -> float:
        """
        Calculate similarity between two keys.
        
        Uses Jaccard similarity on key components.
        """
        parts1 = set(key1.split("_"))
        parts2 = set(key2.split("_"))
        
        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)
        
        return intersection / union if union > 0 else 0
    
    def _save_memories(self):
        """Save memories to disk"""
        os.makedirs(self.storage_path, exist_ok=True)
        
        filepath = os.path.join(self.storage_path, f"{self.agent_id}_memory.json")
        
        data = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "memories": [
                {
                    "timestamp": m.timestamp,
                    "context": m.context,
                    "prediction": m.prediction,
                    "outcome": m.outcome,
                    "similarity_key": m.similarity_key
                }
                for m in self.memories[-200:]  # Keep last 200
            ],
            "success_patterns": {
                k: v[-20:] for k, v in self.success_patterns.items()
            },
            "failure_patterns": {
                k: v[-20:] for k, v in self.failure_patterns.items()
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_memories(self):
        """Load memories from disk"""
        filepath = os.path.join(self.storage_path, f"{self.agent_id}_memory.json")
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            self.memories = [
                MemoryEntry(
                    timestamp=m["timestamp"],
                    context=m["context"],
                    prediction=m["prediction"],
                    outcome=m["outcome"],
                    similarity_key=m["similarity_key"]
                )
                for m in data.get("memories", [])
            ]
            
            self.success_patterns = data.get("success_patterns", {})
            self.failure_patterns = data.get("failure_patterns", {})
            
        except Exception as e:
            print(f"Error loading memories for {self.agent_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "agent_id": self.agent_id,
            "total_memories": len(self.memories),
            "success_patterns": len(self.success_patterns),
            "failure_patterns": len(self.failure_patterns),
            "unique_contexts": len(set(m.similarity_key for m in self.memories))
        }


class TeamMemory:
    """
    Shared memory for the entire agent team.
    Enables cross-agent learning.
    """
    
    def __init__(self, storage_path: str = "data/team_memory"):
        self.storage_path = storage_path
        self.agent_memories: Dict[str, AgentMemory] = {}
        self.shared_patterns: Dict[str, Any] = {}
        
        os.makedirs(storage_path, exist_ok=True)
    
    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """Get or create memory for an agent"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = AgentMemory(agent_id, self.storage_path)
        return self.agent_memories[agent_id]
    
    def share_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        contributing_agents: List[str]
    ):
        """
        Share a learned pattern across agents.
        
        Args:
            pattern_type: Type of pattern (e.g., "reversal_signal")
            pattern_data: Pattern data
            contributing_agents: Agents that contributed to this pattern
        """
        if pattern_type not in self.shared_patterns:
            self.shared_patterns[pattern_type] = []
        
        self.shared_patterns[pattern_type].append({
            "data": pattern_data,
            "contributors": contributing_agents,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_shared_patterns(self, pattern_type: str) -> List[Dict]:
        """Get shared patterns of a type"""
        return self.shared_patterns.get(pattern_type, [])
    
    def aggregate_learnings(self) -> Dict[str, Any]:
        """
        Aggregate learnings across all agents.
        
        Returns:
            Aggregated insights
        """
        all_learnings = []
        
        for agent_id, memory in self.agent_memories.items():
            stats = memory.get_stats()
            success_rate = (
                len(memory.success_patterns) / 
                max(1, len(memory.success_patterns) + len(memory.failure_patterns))
            )
            
            all_learnings.append({
                "agent_id": agent_id,
                "success_rate": success_rate,
                "total_memories": stats["total_memories"]
            })
        
        # Sort by success rate
        all_learnings.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return {
            "agent_rankings": all_learnings,
            "total_agents": len(self.agent_memories),
            "shared_patterns": len(self.shared_patterns)
        }
