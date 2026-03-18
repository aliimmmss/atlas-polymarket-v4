"""
Ensemble Voting Methods for Atlas v4.0
Multiple voting methods for aggregating agent predictions
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class VotingResult:
    """Result of ensemble voting"""
    direction: str
    probability: float
    confidence: float
    method: str
    agreement: float
    details: Dict[str, Any]


class EnsembleVoting:
    """
    Multiple voting methods for agent ensemble.
    
    Methods:
    1. Weighted Average - Standard weighted average
    2. Borda Count - Rank-based voting
    3. Confidence-Weighted - Weight by confidence
    4. Consensus Requiring - Require agreement threshold
    5. Bayesian Model Averaging - Bayesian approach
    6. Trimmed Mean - Remove outliers
    7. Median Aggregation - Use median
    """
    
    @staticmethod
    def weighted_average(predictions: List[Dict]) -> VotingResult:
        """
        Weighted average (current method).
        Weight by agent weight * confidence.
        """
        total_weight = 0
        weighted_up_prob = 0
        
        for pred in predictions:
            weight = pred["weight"]
            confidence = pred["confidence"]
            direction = pred["direction"]
            probability = pred["probability"]
            
            # Weight by both agent weight and confidence
            effective_weight = weight * confidence
            
            # Convert to UP probability
            if direction == "UP":
                up_prob = probability
            elif direction == "DOWN":
                up_prob = 1 - probability
            else:
                up_prob = 0.5
            
            weighted_up_prob += up_prob * effective_weight
            total_weight += effective_weight
        
        final_prob = weighted_up_prob / total_weight if total_weight > 0 else 0.5
        
        direction = "UP" if final_prob > 0.53 else "DOWN" if final_prob < 0.47 else "NEUTRAL"
        confidence = EnsembleVoting._calculate_confidence(predictions, final_prob)
        
        return VotingResult(
            direction=direction,
            probability=final_prob,
            confidence=confidence,
            method="weighted_average",
            agreement=EnsembleVoting._calculate_agreement(predictions),
            details={"total_weight": total_weight, "voters": len(predictions)}
        )
    
    @staticmethod
    def borda_count(predictions: List[Dict]) -> VotingResult:
        """
        Borda count voting.
        Rank predictions and assign scores.
        """
        # Sort by probability (lowest gets 0, highest gets n-1)
        sorted_preds = sorted(predictions, key=lambda p: p["probability"] if p["direction"] == "UP" else 1 - p["probability"])
        
        total_score = 0
        weighted_score = 0
        
        for rank, pred in enumerate(sorted_preds):
            weight = pred["weight"]
            
            # Borda score = rank (higher rank = higher score)
            score = rank * weight
            
            # Get UP probability
            if pred["direction"] == "UP":
                up_prob = pred["probability"]
            elif pred["direction"] == "DOWN":
                up_prob = 1 - pred["probability"]
            else:
                up_prob = 0.5
            
            weighted_score += up_prob * score
            total_score += score if score > 0 else 1  # Avoid division by zero
        
        final_prob = weighted_score / total_score if total_weight > 0 else 0.5
        
        direction = "UP" if final_prob > 0.53 else "DOWN" if final_prob < 0.47 else "NEUTRAL"
        
        return VotingResult(
            direction=direction,
            probability=final_prob,
            confidence=EnsembleVoting._calculate_confidence(predictions, final_prob),
            method="borda_count",
            agreement=EnsembleVoting._calculate_agreement(predictions),
            details={"borda_scores": {p["agent_id"]: i for i, p in enumerate(sorted_preds)}}
        )
    
    @staticmethod
    def confidence_weighted(predictions: List[Dict]) -> VotingResult:
        """
        Confidence-weighted voting.
        Higher confidence predictions get more weight.
        """
        total_confidence = 0
        weighted_prob = 0
        
        for pred in predictions:
            confidence = pred["confidence"]
            direction = pred["direction"]
            probability = pred["probability"]
            
            # Convert to UP probability
            if direction == "UP":
                up_prob = probability
            elif direction == "DOWN":
                up_prob = 1 - probability
            else:
                up_prob = 0.5
            
            # Weight by confidence squared (amplify high confidence)
            effective_weight = confidence ** 2
            
            weighted_prob += up_prob * effective_weight
            total_confidence += effective_weight
        
        final_prob = weighted_prob / total_confidence if total_confidence > 0 else 0.5
        
        direction = "UP" if final_prob > 0.53 else "DOWN" if final_prob < 0.47 else "NEUTRAL"
        
        return VotingResult(
            direction=direction,
            probability=final_prob,
            confidence=EnsembleVoting._calculate_confidence(predictions, final_prob),
            method="confidence_weighted",
            agreement=EnsembleVoting._calculate_agreement(predictions),
            details={"total_confidence_weight": total_confidence}
        )
    
    @staticmethod
    def consensus_requiring(predictions: List[Dict], threshold: float = 0.65) -> VotingResult:
        """
        Consensus-requiring voting.
        Only predict if agreement exceeds threshold.
        """
        agreement = EnsembleVoting._calculate_agreement(predictions)
        
        if agreement < threshold:
            return VotingResult(
                direction="NEUTRAL",
                probability=0.5,
                confidence=0.3,
                method="consensus_requiring",
                agreement=agreement,
                details={"reason": "insufficient_consensus", "threshold": threshold}
            )
        
        # If consensus reached, use weighted average
        result = EnsembleVoting.weighted_average(predictions)
        result.method = "consensus_requiring"
        result.details["consensus_reached"] = True
        
        return result
    
    @staticmethod
    def bayesian_averaging(predictions: List[Dict], prior_up: float = 0.5, prior_strength: float = 2.0) -> VotingResult:
        """
        Bayesian model averaging.
        Incorporate prior belief with observations.
        
        Args:
            predictions: Agent predictions
            prior_up: Prior probability of UP (default 50%)
            prior_strength: How strongly to weight the prior
        """
        # Calculate weighted likelihood from agents
        total_weight = 0
        weighted_likelihood = 0
        
        for pred in predictions:
            weight = pred["weight"] * pred["confidence"]
            direction = pred["direction"]
            probability = pred["probability"]
            
            # Convert to UP probability
            if direction == "UP":
                up_prob = probability
            elif direction == "DOWN":
                up_prob = 1 - probability
            else:
                up_prob = 0.5
            
            weighted_likelihood += up_prob * weight
            total_weight += weight
        
        # Normalize likelihood
        likelihood = weighted_likelihood / total_weight if total_weight > 0 else 0.5
        
        # Bayesian update
        # Posterior = (prior * prior_strength + likelihood * total_weight) / (prior_strength + total_weight)
        posterior = (prior_up * prior_strength + likelihood * total_weight) / (prior_strength + total_weight)
        
        direction = "UP" if posterior > 0.53 else "DOWN" if posterior < 0.47 else "NEUTRAL"
        
        return VotingResult(
            direction=direction,
            probability=posterior,
            confidence=abs(posterior - prior_up) * 2,  # Confidence from deviation from prior
            method="bayesian_averaging",
            agreement=EnsembleVoting._calculate_agreement(predictions),
            details={"prior": prior_up, "likelihood": likelihood, "posterior": posterior}
        )
    
    @staticmethod
    def trimmed_mean(predictions: List[Dict], trim_percent: float = 0.1) -> VotingResult:
        """
        Trimmed mean voting.
        Remove extreme predictions before averaging.
        """
        # Get all UP probabilities
        probs = []
        for pred in predictions:
            if pred["direction"] == "UP":
                up_prob = pred["probability"]
            elif pred["direction"] == "DOWN":
                up_prob = 1 - pred["probability"]
            else:
                up_prob = 0.5
            
            probs.append({
                "agent_id": pred["agent_id"],
                "probability": up_prob,
                "weight": pred["weight"]
            })
        
        # Sort by probability
        probs.sort(key=lambda x: x["probability"])
        
        # Trim extremes
        trim_count = int(len(probs) * trim_percent)
        if trim_count > 0 and len(probs) > trim_count * 2:
            trimmed = probs[trim_count:-trim_count]
        else:
            trimmed = probs
        
        # Calculate weighted mean
        total_weight = 0
        weighted_prob = 0
        
        for p in trimmed:
            weighted_prob += p["probability"] * p["weight"]
            total_weight += p["weight"]
        
        final_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        
        direction = "UP" if final_prob > 0.53 else "DOWN" if final_prob < 0.47 else "NEUTRAL"
        
        return VotingResult(
            direction=direction,
            probability=final_prob,
            confidence=EnsembleVoting._calculate_confidence(predictions, final_prob),
            method="trimmed_mean",
            agreement=EnsembleVoting._calculate_agreement(predictions),
            details={"trimmed_count": trim_count * 2, "remaining": len(trimmed)}
        )
    
    @staticmethod
    def median_aggregation(predictions: List[Dict]) -> VotingResult:
        """
        Median aggregation.
        Use median probability, robust to outliers.
        """
        probs = []
        for pred in predictions:
            if pred["direction"] == "UP":
                up_prob = pred["probability"]
            elif pred["direction"] == "DOWN":
                up_prob = 1 - pred["probability"]
            else:
                up_prob = 0.5
            
            probs.append(up_prob)
        
        # Calculate median
        probs.sort()
        n = len(probs)
        
        if n == 0:
            median_prob = 0.5
        elif n % 2 == 0:
            median_prob = (probs[n//2 - 1] + probs[n//2]) / 2
        else:
            median_prob = probs[n//2]
        
        direction = "UP" if median_prob > 0.53 else "DOWN" if median_prob < 0.47 else "NEUTRAL"
        
        return VotingResult(
            direction=direction,
            probability=median_prob,
            confidence=EnsembleVoting._calculate_confidence(predictions, median_prob),
            method="median_aggregation",
            agreement=EnsembleVoting._calculate_agreement(predictions),
            details={"median": median_prob, "min": min(probs), "max": max(probs)}
        )
    
    @staticmethod
    def supermajority(predictions: List[Dict], threshold: float = 0.6) -> VotingResult:
        """
        Supermajority voting.
        Require supermajority to predict a direction.
        """
        up_votes = sum(1 for p in predictions if p["direction"] == "UP")
        down_votes = sum(1 for p in predictions if p["direction"] == "DOWN")
        total = up_votes + down_votes
        
        up_ratio = up_votes / total if total > 0 else 0.5
        down_ratio = down_votes / total if total > 0 else 0.5
        
        if up_ratio >= threshold:
            direction = "UP"
            probability = up_ratio
        elif down_ratio >= threshold:
            direction = "DOWN"
            probability = 1 - down_ratio
        else:
            direction = "NEUTRAL"
            probability = 0.5
        
        return VotingResult(
            direction=direction,
            probability=probability,
            confidence=max(up_ratio, down_ratio),
            method="supermajority",
            agreement=max(up_ratio, down_ratio),
            details={"up_ratio": up_ratio, "down_ratio": down_ratio, "threshold": threshold}
        )
    
    @staticmethod
    def best_method_selection(predictions: List[Dict]) -> VotingResult:
        """
        Select the best voting method based on conditions.
        
        Rules:
        - High agreement (>70%): Use weighted average
        - Medium agreement (50-70%): Use trimmed mean
        - Low agreement (<50%): Use consensus requiring
        - High confidence variance: Use confidence weighted
        """
        agreement = EnsembleVoting._calculate_agreement(predictions)
        confidence_variance = EnsembleVoting._calculate_confidence_variance(predictions)
        
        # Select method based on conditions
        if confidence_variance > 0.15:
            # High variance in confidence - use confidence weighted
            return EnsembleVoting.confidence_weighted(predictions)
        elif agreement > 0.7:
            # High agreement - use weighted average
            return EnsembleVoting.weighted_average(predictions)
        elif agreement > 0.5:
            # Medium agreement - use trimmed mean
            return EnsembleVoting.trimmed_mean(predictions)
        else:
            # Low agreement - require consensus
            return EnsembleVoting.consensus_requiring(predictions, threshold=0.6)
    
    # Helper methods
    
    @staticmethod
    def _calculate_agreement(predictions: List[Dict]) -> float:
        """Calculate agreement ratio"""
        up_votes = sum(1 for p in predictions if p["direction"] == "UP")
        down_votes = sum(1 for p in predictions if p["direction"] == "DOWN")
        total = up_votes + down_votes
        
        return max(up_votes, down_votes) / total if total > 0 else 0.5
    
    @staticmethod
    def _calculate_confidence(predictions: List[Dict], final_prob: float) -> float:
        """Calculate overall confidence"""
        agreement = EnsembleVoting._calculate_agreement(predictions)
        deviation = abs(final_prob - 0.5)
        
        return agreement * (0.5 + deviation)
    
    @staticmethod
    def _calculate_confidence_variance(predictions: List[Dict]) -> float:
        """Calculate variance of confidence scores"""
        confidences = [p["confidence"] for p in predictions]
        
        if len(confidences) < 2:
            return 0
        
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        
        return variance


def aggregate_with_multiple_methods(predictions: List[Dict]) -> Dict[str, Any]:
    """
    Run multiple voting methods and return all results.
    Useful for comparison and meta-analysis.
    """
    methods = [
        ("weighted_average", EnsembleVoting.weighted_average),
        ("borda_count", EnsembleVoting.borda_count),
        ("confidence_weighted", EnsembleVoting.confidence_weighted),
        ("consensus_requiring", lambda p: EnsembleVoting.consensus_requiring(p, 0.65)),
        ("bayesian_averaging", EnsembleVoting.bayesian_averaging),
        ("trimmed_mean", EnsembleVoting.trimmed_mean),
        ("median_aggregation", EnsembleVoting.median_aggregation),
        ("supermajority", EnsembleVoting.supermajority),
    ]
    
    results = {}
    for name, method in methods:
        try:
            result = method(predictions)
            results[name] = {
                "direction": result.direction,
                "probability": result.probability,
                "confidence": result.confidence,
                "agreement": result.agreement
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    # Add best method selection
    best = EnsembleVoting.best_method_selection(predictions)
    results["best_selection"] = {
        "direction": best.direction,
        "probability": best.probability,
        "confidence": best.confidence,
        "method": best.method
    }
    
    return results
