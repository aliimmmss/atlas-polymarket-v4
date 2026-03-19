"""
Confidence Scorer for Atlas v4.0
Scores signal confidence based on multiple factors

ACADEMIC PAPER IMPLEMENTATIONS:
From "Prediction Markets as Bayesian Inverse Problems" (Madrigal-Cianci, 2026):

1. KL Divergence for Information Gain:
   - KL(P || Q) measures how much distribution P differs from Q
   - Used to measure information gain from market observations
   - Higher KL = more informative predictions

2. Information-Theoretic Metrics:
   - Realized Information Gain: KL(posterior || prior)
   - Expected Information Gain: Mutual information functional
   - Identifiability: KL separation between outcome-conditional laws

3. Bayesian Confidence:
   - Posterior probability with uncertainty bounds
   - Stability under perturbations
   - Outcome identifiability checks
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import math


def kl_divergence(p: float, q: float) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).
    
    For binary outcomes:
    D_KL(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
    
    From academic paper: KL divergence measures:
    1. Information gain when updating beliefs
    2. Distinguishability between outcomes
    3. Quality of predictions
    """
    # Handle boundary cases
    if p <= 0 or p >= 1 or q <= 0 or q >= 1:
        return 0.0
    
    try:
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    except (ValueError, ZeroDivisionError):
        return 0.0


def entropy(p: float) -> float:
    """
    Compute binary entropy H(p) = -p*log(p) - (1-p)*log(1-p).
    
    Entropy measures uncertainty:
    - H(0.5) = 1.0 (maximum uncertainty)
    - H(0) = H(1) = 0 (minimum uncertainty)
    """
    if p <= 0 or p >= 1:
        return 0.0
    
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def mutual_information(p_up: float, p_up_given_signal: float, p_signal: float) -> float:
    """
    Compute mutual information I(Y; X).
    
    From academic paper: Measures expected information gain
    from observing the signal about the outcome.
    
    I(Y; X) = H(Y) - H(Y | X)
    """
    h_y = entropy(p_up)  # Prior entropy
    
    # Conditional entropy
    h_y_given_x = p_signal * entropy(p_up_given_signal) + (1 - p_signal) * entropy(1 - p_up_given_signal)
    
    return h_y - h_y_given_x


@dataclass
class ConfidenceBreakdown:
    """Detailed confidence breakdown"""
    overall: float
    signal_strength: float
    historical_accuracy: float
    confluence: float
    data_quality: float
    regime_alignment: float
    factors: Dict[str, Any]
    # NEW: Academic paper metrics
    information_gain: float = 0.0
    kl_divergence: float = 0.0
    identifiability_score: float = 0.0
    belief_volatility_adjustment: float = 1.0


@dataclass
class BayesianConfidence:
    """
    Bayesian confidence with uncertainty bounds.
    
    From "Prediction Markets as Bayesian Inverse Problems":
    - Posterior probability with credible interval
    - Identifiability assessment
    - Information gain from observations
    """
    posterior_prob: float
    lower_bound: float  # 95% credible interval lower
    upper_bound: float  # 95% credible interval upper
    information_gain: float
    is_identifiable: bool
    stability_score: float
    confidence: float


class ConfidenceScorer:
    """
    Scores signal confidence based on multiple factors.
    
    ENHANCED with Bayesian methods from academic papers.
    
    Factors:
    - Signal strength (how extreme the reading)
    - Historical accuracy (past performance of this signal)
    - Confluence (agreement across indicators)
    - Data quality (freshness and completeness)
    - Regime alignment (signal matches current regime)
    - NEW: Information gain (KL divergence)
    - NEW: Belief volatility adjustment
    """
    
    def __init__(self):
        self.signal_history: Dict[str, List[Dict]] = {}
        self.accuracy_by_signal: Dict[str, float] = {}
        
        # NEW: Track belief evolution for Bayesian updates
        self.belief_history: List[float] = []
        self.prior_prob: float = 0.5
        
        # Track prediction-outcome relationships
        self.prediction_outcome_pairs: List[Dict] = []
    
    def score_signal(
        self,
        signal_type: str,
        value: float,
        context: Dict[str, Any]
    ) -> ConfidenceBreakdown:
        """
        Score a single signal's confidence.
        
        Args:
            signal_type: Type of signal (e.g., "rsi", "macd")
            value: Signal value
            context: Market context
        
        Returns:
            ConfidenceBreakdown with detailed scoring
        """
        factors = {}
        
        # 1. Signal strength (how extreme)
        strength = self._calculate_signal_strength(signal_type, value)
        factors["strength"] = strength
        
        # 2. Historical accuracy
        historical = self.accuracy_by_signal.get(signal_type, 0.5)
        factors["historical_accuracy"] = historical
        
        # 3. Confluence with other signals
        confluence = self._calculate_confluence(signal_type, context)
        factors["confluence"] = confluence
        
        # 4. Data quality
        data_quality = self._calculate_data_quality(context)
        factors["data_quality"] = data_quality
        
        # 5. Regime alignment
        regime_alignment = self._calculate_regime_alignment(signal_type, value, context)
        factors["regime_alignment"] = regime_alignment
        
        # NEW: 6. Information gain via KL divergence
        prior = context.get("prior_probability", 0.5)
        signal_prob = context.get("signal_probability", 0.5)
        info_gain = kl_divergence(signal_prob, prior) if signal_prob != prior else 0.0
        factors["information_gain"] = info_gain
        
        # NEW: 7. Belief volatility adjustment
        belief_vol = context.get("belief_volatility", 0.0)
        vol_adjustment = 1.0 - min(0.3, belief_vol * 0.5)
        factors["belief_volatility_adjustment"] = vol_adjustment
        
        # Calculate weighted overall confidence
        weights = {
            "strength": 0.20,
            "historical_accuracy": 0.15,
            "confluence": 0.20,
            "data_quality": 0.10,
            "regime_alignment": 0.10,
            "information_gain": 0.15,
            "belief_volatility_adjustment": 0.10
        }
        
        overall = sum(
            factors.get(k, 0.5) * weights[k] 
            for k in weights
        )
        
        # NEW: Compute identifiability score
        identifiability = self._compute_identifiability(signal_prob, prior, info_gain)
        
        return ConfidenceBreakdown(
            overall=overall,
            signal_strength=strength,
            historical_accuracy=historical,
            confluence=confluence,
            data_quality=data_quality,
            regime_alignment=regime_alignment,
            factors=factors,
            information_gain=info_gain,
            kl_divergence=info_gain,  # Same as info_gain for binary
            identifiability_score=identifiability,
            belief_volatility_adjustment=vol_adjustment
        )
    
    def compute_bayesian_confidence(
        self,
        prior_prob: float,
        likelihood_up: float,
        likelihood_down: float,
        n_observations: int = 1
    ) -> BayesianConfidence:
        """
        Compute Bayesian posterior confidence with uncertainty bounds.
        
        From "Prediction Markets as Bayesian Inverse Problems":
        
        P(Y=1 | observations) = P(observations | Y=1) * P(Y=1) / P(observations)
        
        This provides:
        1. Posterior probability
        2. Credible intervals (uncertainty bounds)
        3. Information gain (KL divergence from prior to posterior)
        4. Identifiability check (can we distinguish outcomes?)
        
        Args:
            prior_prob: Prior probability P(Y=1)
            likelihood_up: P(observations | Y=1)
            likelihood_down: P(observations | Y=0)
            n_observations: Number of observations (for confidence scaling)
        
        Returns:
            BayesianConfidence with posterior and uncertainty
        """
        # Compute posterior using Bayes' theorem
        # P(Y=1 | obs) = P(obs | Y=1) * P(Y=1) / P(obs)
        # P(obs) = P(obs | Y=1) * P(Y=1) + P(obs | Y=0) * P(Y=0)
        
        marginal = likelihood_up * prior_prob + likelihood_down * (1 - prior_prob)
        
        if marginal > 0:
            posterior = likelihood_up * prior_prob / marginal
        else:
            posterior = prior_prob  # Fallback to prior
        
        # Compute information gain (KL divergence from prior to posterior)
        info_gain = kl_divergence(posterior, prior_prob)
        
        # Compute credible interval using Beta distribution approximation
        # For binary outcomes with n observations, use Beta(n*posterior, n*(1-posterior))
        # 95% CI is approximately posterior ± 1.96 * sqrt(variance)
        if n_observations > 0:
            variance = posterior * (1 - posterior) / (n_observations + 1)
            std = math.sqrt(variance)
            lower = max(0.01, posterior - 1.96 * std)
            upper = min(0.99, posterior + 1.96 * std)
        else:
            lower = max(0.01, prior_prob - 0.1)
            upper = min(0.99, prior_prob + 0.1)
        
        # Check identifiability (can we distinguish UP from DOWN?)
        # High identifiability if KL(likelihood_up || likelihood_down) is large
        identifiability_kl = kl_divergence(likelihood_up, likelihood_down)
        is_identifiable = identifiability_kl > 0.1  # Threshold from academic paper
        
        # Stability score (how stable is the posterior under perturbations?)
        stability = self._compute_stability(posterior, likelihood_up, likelihood_down, prior_prob)
        
        # Overall confidence
        confidence = min(1.0, (info_gain * 2 + identifiability_kl + stability) / 3)
        
        return BayesianConfidence(
            posterior_prob=posterior,
            lower_bound=lower,
            upper_bound=upper,
            information_gain=info_gain,
            is_identifiable=is_identifiable,
            stability_score=stability,
            confidence=confidence
        )
    
    def _compute_identifiability(
        self,
        signal_prob: float,
        prior: float,
        info_gain: float
    ) -> float:
        """
        Compute identifiability score.
        
        From academic paper: Identifiability measures how well
        we can distinguish between outcomes based on observations.
        
        High identifiability = predictions are meaningful
        Low identifiability = predictions are not informative
        """
        if signal_prob <= 0 or signal_prob >= 1:
            return 0.0
        
        # KL divergence between signal and prior
        kl_signal_prior = kl_divergence(signal_prob, prior)
        
        # KL divergence between signal and 0.5 (neutral)
        kl_signal_neutral = kl_divergence(signal_prob, 0.5)
        
        # Combined identifiability
        identifiability = (kl_signal_prior + kl_signal_neutral) / 2
        
        return min(1.0, identifiability * 2)  # Normalize
    
    def _compute_stability(
        self,
        posterior: float,
        likelihood_up: float,
        likelihood_down: float,
        prior: float
    ) -> float:
        """
        Compute stability score.
        
        From academic paper: Stability measures how much the
        posterior changes under small perturbations of inputs.
        
        Stable predictions are more reliable.
        """
        # Perturb likelihoods slightly
        epsilon = 0.01
        
        perturbed_up = likelihood_up + epsilon
        perturbed_down = likelihood_down - epsilon
        
        # Bound
        perturbed_up = max(0.01, min(0.99, perturbed_up))
        perturbed_down = max(0.01, min(0.99, perturbed_down))
        
        # Compute perturbed posterior
        marginal_perturbed = perturbed_up * prior + perturbed_down * (1 - prior)
        if marginal_perturbed > 0:
            posterior_perturbed = perturbed_up * prior / marginal_perturbed
        else:
            posterior_perturbed = prior
        
        # Stability = 1 - change
        change = abs(posterior - posterior_perturbed)
        stability = max(0, 1 - change * 10)  # Scale for interpretation
        
        return stability
    
    def get_overall_confidence(
        self,
        signals: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence across all signals.
        
        ENHANCED: Uses information-theoretic metrics from academic paper.
        
        Args:
            signals: All generated signals
            market_context: Market context data
        
        Returns:
            Overall confidence score (0-1)
        """
        if not signals:
            return 0.5
        
        confidence_scores = []
        info_gains = []
        
        # Score each signal category
        categories = ["technical", "derivatives", "onchain", "sentiment", "multi_timeframe"]
        
        for category in categories:
            if category not in signals:
                continue
            
            cat_signals = signals[category]
            
            if isinstance(cat_signals, dict):
                # Get category confidence
                if category == "technical":
                    vote = cat_signals.get("technical_vote", {})
                    direction = vote.get("direction", "NEUTRAL")
                    up_pct = vote.get("up_percent", 0.5)
                    strength = abs(up_pct - 0.5) * 2
                    confidence_scores.append(strength)
                    
                    # NEW: Compute information gain
                    ig = kl_divergence(up_pct, 0.5)
                    info_gains.append(ig)
                
                elif category == "multi_timeframe":
                    conf = cat_signals.get("confidence", 0.5)
                    confluence = cat_signals.get("confluence_score", 50) / 100
                    confidence_scores.append((conf + confluence) / 2)
                
                else:
                    # Check for confidence indicators
                    if "confidence" in cat_signals:
                        confidence_scores.append(cat_signals["confidence"])
        
        # Adjust for regime alignment
        regime = signals.get("regime", {})
        regime_confidence = regime.get("confidence", 0.5)
        
        # Adjust for data freshness
        timestamp = signals.get("timestamp")
        if timestamp:
            try:
                data_age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
                if data_age > 60:  # More than 1 minute old
                    age_penalty = min(0.2, data_age / 300)  # Up to 20% penalty
                else:
                    age_penalty = 0
            except:
                age_penalty = 0
        else:
            age_penalty = 0
        
        # Calculate overall
        if confidence_scores:
            base_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # NEW: Add information gain bonus
            if info_gains:
                avg_info_gain = sum(info_gains) / len(info_gains)
                info_bonus = min(0.15, avg_info_gain * 0.5)  # Up to 15% bonus
                base_confidence += info_bonus
            
            adjusted_confidence = base_confidence * (1 - age_penalty)
            
            # Boost if regime confidence is high
            if regime_confidence > 0.7:
                adjusted_confidence = min(1, adjusted_confidence * 1.1)
        else:
            adjusted_confidence = 0.5
        
        # NEW: Apply belief volatility adjustment
        belief_vol = market_context.get("belief_volatility", 0.0)
        if belief_vol > 0.3:
            vol_penalty = min(0.2, belief_vol * 0.3)
            adjusted_confidence *= (1 - vol_penalty)
        
        return adjusted_confidence
    
    def compute_expected_information_gain(
        self,
        predicted_prob: float,
        prior_prob: float = 0.5
    ) -> float:
        """
        Compute expected information gain from a prediction.
        
        From "Prediction Markets as Bayesian Inverse Problems":
        Expected information gain is the mutual information between
        the prediction and the outcome.
        
        E[IG] = H(prior) - E[H(posterior)]
        
        Higher values = more informative predictions
        """
        # Entropy of prior
        h_prior = entropy(prior_prob)
        
        # Expected entropy of posterior (simplified)
        # After observing outcome, posterior is updated
        # We use predicted probability as "expected" posterior
        h_expected_posterior = (
            predicted_prob * entropy(predicted_prob) +
            (1 - predicted_prob) * entropy(1 - predicted_prob)
        )
        
        # Expected information gain
        eig = h_prior - h_expected_posterior
        
        return max(0, eig)
    
    def _calculate_signal_strength(self, signal_type: str, value: float) -> float:
        """Calculate signal strength based on value extremity"""
        
        # Define signal strength thresholds
        thresholds = {
            "rsi": {
                "neutral_range": (40, 60),
                "strong_range": (20, 80),
                "extreme_range": (0, 100)
            },
            "stochastic": {
                "neutral_range": (40, 60),
                "strong_range": (20, 80),
                "extreme_range": (0, 100)
            },
            "macd": {
                "neutral_range": (-0.001, 0.001),
                "strong_range": (-0.01, 0.01),
                "extreme_range": (-0.1, 0.1)
            },
            "funding_rate": {
                "neutral_range": (-0.0001, 0.0001),
                "strong_range": (-0.0005, 0.0005),
                "extreme_range": (-0.001, 0.001)
            }
        }
        
        if signal_type not in thresholds:
            return 0.5
        
        thresh = thresholds[signal_type]
        neutral_min, neutral_max = thresh["neutral_range"]
        strong_min, strong_max = thresh["strong_range"]
        
        # In neutral range
        if neutral_min <= value <= neutral_max:
            return 0.3
        
        # In strong range
        if strong_min <= value <= strong_max:
            # Calculate position within strong range
            if value < neutral_min:
                position = (value - strong_min) / (neutral_min - strong_min)
            else:
                position = (strong_max - value) / (strong_max - neutral_max)
            return 0.3 + (0.4 * position)
        
        # Extreme values
        return 0.7 + (0.3 * min(1, abs(value) / abs(strong_max if value > 0 else strong_min)))
    
    def _calculate_confluence(
        self,
        signal_type: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confluence with other signals"""
        
        # Get signal direction
        signal_direction = context.get("signals", {}).get(signal_type, {}).get("signal", "neutral")
        
        # Count agreeing signals
        agreeing = 0
        total = 0
        
        for other_type, other_signal in context.get("signals", {}).items():
            if other_type == signal_type:
                continue
            
            if isinstance(other_signal, dict):
                other_direction = other_signal.get("signal", other_signal.get("trend", "neutral"))
            else:
                other_direction = "neutral"
            
            total += 1
            
            if signal_direction.lower() == other_direction.lower():
                agreeing += 1
        
        if total == 0:
            return 0.5
        
        return agreeing / total
    
    def _calculate_data_quality(self, context: Dict[str, Any]) -> float:
        """Calculate data quality score"""
        
        quality = 1.0
        
        # Check data freshness
        timestamp = context.get("timestamp")
        if timestamp:
            try:
                age = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
                if age > 30:
                    quality *= 0.9
                if age > 60:
                    quality *= 0.8
                if age > 120:
                    quality *= 0.6
            except:
                pass
        
        # Check data completeness
        required_fields = ["prices", "highs", "lows", "closes"]
        missing = sum(1 for f in required_fields if f not in context)
        quality *= (1 - missing * 0.1)
        
        # Check data length
        prices = context.get("prices", [])
        if len(prices) < 30:
            quality *= 0.7
        elif len(prices) < 50:
            quality *= 0.9
        
        return max(0.3, quality)
    
    def _calculate_regime_alignment(
        self,
        signal_type: str,
        value: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate alignment with current market regime"""
        
        regime = context.get("regime", {}).get("regime", "ranging")
        
        # Signal effectiveness by regime
        regime_signal_effectiveness = {
            "trending_up": {
                "momentum": 0.9,
                "macd": 0.85,
                "rsi": 0.5,  # Less reliable in trends
                "bollinger": 0.4,
                "stochastic": 0.4
            },
            "trending_down": {
                "momentum": 0.9,
                "macd": 0.85,
                "rsi": 0.5,
                "bollinger": 0.4,
                "stochastic": 0.4
            },
            "ranging": {
                "rsi": 0.9,
                "stochastic": 0.85,
                "bollinger": 0.85,
                "macd": 0.4,  # Less reliable in ranges
                "momentum": 0.3
            },
            "volatile": {
                "bollinger": 0.7,
                "atr": 0.8,
                "rsi": 0.5,
                "momentum": 0.6
            },
            "breakout": {
                "volume": 0.9,
                "momentum": 0.8,
                "bollinger": 0.7,
                "rsi": 0.5
            },
            "reversal": {
                "rsi": 0.85,
                "stochastic": 0.8,
                "bollinger": 0.75,
                "macd": 0.6
            }
        }
        
        # Get effectiveness for this signal in current regime
        regime_effects = regime_signal_effectiveness.get(regime, {})
        effectiveness = regime_effects.get(signal_type, 0.5)
        
        return effectiveness
    
    def record_signal_outcome(
        self,
        signal_type: str,
        predicted_direction: str,
        actual_direction: str,
        predicted_prob: float = 0.5
    ):
        """
        Record signal outcome for accuracy tracking.
        
        ENHANCED: Also tracks for Bayesian updates.
        """
        
        if signal_type not in self.signal_history:
            self.signal_history[signal_type] = []
        
        correct = predicted_direction.lower() == actual_direction.lower()
        
        self.signal_history[signal_type].append({
            "predicted": predicted_direction,
            "actual": actual_direction,
            "correct": correct,
            "probability": predicted_prob,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update accuracy
        recent = self.signal_history[signal_type][-20:]  # Last 20 signals
        if recent:
            self.accuracy_by_signal[signal_type] = sum(1 for s in recent if s["correct"]) / len(recent)
        
        # NEW: Track for Bayesian calibration
        self.prediction_outcome_pairs.append({
            "predicted_prob": predicted_prob,
            "outcome": 1 if actual_direction.lower() == "up" else 0,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.prediction_outcome_pairs) > 100:
            self.prediction_outcome_pairs = self.prediction_outcome_pairs[-100:]
    
    def get_signal_stats(self, signal_type: str) -> Dict[str, Any]:
        """Get statistics for a signal type"""
        
        history = self.signal_history.get(signal_type, [])
        
        if not history:
            return {"accuracy": 0.5, "count": 0}
        
        correct = sum(1 for h in history if h["correct"])
        
        return {
            "accuracy": correct / len(history),
            "count": len(history),
            "recent_accuracy": self.accuracy_by_signal.get(signal_type, 0.5)
        }
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """
        Get calibration metrics for predictions.
        
        From academic paper: Calibration measures how well predicted
        probabilities match actual outcome frequencies.
        """
        if len(self.prediction_outcome_pairs) < 10:
            return {"calibration_score": 0.5, "samples": 0}
        
        # Bin predictions and compute calibration
        bins = {
            "0.0-0.2": [],
            "0.2-0.4": [],
            "0.4-0.6": [],
            "0.6-0.8": [],
            "0.8-1.0": []
        }
        
        for pair in self.prediction_outcome_pairs:
            prob = pair["predicted_prob"]
            if prob < 0.2:
                bins["0.0-0.2"].append(pair["outcome"])
            elif prob < 0.4:
                bins["0.2-0.4"].append(pair["outcome"])
            elif prob < 0.6:
                bins["0.4-0.6"].append(pair["outcome"])
            elif prob < 0.8:
                bins["0.6-0.8"].append(pair["outcome"])
            else:
                bins["0.8-1.0"].append(pair["outcome"])
        
        # Compute calibration error
        calibration_errors = []
        for bin_name, outcomes in bins.items():
            if len(outcomes) >= 5:
                bin_center = (float(bin_name.split("-")[0]) + float(bin_name.split("-")[1])) / 2
                actual_freq = sum(outcomes) / len(outcomes)
                calibration_errors.append(abs(bin_center - actual_freq))
        
        avg_calibration_error = sum(calibration_errors) / len(calibration_errors) if calibration_errors else 0.5
        calibration_score = max(0, 1 - avg_calibration_error * 2)
        
        return {
            "calibration_score": calibration_score,
            "calibration_error": avg_calibration_error,
            "samples": len(self.prediction_outcome_pairs),
            "bin_details": {k: len(v) for k, v in bins.items()}
        }
