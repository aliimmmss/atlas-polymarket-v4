"""
Atlas v4.0 Risk Management Module
Position sizing, expected value, and risk-adjusted confidence
"""

from .position_sizing import KellyPositionSizer, PositionSize, PositionSizer
from .expected_value import ExpectedValueCalculator, ExpectedValueResult
from .risk_adjusted_confidence import RiskAdjustedConfidence, RiskFactors, AdjustedConfidence

# Backwards compatibility aliases
EVResult = ExpectedValueResult
RiskBreakdown = RiskFactors
AdjustedResult = AdjustedConfidence

__all__ = [
    'KellyPositionSizer', 'PositionSize', 'PositionSizer',
    'ExpectedValueCalculator', 'ExpectedValueResult', 'EVResult',
    'RiskAdjustedConfidence', 'RiskFactors', 'AdjustedConfidence',
    'RiskBreakdown', 'AdjustedResult'
]
