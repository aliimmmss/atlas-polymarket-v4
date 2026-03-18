"""
Atlas v4.0 Analysis Module
Advanced technical analysis and market regime detection
"""

from .technical_indicators import TechnicalIndicators, IndicatorResult
from .regime_detector import RegimeDetector, MarketRegime, RegimeResult
from .multi_timeframe import MultiTimeframeAnalyzer, TimeframeConfluence, TrendDirection
from .signal_generator import SignalGenerator, SignalConfidence
from .confidence_scorer import ConfidenceScorer, ConfidenceBreakdown

__all__ = [
    'TechnicalIndicators',
    'IndicatorResult',
    'RegimeDetector',
    'MarketRegime',
    'RegimeResult',
    'MultiTimeframeAnalyzer',
    'TimeframeConfluence',
    'TrendDirection',
    'SignalGenerator',
    'SignalConfidence',
    'ConfidenceScorer',
    'ConfidenceBreakdown'
]
