"""
Atlas v4.0 Agents Module
Enhanced self-improving agent system with specialization
"""

from .atlas_agent import Agent, AgentTeam, AgentPrompt, AgentPerformance
from .specialized_agents import (
    TrendRider, TrendFader, RangeTrader, BreakoutHunter,
    VolatilityHarvester, RiskGuard, MeanReverter,
    WhaleWatcher, FlowAnalyst, FundingTrader, OITracker,
    SentimentSurfer, NewsReactor,
    get_all_specialized_agents
)
from .ensemble_voting import EnsembleVoting, VotingResult
from .meta_agent import MetaAgent, AgentSelection
from .agent_memory import AgentMemory, TeamMemory, MemoryEntry

__all__ = [
    'Agent', 'AgentTeam', 'AgentPrompt', 'AgentPerformance',
    'TrendRider', 'TrendFader', 'RangeTrader', 'BreakoutHunter',
    'VolatilityHarvester', 'RiskGuard', 'MeanReverter',
    'WhaleWatcher', 'FlowAnalyst', 'FundingTrader', 'OITracker',
    'SentimentSurfer', 'NewsReactor',
    'get_all_specialized_agents',
    'EnsembleVoting', 'VotingResult',
    'MetaAgent', 'AgentSelection',
    'AgentMemory', 'TeamMemory', 'MemoryEntry'
]
