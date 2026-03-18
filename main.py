#!/usr/bin/env python3
"""
Atlas v4.0 - Bitcoin 15-Minute Prediction System
Enhanced with multi-source data, advanced analysis, and backtesting

Main entry point for the prediction system.
"""

# CRITICAL: Fix for Windows + Python 3.10+ + aiohttp/aiodns compatibility
# Must be done BEFORE any asyncio imports
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import argparse
import json
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import all modules
from src.proxy.free_claude_proxy import FreeClaudeProxy
from src.data.binance_feed import BitcoinPriceMonitor, BinanceClient
from src.data.derivatives_feed import DerivativesFeed
from src.data.onchain_feed import OnChainFeed
from src.data.price_aggregator import PriceAggregator
from src.data.sentiment_feed import SentimentFeed
from src.data.market_sync import PolymarketSync, format_countdown, format_timestamp
from src.analysis.signal_generator import SignalGenerator
from src.analysis.regime_detector import RegimeDetector, MarketRegime
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.analysis.confidence_scorer import ConfidenceScorer
from src.agents.atlas_agent import AgentTeam
from src.agents.specialized_agents import get_all_specialized_agents
from src.agents.meta_agent import MetaAgent
from src.agents.ensemble_voting import EnsembleVoting
from src.agents.agent_memory import TeamMemory
from src.risk.position_sizing import KellyPositionSizer
from src.risk.expected_value import ExpectedValueCalculator
from src.risk.risk_adjusted_confidence import RiskAdjustedConfidence
from src.backtest.data_store import HistoricalDataStore
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.attribution import PerformanceAttribution

load_dotenv()
console = Console()


@dataclass
class PredictionRecord:
    """Complete prediction record for tracking and learning"""
    prediction_id: str
    timestamp: str
    
    # Market window info
    market_slug: str
    window_start: int
    window_end: int
    
    # Price data at prediction time
    start_price: float
    ptb: Optional[float]  # Price to Beat
    up_odds: Optional[float]
    down_odds: Optional[float]
    
    # Prediction details
    predicted_direction: str
    predicted_probability: float
    confidence: float
    
    # Agent predictions
    agent_predictions: List[Dict]
    
    # Signal summary
    regime: str
    confluence: float
    
    # Outcome (filled after window closes)
    end_price: Optional[float] = None
    actual_direction: Optional[str] = None
    actual_outcome: Optional[bool] = None  # True = UP won, False = DOWN won
    price_change_percent: Optional[float] = None
    brier_score: Optional[float] = None
    correct: Optional[bool] = None
    resolved_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PredictionRecord":
        return cls(**data)


class PredictionHistory:
    """Manages prediction history storage and retrieval"""
    
    def __init__(self, filepath: str = "data/prediction_history.json"):
        self.filepath = filepath
        self.predictions: List[PredictionRecord] = []
        self._load()
    
    def _load(self):
        """Load prediction history from file"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.predictions = [PredictionRecord.from_dict(p) for p in data.get('predictions', [])]
            except Exception as e:
                console.print(f"[yellow]Could not load prediction history: {e}[/]")
                self.predictions = []
    
    def _save(self):
        """Save prediction history to file"""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump({
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "total_predictions": len(self.predictions),
                "predictions": [p.to_dict() for p in self.predictions]
            }, f, indent=2)
    
    def add_prediction(self, record: PredictionRecord):
        """Add a new prediction record"""
        self.predictions.append(record)
        self._save()
    
    def update_outcome(self, prediction_id: str, outcome_data: Dict):
        """Update a prediction with its outcome"""
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                pred.end_price = outcome_data.get('end_price')
                pred.actual_direction = outcome_data.get('actual_direction')
                pred.actual_outcome = outcome_data.get('actual_outcome')
                pred.price_change_percent = outcome_data.get('price_change_percent')
                pred.brier_score = outcome_data.get('brier_score')
                pred.correct = outcome_data.get('correct')
                pred.resolved_at = outcome_data.get('resolved_at')
                break
        self._save()
    
    def get_unresolved(self) -> List[PredictionRecord]:
        """Get all predictions that haven't been resolved yet"""
        return [p for p in self.predictions if p.resolved_at is None and p.window_end <= int(time.time())]
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics"""
        resolved = [p for p in self.predictions if p.resolved_at is not None]
        
        if not resolved:
            return {
                "total_predictions": len(self.predictions),
                "resolved": 0,
                "pending": len(self.predictions)
            }
        
        correct = sum(1 for p in resolved if p.correct)
        total_brier = sum(p.brier_score for p in resolved if p.brier_score is not None)
        
        return {
            "total_predictions": len(self.predictions),
            "resolved": len(resolved),
            "pending": len(self.predictions) - len(resolved),
            "correct": correct,
            "incorrect": len(resolved) - correct,
            "win_rate": correct / len(resolved) if resolved else 0,
            "average_brier": total_brier / len(resolved) if resolved else 0.25
        }


class AtlasV4:
    """
    Main Atlas v4.0 prediction system.
    
    Integrates:
    - Multi-source data pipeline
    - Advanced technical analysis
    - Regime detection
    - Multi-timeframe analysis
    - Specialized agent team
    - Risk management
    - Backtesting
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
        # Data feeds
        self.price_monitor = BitcoinPriceMonitor()
        self.derivatives_feed = DerivativesFeed()
        self.onchain_feed = OnChainFeed()
        self.price_aggregator = PriceAggregator()
        self.sentiment_feed = SentimentFeed()
        self.polymarket = PolymarketSync()
        
        # Analysis
        self.signal_generator = SignalGenerator()
        self.regime_detector = RegimeDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.confidence_scorer = ConfidenceScorer()
        
        # Agents
        self.agent_team = AgentTeam(llm_client=llm_client)
        self.specialized_agents = get_all_specialized_agents(llm_client)
        self.meta_agent = MetaAgent(llm_client=llm_client)
        self.team_memory = TeamMemory()
        
        # Risk management
        self.position_sizer = KellyPositionSizer()
        self.ev_calculator = ExpectedValueCalculator()
        self.risk_confidence = RiskAdjustedConfidence()
        
        # State
        self.current_market = None
        self.current_prediction = None
        self.running = False
        
        # Prediction history
        self.prediction_history = PredictionHistory()
    
    async def gather_all_data(self) -> Dict[str, Any]:
        """Gather data from all sources"""
        
        console.print("[cyan]Gathering market data...[/]")
        
        # Gather all data in parallel
        tasks = [
            self.price_monitor.get_market_context(),
            self._get_derivatives_data(),
            self._get_onchain_data(),
            self._get_aggregated_price(),
            self._get_sentiment_data(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = results[0] if not isinstance(results[0], Exception) else {}
        derivatives = results[1] if not isinstance(results[1], Exception) else {}
        onchain = results[2] if not isinstance(results[2], Exception) else {}
        price_agg = results[3] if not isinstance(results[3], Exception) else {}
        sentiment = results[4] if not isinstance(results[4], Exception) else {}
        
        # Combine all data
        combined_data = {
            **market_data,
            "derivatives": derivatives,
            "onchain": onchain,
            "price_aggregated": price_agg,
            "sentiment": sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
        return combined_data
    
    async def _get_derivatives_data(self) -> Dict[str, Any]:
        """Get derivatives data"""
        try:
            async with self.derivatives_feed as feed:
                spot_price = 70000.0  # Will be replaced
                return await feed.get_full_derivatives_context(spot_price)
        except Exception as e:
            console.print(f"[yellow]Derivatives feed error: {e}[/]")
            return {}
    
    async def _get_onchain_data(self) -> Dict[str, Any]:
        """Get on-chain data"""
        try:
            async with self.onchain_feed as feed:
                return await feed.get_full_onchain_context()
        except Exception as e:
            console.print(f"[yellow]On-chain feed error: {e}[/]")
            return {}
    
    async def _get_aggregated_price(self) -> Dict[str, Any]:
        """Get aggregated price"""
        try:
            async with self.price_aggregator as agg:
                return await agg.get_aggregated_price()
        except Exception as e:
            console.print(f"[yellow]Price aggregator error: {e}[/]")
            return {}
    
    async def _get_sentiment_data(self) -> Dict[str, Any]:
        """Get sentiment data"""
        try:
            async with self.sentiment_feed as feed:
                return await feed.get_aggregate_sentiment()
        except Exception as e:
            console.print(f"[yellow]Sentiment feed error: {e}[/]")
            return {}
    
    async def make_prediction(self) -> Dict[str, Any]:
        """Make a comprehensive prediction"""
        
        # Gather all data
        market_data = await self.gather_all_data()
        
        # Generate signals
        console.print("[cyan]Analyzing market signals...[/]")
        signals = self.signal_generator.generate_signals(market_data)
        
        # Detect regime
        regime_result = self.regime_detector.detect_regime(
            prices=market_data.get("prices", {}).get("5m", []),
            highs=[c.get("high", 0) for c in market_data.get("candles", {}).get("5m", [])],
            lows=[c.get("low", 0) for c in market_data.get("candles", {}).get("5m", [])],
            closes=market_data.get("prices", {}).get("5m", []),
            volumes=[c.get("volume", 0) for c in market_data.get("candles", {}).get("5m", [])]
        )
        
        signals["regime"] = {
            "regime": regime_result.regime.value,
            "confidence": regime_result.confidence,
            "risk_level": regime_result.risk_level,
            "recommended_agents": regime_result.recommended_agents
        }
        
        # Multi-timeframe analysis
        mtf_result = self.mtf_analyzer.analyze_all_timeframes(
            market_data.get("candles", {})
        )
        
        signals["multi_timeframe"] = {
            "overall_trend": mtf_result.overall_trend.value,
            "confluence_score": mtf_result.confluence_score,
            "trading_signal": mtf_result.trading_signal,
            "confidence": mtf_result.confidence
        }
        
        # Get agent predictions
        console.print("[cyan]Consulting agents...[/]")
        
        # Use meta-agent to select agents
        selection = await self.meta_agent.select_agents(
            signals,
            self.agent_team.get_stats()["agent_performance"],
            self.agent_team.agents
        )
        
        # Get team prediction
        prediction = await self.agent_team.predict(
            signals, 
            current_regime=regime_result.regime.value
        )
        
        # Apply ensemble voting
        voting_results = EnsembleVoting.best_method_selection(
            prediction["agent_predictions"]
        )
        
        # Adjust confidence for risk
        adjusted = self.risk_confidence.calculate_risk_adjusted_confidence(
            base_confidence=prediction["final"]["confidence"],
            market_context=signals,
            agent_predictions=prediction["agent_predictions"]
        )
        
        prediction["final"]["confidence"] = adjusted.adjusted_confidence
        prediction["final"]["should_trade"] = adjusted.should_trade
        
        # Calculate expected value
        if self.current_market:
            ev = self.ev_calculator.calculate_ev(
                predicted_prob=prediction["final"]["probability"],
                market_odds=self.current_market.up_price or 0.5,
                confidence=adjusted.adjusted_confidence,
                direction=prediction["final"]["direction"]
            )
            prediction["ev"] = {
                "expected_value": ev.expected_value,
                "edge": ev.edge,
                "recommendation": ev.recommendation,
                "is_positive_ev": ev.is_positive_ev
            }
        
        # Calculate position size
        if self.current_market:
            pos_size = self.position_sizer.calculate_size(
                probability=prediction["final"]["probability"],
                confidence=adjusted.adjusted_confidence
            )
            prediction["position_size"] = {
                "size": pos_size.size,
                "size_percent": pos_size.size_percent,
                "kelly_fraction": pos_size.kelly_fraction
            }
        
        # Add all analysis to prediction
        prediction["signals"] = signals
        prediction["regime"] = regime_result.regime.value
        prediction["confluence"] = mtf_result.confluence_score
        prediction["voting_method"] = voting_results.method
        prediction["risk_adjustment"] = {
            "original": adjusted.original_confidence,
            "adjusted": adjusted.adjusted_confidence,
            "factors": {
                "volatility_risk": adjusted.risk_breakdown.volatility_risk,
                "agent_disagreement": adjusted.risk_breakdown.agent_disagreement
            }
        }
        
        return prediction
    
    def display_prediction(self, prediction: Dict[str, Any]):
        """Display prediction with rich formatting"""
        
        final = prediction["final"]
        direction = final["direction"]
        
        # Get timestamp
        pred_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        market_slug = PolymarketSync.get_market_slug()
        
        if direction == "UP":
            emoji, color = "📈", "green"
        elif direction == "DOWN":
            emoji, color = "📉", "red"
        else:
            emoji, color = "➡️", "yellow"
        
        # Current price info
        current_price = prediction.get("signals", {}).get("current_price", 0)
        price_change = prediction.get("signals", {}).get("price_change_24h", 0)
        
        # Market info
        ptb = self.current_market.ptb if self.current_market else None
        up_odds = self.current_market.up_price if self.current_market else None
        down_odds = self.current_market.down_price if self.current_market else None
        
        # Main prediction panel
        panel_content = f"""
[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold cyan]Prediction Time:[/] {pred_time}
[bold cyan]Market:[/] {market_slug}
[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

{emoji} [bold]PREDICTION: [{color}]{direction}[/{color}][/]

[bold cyan]Probability:[/] {final['probability']:.1%} UP | {(1-final['probability']):.1%} DOWN
[bold cyan]Confidence:[/] {final['confidence']:.0%} (Risk-adjusted)
[bold cyan]Regime:[/] {prediction.get('regime', 'Unknown')}
[bold cyan]Confluence:[/] {prediction.get('confluence', 0):.0f}%

[bold yellow]Risk Factors:[/]
  Volatility: {prediction.get('risk_adjustment', {}).get('factors', {}).get('volatility_risk', 0):.0%}
  Agent Disagreement: {prediction.get('risk_adjustment', {}).get('factors', {}).get('agent_disagreement', 0):.0%}

[bold]Should Trade:[/] {'✓ Yes' if final.get('should_trade', True) else '✗ No'}
"""
        
        # Add current price info
        if current_price:
            price_color = "green" if price_change >= 0 else "red"
            panel_content += f"""
[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold cyan]Current BTC Price:[/] ${current_price:,.2f} [{price_color}]({price_change:+.2f}% 24h)[/{price_color}]
"""
        
        # Add PTB info
        if ptb:
            panel_content += f"""
[bold magenta]Price to Beat:[/] ${ptb:,.2f}
"""
            if current_price:
                diff = current_price - ptb
                diff_pct = (diff / ptb) * 100
                diff_color = "green" if diff >= 0 else "red"
                direction_hint = "↑ ABOVE" if diff >= 0 else "↓ BELOW"
                panel_content += f"""
[bold]Position vs PTB:[/] [{diff_color}]{direction_hint} PTB by {abs(diff_pct):.3f} ({diff:+,.2f})[/{diff_color}]
"""
        
        # Add market odds
        if up_odds and down_odds:
            panel_content += f"""
[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold cyan]Polymarket Odds:[/] UP [green]{up_odds:.1%}[/] | DOWN [red]{down_odds:.1%}[/]
"""
        
        if "ev" in prediction:
            ev = prediction["ev"]
            ev_color = "green" if ev['expected_value'] > 0 else "red"
            panel_content += f"""
[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
[bold green]Expected Value:[/] [{ev_color}]{ev['expected_value']:.2%}[/{ev_color}]
[bold green]Edge vs Market:[/] [{ev_color}]{ev['edge']:.2%}[/{ev_color}]
[bold green]Recommendation:[/] {ev['recommendation']}
"""
        
        if "position_size" in prediction:
            ps = prediction["position_size"]
            panel_content += f"""
[bold magenta]Kelly Position Size:[/] ${ps['size']:.2f} ({ps['size_percent']:.1f}%)
[bold magenta]Kelly Fraction:[/] {ps['kelly_fraction']:.2f}
"""
        
        console.print(Panel(
            panel_content.strip(),
            title="[bold]⚡ Atlas v4.0 Prediction[/]",
            border_style=color
        ))
        
        # Agent votes table
        table = Table(title="🤖 Agent Votes", show_header=True)
        table.add_column("Agent", style="white", width=20)
        table.add_column("Vote", width=8)
        table.add_column("Prob", width=8)
        table.add_column("Weight", width=8)
        table.add_column("Confidence", width=10)
        
        for ap in prediction["agent_predictions"][:10]:
            vote_color = "green" if ap["direction"] == "UP" else "red" if ap["direction"] == "DOWN" else "white"
            confidence = ap.get("confidence", 0.5)
            table.add_row(
                ap["agent_name"],
                f"[{vote_color}]{ap['direction']}[/]",
                f"{ap['probability']:.0%}",
                f"{ap['weight']:.2f}",
                f"{confidence:.0%}"
            )
        
        console.print(table)
        
        # Technical signals summary
        tech_signals = prediction.get("signals", {}).get("technical", {})
        if tech_signals:
            tech_table = Table(title="📊 Technical Signals", show_header=True)
            tech_table.add_column("Indicator", style="cyan", width=15)
            tech_table.add_column("Value", width=12)
            tech_table.add_column("Signal", width=12)
            
            indicators = ["rsi", "macd", "bollinger", "stochastic", "momentum", "atr"]
            for ind in indicators:
                if ind in tech_signals:
                    data = tech_signals[ind]
                    if isinstance(data, dict):
                        if ind == "rsi":
                            val = f"{data.get('value', 0):.1f}"
                            sig = data.get("signal", "neutral")
                        elif ind == "macd":
                            val = f"{data.get('macd', 0):.2f}"
                            sig = data.get("trend", "neutral")
                        elif ind == "bollinger":
                            val = f"{data.get('position', 0):.1%}"
                            sig = data.get("signal", "neutral")
                        elif ind == "stochastic":
                            val = f"K:{data.get('k', 50):.1f}"
                            sig = data.get("signal", "neutral")
                        elif ind == "momentum":
                            val = f"{data.get('value', 0):.2f}%"
                            sig = data.get("signal", "neutral")
                        elif ind == "atr":
                            val = f"{data.get('percent', 0):.2f}%"
                            sig = "volatility"
                        else:
                            continue
                        
                        sig_color = "green" if sig in ["bullish", "oversold", "up", "weak_bullish"] else "red" if sig in ["bearish", "overbought", "down", "weak_bearish"] else "yellow"
                        tech_table.add_row(ind.upper(), val, f"[{sig_color}]{sig}[/]")
            
            console.print(tech_table)
    
    def save_state(self):
        """Save system state"""
        self.agent_team.save_state("data/agent_state.json")
        self.position_sizer._save_stats() if hasattr(self.position_sizer, '_save_stats') else None
        console.print("[green]State saved successfully[/]")


async def run_single_prediction():
    """Run a single prediction"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold]  ATLAS v4.0 - Bitcoin 15-Minute Prediction System[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")
    
    # Initialize LLM
    try:
        llm_client = FreeClaudeProxy()
        console.print("[green]✓ LLM Provider connected[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ LLM unavailable: {e}[/]")
        llm_client = None
    
    # Initialize Atlas
    atlas = AtlasV4(llm_client=llm_client)
    
    # Load previous state
    if atlas.agent_team.load_state("data/agent_state.json"):
        stats = atlas.agent_team.get_stats()
        console.print(f"[green]✓ Loaded previous learning[/]")
    
    # Make prediction
    prediction = await atlas.make_prediction()
    
    # Display results
    atlas.display_prediction(prediction)
    
    # Save state
    atlas.save_state()


def display_market_info(market=None, current_price: float = None):
    """Display market window information"""
    now = datetime.now()
    now_utc = datetime.now(timezone.utc)
    
    # Get market times
    next_start, next_end = PolymarketSync.get_next_market_times()
    current_start, current_end = PolymarketSync.get_current_market_times()
    
    # Create info table
    info_table = Table(show_header=False, box=None, expand=True)
    info_table.add_column("Key", style="cyan", width=20)
    info_table.add_column("Value", style="white")
    
    # Time info
    info_table.add_row("🕐 Local Time", now.strftime("%Y-%m-%d %H:%M:%S"))
    info_table.add_row("🌐 UTC Time", now_utc.strftime("%Y-%m-%d %H:%M:%S"))
    info_table.add_row("", "")
    
    # Current market window
    info_table.add_row("[bold]CURRENT WINDOW[/]", "")
    info_table.add_row("  Market Slug", f"[yellow]{PolymarketSync.get_market_slug()}[/]")
    info_table.add_row("  Window Start", format_timestamp(current_start))
    info_table.add_row("  Window End", format_timestamp(current_end))
    
    remaining = PolymarketSync.seconds_remaining_in_current()
    info_table.add_row("  Time Remaining", f"[red]{format_countdown(remaining)}[/]")
    
    info_table.add_row("", "")
    
    # Next market window
    info_table.add_row("[bold]NEXT WINDOW[/]", "")
    info_table.add_row("  Market Slug", f"[yellow]{PolymarketSync.get_next_market_slug()}[/]")
    info_table.add_row("  Starts At", format_timestamp(next_start))
    info_table.add_row("  Starts In", f"[green]{format_countdown(PolymarketSync.seconds_until_next_market())}[/]")
    
    if market and market.ptb:
        info_table.add_row("", "")
        info_table.add_row("[bold]MARKET DATA[/]", "")
        info_table.add_row("  Price to Beat", f"[magenta]${market.ptb:,.2f}[/]")
        if current_price:
            diff = current_price - market.ptb
            diff_pct = (diff / market.ptb) * 100
            diff_color = "green" if diff >= 0 else "red"
            info_table.add_row("  Current Price", f"[${current_price:,.2f}] [{diff_color}]({diff_pct:+.3f}%)[/{diff_color}]")
        
        if market.up_price and market.down_price:
            info_table.add_row("  UP Odds", f"[green]{market.up_price:.2%}[/]")
            info_table.add_row("  DOWN Odds", f"[red]{market.down_price:.2%}[/]")
    
    console.print(Panel(
        info_table,
        title="[bold]📊 Market Window Info[/]",
        border_style="blue"
    ))


def display_performance_stats(history: PredictionHistory):
    """Display performance statistics"""
    stats = history.get_stats()
    
    if stats["resolved"] == 0:
        return
    
    console.print("\n" + "═" * 60)
    console.print("[bold cyan]📊 Performance Statistics[/]")
    console.print("═" * 60)
    
    perf_table = Table(show_header=True)
    perf_table.add_column("Metric", style="cyan", width=25)
    perf_table.add_column("Value", style="white", width=20)
    
    perf_table.add_row("Total Predictions", str(stats["total_predictions"]))
    perf_table.add_row("Resolved", str(stats["resolved"]))
    perf_table.add_row("Pending", str(stats["pending"]))
    
    if stats["resolved"] > 0:
        win_color = "green" if stats["win_rate"] > 0.55 else "red" if stats["win_rate"] < 0.50 else "yellow"
        perf_table.add_row("Win Rate", f"[{win_color}]{stats['win_rate']:.1%}[/{win_color}]")
        perf_table.add_row("Correct", f"[green]{stats['correct']}[/]")
        perf_table.add_row("Incorrect", f"[red]{stats['incorrect']}[/]")
        
        brier_color = "green" if stats["average_brier"] < 0.20 else "red" if stats["average_brier"] > 0.25 else "yellow"
        perf_table.add_row("Avg Brier Score", f"[{brier_color}]{stats['average_brier']:.4f}[/{brier_color}]")
    
    console.print(perf_table)


def display_outcome_result(record: PredictionRecord):
    """Display the outcome of a resolved prediction"""
    result_color = "green" if record.correct else "red"
    result_emoji = "✅" if record.correct else "❌"
    
    console.print(f"\n[result_color]{'═' * 60}[/]")
    console.print(f"[bold {result_color}]{result_emoji} MARKET RESOLVED - {'CORRECT!' if record.correct else 'INCORRECT'}[/]")
    console.print(f"[{result_color}]{'═' * 60}[/]")
    
    outcome_table = Table(show_header=False)
    outcome_table.add_column("Key", style="cyan", width=25)
    outcome_table.add_column("Value", style="white")
    
    outcome_table.add_row("Market", record.market_slug)
    outcome_table.add_row("Predicted", f"[bold]{record.predicted_direction}[/] ({record.predicted_probability:.0%})")
    outcome_table.add_row("Actual", f"[bold {result_color}]{record.actual_direction}[/]")
    outcome_table.add_row("Start Price", f"${record.start_price:,.2f}")
    outcome_table.add_row("End Price", f"${record.end_price:,.2f}")
    outcome_table.add_row("Price Change", f"{record.price_change_percent:+.3f}%")
    
    if record.ptb:
        outcome_table.add_row("Price to Beat", f"${record.ptb:,.2f}")
    
    outcome_table.add_row("Brier Score", f"{record.brier_score:.4f}")
    
    console.print(outcome_table)


async def resolve_prediction(atlas: AtlasV4, record: PredictionRecord) -> bool:
    """
    Resolve a prediction by fetching the final price and determining outcome.
    
    Returns True if resolution was successful.
    """
    try:
        console.print(f"\n[cyan]🔄 Resolving prediction {record.prediction_id[:8]}...[/]")
        
        # Fetch current price (which is now the end price since window closed)
        async with BinanceClient() as client:
            end_price = await client.get_current_price()
        
        # Determine actual outcome
        price_change_percent = ((end_price - record.start_price) / record.start_price) * 100
        
        # Determine actual direction based on PTB or price change
        if record.ptb:
            # If we have PTB, use it to determine outcome
            # UP wins if end_price >= PTB, DOWN wins if end_price < PTB
            actual_outcome = end_price >= record.ptb  # True = UP won
            actual_direction = "UP" if actual_outcome else "DOWN"
        else:
            # Fallback to price change direction
            actual_direction = "UP" if end_price >= record.start_price else "DOWN"
            actual_outcome = actual_direction == "UP"
        
        # Determine if prediction was correct
        # The prediction was correct if the predicted direction matches actual
        if record.predicted_direction == "UP":
            correct = actual_outcome  # UP won
        elif record.predicted_direction == "DOWN":
            correct = not actual_outcome  # DOWN won
        else:
            correct = False  # NEUTRAL predictions are never "correct"
        
        # Calculate Brier score
        # For UP prediction: (predicted_prob - 1)^2 if UP won, (predicted_prob - 0)^2 if DOWN won
        if actual_outcome:  # UP won
            brier_score = (record.predicted_probability - 1.0) ** 2
        else:  # DOWN won
            brier_score = (record.predicted_probability - 0.0) ** 2
        
        # Update the record
        outcome_data = {
            "end_price": end_price,
            "actual_direction": actual_direction,
            "actual_outcome": actual_outcome,
            "price_change_percent": price_change_percent,
            "brier_score": brier_score,
            "correct": correct,
            "resolved_at": datetime.now(timezone.utc).isoformat()
        }
        
        atlas.prediction_history.update_outcome(record.prediction_id, outcome_data)
        
        # Record outcome in agent team for learning
        atlas.agent_team.record_outcome(
            prediction_id=record.prediction_id,
            actual_outcome=actual_outcome,
            actual_price_change=price_change_percent
        )
        
        # Update the record for display
        record.end_price = end_price
        record.actual_direction = actual_direction
        record.actual_outcome = actual_outcome
        record.price_change_percent = price_change_percent
        record.brier_score = brier_score
        record.correct = correct
        record.resolved_at = outcome_data["resolved_at"]
        
        # Display result
        display_outcome_result(record)
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error resolving prediction: {e}[/]")
        return False


async def run_paper_mode(max_predictions: int = None):
    """Run paper trading mode with full outcome tracking and learning"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold]  ATLAS v4.0 - Paper Trading Mode with Learning[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")
    
    # Initialize
    try:
        llm_client = FreeClaudeProxy()
        console.print("[green]✓ NVIDIA NIM API connected[/]")
        console.print("[green]✓ LLM Provider connected[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ LLM unavailable: {e}[/]")
        llm_client = None
    
    atlas = AtlasV4(llm_client=llm_client)
    
    # Load previous state
    if atlas.agent_team.load_state("data/agent_state.json"):
        stats = atlas.agent_team.get_stats()
        console.print(f"[green]✓ Loaded previous learning ({stats['total_predictions']} predictions)[/]")
    
    # Display existing performance stats
    display_performance_stats(atlas.prediction_history)
    
    atlas.running = True
    prediction_count = 0
    
    # Queue for pending resolutions
    pending_resolutions: List[PredictionRecord] = []
    
    while atlas.running:
        if max_predictions and prediction_count >= max_predictions:
            console.print(f"\n[yellow]Reached max predictions ({max_predictions}). Stopping.[/]")
            break
        
        # Check for any predictions that need resolution
        unresolved = atlas.prediction_history.get_unresolved()
        for record in unresolved:
            if record not in pending_resolutions:
                pending_resolutions.append(record)
        
        # Resolve any pending predictions whose windows have closed
        resolved_this_round = []
        for record in pending_resolutions:
            if record.window_end <= int(time.time()):
                success = await resolve_prediction(atlas, record)
                if success:
                    resolved_this_round.append(record)
                    # Save state after each resolution
                    atlas.save_state()
        
        # Remove resolved from pending
        for record in resolved_this_round:
            pending_resolutions.remove(record)
        
        # Display market info
        display_market_info()
        
        # Wait for next market window
        seconds_until = PolymarketSync.seconds_until_next_market()
        console.print(f"\n[cyan]⏳ Next market window in {format_countdown(seconds_until)}[/]")
        
        if pending_resolutions:
            console.print(f"[yellow]📋 {len(pending_resolutions)} prediction(s) pending resolution[/]")
        
        await asyncio.sleep(seconds_until)
        
        # Get current market with PTB
        console.print("\n[bold green]🚀 Market Window Open! Fetching data...[/]")
        try:
            market = atlas.polymarket.get_current_market(fetch_ptb=True)
            atlas.current_market = market
        except Exception as e:
            console.print(f"[yellow]⚠ Could not fetch market: {e}[/]")
            market = None
        
        # Make prediction
        prediction = await atlas.make_prediction()
        
        # Get market window times
        window_start, window_end = PolymarketSync.get_current_market_times()
        
        # Create prediction record
        current_price = prediction.get("signals", {}).get("current_price", 0)
        
        record = PredictionRecord(
            prediction_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_slug=PolymarketSync.get_market_slug(),
            window_start=window_start,
            window_end=window_end,
            start_price=current_price,
            ptb=market.ptb if market else None,
            up_odds=market.up_price if market else None,
            down_odds=market.down_price if market else None,
            predicted_direction=prediction["final"]["direction"],
            predicted_probability=prediction["final"]["probability"],
            confidence=prediction["final"]["confidence"],
            agent_predictions=prediction["agent_predictions"],
            regime=prediction.get("regime", "Unknown"),
            confluence=prediction.get("confluence", 0)
        )
        
        # Store prediction in history
        atlas.prediction_history.add_prediction(record)
        
        # Add prediction to agent team's history for later resolution
        atlas.agent_team.prediction_history.append({
            "prediction_id": record.prediction_id,
            "timestamp": record.timestamp,
            "final": prediction["final"],
            "agent_predictions": prediction["agent_predictions"]
        })
        
        # Display market info with current price
        display_market_info(market, current_price)
        
        # Display prediction
        atlas.display_prediction(prediction)
        
        # Add to pending resolutions
        pending_resolutions.append(record)
        
        console.print(f"\n[cyan]📝 Prediction saved with ID: {record.prediction_id}[/]")
        console.print(f"[cyan]⏰ Will resolve after window closes at {format_timestamp(window_end)}[/]")
        
        # Save state
        atlas.save_state()
        
        prediction_count += 1
        
        # Display updated stats
        display_performance_stats(atlas.prediction_history)
    
    # Final resolution of any remaining predictions
    console.print("\n[yellow]Resolving remaining predictions...[/]")
    for record in pending_resolutions:
        if record.resolved_at is None:
            await resolve_prediction(atlas, record)
    
    # Final save
    atlas.save_state()
    
    # Display final stats
    display_performance_stats(atlas.prediction_history)


async def run_backtest(start_date: str, end_date: str):
    """Run backtesting"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold]  ATLAS v4.0 - Backtesting Mode[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")
    
    data_store = HistoricalDataStore()
    stats = data_store.get_stats()
    
    console.print(f"[cyan]Data Store Stats:[/]")
    console.print(f"  Candles: {stats.get('candle_count', 0)}")
    console.print(f"  Funding Rates: {stats.get('funding_rate_count', 0)}")
    console.print(f"  Open Interest: {stats.get('open_interest_count', 0)}")
    
    # Check if data exists
    range_5m = stats.get('time_ranges', {}).get('5m', {})
    if not range_5m.get('start'):
        console.print("\n[yellow]No historical data found. Fetching...[/]")
        await data_store.fetch_and_store(start_date, end_date)
    
    # Run backtest
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date
    )
    
    engine = BacktestEngine(data_store, config)
    
    # Get candles
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    candles = data_store.get_candles("5m", start_ts, end_ts)
    
    if not candles:
        console.print("[red]No candle data available for backtesting[/]")
        return
    
    console.print(f"\n[cyan]Running backtest with {len(candles)} candles...[/]")
    
    # Simple strategy for demo
    def simple_strategy(context):
        signals = context.get("signals", {})
        regime = context.get("regime", "RANGING")
        
        # Simple rule-based strategy
        if regime in ["TRENDING_UP", "TRENDING_DOWN"]:
            momentum = context.get("momentum", 0)
            if momentum > 0.3:
                return {"direction": "UP", "probability": 0.6, "confidence": 0.6}
            elif momentum < -0.3:
                return {"direction": "DOWN", "probability": 0.6, "confidence": 0.6}
        
        return None
    
    result = engine.run_backtest(simple_strategy, candles)
    
    # Display results
    console.print("\n[bold green]═══ Backtest Results ═══[/]")
    console.print(f"Total Trades: {result.total_trades}")
    console.print(f"Win Rate: {result.win_rate:.1%}")
    console.print(f"Total Return: {result.metrics.get('total_return', 0):.1%}")
    console.print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.1%}")
    console.print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")


def show_status():
    """Show system status"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold]  ATLAS v4.0 - System Status[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")
    
    # Load agent state
    team = AgentTeam()
    if team.load_state("data/agent_state.json"):
        stats = team.get_stats()
        
        table = Table(title="📊 Team Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Total Predictions", str(stats["total_predictions"]))
        table.add_row("Average Brier Score", f"{stats['average_brier_score']:.4f}")
        
        console.print(table)
        
        # Agent weights
        weights_table = Table(title="🤖 Agent Weights")
        weights_table.add_column("Agent", style="white")
        weights_table.add_column("Weight", style="magenta")
        weights_table.add_column("Win Rate", style="green")
        
        for name, weight in stats["agent_weights"].items():
            perf = stats["agent_performance"].get(name, {})
            win_rate = perf.get("win_rate", 0.5)
            weights_table.add_row(name, f"{weight:.2f}", f"{win_rate:.0%}")
        
        console.print(weights_table)
    
    # Load prediction history
    history = PredictionHistory()
    hist_stats = history.get_stats()
    
    if hist_stats["resolved"] > 0:
        hist_table = Table(title="📋 Prediction History")
        hist_table.add_column("Metric", style="cyan")
        hist_table.add_column("Value", style="yellow")
        
        hist_table.add_row("Total Predictions", str(hist_stats["total_predictions"]))
        hist_table.add_row("Resolved", str(hist_stats["resolved"]))
        hist_table.add_row("Win Rate", f"{hist_stats['win_rate']:.1%}")
        hist_table.add_row("Avg Brier Score", f"{hist_stats['average_brier']:.4f}")
        
        console.print(hist_table)
    else:
        console.print("[yellow]No resolved predictions yet[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Atlas v4.0 - Bitcoin 15-Minute Prediction System"
    )
    
    parser.add_argument("--predict", "-p", action="store_true",
        help="Make a single prediction")
    parser.add_argument("--paper", action="store_true",
        help="Run paper trading mode")
    parser.add_argument("--backtest", action="store_true",
        help="Run backtesting")
    parser.add_argument("--status", "-s", action="store_true",
        help="Show system status")
    parser.add_argument("--max", type=int, default=None,
        help="Max predictions in paper mode")
    parser.add_argument("--start", type=str, default=None,
        help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
        help="End date for backtest (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.status:
        show_status()
    elif args.paper:
        asyncio.run(run_paper_mode(max_predictions=args.max))
    elif args.backtest:
        if not args.start or not args.end:
            console.print("[red]Error: --start and --end required for backtest[/]")
            return
        asyncio.run(run_backtest(args.start, args.end))
    else:
        asyncio.run(run_single_prediction())


if __name__ == "__main__":
    main()
