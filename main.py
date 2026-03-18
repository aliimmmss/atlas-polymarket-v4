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
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import all modules
from src.proxy.free_claude_proxy import FreeClaudeProxy
from src.data.binance_feed import BitcoinPriceMonitor
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
        
        if direction == "UP":
            emoji, color = "📈", "green"
        elif direction == "DOWN":
            emoji, color = "📉", "red"
        else:
            emoji, color = "➡️", "yellow"
        
        # Main prediction panel
        panel_content = f"""
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
        
        if "ev" in prediction:
            ev = prediction["ev"]
            panel_content += f"""
[bold green]Expected Value:[/] {ev['expected_value']:.2%}
[bold green]Edge:[/] {ev['edge']:.2%}
"""
        
        if "position_size" in prediction:
            ps = prediction["position_size"]
            panel_content += f"""
[bold magenta]Position Size:[/] ${ps['size']:.2f} ({ps['size_percent']:.1f}%)
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
        
        for ap in prediction["agent_predictions"][:10]:
            vote_color = "green" if ap["direction"] == "UP" else "red" if ap["direction"] == "DOWN" else "white"
            table.add_row(
                ap["agent_name"],
                f"[{vote_color}]{ap['direction']}[/]",
                f"{ap['probability']:.0%}",
                f"{ap['weight']:.2f}"
            )
        
        console.print(table)
    
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


async def run_paper_mode(max_predictions: int = None):
    """Run paper trading mode"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold]  ATLAS v4.0 - Paper Trading Mode[/]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/]\n")
    
    # Initialize
    try:
        llm_client = FreeClaudeProxy()
        console.print("[green]✓ LLM Provider connected[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ LLM unavailable: {e}[/]")
        llm_client = None
    
    atlas = AtlasV4(llm_client=llm_client)
    
    if atlas.agent_team.load_state("data/agent_state.json"):
        console.print("[green]✓ Loaded previous learning[/]")
    
    atlas.running = True
    prediction_count = 0
    
    while atlas.running:
        if max_predictions and prediction_count >= max_predictions:
            console.print(f"\n[yellow]Reached max predictions ({max_predictions}). Stopping.[/]")
            break
        
        # Wait for next market window
        seconds_until = PolymarketSync.seconds_until_next_market()
        console.print(f"\n[cyan]⏳ Next market window in {format_countdown(seconds_until)}[/]")
        
        await asyncio.sleep(seconds_until)
        
        # Make prediction
        prediction = await atlas.make_prediction()
        atlas.display_prediction(prediction)
        
        prediction_count += 1
    
    atlas.save_state()


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
    else:
        console.print("[yellow]No previous state found[/]")


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
