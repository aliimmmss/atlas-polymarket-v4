"""
Backtesting Engine for Atlas v4.0
Full-featured event-driven backtesting
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    position_size_percent: float = 2.0  # 2% per trade
    max_positions: int = 1
    fee_percent: float = 0.02  # 2% fee (typical for prediction markets)
    slippage_percent: float = 0.01  # 1% slippage
    prediction_window_seconds: int = 900  # 15 minutes
    
    # Risk management
    max_drawdown_percent: float = 20.0
    kelly_fraction: float = 0.5  # Half Kelly
    
    # Strategy settings
    min_confidence: float = 0.5
    min_edge: float = 0.02  # Minimum 2% edge required


@dataclass
class Trade:
    """Single trade record"""
    trade_id: str
    entry_time: int
    exit_time: int
    direction: str  # "UP" or "DOWN"
    entry_price: float  # Market probability at entry
    exit_price: float  # Market probability at exit
    outcome: bool  # True if correct
    size: float
    pnl: float
    pnl_percent: float
    fee: float
    confidence: float
    agent_votes: Dict[str, str]


@dataclass
class BacktestResult:
    """Complete backtest results"""
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: List[Dict]
    metrics: Dict[str, Any]
    monthly_returns: Dict[str, float]
    agent_performance: Dict[str, Dict]
    regime_performance: Dict[str, Dict]
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.outcome)
    
    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Features:
    - Realistic slippage modeling
    - Fee calculation
    - Multiple strategy comparison
    - Walk-forward optimization
    - Monte Carlo simulation
    """
    
    def __init__(
        self,
        data_store,
        config: BacktestConfig
    ):
        self.data_store = data_store
        self.config = config
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_capital: float = config.initial_capital
        self.peak_capital: float = config.initial_capital
        self.trade_count = 0
    
    def run_backtest(
        self,
        strategy: Callable,
        candles: List[Dict],
        funding_rates: List[Dict] = None,
        open_interest: List[Dict] = None
    ) -> BacktestResult:
        """
        Run full backtest.
        
        Args:
            strategy: Strategy function that takes market data and returns prediction
            candles: Historical candles
            funding_rates: Historical funding rates
            open_interest: Historical open interest
        
        Returns:
            BacktestResult with all metrics
        """
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        
        # Convert to DataFrames for easier manipulation
        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('timestamp')
        
        # Group into 15-minute windows
        window_size = self.config.prediction_window_seconds
        
        # Track open positions
        open_positions: List[Dict] = []
        
        # Iterate through candles
        for i in range(len(df) - 1):
            current_time = df.iloc[i]['timestamp']
            current_price = df.iloc[i]['close']
            
            # Skip if not at 15-minute boundary
            if current_time % window_size != 0:
                continue
            
            # Get market context
            context = self._build_context(df, i, funding_rates, open_interest)
            
            # Get strategy prediction
            prediction = strategy(context)
            
            if prediction is None:
                continue
            
            direction = prediction.get("direction", "NEUTRAL")
            probability = prediction.get("probability", 0.5)
            confidence = prediction.get("confidence", 0.5)
            
            # Skip if confidence too low
            if confidence < self.config.min_confidence:
                continue
            
            # Skip if direction is neutral
            if direction == "NEUTRAL":
                continue
            
            # Calculate edge
            if direction == "UP":
                edge = probability - context.get("market_up_prob", 0.5)
            else:
                edge = (1 - probability) - context.get("market_down_prob", 0.5)
            
            # Skip if edge too small
            if abs(edge) < self.config.min_edge:
                continue
            
            # Check drawdown limit
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if drawdown > self.config.max_drawdown_percent / 100:
                continue
            
            # Calculate position size (Kelly)
            kelly_size = self._calculate_kelly_size(probability, confidence)
            position_size = min(
                kelly_size,
                self.config.position_size_percent / 100 * self.current_capital
            )
            
            # Simulate trade
            exit_time = current_time + window_size
            exit_price_idx = df[df['timestamp'] >= exit_time].index
            
            if len(exit_price_idx) == 0:
                continue
            
            exit_price_idx = exit_price_idx[0]
            exit_price = df.loc[exit_price_idx, 'close']
            
            # Determine outcome
            actual_up = exit_price > current_price
            predicted_up = direction == "UP"
            correct = (predicted_up and actual_up) or (not predicted_up and not actual_up)
            
            # Calculate PnL
            fee = position_size * self.config.fee_percent
            slippage = position_size * self.config.slippage_percent
            
            if correct:
                # Win: Get the probability difference as payout
                if direction == "UP":
                    entry_price = context.get("market_up_prob", 0.5)
                    payout = (1 - entry_price) / max(entry_price, 0.01)
                else:
                    entry_price = context.get("market_down_prob", 0.5)
                    payout = (1 - entry_price) / max(entry_price, 0.01)
                
                pnl = position_size * payout - fee - slippage
            else:
                # Loss: Lose the position
                pnl = -position_size - fee
            
            pnl_percent = pnl / self.current_capital * 100
            
            # Record trade
            trade = Trade(
                trade_id=f"trade_{self.trade_count}",
                entry_time=current_time,
                exit_time=exit_time,
                direction=direction,
                entry_price=current_price,
                exit_price=exit_price,
                outcome=correct,
                size=position_size,
                pnl=pnl,
                pnl_percent=pnl_percent,
                fee=fee,
                confidence=confidence,
                agent_votes=prediction.get("agent_votes", {})
            )
            
            self.trades.append(trade)
            self.trade_count += 1
            
            # Update capital
            self.current_capital += pnl
            self.peak_capital = max(self.peak_capital, self.current_capital)
            
            # Record equity
            self.equity_curve.append({
                "timestamp": current_time,
                "datetime": datetime.fromtimestamp(current_time).isoformat(),
                "equity": self.current_capital,
                "drawdown": (self.peak_capital - self.current_capital) / self.peak_capital * 100
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        monthly_returns = self._calculate_monthly_returns()
        agent_performance = self._calculate_agent_performance()
        regime_performance = self._calculate_regime_performance()
        
        return BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            metrics=metrics,
            monthly_returns=monthly_returns,
            agent_performance=agent_performance,
            regime_performance=regime_performance
        )
    
    def run_walk_forward(
        self,
        strategy_factory: Callable,
        total_periods: int = 12,
        train_period_months: int = 3,
        test_period_months: int = 1
    ) -> List[BacktestResult]:
        """
        Walk-forward optimization.
        
        Splits data into train/test periods and runs backtests sequentially.
        """
        results = []
        
        # Get data range
        start_ts = int(datetime.strptime(self.config.start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(self.config.end_date, "%Y-%m-%d").timestamp())
        
        period_seconds = (train_period_months + test_period_months) * 30 * 24 * 3600
        train_seconds = train_period_months * 30 * 24 * 3600
        test_seconds = test_period_months * 30 * 24 * 3600
        
        for i in range(total_periods):
            period_start = start_ts + i * test_seconds
            train_end = period_start + train_seconds
            test_end = train_end + test_seconds
            
            if test_end > end_ts:
                break
            
            # Get train data
            train_candles = self.data_store.get_candles(
                "5m", period_start, train_end
            )
            
            # Get test data
            test_candles = self.data_store.get_candles(
                "5m", train_end, test_end
            )
            
            if not train_candles or not test_candles:
                continue
            
            # Create optimized strategy for this period
            strategy = strategy_factory(train_candles)
            
            # Run backtest on test period
            test_config = BacktestConfig(
                start_date=datetime.fromtimestamp(train_end).strftime("%Y-%m-%d"),
                end_date=datetime.fromtimestamp(test_end).strftime("%Y-%m-%d"),
                **{k: v for k, v in self.config.__dict__.items() 
                   if k not in ['start_date', 'end_date']}
            )
            
            engine = BacktestEngine(self.data_store, test_config)
            result = engine.run_backtest(strategy, test_candles)
            
            results.append(result)
            
            print(f"Period {i+1}: {result.winning_trades}/{result.total_trades} trades, "
                  f"{result.win_rate:.1%} win rate, {result.metrics.get('total_return', 0):.1%} return")
        
        return results
    
    def run_monte_carlo(
        self,
        trades: List[Trade],
        iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for confidence intervals.
        
        Randomly samples trades with replacement to estimate
        the distribution of possible outcomes.
        """
        if not trades:
            return {"error": "No trades to simulate"}
        
        pnls = [t.pnl for t in trades]
        
        simulated_returns = []
        simulated_max_drawdowns = []
        
        for _ in range(iterations):
            # Sample with replacement
            sampled_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
            
            # Calculate equity curve
            equity = [self.config.initial_capital]
            peak = self.config.initial_capital
            
            for pnl in sampled_pnls:
                new_equity = equity[-1] + pnl
                equity.append(new_equity)
                peak = max(peak, new_equity)
            
            # Calculate return
            total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital
            simulated_returns.append(total_return)
            
            # Calculate max drawdown
            equity_arr = np.array(equity)
            peak_arr = np.maximum.accumulate(equity_arr)
            drawdowns = (peak_arr - equity_arr) / peak_arr
            simulated_max_drawdowns.append(np.max(drawdowns))
        
        # Calculate statistics
        returns_arr = np.array(simulated_returns)
        dd_arr = np.array(simulated_max_drawdowns)
        
        return {
            "iterations": iterations,
            "return_mean": float(np.mean(returns_arr)),
            "return_std": float(np.std(returns_arr)),
            "return_5th": float(np.percentile(returns_arr, 5)),
            "return_25th": float(np.percentile(returns_arr, 25)),
            "return_50th": float(np.percentile(returns_arr, 50)),
            "return_75th": float(np.percentile(returns_arr, 75)),
            "return_95th": float(np.percentile(returns_arr, 95)),
            "max_drawdown_mean": float(np.mean(dd_arr)),
            "max_drawdown_95th": float(np.percentile(dd_arr, 95)),
            "prob_profit": float(np.mean(returns_arr > 0)),
            "prob_double": float(np.mean(returns_arr > 1)),
        }
    
    def _build_context(
        self,
        df: pd.DataFrame,
        idx: int,
        funding_rates: List[Dict],
        open_interest: List[Dict]
    ) -> Dict[str, Any]:
        """Build market context for a point in time"""
        
        current = df.iloc[idx]
        
        # Get recent candles for technical analysis
        lookback = min(100, idx)
        recent = df.iloc[idx-lookback:idx+1]
        
        # Calculate basic indicators
        closes = recent['close'].values
        
        # RSI
        rsi = self._calculate_rsi(closes, 14) if len(closes) > 14 else 50
        
        # Momentum
        if len(closes) > 10:
            momentum = (closes[-1] - closes[-10]) / closes[-10] * 100
        else:
            momentum = 0
        
        # Volatility
        if len(closes) > 20:
            returns = np.diff(closes[-20:]) / closes[-21:-1]
            volatility = np.std(returns) * 100
        else:
            volatility = 0
        
        # Support/Resistance
        if len(recent) >= 20:
            support = recent['low'].tail(20).min()
            resistance = recent['high'].tail(20).max()
        else:
            support = current['low']
            resistance = current['high']
        
        # Market probability (approximate from recent price action)
        # This is a simplified model - in reality would use Polymarket data
        market_up_prob = 0.5 + momentum / 100 * 0.1  # Slight bias from momentum
        market_up_prob = max(0.3, min(0.7, market_up_prob))
        
        return {
            "timestamp": int(current['timestamp']),
            "current_price": float(current['close']),
            "volume": float(current['volume']),
            "rsi": rsi,
            "momentum": momentum,
            "volatility": volatility,
            "support": float(support),
            "resistance": float(resistance),
            "market_up_prob": market_up_prob,
            "market_down_prob": 1 - market_up_prob,
            "candles": recent.to_dict('records')[-30:]  # Last 30 candles
        }
    
    def _calculate_kelly_size(self, probability: float, confidence: float) -> float:
        """Calculate Kelly Criterion position size"""
        
        # Kelly = (p * b - q) / b
        # where p = win prob, q = loss prob, b = odds
        
        # For prediction markets, if we buy at 0.55:
        # Win: get 1.0, profit = 0.45
        # Loss: get 0, loss = 0.55
        # b = 0.45 / 0.55 ≈ 0.82
        
        b = 0.82  # Approximate odds for 55% market
        
        p = probability
        q = 1 - p
        
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Apply Kelly fraction and confidence
        kelly = max(0, kelly * self.config.kelly_fraction * confidence)
        
        # Convert to position size
        return kelly * self.current_capital
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        if not self.trades:
            return {"error": "No trades executed"}
        
        pnls = [t.pnl for t in self.trades]
        
        # Basic metrics
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital
        
        # Win rate
        wins = sum(1 for t in self.trades if t.outcome)
        win_rate = wins / len(self.trades)
        
        # Average win/loss
        wins_pnl = [t.pnl for t in self.trades if t.outcome]
        losses_pnl = [t.pnl for t in self.trades if not t.outcome]
        
        avg_win = np.mean(wins_pnl) if wins_pnl else 0
        avg_loss = np.mean(losses_pnl) if losses_pnl else 0
        
        # Profit factor
        gross_profit = sum(wins_pnl) if wins_pnl else 0
        gross_loss = abs(sum(losses_pnl)) if losses_pnl else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_percent for t in self.trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96) if returns else 0  # 96 15-min periods per day
        
        # Max drawdown
        if self.equity_curve:
            equity = [e['equity'] for e in self.equity_curve]
            peak = np.maximum.accumulate(equity)
            drawdowns = (peak - equity) / peak
            max_drawdown = np.max(drawdowns)
        else:
            max_drawdown = 0
        
        # Brier score (for probability predictions)
        brier_scores = []
        for t in self.trades:
            predicted_prob = t.confidence if t.direction == "UP" else 1 - t.confidence
            actual = 1.0 if t.outcome else 0.0
            brier = (predicted_prob - actual) ** 2
            brier_scores.append(brier)
        
        avg_brier = np.mean(brier_scores) if brier_scores else 0.25
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "total_trades": len(self.trades),
            "winning_trades": wins,
            "losing_trades": len(self.trades) - wins,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_brier_score": avg_brier,
            "final_capital": self.current_capital,
            "total_fees": sum(t.fee for t in self.trades)
        }
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate returns by month"""
        
        monthly = {}
        
        for trade in self.trades:
            dt = datetime.fromtimestamp(trade.entry_time)
            month_key = dt.strftime("%Y-%m")
            
            if month_key not in monthly:
                monthly[month_key] = []
            monthly[month_key].append(trade.pnl)
        
        return {
            k: sum(v) for k, v in monthly.items()
        }
    
    def _calculate_agent_performance(self) -> Dict[str, Dict]:
        """Calculate performance by agent"""
        
        agent_stats = {}
        
        for trade in self.trades:
            for agent_id, vote in trade.agent_votes.items():
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {
                        "votes": 0,
                        "correct": 0,
                        "up_votes": 0,
                        "down_votes": 0
                    }
                
                agent_stats[agent_id]["votes"] += 1
                
                if vote == trade.direction:
                    agent_stats[agent_id]["correct"] += 1
                
                if vote == "UP":
                    agent_stats[agent_id]["up_votes"] += 1
                else:
                    agent_stats[agent_id]["down_votes"] += 1
        
        # Calculate accuracy
        for agent_id, stats in agent_stats.items():
            stats["accuracy"] = stats["correct"] / stats["votes"] if stats["votes"] > 0 else 0
        
        return agent_stats
    
    def _calculate_regime_performance(self) -> Dict[str, Dict]:
        """Calculate performance by market regime"""
        
        # This would require regime labels on each trade
        # Simplified version
        return {
            "overall": {
                "trades": len(self.trades),
                "win_rate": sum(1 for t in self.trades if t.outcome) / len(self.trades) if self.trades else 0
            }
        }
    
    def compare_strategies(
        self,
        strategies: List[Tuple[str, Callable]],
        candles: List[Dict]
    ) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        
        results = {}
        
        for name, strategy in strategies:
            print(f"Testing strategy: {name}")
            
            # Reset state
            self.trades = []
            self.equity_curve = []
            self.current_capital = self.config.initial_capital
            self.peak_capital = self.config.initial_capital
            
            result = self.run_backtest(strategy, candles)
            results[name] = result
            
            print(f"  Return: {result.metrics.get('total_return', 0):.1%}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}")
        
        return results
