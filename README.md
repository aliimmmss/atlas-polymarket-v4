# Atlas v4.0 "Evolution" - Bitcoin 15-Minute Prediction System

## 🚀 Overview

Atlas v4.0 is a comprehensive Bitcoin 15-minute price prediction system designed for Polymarket trading. It features multi-source data integration, advanced technical analysis, regime detection, and a self-improving agent system.

## 📋 Features

### Phase 1: Enhanced Data Pipeline
- **Derivatives Data**: Real-time funding rates, open interest, liquidation levels
- **On-Chain Data**: Exchange flows, whale alerts, Fear & Greed Index
- **Multi-Exchange Prices**: Volume-weighted average from Binance, Coinbase, Kraken, etc.
- **Sentiment Analysis**: News sentiment, Reddit sentiment, Fear & Greed

### Phase 2: Advanced Technical Analysis
- **Regime Detection**: Automatically detect trending, ranging, volatile markets
- **Multi-Timeframe Analysis**: 1m, 5m, 15m, 1h, 4h, 1d confluence scoring
- **Advanced Indicators**: VWAP, Ichimoku, Supertrend, Order Blocks, FVGs
- **Signal Confidence Scoring**: Weight signals by historical accuracy

### Phase 3: Enhanced Agent System
- **13 Specialized Agents**: Regime-specific agents (TrendRider, RangeTrader, etc.)
- **Ensemble Voting**: 7+ voting methods (Kelly, Borda, Bayesian, etc.)
- **Meta-Agent**: Dynamically selects best agents for current conditions
- **Agent Memory**: Agents learn from past mistakes in similar situations

### Phase 4: Backtesting Framework
- **Historical Data Store**: SQLite storage for candles, funding rates, OI
- **Event-Driven Engine**: Realistic slippage and fee modeling
- **Walk-Forward Optimization**: Out-of-sample strategy validation
- **Monte Carlo Simulation**: Confidence intervals for returns

### Phase 5: Risk Management
- **Kelly Criterion**: Optimal position sizing
- **Expected Value Calculator**: Only bet when +EV
- **Risk-Adjusted Confidence**: Adjust for volatility, regime, disagreement

## 🏗️ Project Structure

```
atlas-polymarket-v4-bundle/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── src/
│   ├── data/                    # Data pipeline
│   │   ├── binance_feed.py      # Binance price data
│   │   ├── derivatives_feed.py  # Funding rates, OI
│   │   ├── onchain_feed.py      # Whale alerts, flows
│   │   ├── price_aggregator.py  # Multi-exchange prices
│   │   ├── sentiment_feed.py    # News & social sentiment
│   │   └── market_sync.py       # Polymarket sync
│   │
│   ├── analysis/                # Technical analysis
│   │   ├── technical_indicators.py  # All indicators
│   │   ├── regime_detector.py       # Market regime detection
│   │   ├── multi_timeframe.py       # MTF analysis
│   │   ├── signal_generator.py      # Signal aggregation
│   │   └── confidence_scorer.py     # Confidence scoring
│   │
│   ├── agents/                  # Agent system
│   │   ├── atlas_agent.py       # Core agent framework
│   │   ├── specialized_agents.py   # 13 specialized agents
│   │   ├── ensemble_voting.py   # Multiple voting methods
│   │   ├── meta_agent.py        # Agent selector
│   │   └── agent_memory.py      # Contextual memory
│   │
│   ├── backtest/                # Backtesting
│   │   ├── data_store.py        # Historical data storage
│   │   ├── engine.py            # Backtesting engine
│   │   └── attribution.py       # Performance attribution
│   │
│   ├── risk/                    # Risk management
│   │   ├── position_sizing.py   # Kelly Criterion
│   │   ├── expected_value.py    # EV calculator
│   │   └── risk_adjusted_confidence.py
│   │
│   └── proxy/
│       └── free_claude_proxy.py # NVIDIA NIM API proxy
│
├── config/
│   └── settings.py              # Configuration
│
└── data/                        # Runtime data storage
    ├── agent_state.json
    ├── memory.json
    └── backtest/
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_nim_api_key
CHAINLINK_STREAMS_API_KEY=optional_chainlink_key
CHAINLINK_STREAMS_API_SECRET=optional_chainlink_secret
```

### 3. Run a Single Prediction

```bash
python main.py --predict
```

### 4. Run Paper Trading

```bash
python main.py --paper --max 10  # Max 10 predictions
```

### 5. Run Backtesting

```bash
# First, fetch historical data
python -m src.backtest.data_store 2024-01-01 2024-03-01

# Then run backtest
python main.py --backtest --start 2024-01-01 --end 2024-02-01
```

### 6. Check Status

```bash
python main.py --status
```

## 📊 Agent System

### Specialized Agents by Regime

| Regime | Recommended Agents |
|--------|-------------------|
| Trending Up | TrendRider, MomentumHawk, VolumeWhale |
| Trending Down | TrendFader, MomentumHawk, SupportResist |
| Ranging | RangeTrader, BreakoutHunter, MeanReverter |
| Volatile | VolatilityHarvester, RiskGuard, OrderFlow |
| Breakout | BreakoutHunter, MomentumHawk, VolumeWhale |
| Reversal | MeanReverter, RSIMaster, SentimentSurfer |

### Ensemble Voting Methods

1. **Weighted Average** - Standard method
2. **Borda Count** - Rank-based voting
3. **Confidence Weighted** - Higher confidence = more weight
4. **Consensus Requiring** - Require agreement threshold
5. **Bayesian Averaging** - Incorporate prior beliefs
6. **Trimmed Mean** - Remove outliers
7. **Median Aggregation** - Robust to extremes

## 📈 Performance Targets

| Metric | Baseline | Target |
|--------|----------|--------|
| Accuracy | ~50% | 55-65% |
| Brier Score | 0.25 | 0.15-0.20 |
| Data Sources | 1 | 5+ |
| Specialized Agents | 8 | 13+ |

## ⚙️ Configuration

Edit `config/settings.py` for:

- API endpoints
- Risk parameters
- Agent configurations
- Backtesting settings

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Adding New Agents

1. Create agent in `src/agents/specialized_agents.py`
2. Register in `get_all_specialized_agents()`
3. Add regime mapping in `MetaAgent.REGIME_AGENT_MAP`

### Adding New Data Sources

1. Create feed in `src/data/`
2. Integrate in `AtlasV4.gather_all_data()`
3. Add to signal generation

## 📝 License

MIT License

## 🙏 Acknowledgments

- Polymarket for the prediction market platform
- NVIDIA for NIM API access
- Chainlink for data streams infrastructure
