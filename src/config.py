"""
Centralized Configuration for Atlas v4.0
Manages all settings and API credentials
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """LLM Provider Configuration"""
    nvidia_api_key: Optional[str] = None
    nim_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"
    provider_priority: List[str] = field(default_factory=lambda: ["nvidia", "anthropic", "openai"])
    max_tokens: int = 500
    temperature: float = 0.3
    
    def __post_init__(self):
        self.nvidia_api_key = self.nvidia_api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        self.nim_api_key = self.nim_api_key or self.nvidia_api_key
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", self.anthropic_model)
    
    def get_available_provider(self) -> Optional[str]:
        for provider in self.provider_priority:
            if provider == "nvidia" and self.nvidia_api_key:
                return "nvidia"
            elif provider == "anthropic" and self.anthropic_api_key:
                return "anthropic"
            elif provider == "openai" and self.openai_api_key:
                return "openai"
        return None
    
    def is_available(self) -> bool:
        return self.get_available_provider() is not None


@dataclass
class ChainlinkConfig:
    """Chainlink Data Streams Configuration"""
    access_key: Optional[str] = None
    secret: Optional[str] = None
    stream_id: str = "BTC_USD"
    
    def __post_init__(self):
        self.access_key = self.access_key or os.getenv("CHAINLINK_ACCESS_KEY")
        self.secret = self.secret or os.getenv("CHAINLINK_SECRET")
        self.stream_id = os.getenv("CHAINLINK_STREAM_ID", self.stream_id)
    
    def is_available(self) -> bool:
        return bool(self.access_key and self.secret)


@dataclass
class TradingConfig:
    """Trading Parameters"""
    paper_trading: bool = True
    initial_capital: float = 10000.0
    max_position_percent: float = 10.0
    min_position_percent: float = 0.5
    kelly_fraction: float = 0.5
    min_confidence: float = 0.5
    min_edge: float = 0.02
    max_drawdown_percent: float = 20.0
    prediction_window_seconds: int = 900
    fee_percent: float = 0.02
    slippage_percent: float = 0.01
    
    def __post_init__(self):
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.initial_capital = float(os.getenv("INITIAL_CAPITAL", self.initial_capital))
        self.max_position_percent = float(os.getenv("MAX_POSITION_PERCENT", self.max_position_percent))
        self.kelly_fraction = float(os.getenv("KELLY_FRACTION", self.kelly_fraction))
        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", self.min_confidence))
        self.min_edge = float(os.getenv("MIN_EDGE", self.min_edge))
        self.max_drawdown_percent = float(os.getenv("MAX_DRAWDOWN_PERCENT", self.max_drawdown_percent))


@dataclass
class FeatureFlags:
    """Feature Toggle Configuration"""
    enable_backtest: bool = False
    enable_agent_improvement: bool = True
    enable_regime_selection: bool = True
    enable_mtf_analysis: bool = True
    enable_derivatives: bool = True
    enable_onchain: bool = True
    enable_sentiment: bool = True
    
    def __post_init__(self):
        self.enable_backtest = os.getenv("ENABLE_BACKTEST", "false").lower() == "true"
        self.enable_agent_improvement = os.getenv("ENABLE_AGENT_IMPROVEMENT", "true").lower() == "true"
        self.enable_regime_selection = os.getenv("ENABLE_REGIME_SELECTION", "true").lower() == "true"
        self.enable_mtf_analysis = os.getenv("ENABLE_MTF_ANALYSIS", "true").lower() == "true"
        self.enable_derivatives = os.getenv("ENABLE_DERIVATIVES", "true").lower() == "true"
        self.enable_onchain = os.getenv("ENABLE_ONCHAIN", "true").lower() == "true"
        self.enable_sentiment = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"


@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    
    def __post_init__(self):
        self.level = os.getenv("LOG_LEVEL", self.level)
    
    def setup(self):
        handlers = [logging.StreamHandler()]
        if self.file_path:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            handlers.append(logging.FileHandler(self.file_path))
        logging.basicConfig(
            level=getattr(logging, self.level.upper()),
            format=self.format,
            handlers=handlers
        )


class Config:
    """Centralized configuration manager for Atlas v4.0."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.llm = LLMConfig()
        self.chainlink = ChainlinkConfig()
        self.trading = TradingConfig()
        self.features = FeatureFlags()
        self.logging = LoggingConfig()
        self._initialized = True
    
    def setup_logging(self):
        self.logging.setup()
    
    def get_proxy_config(self) -> Optional[Dict[str, str]]:
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy or https_proxy:
            return {"http": http_proxy, "https": https_proxy}
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_available": self.llm.is_available(),
            "llm_provider": self.llm.get_available_provider(),
            "chainlink_available": self.chainlink.is_available(),
            "paper_trading": self.trading.paper_trading,
            "features": {
                "agent_improvement": self.features.enable_agent_improvement,
                "regime_selection": self.features.enable_regime_selection,
                "mtf_analysis": self.features.enable_mtf_analysis,
                "derivatives": self.features.enable_derivatives,
                "onchain": self.features.enable_onchain,
                "sentiment": self.features.enable_sentiment,
            },
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "kelly_fraction": self.trading.kelly_fraction,
                "min_confidence": self.trading.min_confidence,
            }
        }
    
    def print_status(self):
        print("\n" + "="*60)
        print("ATLAS v4.0 Configuration Status")
        print("="*60)
        print(f"LLM Provider: {self.llm.get_available_provider() or 'None (using rule-based)'}")
        print(f"Chainlink: {'Available' if self.chainlink.is_available() else 'Not configured'}")
        print(f"Paper Trading: {self.trading.paper_trading}")
        print(f"Initial Capital: ${self.trading.initial_capital:,.2f}")
        print(f"Kelly Fraction: {self.trading.kelly_fraction}")
        print("\nFeatures:")
        print(f"  Agent Improvement: {self.features.enable_agent_improvement}")
        print(f"  Regime Selection: {self.features.enable_regime_selection}")
        print(f"  Multi-Timeframe Analysis: {self.features.enable_mtf_analysis}")
        print(f"  Derivatives Data: {self.features.enable_derivatives}")
        print(f"  On-Chain Data: {self.features.enable_onchain}")
        print(f"  Sentiment Analysis: {self.features.enable_sentiment}")
        print("="*60 + "\n")


# Global config instance
config = Config()
