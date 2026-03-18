"""
Technical Indicators for Atlas v4.0
Comprehensive set of technical analysis indicators optimized for 15-minute BTC prediction
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class IndicatorResult:
    """Standardized indicator result"""
    value: float
    signal: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    name: str
    metadata: Dict[str, Any] = None


class TechnicalIndicators:
    """
    Core technical indicators for Bitcoin price prediction.
    Optimized for 15-minute timeframe analysis.
    """
    
    # ==================== Moving Averages ====================
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return []
        
        result = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            result.append(avg)
        
        return result
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        result = [sum(prices[:period]) / period]
        
        for price in prices[period:]:
            ema = (price - result[-1]) * multiplier + result[-1]
            result.append(ema)
        
        return result
    
    @staticmethod
    def wma(prices: List[float], period: int) -> List[float]:
        """Weighted Moving Average"""
        if len(prices) < period:
            return []
        
        weights = [i + 1 for i in range(period)]
        weight_sum = sum(weights)
        
        result = []
        for i in range(period - 1, len(prices)):
            weighted_sum = sum(prices[i - period + 1 + j] * weights[j] for j in range(period))
            result.append(weighted_sum / weight_sum)
        
        return result
    
    # ==================== Momentum Indicators ====================
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        """
        Relative Strength Index (0-100)
        >70 = Overbought (potential down)
        <30 = Oversold (potential up)
        """
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def rsi_with_signal(prices: List[float], period: int = 14) -> IndicatorResult:
        """RSI with trading signal"""
        rsi_value = TechnicalIndicators.rsi(prices, period)
        
        if rsi_value >= 70:
            signal = "bearish"
            strength = (rsi_value - 70) / 30  # 0-1 scale
        elif rsi_value <= 30:
            signal = "bullish"
            strength = (30 - rsi_value) / 30
        else:
            signal = "neutral"
            strength = 0
        
        return IndicatorResult(
            value=rsi_value,
            signal=signal,
            strength=min(1, strength),
            name="RSI",
            metadata={"period": period}
        )
    
    @staticmethod
    def stochastic(
        prices: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, Any]:
        """
        Stochastic Oscillator
        %K > 80 = Overbought
        %K < 20 = Oversold
        """
        if len(prices) < k_period:
            return {"k": 50, "d": 50, "signal": "neutral"}
        
        k_values = []
        for i in range(k_period - 1, len(prices)):
            period_prices = prices[i - k_period + 1:i + 1]
            high = max(period_prices)
            low = min(period_prices)
            
            if high != low:
                k = ((prices[i] - low) / (high - low)) * 100
            else:
                k = 50
            
            k_values.append(k)
        
        if len(k_values) < d_period:
            return {"k": k_values[-1] if k_values else 50, "d": 50, "signal": "neutral"}
        
        d = sum(k_values[-d_period:]) / d_period
        k = k_values[-1]
        
        if k > 80:
            signal = "overbought"
        elif k < 20:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {"k": k, "d": d, "signal": signal}
    
    @staticmethod
    def williams_r(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> Dict[str, Any]:
        """
        Williams %R (-100 to 0)
        -20 to 0 = Overbought
        -100 to -80 = Oversold
        """
        if len(closes) < period:
            return {"value": -50, "signal": "neutral"}
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest = max(recent_highs)
        lowest = min(recent_lows)
        
        if highest == lowest:
            return {"value": -50, "signal": "neutral"}
        
        wr = ((highest - current_close) / (highest - lowest)) * -100
        
        if wr > -20:
            signal = "overbought"
        elif wr < -80:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {"value": wr, "signal": signal}
    
    @staticmethod
    def momentum(prices: List[float], period: int = 10) -> float:
        """
        Price momentum as percentage.
        Positive = upward, Negative = downward
        """
        if len(prices) < period:
            return 0.0
        
        return ((prices[-1] - prices[-period]) / prices[-period]) * 100
    
    @staticmethod
    def rate_of_change(prices: List[float], period: int = 10) -> float:
        """Rate of Change (ROC)"""
        if len(prices) < period:
            return 0.0
        
        return ((prices[-1] / prices[-period]) - 1) * 100
    
    # ==================== Trend Indicators ====================
    
    @staticmethod
    def macd(
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, Any]:
        """
        MACD (Moving Average Convergence Divergence)
        Returns: macd_line, signal_line, histogram, trend
        """
        if len(prices) < slow + signal:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        # MACD line
        macd_line = [f - s for f, s in zip(ema_fast[-(len(ema_slow)):], ema_slow)]
        
        # Signal line
        signal_line = TechnicalIndicators.sma(macd_line, signal)
        
        if not macd_line or not signal_line:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral"}
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        histogram = current_macd - current_signal
        
        # Determine trend
        if histogram > 0 and len(signal_line) > 1 and signal_line[-1] > signal_line[-2]:
            trend = "bullish"
        elif histogram < 0 and len(signal_line) > 1 and signal_line[-1] < signal_line[-2]:
            trend = "bearish"
        elif histogram > 0:
            trend = "weak_bullish"
        else:
            trend = "weak_bearish"
        
        return {
            "macd": current_macd,
            "signal": current_signal,
            "histogram": histogram,
            "trend": trend
        }
    
    @staticmethod
    def adx(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> Dict[str, Any]:
        """
        Average Directional Index (ADX)
        ADX > 25 = Strong trend
        ADX < 20 = Weak/no trend
        """
        if len(closes) < period * 2:
            return {"adx": 25, "di_plus": 25, "di_minus": 25, "trend": "neutral"}
        
        # Calculate True Range and Directional Movement
        tr_list = []
        dm_plus_list = []
        dm_minus_list = []
        
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)
            
            dm_plus = highs[i] - highs[i-1] if highs[i] - highs[i-1] > 0 else 0
            dm_minus = lows[i-1] - lows[i] if lows[i-1] - lows[i] > 0 else 0
            
            dm_plus_list.append(dm_plus)
            dm_minus_list.append(dm_minus)
        
        # Smooth the values
        def smooth(values: List[float], period: int) -> List[float]:
            smoothed = [sum(values[:period])]
            for i in range(period, len(values)):
                smoothed.append(smoothed[-1] - smoothed[-1] / period + values[i])
            return smoothed
        
        tr_smooth = smooth(tr_list, period)
        dm_plus_smooth = smooth(dm_plus_list, period)
        dm_minus_smooth = smooth(dm_minus_list, period)
        
        # Calculate DI
        di_plus = [(dm_plus_smooth[i] / tr_smooth[i]) * 100 if tr_smooth[i] > 0 else 0 
                   for i in range(len(tr_smooth))]
        di_minus = [(dm_minus_smooth[i] / tr_smooth[i]) * 100 if tr_smooth[i] > 0 else 0 
                    for i in range(len(tr_smooth))]
        
        # Calculate DX and ADX
        dx = [(abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100 
              if (di_plus[i] + di_minus[i]) > 0 else 0 
              for i in range(len(di_plus))]
        
        adx = sum(dx[-period:]) / period if len(dx) >= period else 25
        
        # Determine trend
        if adx > 25:
            if di_plus[-1] > di_minus[-1]:
                trend = "trending_up"
            else:
                trend = "trending_down"
        else:
            trend = "ranging"
        
        return {
            "adx": adx,
            "di_plus": di_plus[-1] if di_plus else 25,
            "di_minus": di_minus[-1] if di_minus else 25,
            "trend": trend
        }
    
    # ==================== Volatility Indicators ====================
    
    @staticmethod
    def bollinger_bands(
        prices: List[float],
        period: int = 20,
        std_dev: float = 2
    ) -> Dict[str, Any]:
        """
        Bollinger Bands
        Returns: upper, middle, lower, position (0-1 where price is relative to bands)
        """
        if len(prices) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "position": 0.5, "signal": "neutral"}
        
        recent = prices[-period:]
        middle = sum(recent) / period
        
        variance = sum((p - middle) ** 2 for p in recent) / period
        std = math.sqrt(variance)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        current_price = prices[-1]
        
        # Position within bands (0 = at lower, 1 = at upper)
        if upper != lower:
            position = (current_price - lower) / (upper - lower)
        else:
            position = 0.5
        
        # Bandwidth (volatility indicator)
        bandwidth = (upper - lower) / middle if middle > 0 else 0
        
        # Signal
        if position > 0.9:
            signal = "overbought"
        elif position < 0.1:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "position": position,
            "bandwidth": bandwidth,
            "signal": signal
        }
    
    @staticmethod
    def atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> float:
        """
        Average True Range - Volatility indicator
        """
        if len(closes) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        return sum(true_ranges[-period:]) / period
    
    @staticmethod
    def keltner_channels(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 20,
        atr_mult: float = 2
    ) -> Dict[str, Any]:
        """
        Keltner Channels
        Price above upper = Strong uptrend
        Price below lower = Strong downtrend
        """
        if len(closes) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "signal": "neutral"}
        
        # Middle line = EMA
        ema_values = TechnicalIndicators.ema(closes, period)
        middle = ema_values[-1] if ema_values else closes[-1]
        
        # ATR
        atr = TechnicalIndicators.atr(highs, lows, closes, period)
        
        upper = middle + (atr_mult * atr)
        lower = middle - (atr_mult * atr)
        
        current = closes[-1]
        
        if current > upper:
            signal = "strong_up"
        elif current < lower:
            signal = "strong_down"
        else:
            signal = "neutral"
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "atr": atr,
            "signal": signal
        }
    
    # ==================== Volume Indicators ====================
    
    @staticmethod
    def obv(prices: List[float], volumes: List[float]) -> List[float]:
        """On-Balance Volume"""
        if len(prices) != len(volumes) or len(prices) < 2:
            return []
        
        obv = [0]
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif prices[i] < prices[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        
        return obv
    
    @staticmethod
    def mfi(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        period: int = 14
    ) -> Dict[str, Any]:
        """
        Money Flow Index (0-100)
        >80 = Overbought
        <20 = Oversold
        """
        if len(closes) < period + 1:
            return {"value": 50, "signal": "neutral"}
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_prices)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_flow.append(typical_prices[i] * volumes[i])
                negative_flow.append(0)
            else:
                positive_flow.append(0)
                negative_flow.append(typical_prices[i] * volumes[i])
        
        if len(positive_flow) < period:
            return {"value": 50, "signal": "neutral"}
        
        pos_mf = sum(positive_flow[-period:])
        neg_mf = sum(negative_flow[-period:])
        
        if neg_mf == 0:
            mfi_value = 100
        else:
            mfi_value = 100 - (100 / (1 + pos_mf / neg_mf))
        
        if mfi_value > 80:
            signal = "overbought"
        elif mfi_value < 20:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return {"value": mfi_value, "signal": signal}
    
    # ==================== Support/Resistance ====================
    
    @staticmethod
    def support_resistance(
        prices: List[float],
        window: int = 20
    ) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        if len(prices) < window:
            return {"support": prices[-1], "resistance": prices[-1], "distance": 0}
        
        recent = prices[-window:]
        support = min(recent)
        resistance = max(recent)
        current = prices[-1]
        
        # Distance from middle
        middle = (support + resistance) / 2
        distance = (current - middle) / (resistance - support) if resistance != support else 0
        
        return {
            "support": support,
            "resistance": resistance,
            "current": current,
            "distance": distance,  # -1 at support, +1 at resistance
            "range_percent": ((resistance - support) / support) * 100
        }
    
    @staticmethod
    def pivot_points(
        high: float,
        low: float,
        close: float
    ) -> Dict[str, float]:
        """Calculate pivot points"""
        pivot = (high + low + close) / 3
        
        return {
            "pivot": pivot,
            "r1": 2 * pivot - low,
            "r2": pivot + (high - low),
            "r3": high + 2 * (high - low),
            "s1": 2 * pivot - high,
            "s2": pivot - (high - low),
            "s3": low - 2 * (high - low)
        }


class AdvancedIndicators:
    """
    Advanced technical indicators for professional analysis.
    """
    
    @staticmethod
    def vwap(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> Dict[str, Any]:
        """
        Volume Weighted Average Price
        Price above VWAP = Bullish
        Price below VWAP = Bearish
        """
        if len(closes) < 2:
            return {"vwap": closes[-1] if closes else 0, "signal": "neutral"}
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        
        cum_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
        cum_vol = sum(volumes)
        
        vwap = cum_tp_vol / cum_vol if cum_vol > 0 else closes[-1]
        
        current = closes[-1]
        
        if current > vwap * 1.002:  # 0.2% above VWAP
            signal = "bullish"
        elif current < vwap * 0.998:
            signal = "bearish"
        else:
            signal = "neutral"
        
        return {
            "vwap": vwap,
            "current": current,
            "deviation_percent": ((current - vwap) / vwap) * 100,
            "signal": signal
        }
    
    @staticmethod
    def ichimoku_cloud(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ) -> Dict[str, Any]:
        """
        Ichimoku Cloud Analysis
        
        Signals:
        - Price above cloud = Bullish
        - Price below cloud = Bearish
        - Price in cloud = Neutral/Consolidation
        - TK Cross = Tenkan/Kijun crossover signal
        """
        if len(closes) < senkou_b_period:
            return {"signal": "neutral", "cloud_top": 0, "cloud_bottom": 0}
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = max(highs[-tenkan_period:])
        tenkan_low = min(lows[-tenkan_period:])
        tenkan = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = max(highs[-kijun_period:])
        kijun_low = min(lows[-kijun_period:])
        kijun = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan + kijun) / 2
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = max(highs[-senkou_b_period:])
        senkou_b_low = min(lows[-senkou_b_period:])
        senkou_b = (senkou_b_high + senkou_b_low) / 2
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        current = closes[-1]
        
        # Determine signal
        if current > cloud_top:
            signal = "bullish"
            strength = (current - cloud_top) / cloud_top * 100
        elif current < cloud_bottom:
            signal = "bearish"
            strength = (cloud_bottom - current) / cloud_bottom * 100
        else:
            signal = "neutral"
            strength = 0
        
        # TK Cross
        tk_cross = "bullish" if tenkan > kijun else "bearish"
        
        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "tk_cross": tk_cross,
            "signal": signal,
            "strength": min(5, strength)  # Cap at 5%
        }
    
    @staticmethod
    def supertrend(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        atr_period: int = 10,
        multiplier: float = 3
    ) -> Dict[str, Any]:
        """
        Supertrend Indicator
        
        Price above supertrend = Uptrend (Buy signal)
        Price below supertrend = Downtrend (Sell signal)
        """
        if len(closes) < atr_period + 1:
            return {"value": closes[-1], "signal": "neutral", "trend": "neutral"}
        
        # Calculate ATR
        atr = TechnicalIndicators.atr(highs, lows, closes, atr_period)
        
        # Calculate basic bands
        hl2 = (highs[-1] + lows[-1]) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        current = closes[-1]
        
        # Determine trend
        if current > upper_band:
            trend = "up"
            supertrend = lower_band
            signal = "bullish"
        elif current < lower_band:
            trend = "down"
            supertrend = upper_band
            signal = "bearish"
        else:
            trend = "neutral"
            supertrend = (upper_band + lower_band) / 2
            signal = "neutral"
        
        return {
            "value": supertrend,
            "atr": atr,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "trend": trend,
            "signal": signal
        }
    
    @staticmethod
    def detect_order_blocks(
        candles: List[Dict[str, float]],
        lookback: int = 10
    ) -> Dict[str, Any]:
        """
        Detect Order Blocks (institutional footprints)
        
        Bullish OB: Last down candle before strong up move
        Bearish OB: Last up candle before strong down move
        """
        if len(candles) < lookback + 5:
            return {"bullish_obs": [], "bearish_obs": []}
        
        bullish_obs = []
        bearish_obs = []
        
        for i in range(len(candles) - lookback, len(candles) - 2):
            candle = candles[i]
            next_candle = candles[i + 1]
            
            is_bullish = candle["close"] > candle["open"]
            is_bearish = candle["close"] < candle["open"]
            
            # Check for bullish OB (down candle followed by strong up)
            if is_bearish:
                next_move = (next_candle["close"] - next_candle["open"]) / next_candle["open"] * 100
                if next_move > 0.5:  # Strong up move
                    bullish_obs.append({
                        "high": candle["high"],
                        "low": candle["low"],
                        "open": candle["open"],
                        "close": candle["close"],
                        "index": i
                    })
            
            # Check for bearish OB (up candle followed by strong down)
            if is_bullish:
                next_move = (next_candle["open"] - next_candle["close"]) / next_candle["open"] * 100
                if next_move > 0.5:  # Strong down move
                    bearish_obs.append({
                        "high": candle["high"],
                        "low": candle["low"],
                        "open": candle["open"],
                        "close": candle["close"],
                        "index": i
                    })
        
        return {
            "bullish_obs": bullish_obs[-3:],  # Last 3
            "bearish_obs": bearish_obs[-3:]
        }
    
    @staticmethod
    def detect_fvg(
        candles: List[Dict[str, float]],
        min_size_percent: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Detect Fair Value Gaps (Imbalance zones)
        
        Bullish FVG: Gap between candle 1 high and candle 3 low
        Bearish FVG: Gap between candle 1 low and candle 3 high
        """
        if len(candles) < 3:
            return []
        
        fvgs = []
        current_price = candles[-1]["close"]
        
        for i in range(len(candles) - 3):
            c1 = candles[i]
            c2 = candles[i + 1]
            c3 = candles[i + 2]
            
            # Bullish FVG
            if c3["low"] > c1["high"]:
                gap_size = c3["low"] - c1["high"]
                gap_percent = (gap_size / c1["high"]) * 100
                
                if gap_percent >= min_size_percent:
                    fvgs.append({
                        "type": "bullish",
                        "top": c3["low"],
                        "bottom": c1["high"],
                        "size_percent": gap_percent,
                        "filled": current_price < c1["high"],  # Price has returned
                        "index": i
                    })
            
            # Bearish FVG
            if c3["high"] < c1["low"]:
                gap_size = c1["low"] - c3["high"]
                gap_percent = (gap_size / c1["low"]) * 100
                
                if gap_percent >= min_size_percent:
                    fvgs.append({
                        "type": "bearish",
                        "top": c1["low"],
                        "bottom": c3["high"],
                        "size_percent": gap_percent,
                        "filled": current_price > c1["low"],
                        "index": i
                    })
        
        return fvgs[-5:]  # Last 5 FVGs
