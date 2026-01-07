"""Feature extraction from market data for ML models.

The agent doesn't know which features matter - it learns that through training.
We compute many potential features and let the model discover what's predictive.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np

from stockbot.core.models import Bar
from stockbot.core.types import Price


@dataclass
class MarketFeatures:
    """Extracted features from market data.

    All features are normalized to roughly [-1, 1] or [0, 1] range
    to help the learning algorithm.
    """

    # Raw feature vector for ML model
    vector: np.ndarray

    # Feature names (for interpretability)
    names: list[str]

    # Current price (for position sizing)
    price: float

    # Timestamp
    timestamp: int


class FeatureExtractor:
    """Extracts features from price data for ML models.

    The model will learn which features are actually predictive.
    We just provide many potential signals.
    """

    def __init__(
        self,
        lookback_periods: list[int] = [5, 10, 20, 50],
        include_volume: bool = True,
        include_volatility: bool = True,
        include_momentum: bool = True,
        include_mean_reversion: bool = True,
    ) -> None:
        """Initialize feature extractor.

        Args:
            lookback_periods: Periods for moving averages, etc.
            include_volume: Include volume-based features
            include_volatility: Include volatility features
            include_momentum: Include momentum/trend features
            include_mean_reversion: Include mean reversion features
        """
        self._lookback = lookback_periods
        self._include_volume = include_volume
        self._include_volatility = include_volatility
        self._include_momentum = include_momentum
        self._include_mean_reversion = include_mean_reversion

        # Build feature name list
        self._feature_names = self._build_feature_names()

    @property
    def feature_count(self) -> int:
        """Number of features extracted."""
        return len(self._feature_names)

    @property
    def feature_names(self) -> list[str]:
        """Names of all features."""
        return self._feature_names.copy()

    def _build_feature_names(self) -> list[str]:
        """Build list of feature names."""
        names = []

        # Price-based features
        for p in self._lookback:
            names.append(f"return_{p}d")  # N-day return
            names.append(f"sma_ratio_{p}d")  # Price / SMA ratio

        if self._include_momentum:
            for p in self._lookback:
                names.append(f"momentum_{p}d")  # Rate of change
                names.append(f"rsi_{p}d")  # RSI
            names.append("macd")  # MACD signal
            names.append("macd_hist")  # MACD histogram

        if self._include_volatility:
            for p in self._lookback:
                names.append(f"volatility_{p}d")  # Std dev of returns
                names.append(f"atr_ratio_{p}d")  # ATR / price
            names.append("bollinger_pos")  # Position in Bollinger bands

        if self._include_mean_reversion:
            for p in self._lookback:
                names.append(f"zscore_{p}d")  # Z-score from mean
            names.append("high_low_pos")  # Position in recent range

        if self._include_volume:
            for p in self._lookback:
                names.append(f"volume_ratio_{p}d")  # Volume / avg volume
            names.append("volume_trend")  # Volume increasing/decreasing

        # Day of week (one-hot, 5 features)
        names.extend([f"dow_{i}" for i in range(5)])

        return names

    def extract(self, bars: list[Bar]) -> Optional[MarketFeatures]:
        """Extract features from a list of bars.

        Args:
            bars: Historical bars (oldest first)

        Returns:
            MarketFeatures or None if insufficient data
        """
        max_lookback = max(self._lookback) + 10  # Buffer for calculations

        if len(bars) < max_lookback:
            return None

        # Convert to numpy arrays
        closes = np.array([float(b.close) for b in bars])
        highs = np.array([float(b.high) for b in bars])
        lows = np.array([float(b.low) for b in bars])
        volumes = np.array([float(b.volume) for b in bars])

        features = []

        # Price-based features
        for p in self._lookback:
            # N-day return (normalized)
            ret = (closes[-1] / closes[-p] - 1) * 10  # Scale up small returns
            features.append(np.clip(ret, -3, 3))

            # Price / SMA ratio
            sma = np.mean(closes[-p:])
            sma_ratio = (closes[-1] / sma - 1) * 10
            features.append(np.clip(sma_ratio, -3, 3))

        if self._include_momentum:
            for p in self._lookback:
                # Momentum (rate of change)
                momentum = (closes[-1] - closes[-p]) / closes[-p] * 10
                features.append(np.clip(momentum, -3, 3))

                # RSI
                rsi = self._calculate_rsi(closes, p)
                features.append((rsi - 50) / 50)  # Normalize to [-1, 1]

            # MACD
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            macd = ema12 - ema26
            signal = self._ema(np.array([macd]), 9) if len(closes) > 35 else macd
            macd_norm = macd / closes[-1] * 100
            features.append(np.clip(macd_norm, -3, 3))
            features.append(np.clip((macd - signal) / closes[-1] * 100, -3, 3))

        if self._include_volatility:
            returns = np.diff(closes) / closes[:-1]

            for p in self._lookback:
                # Volatility (annualized)
                vol = np.std(returns[-p:]) * np.sqrt(252)
                features.append(np.clip(vol, 0, 2))  # Cap at 200% annual vol

                # ATR ratio
                atr = self._calculate_atr(highs, lows, closes, p)
                atr_ratio = atr / closes[-1] * 100
                features.append(np.clip(atr_ratio, 0, 3))

            # Bollinger band position
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:])
            if std20 > 0:
                bb_pos = (closes[-1] - sma20) / (2 * std20)
            else:
                bb_pos = 0
            features.append(np.clip(bb_pos, -2, 2))

        if self._include_mean_reversion:
            for p in self._lookback:
                # Z-score
                mean = np.mean(closes[-p:])
                std = np.std(closes[-p:])
                if std > 0:
                    zscore = (closes[-1] - mean) / std
                else:
                    zscore = 0
                features.append(np.clip(zscore, -3, 3))

            # Position in high-low range
            recent_high = np.max(highs[-20:])
            recent_low = np.min(lows[-20:])
            if recent_high > recent_low:
                hl_pos = (closes[-1] - recent_low) / (recent_high - recent_low)
            else:
                hl_pos = 0.5
            features.append(hl_pos * 2 - 1)  # Normalize to [-1, 1]

        if self._include_volume:
            for p in self._lookback:
                # Volume ratio
                avg_vol = np.mean(volumes[-p:])
                if avg_vol > 0:
                    vol_ratio = volumes[-1] / avg_vol
                else:
                    vol_ratio = 1
                features.append(np.clip(vol_ratio - 1, -2, 2))

            # Volume trend
            vol_sma5 = np.mean(volumes[-5:])
            vol_sma20 = np.mean(volumes[-20:])
            if vol_sma20 > 0:
                vol_trend = vol_sma5 / vol_sma20 - 1
            else:
                vol_trend = 0
            features.append(np.clip(vol_trend, -2, 2))

        # Day of week (one-hot)
        from datetime import datetime, timezone
        ts = bars[-1].timestamp
        dt = datetime.fromtimestamp(ts / 1_000_000_000, tz=timezone.utc)
        dow = dt.weekday()
        for i in range(5):
            features.append(1.0 if i == dow else 0.0)

        return MarketFeatures(
            vector=np.array(features, dtype=np.float32),
            names=self._feature_names,
            price=float(bars[-1].close),
            timestamp=bars[-1].timestamp,
        )

    def _calculate_rsi(self, closes: np.ndarray, period: int) -> float:
        """Calculate RSI indicator."""
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int,
    ) -> float:
        """Calculate Average True Range."""
        tr_list = []
        for i in range(-period, 0):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr_list.append(max(high_low, high_close, low_close))
        return np.mean(tr_list)

    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return data[-1]

        multiplier = 2 / (period + 1)
        ema = data[-period]

        for price in data[-(period-1):]:
            ema = (price - ema) * multiplier + ema

        return ema
