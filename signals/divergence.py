import logging
import numpy as np
import pandas as pd
from .base import Signal

logger = logging.getLogger(__name__)


class DivergenceDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lookback = 30

    def detect(self, df: pd.DataFrame, timeframe: str):
        signals = []
        if len(df) < self.lookback + 5:
            return signals

        recent = df.iloc[-self.lookback:]

        for indicator in ["rsi", "macd_hist"]:
            s = self._check(recent, indicator, timeframe)
            if s:
                signals.append(s)

        return signals

    def _check(self, df: pd.DataFrame, indicator: str, tf: str):
        if indicator not in df.columns:
            return None

        close = df["close"].values
        ind   = df[indicator].fillna(0).values

        low_idx  = self._local_min(close, 4)
        high_idx = self._local_max(close, 4)

        # Bullische Divergenz
        if len(low_idx) >= 2:
            i1, i2 = low_idx[-2], low_idx[-1]
            if i1 < len(ind) and i2 < len(ind):
                if close[i2] < close[i1] and ind[i2] > ind[i1]:
                    strength = min(abs(ind[i2] - ind[i1]) / (abs(ind[i1]) + 1e-9), 1.0)
                    return Signal(
                        type=f"DIV_BULL_{indicator.upper()}",
                        direction="long",
                        timeframe=tf,
                        price=close[-1],
                        strength=round(0.5 + strength * 0.5, 2),
                        description=f"Bullische Divergenz ({indicator.upper()})",
                    )

        # Bärische Divergenz
        if len(high_idx) >= 2:
            i1, i2 = high_idx[-2], high_idx[-1]
            if i1 < len(ind) and i2 < len(ind):
                if close[i2] > close[i1] and ind[i2] < ind[i1]:
                    strength = min(abs(ind[i2] - ind[i1]) / (abs(ind[i1]) + 1e-9), 1.0)
                    return Signal(
                        type=f"DIV_BEAR_{indicator.upper()}",
                        direction="short",
                        timeframe=tf,
                        price=close[-1],
                        strength=round(0.5 + strength * 0.5, 2),
                        description=f"Bärische Divergenz ({indicator.upper()})",
                    )

        return None

    @staticmethod
    def _local_min(arr: np.ndarray, order: int = 3):
        idx = []
        for i in range(order, len(arr) - order):
            window = arr[i - order : i + order + 1]
            if arr[i] == window.min():
                idx.append(i)
        return idx

    @staticmethod
    def _local_max(arr: np.ndarray, order: int = 3):
        idx = []
        for i in range(order, len(arr) - order):
            window = arr[i - order : i + order + 1]
            if arr[i] == window.max():
                idx.append(i)
        return idx
