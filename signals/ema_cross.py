import logging
import pandas as pd
from .base import Signal

logger = logging.getLogger(__name__)


class EMACrossDetector:
    """Golden Cross / Death Cross EMA mit EMA200-Trendfilter."""

    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, df: pd.DataFrame, timeframe: str):
        signals = []
        if "ema_short" not in df.columns or "ema_long" not in df.columns:
            return signals
        if len(df) < 3:
            return signals

        es = df["ema_short"]
        el = df["ema_long"]
        close = df["close"]

        prev_diff = es.iloc[-2] - el.iloc[-2]
        curr_diff = es.iloc[-1] - el.iloc[-1]

        trend_col = "ema_trend" in df.columns

        if prev_diff <= 0 < curr_diff:
            above = close.iloc[-1] > df["ema_trend"].iloc[-1] if trend_col else True
            signals.append(Signal(
                type="EMA_GOLDEN_CROSS",
                direction="long",
                timeframe=timeframe,
                price=close.iloc[-1],
                strength=0.70 if above else 0.40,
                description=f"Golden Cross EMA{self.cfg.EMA_SHORT}/EMA{self.cfg.EMA_LONG}",
            ))

        elif prev_diff >= 0 > curr_diff:
            below = close.iloc[-1] < df["ema_trend"].iloc[-1] if trend_col else True
            signals.append(Signal(
                type="EMA_DEATH_CROSS",
                direction="short",
                timeframe=timeframe,
                price=close.iloc[-1],
                strength=0.70 if below else 0.40,
                description=f"Death Cross EMA{self.cfg.EMA_SHORT}/EMA{self.cfg.EMA_LONG}",
            ))

        return signals
