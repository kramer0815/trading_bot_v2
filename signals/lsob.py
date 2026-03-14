import logging
import pandas as pd
from .base import Signal

logger = logging.getLogger(__name__)


class LSOBDetector:
    """
    Liquidity Sweep Order Block:
    - Preis swept unter lokales Tief / über lokales Hoch
    - Dreht sofort mit überdurchschnittlichem Volumen zurück
    - Signal entsteht wenn aktueller Preis noch nah am OB ist
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, df: pd.DataFrame, timeframe: str):
        signals = []
        cfg = self.cfg
        lookback = cfg.LSOB_LOOKBACK

        if len(df) < lookback + 5:
            return signals

        recent   = df.iloc[-(lookback + 10):]
        closes   = recent["close"].values
        highs    = recent["high"].values
        lows     = recent["low"].values
        opens    = recent["open"].values
        volumes  = recent["volume"].values
        vol_ma   = recent["vol_ma"].values
        n        = len(recent)
        price_now = closes[-1]

        # ── Bullisches LSOB ──────────────────────────────────────
        for i in range(5, n - 2):
            swing_low  = float(min(lows[i - 5 : i]))
            swept      = lows[i] < swing_low
            reversal   = closes[i] > opens[i]
            vol_ok     = vol_ma[i] > 0 and volumes[i] > vol_ma[i] * cfg.LSOB_VOLUME_MULT

            if swept and reversal and vol_ok:
                if abs(price_now - closes[i]) / closes[i] < 0.025:
                    strength = min(volumes[i] / (vol_ma[i] * 3), 1.0)
                    signals.append(Signal(
                        type="LSOB_LONG",
                        direction="long",
                        timeframe=timeframe,
                        price=price_now,
                        strength=round(strength, 2),
                        description=f"Liquidity Sweep unter {swing_low:,.0f} + V-Umkehr",
                        extra={"ob_high": highs[i], "ob_low": lows[i]},
                    ))
                    break

        # ── Bärisches LSOB ───────────────────────────────────────
        for i in range(5, n - 2):
            swing_high = float(max(highs[i - 5 : i]))
            swept      = highs[i] > swing_high
            reversal   = closes[i] < opens[i]
            vol_ok     = vol_ma[i] > 0 and volumes[i] > vol_ma[i] * cfg.LSOB_VOLUME_MULT

            if swept and reversal and vol_ok:
                if abs(price_now - closes[i]) / closes[i] < 0.025:
                    strength = min(volumes[i] / (vol_ma[i] * 3), 1.0)
                    signals.append(Signal(
                        type="LSOB_SHORT",
                        direction="short",
                        timeframe=timeframe,
                        price=price_now,
                        strength=round(strength, 2),
                        description=f"Liquidity Sweep über {swing_high:,.0f} + Umkehr",
                        extra={"ob_high": highs[i], "ob_low": lows[i]},
                    ))
                    break

        return signals
