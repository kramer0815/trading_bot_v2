import logging
import pandas as pd
import numpy as np
import ccxt

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    line = _ema(series, fast) - _ema(series, slow)
    sig  = _ema(line, signal)
    return line, sig, line - sig


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


class DataFetcher:
    def __init__(self, cfg):
        self.cfg = cfg
        exchange_class = getattr(ccxt, cfg.EXCHANGE)
        self.exchange = exchange_class({
            "apiKey": cfg.API_KEY,
            "secret": cfg.API_SECRET,
            "enableRateLimit": True,
        })

    def get_ohlcv(self, timeframe: str, limit: int = 300):
        try:
            raw = self.exchange.fetch_ohlcv(self.cfg.SYMBOL, timeframe, limit=limit)
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            self._add_indicators(df)
            return df
        except Exception as e:
            logger.error(f"Fehler beim Laden ({timeframe}): {e}")
            return None

    def _add_indicators(self, df: pd.DataFrame):
        cfg = self.cfg

        # RSI
        df["rsi"] = _rsi(df["close"], cfg.RSI_PERIOD)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
            df["close"], cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL
        )

        # Konfigurierbbare EMAs (für Chart-Anzeige & EMA Cross)
        df["ema_short"] = _ema(df["close"], cfg.EMA_SHORT)   # default 9
        df["ema_long"]  = _ema(df["close"], cfg.EMA_LONG)    # default 21
        df["ema_trend"] = _ema(df["close"], cfg.EMA_TREND)   # default 200

        # Feste EMAs für GUSS-Strategie
        df["ema21"] = _ema(df["close"], 21)
        df["ema50"] = _ema(df["close"], 50)

        # ATR & Volumen
        df["atr"]    = _atr(df["high"], df["low"], df["close"], cfg.ATR_PERIOD)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        # Swing-Punkte (für andere Detektoren)
        df["swing_high"] = df["high"].where(
            (df["high"] > df["high"].shift(1)) &
            (df["high"] > df["high"].shift(2)) &
            (df["high"] > df["high"].shift(-1)) &
            (df["high"] > df["high"].shift(-2))
        )
        df["swing_low"] = df["low"].where(
            (df["low"] < df["low"].shift(1)) &
            (df["low"] < df["low"].shift(2)) &
            (df["low"] < df["low"].shift(-1)) &
            (df["low"] < df["low"].shift(-2))
        )
