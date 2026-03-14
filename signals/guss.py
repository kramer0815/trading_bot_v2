import logging
import pandas as pd
import numpy as np
from .base import Signal

logger = logging.getLogger(__name__)


class GUSSDetector:
    """
    GUSS — Geordneter Unidirektionaler Swing zum Support

    GUSS LONG:
      1. Preis macht ein neues Swing-Hoch (letztes Hoch in Lookback)
      2. Preis fällt danach mit AUSSCHLIESSLICH bärischen Kerzen (close < open)
         zum EMA50 herunter
      3. Aktuelle Kerze berührt oder unterschreitet EMA50 leicht
      → Long Entry @ EMA50
      → SL: EMA50 - ATR * Multiplikator  (Platz für Wicks)
      → TP1: EMA21
      → TP2: letztes Hoch - (letztes Hoch - Entry) * 0.5   (50% zurück)
      → TP3: letztes Hoch

    Reverse GUSS SHORT:
      1. Preis macht ein neues Swing-Tief
      2. Preis steigt danach mit AUSSCHLIESSLICH bullischen Kerzen zum EMA50
      3. Aktuelle Kerze berührt oder überschreitet EMA50 leicht
      → Short Entry @ EMA50
      → SL: EMA50 + ATR * Multiplikator
      → TP1: EMA21
      → TP2: letztes Tief + (Entry - letztes Tief) * 0.5   (50% zurück)
      → TP3: letztes Tief
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Wie viele Kerzen nach dem Hoch/Tief darf der "Guss" maximal dauern
        self.max_guss_candles = int(getattr(cfg, "GUSS_MAX_CANDLES", 15))
        # Wie viele % Toleranz beim EMA50-Touch
        self.ema_touch_pct    = float(getattr(cfg, "GUSS_EMA_TOUCH_PCT", 0.003))
        # Swing-Hoch/Tief Lookback
        self.swing_lookback   = int(getattr(cfg, "GUSS_SWING_LOOKBACK", 50))

    # ── Hauptmethode ────────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame, timeframe: str):
        signals = []

        # Benötigte Spalten prüfen
        required = {"open", "high", "low", "close", "ema_long", "atr"}
        if not required.issubset(df.columns):
            logger.debug(f"[GUSS] Fehlende Spalten: {required - set(df.columns)}")
            return signals

        # EMA50 berechnen (wird in fetcher als ema_long mit period=50 gesetzt,
        # falls nicht vorhanden → on-the-fly berechnen)
        if "ema50" not in df.columns:
            df = df.copy()
            df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        # EMA21 (für TPs)
        if "ema21" not in df.columns:
            df = df.copy()
            df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

        sig_long  = self._detect_guss_long(df, timeframe)
        sig_short = self._detect_guss_short(df, timeframe)

        if sig_long:
            signals.append(sig_long)
        if sig_short:
            signals.append(sig_short)

        return signals

    # ── GUSS LONG ───────────────────────────────────────────────────────────
    def _detect_guss_long(self, df: pd.DataFrame, timeframe: str):
        n          = len(df)
        closes     = df["close"].values
        opens      = df["open"].values
        highs      = df["high"].values
        lows       = df["low"].values
        ema50      = df["ema50"].values
        ema21      = df["ema21"].values
        atr        = df["atr"].values

        lookback_start = max(0, n - self.swing_lookback)

        # 1. Letztes Swing-Hoch finden (höchstes High im Lookback-Fenster,
        #    mindestens 5 Kerzen zurück damit der "Guss" danach stattfinden kann)
        search_end = n - 3
        search_start = lookback_start

        swing_high_idx = None
        swing_high_val = -np.inf

        for i in range(search_start, search_end):
            # Lokales Hoch: höher als die 2 Nachbarn links und rechts
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2]
                    and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                if highs[i] > swing_high_val:
                    swing_high_val = highs[i]
                    swing_high_idx = i

        if swing_high_idx is None:
            return None

        # 2. Kerzen NACH dem Swing-Hoch bis jetzt müssen ausschließlich bärisch sein
        guss_start = swing_high_idx + 1
        guss_end   = n - 1  # letzte vollständige Kerze

        if guss_end - guss_start < 1:
            return None
        if guss_end - guss_start > self.max_guss_candles:
            return None

        # Alle Kerzen im Guss müssen bärisch sein (close < open)
        for i in range(guss_start, guss_end + 1):
            if closes[i] >= opens[i]:   # bullische Kerze → kein GUSS
                return None

        # 3. Aktuelle (letzte) Kerze berührt EMA50
        current_low   = lows[-1]
        current_close = closes[-1]
        current_ema50 = ema50[-1]
        current_ema21 = ema21[-1]
        current_atr   = atr[-1]

        # Touch-Bedingung: Low oder Close liegt innerhalb der Toleranz des EMA50
        # oder hat ihn leicht unterschritten
        lower_bound = current_ema50 * (1 - self.ema_touch_pct)
        upper_bound = current_ema50 * (1 + self.ema_touch_pct)

        ema50_touched = lower_bound <= current_low <= upper_bound \
                     or lower_bound <= current_close <= upper_bound

        if not ema50_touched:
            return None

        # 4. EMA50 muss UNTER EMA21 liegen (Abwärtstrend zum EMA — sonst kein Guss)
        #    → Preis kam von oben (Hoch) herunter
        if current_ema50 >= current_ema21:
            return None

        # 5. Trade-Parameter berechnen
        entry = current_ema50

        # SL: ATR-Puffer unter EMA50 für Wicks
        sl_mult = float(getattr(self.cfg, "GUSS_SL_ATR_MULT", 1.8))
        stop_loss = entry - current_atr * sl_mult

        # TP1: EMA21
        tp1 = current_ema21

        # TP2: 50% des Weges vom Entry zum letzten Hoch
        tp2 = entry + (swing_high_val - entry) * 0.5

        # TP3: letztes Hoch
        tp3 = swing_high_val

        # Validierung: TPs müssen über Entry liegen
        if tp1 <= entry or tp2 <= entry or tp3 <= entry:
            return None

        risk = entry - stop_loss
        if risk <= 0:
            return None

        rr1 = (tp1 - entry) / risk
        rr2 = (tp2 - entry) / risk
        rr3 = (tp3 - entry) / risk

        # Stärke: Anzahl bärischer Kerzen im Guss (mehr = sauberer)
        guss_len  = guss_end - guss_start + 1
        strength  = min(0.5 + guss_len * 0.06, 0.95)

        return Signal(
            type="GUSS_LONG",
            direction="long",
            timeframe=timeframe,
            price=current_close,
            strength=round(strength, 2),
            description=(
                f"GUSS Long: {guss_len} bärische Kerzen vom Hoch "
                f"({swing_high_val:,.0f}) zum EMA50 ({current_ema50:,.0f})"
            ),
            extra={
                "entry":          entry,
                "stop_loss":      stop_loss,
                "take_profits":   [tp1, tp2, tp3],
                "rr_ratios":      [round(rr1,2), round(rr2,2), round(rr3,2)],
                "swing_high":     swing_high_val,
                "swing_high_idx": int(swing_high_idx),
                "guss_len":       guss_len,
                "ema50":          current_ema50,
                "ema21":          current_ema21,
                # Für Chart-Markierung
                "ob_high": current_ema50 * 1.001,
                "ob_low":  stop_loss,
            },
        )

    # ── REVERSE GUSS SHORT ──────────────────────────────────────────────────
    def _detect_guss_short(self, df: pd.DataFrame, timeframe: str):
        n          = len(df)
        closes     = df["close"].values
        opens      = df["open"].values
        highs      = df["high"].values
        lows       = df["low"].values
        ema50      = df["ema50"].values
        ema21      = df["ema21"].values
        atr        = df["atr"].values

        lookback_start = max(0, n - self.swing_lookback)
        search_end     = n - 3

        # 1. Letztes Swing-Tief finden
        swing_low_idx = None
        swing_low_val = np.inf

        for i in range(lookback_start, search_end):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2]
                    and lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                if lows[i] < swing_low_val:
                    swing_low_val = lows[i]
                    swing_low_idx = i

        if swing_low_idx is None:
            return None

        # 2. Kerzen nach dem Swing-Tief ausschließlich bullisch
        guss_start = swing_low_idx + 1
        guss_end   = n - 1

        if guss_end - guss_start < 1:
            return None
        if guss_end - guss_start > self.max_guss_candles:
            return None

        for i in range(guss_start, guss_end + 1):
            if closes[i] <= opens[i]:   # bärische Kerze → kein Reverse GUSS
                return None

        # 3. Aktuelle Kerze berührt EMA50
        current_high  = highs[-1]
        current_close = closes[-1]
        current_ema50 = ema50[-1]
        current_ema21 = ema21[-1]
        current_atr   = atr[-1]

        lower_bound = current_ema50 * (1 - self.ema_touch_pct)
        upper_bound = current_ema50 * (1 + self.ema_touch_pct)

        ema50_touched = lower_bound <= current_high <= upper_bound \
                     or lower_bound <= current_close <= upper_bound

        if not ema50_touched:
            return None

        # 4. EMA50 muss ÜBER EMA21 liegen (Aufwärtstrend — Preis kam von unten)
        if current_ema50 <= current_ema21:
            return None

        # 5. Trade-Parameter
        entry = current_ema50

        sl_mult   = float(getattr(self.cfg, "GUSS_SL_ATR_MULT", 1.8))
        stop_loss = entry + current_atr * sl_mult

        # TP1: EMA21
        tp1 = current_ema21

        # TP2: 50% des Weges vom Entry zum letzten Tief
        tp2 = entry - (entry - swing_low_val) * 0.5

        # TP3: letztes Tief
        tp3 = swing_low_val

        if tp1 >= entry or tp2 >= entry or tp3 >= entry:
            return None

        risk = stop_loss - entry
        if risk <= 0:
            return None

        rr1 = (entry - tp1) / risk
        rr2 = (entry - tp2) / risk
        rr3 = (entry - tp3) / risk

        guss_len = guss_end - guss_start + 1
        strength = min(0.5 + guss_len * 0.06, 0.95)

        return Signal(
            type="GUSS_SHORT",
            direction="short",
            timeframe=timeframe,
            price=current_close,
            strength=round(strength, 2),
            description=(
                f"Reverse GUSS Short: {guss_len} bullische Kerzen vom Tief "
                f"({swing_low_val:,.0f}) zum EMA50 ({current_ema50:,.0f})"
            ),
            extra={
                "entry":         entry,
                "stop_loss":     stop_loss,
                "take_profits":  [tp1, tp2, tp3],
                "rr_ratios":     [round(rr1,2), round(rr2,2), round(rr3,2)],
                "swing_low":     swing_low_val,
                "swing_low_idx": int(swing_low_idx),
                "guss_len":      guss_len,
                "ema50":         current_ema50,
                "ema21":         current_ema21,
                "ob_high":       stop_loss,
                "ob_low":        current_ema50 * 0.999,
            },
        )
