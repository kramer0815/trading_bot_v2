"""
Trendline Breakout Detector
============================
Erkennt signifikante Trendlinien und generiert Signale bei Breakouts.

Algorithmus:
  1. Pivot-Hochs und Pivot-Tiefs identifizieren (5-Kerzen-Fenster)
  2. Abwärtstrendlinien: lineare Regression über Pivot-Hochs
     Aufwärtstrendlinien: lineare Regression über Pivot-Tiefs
  3. Nur Linien mit ≥2 bestätigten Berührungen und negativer/positiver Steigung
  4. Breakout-Bedingungen:
     LONG:  Close bricht über Abwärtstrendlinie UND Volumen > 1.2x MA UND RSI > 45
     SHORT: Close bricht unter Aufwärtstrendlinie UND Volumen > 1.2x MA UND RSI < 55
  5. Kein Breakout durch Wick — nur Close-Basis
  6. Entry: aktueller Close
     SL: Trendlinie - ATR×1.5 (Long) / Trendlinie + ATR×1.5 (Short)
     TP1: nächster Widerstand/Support aus Swing-Struktur
     TP2: 1.618x ATR-Projektion
     TP3: 2.618x ATR-Projektion (Fibonacci Extension)
"""

import logging

import numpy as np
import pandas as pd

from .base import Signal

logger = logging.getLogger(__name__)

# ── Konfiguration ──────────────────────────────────────────────────────────

PIVOT_WINDOW      = 5      # Kerzen links + rechts für Pivot-Erkennung
MIN_PIVOTS        = 2      # Mindestanzahl Berührungspunkte für gültige Trendlinie
MAX_TOUCH_DIST    = 0.003  # Max. Abstand Pivot→Linie (0.3%) gilt als "Berührung"
LOOKBACK          = 80     # Wie viele Kerzen rückwärts für Trendlinien-Suche
MIN_SLOPE_PCT     = 0.0002 # Mindeststeigung pro Kerze (filtert flache Linien)
VOL_MULT          = 1.2    # Volumen muss X-faches des MA sein
BREAKOUT_CONFIRM  = 1      # Anzahl Kerzen die über/unter der Linie schließen müssen
SL_ATR_MULT       = 1.5    # ATR-Multiplikator für Stop Loss
MAX_BREAKOUT_DIST = 0.015  # Breakout darf max. 1.5% von der Linie entfernt sein


class TrendlineDetector:

    def __init__(self, cfg):
        self.cfg          = cfg
        self.pivot_window = int(getattr(cfg, "TL_PIVOT_WINDOW",   PIVOT_WINDOW))
        self.min_pivots   = int(getattr(cfg, "TL_MIN_PIVOTS",     MIN_PIVOTS))
        self.lookback     = int(getattr(cfg, "TL_LOOKBACK",       LOOKBACK))
        self.vol_mult     = float(getattr(cfg, "TL_VOL_MULT",     VOL_MULT))
        self.sl_atr_mult  = float(getattr(cfg, "TL_SL_ATR_MULT",  SL_ATR_MULT))

    # ── Hauptmethode ──────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame, timeframe: str) -> list:
        signals = []
        required = {"open", "high", "low", "close", "volume", "atr"}
        if not required.issubset(df.columns):
            logger.debug(f"[TL] Fehlende Spalten: {required - set(df.columns)}")
            return signals

        df = df.copy()
        if "rsi" not in df.columns:
            df["rsi"] = self._rsi(df["close"])
        if "vol_ma" not in df.columns:
            df["vol_ma"] = df["volume"].rolling(20).mean()

        n = len(df)
        if n < self.lookback + self.pivot_window:
            return signals

        # Arbeitsbereich: letzten LOOKBACK Kerzen
        work = df.iloc[-(self.lookback + self.pivot_window):]

        # Pivot-Punkte finden
        pivot_highs = self._find_pivots(work["high"].values,  "high")
        pivot_lows  = self._find_pivots(work["low"].values,   "low")

        # Trendlinien berechnen
        down_lines = self._fit_trendlines(pivot_highs, work["high"].values,  "down")
        up_lines   = self._fit_trendlines(pivot_lows,  work["low"].values,   "up")

        # Breakout prüfen
        last_close  = float(df["close"].iloc[-1])
        last_atr    = float(df["atr"].iloc[-1])
        last_vol    = float(df["volume"].iloc[-1])
        last_vol_ma = float(df["vol_ma"].iloc[-1]) if df["vol_ma"].iloc[-1] > 0 else 1
        last_rsi    = float(df["rsi"].iloc[-1])
        prev_close  = float(df["close"].iloc[-2])
        n_work      = len(work)

        for line in down_lines:
            sig = self._check_long_breakout(
                line, df, work, last_close, prev_close,
                last_atr, last_vol, last_vol_ma, last_rsi,
                n_work, timeframe
            )
            if sig:
                signals.append(sig)
                break  # Stärkste Linie reicht

        for line in up_lines:
            sig = self._check_short_breakout(
                line, df, work, last_close, prev_close,
                last_atr, last_vol, last_vol_ma, last_rsi,
                n_work, timeframe
            )
            if sig:
                signals.append(sig)
                break

        return signals

    # ── Pivot-Erkennung ───────────────────────────────────────────────────

    def _find_pivots(self, values: np.ndarray, pivot_type: str) -> list:
        """Findet Pivot-Hochs oder -Tiefs im Array."""
        pivots = []
        w = self.pivot_window
        n = len(values)
        for i in range(w, n - w):
            window = values[i-w:i+w+1]
            if pivot_type == "high" and values[i] == np.max(window):
                pivots.append((i, float(values[i])))
            elif pivot_type == "low" and values[i] == np.min(window):
                pivots.append((i, float(values[i])))
        return pivots

    # ── Trendlinien-Berechnung ────────────────────────────────────────────

    def _fit_trendlines(self, pivots: list, values: np.ndarray,
                        direction: str) -> list:
        """
        Berechnet die besten Trendlinien aus Pivot-Punkten.
        direction: 'down' (über Hochs) oder 'up' (unter Tiefs)
        Gibt sortierte Liste von Linien zurück (beste zuerst).
        """
        if len(pivots) < self.min_pivots:
            return []

        lines = []
        n     = len(pivots)

        # Alle Paare von Pivots als mögliche Trendlinien
        for i in range(n - 1):
            for j in range(i + 1, n):
                x1, y1 = pivots[i]
                x2, y2 = pivots[j]

                if x2 == x1:
                    continue

                slope     = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Steigung prüfen
                slope_pct = abs(slope) / ((y1 + y2) / 2)
                if slope_pct < MIN_SLOPE_PCT:
                    continue  # Zu flach

                # Richtung prüfen
                if direction == "down" and slope >= 0:
                    continue  # Abwärtstrendlinie muss fallend sein
                if direction == "up" and slope <= 0:
                    continue  # Aufwärtstrendlinie muss steigend sein

                # Alle Pivots zählen die diese Linie berühren
                touches = []
                for px, py in pivots:
                    line_y   = slope * px + intercept
                    dist_pct = abs(py - line_y) / line_y
                    if dist_pct <= MAX_TOUCH_DIST:
                        touches.append((px, py))

                if len(touches) < self.min_pivots:
                    continue

                # Prüfen: Kerzen sollen auf der "richtigen" Seite der Linie liegen
                # (Downtrend: Kerzen unter der Linie, Uptrend: Kerzen über der Linie)
                violations = 0
                for k in range(x1, len(values)):
                    line_y = slope * k + intercept
                    if direction == "down" and values[k] > line_y * 1.001:
                        violations += 1
                    elif direction == "up" and values[k] < line_y * 0.999:
                        violations += 1

                if violations > len(values) * 0.05:  # Max 5% Verletzungen
                    continue

                lines.append({
                    "slope":     slope,
                    "intercept": intercept,
                    "touches":   touches,
                    "n_touches": len(touches),
                    "direction": direction,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    # Aktuelle Linienwert (extrapoliert auf letzte Kerze)
                    "current_y": slope * (len(values) - 1) + intercept,
                    "slope_pct_per_candle": slope / ((y1 + y2) / 2) * 100,
                })

        # Sortieren: mehr Berührungen = besser, dann nach Aktualität
        lines.sort(key=lambda l: (-l["n_touches"], -l["x2"]))
        return lines[:3]  # Top 3 Linien

    # ── Breakout-Prüfung LONG ─────────────────────────────────────────────

    def _check_long_breakout(self, line, df, work, last_close, prev_close,
                              last_atr, last_vol, last_vol_ma, last_rsi,
                              n_work, timeframe):
        """Prüft ob der Preis eine Abwärtstrendlinie von unten nach oben bricht."""

        line_now  = line["current_y"]
        line_prev = line["slope"] * (n_work - 2) + line["intercept"]

        # Kernbedingung: vorherige Kerze unter der Linie, aktuelle darüber
        if not (prev_close <= line_prev * 1.001 and last_close > line_now):
            return None

        # Breakout darf nicht zu weit von der Linie entfernt sein (kein Gap-Breakout)
        breakout_dist = (last_close - line_now) / line_now
        if breakout_dist > MAX_BREAKOUT_DIST:
            return None

        # Volumen-Bestätigung
        vol_ok = last_vol >= last_vol_ma * self.vol_mult

        # RSI-Bestätigung (nicht überkauft)
        rsi_ok = last_rsi > 45 and last_rsi < 75

        if not (vol_ok and rsi_ok):
            logger.debug(f"[TL LONG] Breakout ohne Bestätigung: vol={last_vol/last_vol_ma:.1f}x rsi={last_rsi:.0f}")
            return None

        # Trade-Parameter
        entry     = last_close
        stop_loss = line_now - last_atr * self.sl_atr_mult

        if stop_loss >= entry:
            return None

        risk = entry - stop_loss
        if risk <= 0:
            return None

        # TPs: Swing-Struktur + Fibonacci-Extensionen
        tp1, tp2, tp3 = self._calc_long_tps(df, entry, risk)

        if tp1 <= entry:
            return None

        rr1 = (tp1 - entry) / risk
        rr2 = (tp2 - entry) / risk
        rr3 = (tp3 - entry) / risk

        n_touches  = line["n_touches"]
        slope_deg  = abs(line["slope_pct_per_candle"])
        strength   = min(0.4 + n_touches * 0.1 + (vol_ok * 0.15) + (slope_deg * 2), 0.95)

        return Signal(
            type="TL_BREAKOUT_LONG",
            direction="long",
            timeframe=timeframe,
            price=last_close,
            strength=round(strength, 2),
            description=(
                f"Trendlinie-Breakout Long: Abwärtstrendlinie bei {line_now:,.0f} "
                f"gebrochen ({n_touches} Berührungen, "
                f"Steigung {line['slope_pct_per_candle']:.3f}%/Kerze)"
            ),
            extra={
                "entry":          entry,
                "stop_loss":      stop_loss,
                "take_profits":   [tp1, tp2, tp3],
                "rr_ratios":      [round(rr1,2), round(rr2,2), round(rr3,2)],
                "trendline_y":    round(line_now, 0),
                "trendline_slope": round(line["slope"], 2),
                "trendline_touches": n_touches,
                "trendline_direction": "down",
                # Für Chart-Zeichnung
                "tl_x1":     line["x1"],
                "tl_y1":     round(line["y1"], 0),
                "tl_x2":     n_work - 1,
                "tl_y2":     round(line_now, 0),
                "vol_ratio": round(last_vol / last_vol_ma, 2),
                "rsi_at_breakout": round(last_rsi, 1),
                "ob_high":   entry * 1.001,
                "ob_low":    stop_loss,
            },
        )

    # ── Breakout-Prüfung SHORT ────────────────────────────────────────────

    def _check_short_breakout(self, line, df, work, last_close, prev_close,
                               last_atr, last_vol, last_vol_ma, last_rsi,
                               n_work, timeframe):
        """Prüft ob der Preis eine Aufwärtstrendlinie von oben nach unten bricht."""

        line_now  = line["current_y"]
        line_prev = line["slope"] * (n_work - 2) + line["intercept"]

        # Kernbedingung: vorherige Kerze über der Linie, aktuelle darunter
        if not (prev_close >= line_prev * 0.999 and last_close < line_now):
            return None

        breakout_dist = (line_now - last_close) / line_now
        if breakout_dist > MAX_BREAKOUT_DIST:
            return None

        vol_ok = last_vol >= last_vol_ma * self.vol_mult
        rsi_ok = last_rsi < 55 and last_rsi > 25

        if not (vol_ok and rsi_ok):
            return None

        entry     = last_close
        stop_loss = line_now + last_atr * self.sl_atr_mult

        if stop_loss <= entry:
            return None

        risk = stop_loss - entry
        if risk <= 0:
            return None

        tp1, tp2, tp3 = self._calc_short_tps(df, entry, risk)

        if tp1 >= entry:
            return None

        rr1 = (entry - tp1) / risk
        rr2 = (entry - tp2) / risk
        rr3 = (entry - tp3) / risk

        n_touches = line["n_touches"]
        slope_deg = abs(line["slope_pct_per_candle"])
        strength  = min(0.4 + n_touches * 0.1 + (vol_ok * 0.15) + (slope_deg * 2), 0.95)

        return Signal(
            type="TL_BREAKOUT_SHORT",
            direction="short",
            timeframe=timeframe,
            price=last_close,
            strength=round(strength, 2),
            description=(
                f"Trendlinie-Breakout Short: Aufwärtstrendlinie bei {line_now:,.0f} "
                f"gebrochen ({n_touches} Berührungen, "
                f"Steigung {line['slope_pct_per_candle']:.3f}%/Kerze)"
            ),
            extra={
                "entry":          entry,
                "stop_loss":      stop_loss,
                "take_profits":   [tp1, tp2, tp3],
                "rr_ratios":      [round(rr1,2), round(rr2,2), round(rr3,2)],
                "trendline_y":    round(line_now, 0),
                "trendline_slope": round(line["slope"], 2),
                "trendline_touches": n_touches,
                "trendline_direction": "up",
                "tl_x1":     line["x1"],
                "tl_y1":     round(line["y1"], 0),
                "tl_x2":     n_work - 1,
                "tl_y2":     round(line_now, 0),
                "vol_ratio": round(last_vol / last_vol_ma, 2),
                "rsi_at_breakout": round(last_rsi, 1),
                "ob_high":   stop_loss,
                "ob_low":    entry * 0.999,
            },
        )

    # ── Take-Profit Berechnung ────────────────────────────────────────────

    def _calc_long_tps(self, df, entry: float, risk: float):
        """TP1: nächster Swing-High, TP2: 1.618R, TP3: 2.618R"""
        # Letzten Swing-High über Entry finden
        highs  = df["high"].values[-self.lookback:]
        swings = [h for h in highs if h > entry * 1.001]
        tp1 = min(swings) if swings else entry + risk * 1.5
        tp1 = max(tp1, entry + risk * 1.2)  # Mindest-RR 1.2

        tp2 = entry + risk * 1.618
        tp3 = entry + risk * 2.618
        return round(tp1, 0), round(tp2, 0), round(tp3, 0)

    def _calc_short_tps(self, df, entry: float, risk: float):
        """TP1: nächster Swing-Low, TP2: 1.618R, TP3: 2.618R"""
        lows   = df["low"].values[-self.lookback:]
        swings = [l for l in lows if l < entry * 0.999]
        tp1 = max(swings) if swings else entry - risk * 1.5
        tp1 = min(tp1, entry - risk * 1.2)

        tp2 = entry - risk * 1.618
        tp3 = entry - risk * 2.618
        return round(tp1, 0), round(tp2, 0), round(tp3, 0)

    # ── Hilfsfunktionen ───────────────────────────────────────────────────

    def _rsi(self, s: pd.Series, n: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + g / (l + 1e-9))
