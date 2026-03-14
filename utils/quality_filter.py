"""
Elite Trade Quality Filter
==========================
Simuliert die Denkweise eines erfahrenen BTC-Traders.
Jeder potenzielle Trade wird auf 8 Dimensionen bewertet.
Nur Trades mit ausreichend hohem Score erhalten einen Chart.

Bewertungsdimensionen (max. 100 Punkte gesamt):
  1. Risk/Reward Ratio          (25 Punkte)
  2. Trend-Alignment HTF        (20 Punkte)
  3. Kerzen-Bestätigung Entry   (15 Punkte)
  4. RSI-Kontext                (10 Punkte)
  5. Volumen-Bestätigung        (10 Punkte)
  6. Freier Weg zu TP1          (10 Punkte)
  7. Marktstruktur / ATR-Kontext ( 5 Punkte)
  8. Session-Timing             ( 5 Punkte)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from signals.base import Signal
from utils.risk_manager import Trade

logger = logging.getLogger(__name__)


# ── Konfiguration ─────────────────────────────────────────────────────────────
MIN_SCORE          = 62   # Mindest-Score für Chart-Ausgabe (0–100)
MIN_RR             = 1.8  # Absolutes Minimum RR — unter diesem Wert sofort ablehnen
PREFERRED_RR       = 2.5  # Ab hier volle Punkte für RR
EXCELLENT_RR       = 3.5  # Bonus

# Aktive Handelssessions (UTC-Stunden) — London + New York Overlap ist Gold
SESSIONS = {
    "london":        (7,  16),   # 07:00–16:00 UTC
    "new_york":      (13, 22),   # 13:00–22:00 UTC
    "london_ny_overlap": (13, 17),  # Beste Liquidität
    "asia":          (0,  8),    # 00:00–08:00 UTC
}

# Timeframe-Gewichtung für Trend-Alignment
TF_ORDER = ["1m","3m","5m","15m","30m","1h","2h","4h","8h","1d","3d","1w"]


@dataclass
class QualityResult:
    passed: bool
    total_score: float
    max_score: float = 100.0
    scores: dict = field(default_factory=dict)
    reasons_pass: list = field(default_factory=list)
    reasons_fail: list = field(default_factory=list)
    grade: str = ""

    @property
    def pct(self) -> float:
        return self.total_score / self.max_score * 100

    def __str__(self):
        status = "✅ APPROVED" if self.passed else "❌ REJECTED"
        grade_str = f" [{self.grade}]" if self.grade else ""
        lines = [
            f"{status}{grade_str}  Score: {self.total_score:.0f}/{self.max_score:.0f} ({self.pct:.0f}%)",
        ]
        for dim, (score, max_s, note) in self.scores.items():
            bar = "█" * int(score / max_s * 10) + "░" * (10 - int(score / max_s * 10))
            lines.append(f"  {bar}  {dim:<30} {score:4.1f}/{max_s:.0f}  {note}")
        if self.reasons_fail:
            lines.append("  Abgelehnt wegen: " + " | ".join(self.reasons_fail))
        return "\n".join(lines)


# ── Haupt-Klasse ──────────────────────────────────────────────────────────────

class TradeQualityFilter:
    """
    Bewertet jeden Trade auf 8 Qualitätsdimensionen.
    Gibt QualityResult zurück — bei passed=False keinen Chart erzeugen.
    """

    def __init__(self, cfg):
        self.cfg        = cfg
        self.min_score  = float(getattr(cfg, "QF_MIN_SCORE",  MIN_SCORE))
        self.min_rr     = float(getattr(cfg, "QF_MIN_RR",     MIN_RR))

    def evaluate(self, signal: Signal, trade: Trade, df: pd.DataFrame) -> QualityResult:
        scores = {}
        reasons_pass = []
        reasons_fail = []

        # ── 1. Risk / Reward ─────────────────────────────────── max 25
        rr_score, rr_note = self._score_rr(trade)
        scores["Risk/Reward"] = (rr_score, 25, rr_note)

        # Hartes Minimum: unter MIN_RR sofort ablehnen
        best_rr = self._best_rr(trade)
        if best_rr < self.min_rr:
            return QualityResult(
                passed=False,
                total_score=rr_score,
                scores=scores,
                reasons_fail=[f"RR {best_rr:.2f} unter Minimum {self.min_rr}"],
                grade="F",
            )

        # ── 2. Trend-Alignment HTF ───────────────────────────── max 20
        ta_score, ta_note = self._score_trend_alignment(signal, df)
        scores["Trend-Alignment HTF"] = (ta_score, 20, ta_note)

        # ── 3. Entry-Kerzen-Bestätigung ──────────────────────── max 15
        ec_score, ec_note = self._score_entry_candle(signal, df)
        scores["Entry-Bestätigung"] = (ec_score, 15, ec_note)

        # ── 4. RSI-Kontext ───────────────────────────────────── max 10
        rsi_score, rsi_note = self._score_rsi(signal, df)
        scores["RSI-Kontext"] = (rsi_score, 10, rsi_note)

        # ── 5. Volumen-Bestätigung ───────────────────────────── max 10
        vol_score, vol_note = self._score_volume(signal, df)
        scores["Volumen"] = (vol_score, 10, vol_note)

        # ── 6. Freier Weg zu TP1 ─────────────────────────────── max 10
        tp_score, tp_note = self._score_clear_path(signal, trade, df)
        scores["Freier Weg TP1"] = (tp_score, 10, tp_note)

        # ── 7. ATR / Marktstruktur ───────────────────────────── max 5
        atr_score, atr_note = self._score_atr_context(signal, trade, df)
        scores["ATR/Struktur"] = (atr_score, 5, atr_note)

        # ── 8. Session-Timing ────────────────────────────────── max 5
        sess_score, sess_note = self._score_session(df)
        scores["Session-Timing"] = (sess_score, 5, sess_note)

        # ── Gesamt ───────────────────────────────────────────────
        total = sum(s for s, _, _ in scores.values())
        passed = total >= self.min_score

        # Grade
        pct = total / 100
        if pct >= 0.90: grade = "S"
        elif pct >= 0.80: grade = "A"
        elif pct >= 0.70: grade = "B"
        elif pct >= 0.62: grade = "C"
        else: grade = "F"

        # Sammle Pass/Fail-Gründe
        for dim, (score, max_s, note) in scores.items():
            if score >= max_s * 0.7:
                reasons_pass.append(f"{dim}: {note}")
            elif score < max_s * 0.4:
                reasons_fail.append(f"{dim}: {note}")

        return QualityResult(
            passed=passed,
            total_score=round(total, 1),
            scores=scores,
            reasons_pass=reasons_pass,
            reasons_fail=reasons_fail,
            grade=grade,
        )

    # ── Scoring-Methoden ──────────────────────────────────────────────────────

    def _best_rr(self, trade: Trade) -> float:
        """Bestes RR über alle TPs."""
        if not trade.rr_ratios:
            return 0.0
        return max(trade.rr_ratios)

    def _score_rr(self, trade: Trade) -> tuple:
        """
        RR1 (zu TP1) zählt am meisten — das ist der realistischste Exit.
        Bonus für RR3 (voller Move).
        """
        if not trade.rr_ratios:
            return 0.0, "Kein RR berechenbar"

        rr1 = trade.rr_ratios[0] if len(trade.rr_ratios) > 0 else 0
        rr3 = trade.rr_ratios[-1] if len(trade.rr_ratios) > 2 else rr1

        # RR1 Scoring (max 18 Punkte)
        if rr1 >= PREFERRED_RR:
            rr1_score = 18.0
        elif rr1 >= self.min_rr:
            rr1_score = 18 * (rr1 - self.min_rr) / (PREFERRED_RR - self.min_rr)
        else:
            rr1_score = 0.0

        # RR3 Bonus (max 7 Punkte)
        if rr3 >= EXCELLENT_RR:
            rr3_score = 7.0
        elif rr3 >= PREFERRED_RR:
            rr3_score = 7 * (rr3 - PREFERRED_RR) / (EXCELLENT_RR - PREFERRED_RR)
        else:
            rr3_score = 0.0

        total = min(rr1_score + rr3_score, 25.0)
        note  = f"RR1={rr1:.1f}R  RR3={rr3:.1f}R"
        return round(total, 1), note

    def _score_trend_alignment(self, signal: Signal, df: pd.DataFrame) -> tuple:
        """
        Prüft ob der Trade MIT dem EMA200-Trend geht (höchste Gewichtung).
        Zusätzlich: EMA50 über/unter EMA200 als Trendbestätigung.
        """
        score = 0.0
        notes = []

        close   = df["close"].iloc[-1]
        ema200  = df["ema_trend"].iloc[-1] if "ema_trend" in df.columns else None
        ema50   = df["ema50"].iloc[-1]     if "ema50"    in df.columns else None
        ema21   = df["ema21"].iloc[-1]     if "ema21"    in df.columns else None

        direction = signal.direction

        if ema200 is not None:
            if direction == "long" and close > ema200:
                score += 10
                notes.append("Preis > EMA200 ✓")
            elif direction == "short" and close < ema200:
                score += 10
                notes.append("Preis < EMA200 ✓")
            else:
                notes.append("Gegen EMA200-Trend ✗")

        if ema50 is not None and ema200 is not None:
            if direction == "long" and ema50 > ema200:
                score += 5
                notes.append("EMA50 > EMA200 ✓")
            elif direction == "short" and ema50 < ema200:
                score += 5
                notes.append("EMA50 < EMA200 ✓")
            else:
                notes.append("EMA50/200 gegen Richtung")

        if ema21 is not None and ema50 is not None:
            if direction == "long" and ema21 > ema50:
                score += 5
                notes.append("EMA21 > EMA50 ✓")
            elif direction == "short" and ema21 < ema50:
                score += 5
                notes.append("EMA21 < EMA50 ✓")

        return round(min(score, 20.0), 1), " | ".join(notes) if notes else "keine EMA-Daten"

    def _score_entry_candle(self, signal: Signal, df: pd.DataFrame) -> tuple:
        """
        Bewertet die letzte Kerze (Entry-Kerze):
        - Pinbar / Hammer / Shooting Star
        - Engulfing
        - Starker Kerzenkörper (> 60% der Range)
        - Kerze schließt in Richtung des Trades
        """
        score = 0.0
        notes = []

        if len(df) < 3:
            return 0.0, "Zu wenig Daten"

        last   = df.iloc[-1]
        prev   = df.iloc[-2]
        o, h, l, c = last["open"], last["high"], last["low"], last["close"]
        p_o, p_h, p_l, p_c = prev["open"], prev["high"], prev["low"], prev["close"]

        candle_range = h - l
        body         = abs(c - o)
        upper_wick   = h - max(o, c)
        lower_wick   = min(o, c) - l

        if candle_range < 1e-9:
            return 0.0, "Doji / keine Range"

        body_ratio  = body / candle_range
        direction   = signal.direction

        # Kerze schließt in Trade-Richtung
        if direction == "long" and c > o:
            score += 4
            notes.append("Bullische Schlusskerze ✓")
        elif direction == "short" and c < o:
            score += 4
            notes.append("Bärische Schlusskerze ✓")

        # Pinbar / Hammer (langer Wick, kleiner Körper)
        if direction == "long" and lower_wick > body * 2 and lower_wick > candle_range * 0.5:
            score += 5
            notes.append("Hammer/Pinbar ✓")
        elif direction == "short" and upper_wick > body * 2 and upper_wick > candle_range * 0.5:
            score += 5
            notes.append("Shooting Star/Pinbar ✓")

        # Starker Körper
        if body_ratio > 0.65:
            score += 3
            notes.append(f"Starker Körper ({body_ratio:.0%}) ✓")

        # Engulfing
        if direction == "long" and c > p_h and o < p_l:
            score += 3
            notes.append("Bullisches Engulfing ✓")
        elif direction == "short" and c < p_l and o > p_h:
            score += 3
            notes.append("Bärisches Engulfing ✓")

        return round(min(score, 15.0), 1), " | ".join(notes) if notes else "Neutrale Kerze"

    def _score_rsi(self, signal: Signal, df: pd.DataFrame) -> tuple:
        """
        RSI soll NICHT gegen den Trade arbeiten:
        Long: RSI nicht überkauft (>75)
        Short: RSI nicht überverkauft (<25)
        Bonus: RSI dreht in Trade-Richtung (Momentum)
        """
        if "rsi" not in df.columns:
            return 5.0, "RSI nicht verfügbar (neutral)"

        rsi_now  = df["rsi"].iloc[-1]
        rsi_prev = df["rsi"].iloc[-2]
        direction = signal.direction

        score = 0.0
        notes = []

        if direction == "long":
            if rsi_now < 70:
                score += 6
                notes.append(f"RSI {rsi_now:.0f} — nicht überkauft ✓")
            elif rsi_now < 80:
                score += 2
                notes.append(f"RSI {rsi_now:.0f} — leicht überkauft")
            else:
                notes.append(f"RSI {rsi_now:.0f} — stark überkauft ✗")

            if rsi_now < 50 and rsi_now > rsi_prev:
                score += 4
                notes.append("RSI dreht bullisch aus Tiefe ✓")
            elif rsi_now > rsi_prev:
                score += 2
                notes.append("RSI steigt ✓")

        else:  # short
            if rsi_now > 30:
                score += 6
                notes.append(f"RSI {rsi_now:.0f} — nicht überverkauft ✓")
            elif rsi_now > 20:
                score += 2
                notes.append(f"RSI {rsi_now:.0f} — leicht überverkauft")
            else:
                notes.append(f"RSI {rsi_now:.0f} — stark überverkauft ✗")

            if rsi_now > 50 and rsi_now < rsi_prev:
                score += 4
                notes.append("RSI dreht bärisch aus Höhe ✓")
            elif rsi_now < rsi_prev:
                score += 2
                notes.append("RSI fällt ✓")

        return round(min(score, 10.0), 1), " | ".join(notes)

    def _score_volume(self, signal: Signal, df: pd.DataFrame) -> tuple:
        """
        Volumen der letzten 1–3 Kerzen vs. gleitender Durchschnitt.
        Überdurchschnittliches Volumen am Entry = Überzeugung.
        """
        if "vol_ma" not in df.columns or "volume" not in df.columns:
            return 5.0, "Volumen nicht verfügbar (neutral)"

        score = 0.0
        notes = []

        vol_now  = df["volume"].iloc[-1]
        vol_prev = df["volume"].iloc[-2]
        vol_ma   = df["vol_ma"].iloc[-1]

        if vol_ma <= 0:
            return 5.0, "Vol-MA = 0"

        ratio_now  = vol_now  / vol_ma
        ratio_prev = vol_prev / vol_ma

        # Aktuelle Kerze
        if ratio_now >= 1.8:
            score += 6
            notes.append(f"Vol ×{ratio_now:.1f} — sehr stark ✓✓")
        elif ratio_now >= 1.2:
            score += 4
            notes.append(f"Vol ×{ratio_now:.1f} — überdurchschnittlich ✓")
        elif ratio_now >= 0.8:
            score += 2
            notes.append(f"Vol ×{ratio_now:.1f} — normal")
        else:
            notes.append(f"Vol ×{ratio_now:.1f} — schwach ✗")

        # Vorherige Kerze (Guss-Kerzen Volumen)
        if ratio_prev >= 1.3:
            score += 4
            notes.append(f"Vorkerze Vol ×{ratio_prev:.1f} ✓")

        return round(min(score, 10.0), 1), " | ".join(notes)

    def _score_clear_path(self, signal: Signal, trade: Trade, df: pd.DataFrame) -> tuple:
        """
        Prüft ob zwischen Entry und TP1 wichtige Widerstände / Unterstützungen liegen.
        Methode: Swing-Hochs/-Tiefs und EMA-Cluster im TP1-Bereich zählen.
        """
        if not trade.take_profits:
            return 5.0, "Kein TP definiert"

        entry = trade.entry
        tp1   = trade.take_profits[0]
        score = 10.0
        notes = []

        # Zone zwischen Entry und TP1
        zone_lo = min(entry, tp1)
        zone_hi = max(entry, tp1)

        obstacles = 0

        # Swing-Hochs/-Tiefs in der Zone
        for col in ["swing_high", "swing_low"]:
            if col in df.columns:
                vals = df[col].dropna()
                in_zone = vals[(vals >= zone_lo * 0.999) & (vals <= zone_hi * 1.001)]
                obstacles += len(in_zone)

        # EMAs in der Zone
        for ema_col in ["ema21", "ema50", "ema_trend"]:
            if ema_col in df.columns:
                ema_val = df[ema_col].iloc[-1]
                if zone_lo * 0.998 <= ema_val <= zone_hi * 1.002:
                    obstacles += 1
                    notes.append(f"{ema_col} im Weg")

        if obstacles == 0:
            notes.append("Freier Weg zu TP1 ✓✓")
        elif obstacles <= 2:
            score -= 3
            notes.append(f"{obstacles} Hindernisse im Weg")
        else:
            score -= 7
            notes.append(f"{obstacles} Hindernisse — TP1 blockiert ✗")

        return round(max(score, 0.0), 1), " | ".join(notes) if notes else "Weg zu TP1 klar ✓"

    def _score_atr_context(self, signal: Signal, trade: Trade, df: pd.DataFrame) -> tuple:
        """
        ATR-Kontext: Ist der Trade in einem liquiden, bewegungsreichen Markt?
        - ATR sollte ausreichend groß sein (> 0.3% des Preises)
        - Nicht in extremer Kontraktion (Squeeze)
        """
        if "atr" not in df.columns:
            return 3.0, "ATR nicht verfügbar"

        close = df["close"].iloc[-1]
        atr   = df["atr"].iloc[-1]
        atr_pct = atr / close * 100

        # ATR-Trend (dehnt der Markt aus oder zieht er sich zusammen?)
        atr_5  = df["atr"].iloc[-5:].mean()
        atr_20 = df["atr"].iloc[-20:].mean() if len(df) >= 20 else atr_5

        score = 0.0
        notes = []

        # Ausreichend Volatilität
        if atr_pct >= 0.5:
            score += 3
            notes.append(f"ATR {atr_pct:.2f}% — gute Volatilität ✓")
        elif atr_pct >= 0.3:
            score += 1.5
            notes.append(f"ATR {atr_pct:.2f}% — akzeptabel")
        else:
            notes.append(f"ATR {atr_pct:.2f}% — zu gering ✗")

        # ATR expandiert (gut) vs. kontrahiert (schlecht)
        if atr_5 > atr_20 * 1.1:
            score += 2
            notes.append("ATR expandiert ✓")
        elif atr_5 < atr_20 * 0.7:
            notes.append("ATR Squeeze ✗")

        return round(min(score, 5.0), 1), " | ".join(notes)

    def _score_session(self, df: pd.DataFrame) -> tuple:
        """
        Präferiert Trades während London/New York Sessions.
        London-NY Overlap (13–17 UTC) ist der beste Zeitpunkt.
        """
        try:
            last_ts = df.index[-1]
            if hasattr(last_ts, "hour"):
                hour = last_ts.hour
            else:
                hour = datetime.utcnow().hour
        except Exception:
            return 3.0, "Session unbekannt (neutral)"

        if SESSIONS["london_ny_overlap"][0] <= hour < SESSIONS["london_ny_overlap"][1]:
            return 5.0, f"London/NY Overlap {hour:02d}:xx UTC ✓✓"
        elif SESSIONS["new_york"][0] <= hour < SESSIONS["new_york"][1]:
            return 3.5, f"New York Session {hour:02d}:xx UTC ✓"
        elif SESSIONS["london"][0] <= hour < SESSIONS["london"][1]:
            return 3.0, f"London Session {hour:02d}:xx UTC ✓"
        elif SESSIONS["asia"][0] <= hour < SESSIONS["asia"][1]:
            return 1.5, f"Asia Session {hour:02d}:xx UTC (niedrige Liquidität)"
        else:
            return 2.0, f"Off-Hours {hour:02d}:xx UTC"
