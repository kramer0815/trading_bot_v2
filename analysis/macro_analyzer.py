import os
"""
Macro BTC Analyzer — 1W + 1D
==============================
Professionelle Chartanalyse für Bitcoin.
Analysiert einmal täglich und cached das Ergebnis als JSON.

Indikatoren & Werkzeuge:
  - EMA 21, 50, 100, 200
  - RSI(14) mit Divergenz-Erkennung
  - MACD(12,26,9) + Histogramm-Momentum
  - Fibonacci Retracement (0, 0.236, 0.382, 0.5, 0.618, 0.65, 0.786, 0.822, 0.941, 1.0)
  - Volumen + OBV-Trend
  - Marktstruktur (HH/HL/LH/LL, MSB/CHoCH)
  - Liquiditätszonen (Equal Highs/Lows, BSL/SSL)
  - Support/Resistance Cluster
  - Trendphase (Accumulation / Markup / Distribution / Markdown)
  - ATR-basierte Volatilitätsanalyse
  - Gesamt-Bias Score + narrativer Kommentar
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.65, 0.786, 0.822, 0.941, 1.0]
FIB_NAMES  = {
    0:     "0 — Swing Low",
    0.236: "0.236",
    0.382: "0.382 — Goldene Zone",
    0.5:   "0.5 — Mittelwert",
    0.618: "0.618 — Goldener Schnitt",
    0.65:  "0.65",
    0.786: "0.786",
    0.822: "0.822",
    0.941: "0.941",
    1.0:   "1.0 — Swing High",
}


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def _atr(h, l, c, n=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _obv(c, v):
    sign = np.sign(c.diff()).fillna(0)
    return (sign * v).cumsum()

def _macd(s, fast=12, slow=26, sig=9):
    line   = _ema(s, fast) - _ema(s, slow)
    signal = _ema(line, sig)
    return line, signal, line - signal


class MacroAnalyzer:

    def __init__(self, cfg):
        self.cfg       = cfg
        self.cache_dir = Path(os.getenv("MACRO_CACHE_DIR", "/app/macro_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ex_cls = getattr(ccxt, cfg.EXCHANGE)
        self.exchange  = ex_cls({
            "apiKey": cfg.API_KEY,
            "secret": cfg.API_SECRET,
            "enableRateLimit": True,
        })

    def get_analysis(self, force=False) -> dict:
        """Gibt gecachte Analyse zurück oder erstellt neue wenn > 24h alt."""
        cache_file = self.cache_dir / "macro_analysis.json"
        if not force and cache_file.exists():
            age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if age < 86400:
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    data["from_cache"]      = True
                    data["cache_age_hours"] = round(age / 3600, 1)
                    return data
                except Exception:
                    pass

        result = self._run_analysis()
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Cache-Schreiben: {e}")

        result["from_cache"]      = False
        result["cache_age_hours"] = 0
        return result

    def _fetch(self, tf, limit=300):
        raw = self.exchange.fetch_ohlcv(self.cfg.SYMBOL, tf, limit=limit)
        df  = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df.astype(float)

    def _run_analysis(self) -> dict:
        logger.info("Starte Makro-Analyse 1W + 1D …")
        result = {
            "generated_at": datetime.now().isoformat(),
            "symbol":       self.cfg.SYMBOL,
            "weekly":       {},
            "daily":        {},
            "fibonacci":    {},
            "bias":         {},
            "key_levels":   [],
        }

        try:
            df_w = self._fetch("1w", 200)
            result["weekly"] = self._analyze_tf(df_w, "1W")
        except Exception as e:
            logger.error(f"Weekly: {e}")
            result["weekly"] = {"error": str(e)}

        try:
            df_d = self._fetch("1d", 365)
            result["daily"] = self._analyze_tf(df_d, "1D")
        except Exception as e:
            logger.error(f"Daily: {e}")
            result["daily"] = {"error": str(e)}

        try:
            df_w = self._fetch("1w", 200)
            result["fibonacci"] = self._calc_fibonacci(df_w)
        except Exception as e:
            result["fibonacci"] = {"error": str(e)}

        result["bias"]       = self._calc_bias(result)
        result["key_levels"] = self._merge_key_levels(result)
        logger.info("Makro-Analyse abgeschlossen.")
        return result

    # ── Timeframe-Analyse ──────────────────────────────────────────────────

    def _analyze_tf(self, df: pd.DataFrame, label: str) -> dict:
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        ema21  = _ema(c, 21);  ema50  = _ema(c, 50)
        ema100 = _ema(c, 100); ema200 = _ema(c, 200)
        rsi_s  = _rsi(c)
        macd_line, macd_sig, macd_hist = _macd(c)
        atr_s  = _atr(h, l, c)
        obv_s  = _obv(c, v)
        vol_ma = v.rolling(20).mean()

        price  = float(c.iloc[-1])
        atr_v  = float(atr_s.iloc[-1])
        rsi_v  = float(rsi_s.iloc[-1])

        ema_data = {
            "ema21":  round(float(ema21.iloc[-1]),  0),
            "ema50":  round(float(ema50.iloc[-1]),  0),
            "ema100": round(float(ema100.iloc[-1]), 0),
            "ema200": round(float(ema200.iloc[-1]), 0),
        }
        ema_trend = (
            "Bullisch" if price > ema21.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]
            else "Bärisch" if price < ema21.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]
            else "Gemischt"
        )

        macd_v   = float(macd_line.iloc[-1])
        macd_sv  = float(macd_sig.iloc[-1])
        macd_hv  = float(macd_hist.iloc[-1])
        macd_cross = (
            "Bullisches Crossover"
            if macd_v > macd_sv and float(macd_line.iloc[-2]) <= float(macd_sig.iloc[-2])
            else "Bärisches Crossover"
            if macd_v < macd_sv and float(macd_line.iloc[-2]) >= float(macd_sig.iloc[-2])
            else "Über Signal" if macd_v > macd_sv else "Unter Signal"
        )

        vol_ratio = float(v.iloc[-1] / vol_ma.iloc[-1]) if float(vol_ma.iloc[-1]) > 0 else 1.0
        obv_slope = float(obv_s.iloc[-1] - obv_s.iloc[-10]) / (abs(float(obv_s.iloc[-10])) + 1)

        atr_pct    = atr_v / price * 100
        atr_20_avg = float(atr_s.iloc[-20:].mean())
        atr_state  = ("Expansiv" if atr_v > atr_20_avg * 1.2
                      else "Kontraktiv" if atr_v < atr_20_avg * 0.8
                      else "Normal")

        return {
            "label":     label,
            "price":     round(price, 0),
            "atr":       round(atr_v, 0),
            "atr_pct":   round(atr_pct, 2),
            "atr_state": atr_state,
            "emas":      ema_data,
            "ema_trend": ema_trend,
            "rsi":       round(rsi_v, 1),
            "rsi_zone":  ("Überkauft"    if rsi_v > 70
                          else "Überverkauft" if rsi_v < 30
                          else "Bullisch" if rsi_v > 55
                          else "Neutral"),
            "rsi_divergence": self._rsi_divergence(c, rsi_s),
            "macd": {
                "line":      round(macd_v,  1),
                "signal":    round(macd_sv, 1),
                "hist":      round(macd_hv, 1),
                "cross":     macd_cross,
                "momentum":  ("Zunehmend" if abs(macd_hv) > abs(float(macd_hist.iloc[-2]))
                               else "Abnehmend"),
                "above_zero": macd_v > 0,
            },
            "volume": {
                "ratio":    round(vol_ratio, 2),
                "vs_avg":   ("Hoch"    if vol_ratio > 1.5
                             else "Niedrig" if vol_ratio < 0.7
                             else "Normal"),
                "obv_bias": "Bullisch" if obv_slope > 0 else "Bärisch",
            },
            "structure": self._market_structure(h, l, c),
            "liquidity": self._liquidity_zones(h, l, price),
            "sr_levels": self._sr_clusters(h, l, c, price),
            "phase":     self._market_phase(c, v, ema50, ema200, rsi_s),
        }

    # ── Marktstruktur ──────────────────────────────────────────────────────

    def _market_structure(self, h, l, c) -> dict:
        highs = h.values
        lows  = l.values
        n     = len(highs)

        swing_highs, swing_lows = [], []
        for i in range(2, n - 2):
            if highs[i] == max(highs[i-2:i+3]):
                swing_highs.append((i, float(highs[i])))
            if lows[i] == min(lows[i-2:i+3]):
                swing_lows.append((i, float(lows[i])))

        last_sh = swing_highs[-4:] if len(swing_highs) >= 4 else swing_highs
        last_sl = swing_lows[-4:]  if len(swing_lows)  >= 4 else swing_lows

        if len(last_sh) >= 2 and len(last_sl) >= 2:
            hh = last_sh[-1][1] > last_sh[-2][1]
            hl = last_sl[-1][1] > last_sl[-2][1]
            if   hh and hl:       struct = "Bullisch (HH/HL)"
            elif not hh and not hl: struct = "Bärisch (LH/LL)"
            elif hh and not hl:   struct = "Distribution (HH/LL)"
            else:                 struct = "Akkumulation (LH/HL)"
        else:
            struct = "Unzureichend Daten"

        price_now = float(c.iloc[-1])
        msb = None
        if swing_lows   and price_now < swing_lows[-1][1]:
            msb = f"MSB Bearish — Tief {swing_lows[-1][1]:,.0f} gebrochen"
        elif swing_highs and price_now > swing_highs[-1][1]:
            msb = f"MSB Bullish — Hoch {swing_highs[-1][1]:,.0f} gebrochen"

        return {
            "type":        struct,
            "msb":         msb,
            "last_high":   round(swing_highs[-1][1], 0) if swing_highs else None,
            "last_low":    round(swing_lows[-1][1],  0) if swing_lows  else None,
            "swing_highs": [round(x[1], 0) for x in swing_highs[-3:]],
            "swing_lows":  [round(x[1], 0) for x in swing_lows[-3:]],
        }

    # ── Fibonacci ──────────────────────────────────────────────────────────

    def _calc_fibonacci(self, df: pd.DataFrame) -> dict:
        n       = len(df)
        highs   = df["high"].values
        lows    = df["low"].values
        close   = float(df["close"].iloc[-1])
        window  = min(52, n)

        swing_high_val = float(np.max(highs[-window:]))
        swing_low_val  = float(np.min(lows[-window:]))
        sh_idx = int(np.argmax(highs[-window:]))
        sl_idx = int(np.argmin(lows[-window:]))

        is_bullish = sl_idx < sh_idx
        diff       = swing_high_val - swing_low_val

        levels = {}
        for lvl in FIB_LEVELS:
            price_at = (swing_high_val - lvl * diff) if is_bullish else (swing_low_val + lvl * diff)
            dist_pct  = abs(price_at - close) / close * 100
            levels[str(lvl)] = {
                "level":        lvl,
                "name":         FIB_NAMES.get(lvl, str(lvl)),
                "price":        round(price_at, 0),
                "distance_pct": round(dist_pct, 2),
                "active":       dist_pct < 3.0,
                "above":        price_at > close,
            }

        below = [(k, v) for k, v in levels.items() if not v["above"]]
        above = [(k, v) for k, v in levels.items() if v["above"]]
        next_sup = max(below, key=lambda x: x[1]["price"])[1] if below else None
        next_res = min(above, key=lambda x: x[1]["price"])[1] if above else None

        return {
            "swing_high":      round(swing_high_val, 0),
            "swing_low":       round(swing_low_val,  0),
            "direction":       "Bullisch" if is_bullish else "Bärisch",
            "range":           round(diff, 0),
            "current_price":   round(close, 0),
            "levels":          levels,
            "next_support":    next_sup,
            "next_resistance": next_res,
        }

    # ── Liquiditätszonen ───────────────────────────────────────────────────

    def _liquidity_zones(self, h, l, price) -> dict:
        highs = h.values[-60:]
        lows  = l.values[-60:]
        bsl, ssl = [], []

        for i in range(len(highs) - 3):
            cluster = [highs[j] for j in range(i, min(i+10, len(highs)))
                       if abs(highs[j] - highs[i]) / highs[i] < 0.005]
            if len(cluster) >= 2:
                val = round(float(np.mean(cluster)), 0)
                if val > price and not any(abs(val - z) / val < 0.012 for z in bsl):
                    bsl.append(val)

        for i in range(len(lows) - 3):
            cluster = [lows[j] for j in range(i, min(i+10, len(lows)))
                       if abs(lows[j] - lows[i]) / lows[i] < 0.005]
            if len(cluster) >= 2:
                val = round(float(np.mean(cluster)), 0)
                if val < price and not any(abs(val - z) / val < 0.012 for z in ssl):
                    ssl.append(val)

        return {
            "bsl": sorted(bsl)[:4],
            "ssl": sorted(ssl, reverse=True)[:4],
        }

    # ── S/R Cluster ────────────────────────────────────────────────────────

    def _sr_clusters(self, h, l, c, price) -> list:
        all_vals = list(h.values[-120:]) + list(l.values[-120:])
        all_vals.sort()
        clusters = []
        for v in all_vals:
            matched = False
            for cl in clusters:
                if abs(v - cl["center"]) / cl["center"] < 0.012:
                    cl["count"] += 1
                    cl["center"] = (cl["center"] * (cl["count"]-1) + v) / cl["count"]
                    matched = True
                    break
            if not matched:
                clusters.append({"center": v, "count": 1})

        strong = [cl for cl in clusters if cl["count"] >= 3]
        strong.sort(key=lambda x: -x["count"])
        result = []
        for cl in strong[:10]:
            val = round(float(cl["center"]), 0)
            result.append({
                "price":    val,
                "strength": cl["count"],
                "type":     "Resistance" if val > price else "Support",
            })
        return sorted(result, key=lambda x: x["price"])

    # ── Marktphase ─────────────────────────────────────────────────────────

    def _market_phase(self, c, v, ema50, ema200, rsi) -> dict:
        price  = float(c.iloc[-1])
        e50    = float(ema50.iloc[-1])
        e200   = float(ema200.iloc[-1])
        rsi_v  = float(rsi.iloc[-1])
        p50    = (price - e50)  / e50  * 100
        p200   = (price - e200) / e200 * 100
        spread = (e50 - e200)   / e200 * 100
        vol_20 = float(c.pct_change().dropna().iloc[-20:].std() * 100)

        if   price > e50 > e200 and p200 > 10 and rsi_v > 55:
            phase, desc, color = "Markup",    "Klarer Aufwärtstrend — Preis über EMA50 & 200.", "#26a69a"
        elif price < e50 < e200 and p200 < -10 and rsi_v < 45:
            phase, desc, color = "Markdown",  "Klarer Abwärtstrend — Preis unter EMA50 & 200.", "#ef5350"
        elif abs(p50) < 5 and abs(spread) < 5 and vol_20 < 2.5:
            phase, desc, color = "Akkumulation/Distribution", "Seitwärtsphase — EMAs eng, niedrige Volatilität.", "#ff9800"
        elif price > e200 and price < e50:
            phase, desc, color = "Retest",    "Preis testet EMA50 von oben — mögliche Bounce-Zone.", "#4fc3f7"
        else:
            phase, desc, color = "Transition","Keine eindeutige Phase erkennbar.", "#90a4ae"

        return {
            "phase":         phase,
            "description":   desc,
            "color":         color,
            "price_vs_50":   round(p50,    1),
            "price_vs_200":  round(p200,   1),
            "ema_spread":    round(spread,  1),
            "volatility_20": round(vol_20,  2),
        }

    # ── RSI Divergenz ──────────────────────────────────────────────────────

    def _rsi_divergence(self, c, rsi) -> dict:
        n      = 20
        closes = c.iloc[-n:].values
        rsi_v  = rsi.iloc[-n:].values
        ph_idx = int(np.argmax(closes));  pl_idx = int(np.argmin(closes))
        rh_idx = int(np.argmax(rsi_v));   rl_idx = int(np.argmin(rsi_v))

        div, detail = "Keine", ""
        if (ph_idx > n//2 and rh_idx < n//2
                and closes[-1] > np.mean(closes[:n//2])
                and rsi_v[-1]  < np.mean(rsi_v[:n//2])):
            div, detail = "Bärisch", "Preis steigt, RSI fällt — Schwächesignal"
        elif (pl_idx > n//2 and rl_idx < n//2
              and closes[-1] < np.mean(closes[:n//2])
              and rsi_v[-1]  > np.mean(rsi_v[:n//2])):
            div, detail = "Bullisch", "Preis fällt, RSI steigt — Stärkesignal"

        return {"type": div, "detail": detail}

    # ── Gesamt-Bias ────────────────────────────────────────────────────────

    def _calc_bias(self, result) -> dict:
        w = result.get("weekly", {})
        d = result.get("daily",  {})
        if "error" in w or "error" in d:
            return {"direction": "Unbekannt", "strength": "neutral", "score": 0, "factors": [], "summary": "Analysefehler"}

        score, factors = 0, []
        checks = [
            (w.get("ema_trend") == "Bullisch",    3, "W EMA-Stack bullisch ✓"),
            (w.get("ema_trend") == "Bärisch",    -3, "W EMA-Stack bärisch ✗"),
            (d.get("ema_trend") == "Bullisch",    2, "D EMA-Stack bullisch ✓"),
            (d.get("ema_trend") == "Bärisch",    -2, "D EMA-Stack bärisch ✗"),
            (w.get("rsi", 50) > 60,               1, f"W RSI {w.get('rsi',50):.0f} bullisch"),
            (w.get("rsi", 50) < 40,              -1, f"W RSI {w.get('rsi',50):.0f} bärisch"),
            (d.get("macd", {}).get("above_zero"), 1, "D MACD über Null ✓"),
            (not d.get("macd", {}).get("above_zero"), -1, "D MACD unter Null ✗"),
            ("Bullisches" in d.get("macd", {}).get("cross", ""), 1, "D MACD Crossover ✓"),
            ("Bärisches"  in d.get("macd", {}).get("cross", ""), -1, "D MACD Crossover ✗"),
            ("Bullisch" in w.get("structure", {}).get("type", ""),  2, "W Struktur HH/HL ✓"),
            ("Bärisch"  in w.get("structure", {}).get("type", ""), -2, "W Struktur LH/LL ✗"),
            (w.get("volume", {}).get("obv_bias") == "Bullisch",  1, "W OBV bullisch ✓"),
            (w.get("volume", {}).get("obv_bias") == "Bärisch",  -1, "W OBV bärisch ✗"),
            (d.get("phase", {}).get("phase") == "Markup",   1, "Phase: Markup ✓"),
            (d.get("phase", {}).get("phase") == "Markdown", -1, "Phase: Markdown ✗"),
        ]
        for cond, pts, label in checks:
            if cond:
                score += pts
                factors.append(label)

        norm = max(-10, min(10, score))
        if   norm >= 6:  direction, strength = "Stark Bullisch",  "bullish-strong"
        elif norm >= 3:  direction, strength = "Bullisch",        "bullish"
        elif norm >= 1:  direction, strength = "Leicht Bullisch", "bullish-weak"
        elif norm <= -6: direction, strength = "Stark Bärisch",   "bearish-strong"
        elif norm <= -3: direction, strength = "Bärisch",         "bearish"
        elif norm <= -1: direction, strength = "Leicht Bärisch",  "bearish-weak"
        else:            direction, strength = "Neutral",         "neutral"

        price   = d.get("price", 0)
        phase   = d.get("phase", {}).get("phase", "—")
        w_struct = w.get("structure", {}).get("type", "—")
        fib     = result.get("fibonacci", {})
        fib_s   = fib.get("next_support",    {}) or {}
        fib_r   = fib.get("next_resistance", {}) or {}

        parts = [
            f"BTC handelt bei ${price:,.0f} in einer {phase}-Phase.",
            f"Wöchentliche Struktur: {w_struct}.",
            f"RSI Weekly {w.get('rsi',50):.0f} / Daily {d.get('rsi',50):.0f}.",
        ]
        if fib_s.get("price"):
            parts.append(f"Nächstes Fib-Support ${fib_s['price']:,.0f}" +
                         (f", Resistance ${fib_r['price']:,.0f}." if fib_r.get("price") else "."))

        return {
            "direction": direction,
            "strength":  strength,
            "score":     norm,
            "max_score": 10,
            "factors":   factors,
            "summary":   " ".join(parts),
        }

    # ── Key Levels ─────────────────────────────────────────────────────────

    def _merge_key_levels(self, result) -> list:
        price  = result.get("daily", {}).get("price", 0)
        levels = []

        for sr in result.get("daily", {}).get("sr_levels", []):
            levels.append({"price": sr["price"], "type": sr["type"],
                           "source": "S/R Cluster", "strength": sr["strength"]})

        fib = result.get("fibonacci", {})
        for k, v in fib.get("levels", {}).items():
            if v.get("active"):
                levels.append({"price": v["price"],
                               "type": "Resistance" if v["above"] else "Support",
                               "source": f"Fib {v['level']}", "strength": 3})

        for lq in result.get("daily", {}).get("liquidity", {}).get("bsl", []):
            levels.append({"price": lq, "type": "BSL", "source": "Buy Side Liquidity", "strength": 2})
        for lq in result.get("daily", {}).get("liquidity", {}).get("ssl", []):
            levels.append({"price": lq, "type": "SSL", "source": "Sell Side Liquidity", "strength": 2})

        emas = result.get("daily", {}).get("emas", {})
        for name, val in emas.items():
            if val:
                levels.append({"price": val, "type": "EMA", "source": name.upper(), "strength": 4})

        unique = []
        for lv in sorted(levels, key=lambda x: -x.get("strength", 0)):
            if not any(abs(lv["price"] - u["price"]) / max(u["price"], 1) < 0.01 for u in unique):
                unique.append(lv)

        return sorted(unique, key=lambda x: x["price"])
