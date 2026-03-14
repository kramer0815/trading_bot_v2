"""
Liquidity Analyzer
==================
Synthetische Liquiditätskarte für BTC/USDT basierend auf:

1. Order Book Depth (Binance Spot + Futures, Top 500 Levels)
   → Zeigt wo echte Limit-Orders cluster liegen

2. Geschätzte Liquidation Levels
   → Berechnet aus Open Interest, Funding Rate und typischen Hebeln (3x,5x,10x,20x,50x,100x)
   → Long-Liquidationen unter dem Preis, Short-Liquidationen darüber

3. Cumulative Delta
   → Netto-Kaufdruck vs. Verkaufsdruck aus Trade-History
   → Zeigt wo aggressiv akkumuliert/distribuiert wurde

Gecacht für 5 Minuten (Order Book ist dynamisch).
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

CACHE_SECONDS = 300   # 5 Minuten
LEVERS        = [3, 5, 10, 20, 50, 100]
PRICE_RANGE   = 0.15  # ±15% vom aktuellen Preis für die Karte


class LiquidityAnalyzer:

    def __init__(self, cfg):
        self.cfg       = cfg
        self.cache_dir = Path(os.getenv("MACRO_CACHE_DIR", "/app/macro_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "liquidity.json"

        # Spot Exchange für Order Book
        ex_cls = getattr(ccxt, cfg.EXCHANGE)
        self.spot = ex_cls({
            "apiKey":          cfg.API_KEY,
            "secret":          cfg.API_SECRET,
            "enableRateLimit": True,
        })

        # Binance Futures für OI + Funding
        try:
            self.futures = ccxt.binanceusdm({
                "apiKey":          cfg.API_KEY,
                "secret":          cfg.API_SECRET,
                "enableRateLimit": True,
            })
        except Exception:
            self.futures = None

    def get_analysis(self, force=False) -> dict:
        if not force and self.cache_file.exists():
            age = time.time() - self.cache_file.stat().st_mtime
            if age < CACHE_SECONDS:
                try:
                    with open(self.cache_file) as f:
                        data = json.load(f)
                    data["from_cache"]     = True
                    data["cache_age_sec"]  = int(age)
                    return data
                except Exception:
                    pass

        result = self._run()
        try:
            with open(self.cache_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Liquidity Cache-Fehler: {e}")

        result["from_cache"]    = False
        result["cache_age_sec"] = 0
        return result

    def _run(self) -> dict:
        logger.info("Starte Liquiditäts-Analyse …")

        price   = self._get_price()
        ob      = self._fetch_orderbook(price)
        oi_data = self._fetch_oi_funding(price)
        liq     = self._calc_liquidation_levels(price, oi_data)
        delta   = self._calc_cumulative_delta(price)
        heatmap = self._build_heatmap(price, ob, liq)
        bias    = self._calc_liquidity_bias(price, ob, liq, oi_data)

        return {
            "generated_at": pd.Timestamp.now().isoformat(),
            "symbol":       self.cfg.SYMBOL,
            "price":        round(price, 0),
            "orderbook":    ob,
            "liquidations": liq,
            "oi":           oi_data,
            "cumulative_delta": delta,
            "heatmap":      heatmap,
            "bias":         bias,
        }

    # ── Preis ─────────────────────────────────────────────────────────────

    def _get_price(self) -> float:
        ticker = self.spot.fetch_ticker(self.cfg.SYMBOL)
        return float(ticker["last"])

    # ── Order Book ────────────────────────────────────────────────────────

    def _fetch_orderbook(self, price: float) -> dict:
        try:
            ob = self.spot.fetch_order_book(self.cfg.SYMBOL, limit=500)
        except Exception as e:
            logger.error(f"Order Book: {e}")
            return {"bids": [], "asks": [], "bid_wall": None, "ask_wall": None,
                    "bid_total": 0, "ask_total": 0, "imbalance": 0}

        lo = price * (1 - PRICE_RANGE)
        hi = price * (1 + PRICE_RANGE)

        # Bids/Asks in Range filtern und clustern
        bids = [(float(p), float(q)) for p, q in ob["bids"] if lo <= float(p) <= price]
        asks = [(float(p), float(q)) for p, q in ob["asks"] if price <= float(p) <= hi]

        bid_clusters = self._cluster_levels(bids, price, side="bid")
        ask_clusters = self._cluster_levels(asks, price, side="ask")

        bid_total = sum(q for _, q in bids)
        ask_total = sum(q for _, q in asks)
        imbalance = (bid_total - ask_total) / (bid_total + ask_total + 1e-9) * 100

        # Größte Wand
        bid_wall = max(bid_clusters, key=lambda x: x["volume"], default=None)
        ask_wall = max(ask_clusters, key=lambda x: x["volume"], default=None)

        return {
            "bids":        bid_clusters[:30],
            "asks":        ask_clusters[:30],
            "bid_wall":    bid_wall,
            "ask_wall":    ask_wall,
            "bid_total":   round(bid_total, 2),
            "ask_total":   round(ask_total, 2),
            "imbalance":   round(imbalance, 1),
        }

    def _cluster_levels(self, levels: list, price: float, side: str) -> list:
        """Fasst Orders innerhalb 0.15% zu Clustern zusammen."""
        if not levels:
            return []
        levels_sorted = sorted(levels, key=lambda x: x[0], reverse=(side == "bid"))
        clusters = []
        for p, q in levels_sorted:
            if clusters and abs(p - clusters[-1]["price"]) / clusters[-1]["price"] < 0.0015:
                clusters[-1]["volume"]    += q
                clusters[-1]["usd_value"] += q * p
                clusters[-1]["count"]     += 1
            else:
                clusters.append({
                    "price":     round(p, 0),
                    "volume":    round(q, 4),
                    "usd_value": round(q * p, 0),
                    "count":     1,
                    "dist_pct":  round(abs(p - price) / price * 100, 2),
                })
        # Stärke normalisieren
        if clusters:
            max_vol = max(c["volume"] for c in clusters)
            for c in clusters:
                c["strength"] = round(c["volume"] / max_vol * 100, 1) if max_vol > 0 else 0
        return clusters

    # ── OI + Funding ──────────────────────────────────────────────────────

    def _fetch_oi_funding(self, price: float) -> dict:
        result = {
            "open_interest_usd": None,
            "funding_rate":      None,
            "funding_bias":      "Neutral",
            "oi_change_24h":     None,
        }
        if not self.futures:
            return result
        try:
            sym = self.cfg.SYMBOL.replace("/", "")  # BTC/USDT → BTCUSDT
            oi  = self.futures.fetch_open_interest(self.cfg.SYMBOL)
            result["open_interest_usd"] = round(float(oi.get("openInterestValue", 0) or 0), 0)

            fr = self.futures.fetch_funding_rate(self.cfg.SYMBOL)
            rate = float(fr.get("fundingRate", 0) or 0)
            result["funding_rate"] = round(rate * 100, 4)
            result["funding_bias"] = (
                "Long-Bias (Longs zahlen)" if rate > 0.0001
                else "Short-Bias (Shorts zahlen)" if rate < -0.0001
                else "Neutral"
            )
        except Exception as e:
            logger.warning(f"OI/Funding: {e}")
        return result

    # ── Liquidation Levels ────────────────────────────────────────────────

    def _calc_liquidation_levels(self, price: float, oi_data: dict) -> dict:
        """
        Schätzt Liquidations-Cluster basierend auf:
        - Hebeln: 3x, 5x, 10x, 20x, 50x, 100x
        - Annahme: Long-Positionen wurden bei aktuellem Preis eröffnet
          → Liquidation bei Entry × (1 - 1/Hebel × 0.9)  [mit 10% Maintenance Margin Buffer]
        - Short-Positionen analog nach oben
        - Gewichtung: höhere Hebel haben mehr Volumen (realistisch für Retail)
        """
        oi_usd = oi_data.get("open_interest_usd") or (price * 50000)  # Fallback ~50k BTC
        lever_weights = {3: 0.08, 5: 0.12, 10: 0.25, 20: 0.30, 50: 0.15, 100: 0.10}

        long_liqs  = []
        short_liqs = []

        for lever, weight in lever_weights.items():
            vol = oi_usd * weight * 0.5  # 50% Long, 50% Short (vereinfacht)

            # Long-Liquidation: unter dem Einstiegspreis
            # Bei Hebel L: Liquidation ≈ entry × (1 - 0.9/L)
            liq_long  = price * (1 - 0.9 / lever)
            liq_short = price * (1 + 0.9 / lever)

            if liq_long > price * (1 - PRICE_RANGE):
                long_liqs.append({
                    "price":   round(liq_long, 0),
                    "lever":   lever,
                    "volume":  round(vol, 0),
                    "dist_pct": round((price - liq_long) / price * 100, 2),
                    "type":    "LONG LIQ",
                    "color":   "#ef5350",
                })

            if liq_short < price * (1 + PRICE_RANGE):
                short_liqs.append({
                    "price":   round(liq_short, 0),
                    "lever":   lever,
                    "volume":  round(vol, 0),
                    "dist_pct": round((liq_short - price) / price * 100, 2),
                    "type":    "SHORT LIQ",
                    "color":   "#26a69a",
                })

        return {
            "long_liquidations":  sorted(long_liqs,  key=lambda x: -x["price"]),
            "short_liquidations": sorted(short_liqs, key=lambda x:  x["price"]),
            "total_long_liq_usd":  sum(x["volume"] for x in long_liqs),
            "total_short_liq_usd": sum(x["volume"] for x in short_liqs),
        }

    # ── Cumulative Delta ──────────────────────────────────────────────────

    def _calc_cumulative_delta(self, price: float) -> dict:
        """
        Berechnet Cumulative Delta aus den letzten 1000 Trades.
        Buy-initiierte Trades (taker = buyer) zählen positiv,
        Sell-initiierte Trades negativ.
        """
        try:
            trades = self.spot.fetch_trades(self.cfg.SYMBOL, limit=1000)
        except Exception as e:
            logger.warning(f"Trades: {e}")
            return {"delta": 0, "buy_volume": 0, "sell_volume": 0, "bias": "Neutral", "levels": []}

        buy_vol  = sum(float(t["amount"]) for t in trades if t.get("side") == "buy")
        sell_vol = sum(float(t["amount"]) for t in trades if t.get("side") == "sell")
        delta    = buy_vol - sell_vol
        total    = buy_vol + sell_vol

        # Delta pro Preis-Bucket (für Heatmap)
        lo     = price * (1 - PRICE_RANGE)
        hi     = price * (1 + PRICE_RANGE)
        n_buckets = 40
        bucket_size = (hi - lo) / n_buckets
        buckets = [0.0] * n_buckets

        for t in trades:
            p = float(t.get("price", 0))
            a = float(t.get("amount", 0))
            s = 1 if t.get("side") == "buy" else -1
            idx = int((p - lo) / bucket_size)
            if 0 <= idx < n_buckets:
                buckets[idx] += a * s

        levels = []
        for i, d in enumerate(buckets):
            p_center = lo + (i + 0.5) * bucket_size
            levels.append({
                "price":  round(p_center, 0),
                "delta":  round(d, 4),
                "is_buy": d > 0,
            })

        # Normalisieren
        max_abs = max((abs(d) for d in buckets), default=1)
        for lv in levels:
            lv["strength"] = round(abs(lv["delta"]) / max_abs * 100, 1)

        return {
            "delta":      round(delta, 4),
            "buy_volume": round(buy_vol,  4),
            "sell_volume":round(sell_vol, 4),
            "buy_pct":    round(buy_vol / total * 100, 1) if total > 0 else 50,
            "bias":       ("Kauf-Druck" if delta > total * 0.05
                           else "Verkaufs-Druck" if delta < -total * 0.05
                           else "Ausgewogen"),
            "levels":     levels,
        }

    # ── Heatmap Aufbau ────────────────────────────────────────────────────

    def _build_heatmap(self, price: float, ob: dict, liq: dict) -> list:
        """
        Kombiniert Order Book + Liquidation Levels in eine
        einheitliche Heatmap-Struktur (pro Preis-Bucket).
        """
        lo = price * (1 - PRICE_RANGE)
        hi = price * (1 + PRICE_RANGE)
        n  = 80
        bucket_size = (hi - lo) / n

        buckets = {}

        # Order Book Bids (Kauf-Liquidität)
        for bid in ob.get("bids", []):
            p = bid["price"]
            idx = int((p - lo) / bucket_size)
            if 0 <= idx < n:
                buckets.setdefault(idx, {"bid": 0, "ask": 0, "liq_long": 0, "liq_short": 0})
                buckets[idx]["bid"] += bid["volume"]

        # Order Book Asks (Verkaufs-Liquidität)
        for ask in ob.get("asks", []):
            p = ask["price"]
            idx = int((p - lo) / bucket_size)
            if 0 <= idx < n:
                buckets.setdefault(idx, {"bid": 0, "ask": 0, "liq_long": 0, "liq_short": 0})
                buckets[idx]["ask"] += ask["volume"]

        # Liquidations
        for ll in liq.get("long_liquidations", []):
            p   = ll["price"]
            idx = int((p - lo) / bucket_size)
            if 0 <= idx < n:
                buckets.setdefault(idx, {"bid": 0, "ask": 0, "liq_long": 0, "liq_short": 0})
                buckets[idx]["liq_long"] += ll["volume"] / price  # in BTC

        for sl in liq.get("short_liquidations", []):
            p   = sl["price"]
            idx = int((p - lo) / bucket_size)
            if 0 <= idx < n:
                buckets.setdefault(idx, {"bid": 0, "ask": 0, "liq_long": 0, "liq_short": 0})
                buckets[idx]["liq_short"] += sl["volume"] / price

        # Normalisieren und Liste aufbauen
        max_bid  = max((v["bid"]       for v in buckets.values()), default=1) or 1
        max_ask  = max((v["ask"]       for v in buckets.values()), default=1) or 1
        max_liq  = max((v["liq_long"] + v["liq_short"] for v in buckets.values()), default=1) or 1

        result = []
        for i in range(n):
            p_center = lo + (i + 0.5) * bucket_size
            b = buckets.get(i, {"bid": 0, "ask": 0, "liq_long": 0, "liq_short": 0})
            result.append({
                "price":        round(p_center, 0),
                "bid":          round(b["bid"],       4),
                "ask":          round(b["ask"],       4),
                "liq_long":     round(b["liq_long"],  4),
                "liq_short":    round(b["liq_short"], 4),
                "bid_str":      round(b["bid"]      / max_bid * 100, 1),
                "ask_str":      round(b["ask"]      / max_ask * 100, 1),
                "liq_str":      round((b["liq_long"] + b["liq_short"]) / max_liq * 100, 1),
                "above_price":  p_center > price,
            })

        return result

    # ── Bias ──────────────────────────────────────────────────────────────

    def _calc_liquidity_bias(self, price: float, ob: dict, liq: dict, oi: dict) -> dict:
        """
        Wo liegt mehr Liquidität — oben oder unten?
        Market Maker tendieren dazu, den Kurs in Richtung der
        größten Liquiditäts-Konzentration zu treiben.
        """
        bid_total = ob.get("bid_total", 0)
        ask_total = ob.get("ask_total", 0)
        ob_imbalance = ob.get("imbalance", 0)

        long_liq_usd  = liq.get("total_long_liq_usd",  0)
        short_liq_usd = liq.get("total_short_liq_usd", 0)

        funding = oi.get("funding_rate", 0) or 0

        # Wo ist mehr Liquidität zum "holen"?
        # Mehr Long-Liqs unten → MM könnte nach unten drücken
        # Mehr Short-Liqs oben → MM könnte nach oben drücken
        liq_score = (short_liq_usd - long_liq_usd) / (short_liq_usd + long_liq_usd + 1) * 10

        # Funding: positiv = Longs zahlen = crowded long = Gefahr nach unten
        funding_score = -funding * 1000  # negativ wenn bullish funding

        # OB Imbalance: positiv = mehr Bids = bullisch
        ob_score = ob_imbalance / 10

        total = liq_score + funding_score + ob_score

        if   total > 3:   direction, color = "Upside Liquidity",   "#26a69a"
        elif total > 1:   direction, color = "Leicht bullisch",     "#4fc3f7"
        elif total < -3:  direction, color = "Downside Liquidity",  "#ef5350"
        elif total < -1:  direction, color = "Leicht bärisch",      "#ff9800"
        else:             direction, color = "Ausgeglichen",        "#90a4ae"

        bid_wall = ob.get("bid_wall")
        ask_wall = ob.get("ask_wall")

        summary_parts = [f"OB Imbalance: {ob_imbalance:+.1f}%"]
        if funding != 0:
            summary_parts.append(f"Funding: {funding:+.4f}%")
        if bid_wall:
            summary_parts.append(f"Größte Bid-Wand: ${bid_wall['price']:,.0f} ({bid_wall['volume']:.1f} BTC)")
        if ask_wall:
            summary_parts.append(f"Größte Ask-Wand: ${ask_wall['price']:,.0f} ({ask_wall['volume']:.1f} BTC)")

        return {
            "direction":   direction,
            "color":       color,
            "score":       round(total, 2),
            "liq_score":   round(liq_score,    2),
            "funding_score": round(funding_score, 2),
            "ob_score":    round(ob_score,     2),
            "summary":     " · ".join(summary_parts),
            "mm_target":   (
                f"MM-Ziel: Short-Liquidationen über ${min((x['price'] for x in liq.get('short_liquidations', [{'price':0}])), default=0):,.0f}"
                if total > 1
                else f"MM-Ziel: Long-Liquidationen unter ${max((x['price'] for x in liq.get('long_liquidations', [{'price':0}])), default=0):,.0f}"
                if total < -1
                else "Kein klares MM-Ziel erkennbar"
            ),
        }
