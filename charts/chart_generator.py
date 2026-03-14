import logging
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from signals.base import Signal
from utils.risk_manager import Trade
from utils.quality_filter import QualityResult

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

C_BULL   = "#26a69a"
C_BEAR   = "#ef5350"
C_BG     = "#0d1117"
C_TEXT   = "#e6edf3"
C_GRID   = "#21262d"
C_ENTRY  = "#f0c040"
C_SL     = "#ff4444"
C_TP     = ["#44cc44", "#22aa22", "#118811"]
C_EMA_S  = "#ff9800"
C_EMA_L  = "#2196f3"
C_EMA21  = "#ab47bc"   # Lila — EMA21 für GUSS TPs
C_EMA50  = "#26c6da"   # Cyan — EMA50 als Entry-Linie bei GUSS
C_TREND  = "#607d8b"
C_SWING  = "#ffd54f"   # Gelb — Swing High/Low Markierung
C_TL_DOWN = "#ef5350"   # Rot — Abwärtstrendlinie
C_TL_UP   = "#26a69a"   # Grün — Aufwärtstrendlinie
C_TL_BRK  = "#ffffff"   # Weiß — Breakout-Markierung
C_GUSS   = "#ff7043"   # Orange — Guss-Kerzen-Highlight


class ChartGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        Path(cfg.CHART_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def create_trade_chart(self, df: pd.DataFrame, signal: Signal, trade: Trade, timeframe: str, quality: QualityResult = None) -> str:
        is_guss      = signal.type in ("GUSS_LONG", "GUSS_SHORT")
        is_trendline = signal.type in ("TL_BREAKOUT_LONG", "TL_BREAKOUT_SHORT")

        view  = df.iloc[-self.cfg.CHART_CANDLES:].copy()
        x     = np.arange(len(view))
        dates = view.index

        fig = plt.figure(figsize=(18, 12), facecolor=C_BG)
        fig.subplots_adjust(left=0.05, right=0.93, top=0.92, bottom=0.06, hspace=0.05)
        gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[6, 1.5, 2, 2])

        ax_m = fig.add_subplot(gs[0])
        ax_v = fig.add_subplot(gs[1], sharex=ax_m)
        ax_r = fig.add_subplot(gs[2], sharex=ax_m)
        ax_c = fig.add_subplot(gs[3], sharex=ax_m)

        c = view["close"].values
        o = view["open"].values
        h = view["high"].values
        l = view["low"].values
        v = view["volume"].values
        n = len(view)

        ex = signal.extra

        # ── Guss-Kerzen-Bereich bestimmen ─────────────────────────
        guss_start_x = None
        if is_guss and "swing_high_idx" in ex:
            # swing_high_idx ist relativ zum gesamten df — in view-Index umrechnen
            df_len   = len(df)
            view_off = df_len - len(view)   # wie viele Kerzen sind vor view
            sh_in_view = ex.get("swing_high_idx", -1) - view_off
            if 0 <= sh_in_view < n:
                guss_start_x = sh_in_view + 1   # erste Kerze nach dem Hoch

        if is_guss and "swing_low_idx" in ex:
            df_len   = len(df)
            view_off = df_len - len(view)
            sl_in_view = ex.get("swing_low_idx", -1) - view_off
            if 0 <= sl_in_view < n:
                guss_start_x = sl_in_view + 1

        # ── Candlesticks ──────────────────────────────────────────
        for i in range(n):
            # Guss-Kerzen farblich hervorheben
            if guss_start_x is not None and i >= guss_start_x:
                col = C_GUSS if (signal.type == "GUSS_LONG" and c[i] < o[i]) \
                             or (signal.type == "GUSS_SHORT" and c[i] > o[i]) \
                             else (C_BULL if c[i] >= o[i] else C_BEAR)
            else:
                col = C_BULL if c[i] >= o[i] else C_BEAR

            ax_m.plot([x[i], x[i]], [l[i], h[i]], color=col, lw=0.8, zorder=2)
            bl = min(o[i], c[i])
            bh = max(o[i], c[i])
            ax_m.add_patch(plt.Rectangle(
                (x[i] - 0.4, bl), 0.8, max(bh - bl, h[i] * 0.0002),
                color=col, zorder=3))

        # ── Guss-Bereich hinterlegen ──────────────────────────────
        if guss_start_x is not None:
            ax_m.axvspan(guss_start_x - 0.5, n - 0.5,
                         alpha=0.06, color=C_GUSS, zorder=1)

        # ── EMAs ─────────────────────────────────────────────────
        if "ema_short" in view.columns:
            ax_m.plot(x, view["ema_short"].values, color=C_EMA_S, lw=1.0,
                      label=f"EMA{self.cfg.EMA_SHORT}", zorder=4, alpha=0.8)
        if "ema_long" in view.columns and not is_guss:
            ax_m.plot(x, view["ema_long"].values, color=C_EMA_L, lw=1.0,
                      label=f"EMA{self.cfg.EMA_LONG}", zorder=4, alpha=0.8)
        if "ema_trend" in view.columns:
            ax_m.plot(x, view["ema_trend"].values, color=C_TREND, lw=0.9,
                      ls="--", alpha=0.6, label="EMA200", zorder=4)

        # Bei GUSS: EMA21 und EMA50 prominent zeigen
        if is_guss:
            if "ema50" in view.columns:
                ax_m.plot(x, view["ema50"].values, color=C_EMA50, lw=2.0,
                          label="EMA50 (Entry)", zorder=5)
            if "ema21" in view.columns:
                ax_m.plot(x, view["ema21"].values, color=C_EMA21, lw=1.6,
                          ls="--", label="EMA21 (TP1)", zorder=5)


        # ── Trendlinien zeichnen ──────────────────────────────────
        if is_trendline and "tl_x1" in ex:
            df_len   = len(df)
            view_off = df_len - len(view)
            tl_x1_view = ex["tl_x1"] - view_off
            tl_x2_view = ex["tl_x2"] - view_off
            tl_color = C_TL_DOWN if ex.get("trendline_direction") == "down" else C_TL_UP

            if tl_x1_view < n:
                x_start = max(0, tl_x1_view)
                x_end   = min(n + 8, tl_x2_view + 6)  # etwas über aktuellen Preis hinaus
                xs = np.array([x_start, x_end])
                # Linie verlängern mit Steigung
                slope = ex["trendline_slope"]
                y_start = ex["tl_y1"] + slope * (x_start - ex["tl_x1"])
                y_end   = ex["tl_y1"] + slope * (x_end   - ex["tl_x1"])
                ax_m.plot(xs, [y_start, y_end],
                          color=tl_color, lw=2.0, ls="--", alpha=0.85, zorder=5,
                          label=f"Trendlinie ({ex.get('trendline_touches','?')}x)")
                # Breakout-Punkt markieren
                ax_m.axvline(x=n-1, color=C_TL_BRK, lw=0.8, ls=":", alpha=0.4, zorder=4)
                ax_m.annotate(
                    f"BREAKOUT\n{ex.get('trendline_touches','?')} Touches",
                    xy=(n-1, ex["trendline_y"]),
                    xytext=(n-1 - 8, ex["trendline_y"] * (0.994 if signal.direction=="long" else 1.006)),
                    color=C_TL_BRK, fontsize=7, fontfamily="monospace", ha="center",
                    arrowprops=dict(arrowstyle="->", color=C_TL_BRK, lw=1.0),
                    zorder=6,
                )
                # Vol-Ratio und RSI im Chart
                info_txt = (f"Vol: {ex.get('vol_ratio','?')}x MA  |  "
                            f"RSI: {ex.get('rsi_at_breakout','?')}")
                ax_m.text(0.01, 0.95, info_txt,
                          transform=ax_m.transAxes,
                          fontsize=7.5, fontfamily="monospace",
                          color=tl_color, alpha=0.9,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                                    edgecolor=tl_color, alpha=0.8))

        # ── Swing High / Low markieren ────────────────────────────
        if is_guss:
            if "swing_high" in ex:
                sh_val = ex["swing_high"]
                sh_x   = ex.get("swing_high_idx", -1) - (len(df) - n)
                if 0 <= sh_x < n:
                    ax_m.annotate(
                        f"Swing High\n{sh_val:,.0f}",
                        xy=(sh_x, h[sh_x] if sh_x < n else sh_val),
                        xytext=(sh_x, h[sh_x] * 1.003 if sh_x < n else sh_val * 1.003),
                        color=C_SWING, fontsize=7.5, fontfamily="monospace",
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color=C_SWING, lw=1.2),
                        zorder=6,
                    )
                    # Horizontale gestrichelte Linie beim letzten Hoch
                    ax_m.axhline(sh_val, color=C_SWING, lw=0.8, ls=":", alpha=0.7)

            if "swing_low" in ex:
                sl_val = ex["swing_low"]
                sl_x   = ex.get("swing_low_idx", -1) - (len(df) - n)
                if 0 <= sl_x < n:
                    ax_m.annotate(
                        f"Swing Low\n{sl_val:,.0f}",
                        xy=(sl_x, l[sl_x] if sl_x < n else sl_val),
                        xytext=(sl_x, l[sl_x] * 0.997 if sl_x < n else sl_val * 0.997),
                        color=C_SWING, fontsize=7.5, fontfamily="monospace",
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color=C_SWING, lw=1.2),
                        zorder=6,
                    )
                    ax_m.axhline(sl_val, color=C_SWING, lw=0.8, ls=":", alpha=0.7)

        # ── Entry / SL / TPs ─────────────────────────────────────
        xl = n - 1

        def hline(y, col, label, lw=1.5, ls="-"):
            ax_m.axhline(y=y, color=col, lw=lw, ls=ls, zorder=5, alpha=0.92)
            ax_m.text(xl + 0.7, y, f"  {label}\n  {y:,.1f}",
                      color=col, fontsize=7.5, va="center", fontweight="bold",
                      fontfamily="monospace")

        hline(trade.entry,     C_ENTRY, "ENTRY @ EMA50" if is_guss else "ENTRY", lw=2.2)
        hline(trade.stop_loss, C_SL,    "SL",   lw=1.5, ls="--")

        tp_labels = ["TP1 EMA21", "TP2 50%", "TP3 Hoch"] if signal.direction == "long" \
               else ["TP1 EMA21", "TP2 50%", "TP3 Tief"]
        for i, (tp, rr) in enumerate(zip(trade.take_profits, trade.rr_ratios)):
            label = f"{tp_labels[i]} ({rr:.1f}R)" if is_guss else f"TP{i+1} ({rr:.1f}R)"
            hline(tp, C_TP[i], label, lw=1.2)

        # SL-Risikozone
        ax_m.axhspan(min(trade.entry, trade.stop_loss),
                     max(trade.entry, trade.stop_loss), alpha=0.07, color=C_SL)
        # TP1-Zone
        if trade.take_profits:
            ax_m.axhspan(min(trade.entry, trade.take_profits[0]),
                         max(trade.entry, trade.take_profits[0]), alpha=0.05, color=C_TP[0])

        # ── Volume ────────────────────────────────────────────────
        for i in range(n):
            if guss_start_x is not None and i >= guss_start_x:
                col = C_GUSS
            else:
                col = C_BULL if c[i] >= o[i] else C_BEAR
            ax_v.bar(x[i], v[i], color=col, alpha=0.75, width=0.8)
        if "vol_ma" in view.columns:
            ax_v.plot(x, view["vol_ma"].values, color="#ffffff", lw=0.8, alpha=0.5)

        # ── RSI ───────────────────────────────────────────────────
        if "rsi" in view.columns:
            rv = view["rsi"].values
            ax_r.plot(x, rv, color="#ba68c8", lw=1.2)
            ax_r.axhline(70, color=C_BEAR, lw=0.7, ls="--", alpha=0.6)
            ax_r.axhline(30, color=C_BULL, lw=0.7, ls="--", alpha=0.6)
            ax_r.fill_between(x, rv, 70, where=(rv >= 70), alpha=0.15, color=C_BEAR)
            ax_r.fill_between(x, rv, 30, where=(rv <= 30), alpha=0.15, color=C_BULL)
            ax_r.set_ylim(0, 100)
        ax_r.set_ylabel("RSI", color=C_TEXT, fontsize=8)

        # ── MACD ──────────────────────────────────────────────────
        if "macd" in view.columns:
            hist = view["macd_hist"].values
            ax_c.bar(x, hist, color=[C_BULL if v >= 0 else C_BEAR for v in hist],
                     alpha=0.7, width=0.8)
            ax_c.plot(x, view["macd"].values,        color="#4fc3f7", lw=1.0)
            ax_c.plot(x, view["macd_signal"].values, color="#ff8a65", lw=1.0)
            ax_c.axhline(0, color=C_GRID, lw=0.5)
        ax_c.set_ylabel("MACD", color=C_TEXT, fontsize=8)

        # ── Achsen-Stil ───────────────────────────────────────────
        step  = max(1, n // 10)
        ticks = x[::step]
        tlabs = [dates[i].strftime("%m/%d %H:%M") for i in ticks]

        for ax in [ax_m, ax_v, ax_r, ax_c]:
            ax.set_facecolor(C_BG)
            ax.tick_params(colors=C_TEXT, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(C_GRID)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("left")
            ax.grid(True, color=C_GRID, lw=0.5, alpha=0.5)
            plt.setp(ax.get_xticklabels(), visible=False)

        ax_c.set_xticks(ticks)
        ax_c.set_xticklabels(tlabs, rotation=30, ha="right", fontsize=6.5)
        plt.setp(ax_c.get_xticklabels(), visible=True)

        ax_m.set_xlim(-1, n + 11)
        ax_m.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))

        # ── Legende ───────────────────────────────────────────────
        handles = [
            Line2D([0], [0], color=C_ENTRY, lw=2,
                   label=f"Entry  {trade.entry:,.0f}"),
            Line2D([0], [0], color=C_SL, lw=2, ls="--",
                   label=f"SL     {trade.stop_loss:,.0f}"),
        ]
        for i, (tp, rr) in enumerate(zip(trade.take_profits, trade.rr_ratios)):
            lbl = f"{tp_labels[i]}  {tp:,.0f}  ({rr:.1f}R)" if is_guss \
                  else f"TP{i+1}   {tp:,.0f}  ({rr:.1f}R)"
            handles.append(Line2D([0], [0], color=C_TP[i], lw=1.5, label=lbl))

        if is_guss:
            handles += [
                Line2D([0], [0], color=C_EMA50, lw=2.0, label="EMA50 (Entry)"),
                Line2D([0], [0], color=C_EMA21, lw=1.6, ls="--", label="EMA21 (TP1)"),
                Line2D([0], [0], color=C_GUSS,  lw=4, alpha=0.5,
                       label=f"Guss ({ex.get('guss_len','?')} Kerzen)"),
            ]
        else:
            if "ema_short" in view.columns:
                handles += [
                    Line2D([0], [0], color=C_EMA_S, lw=1.2,
                           label=f"EMA{self.cfg.EMA_SHORT}"),
                    Line2D([0], [0], color=C_EMA_L, lw=1.2,
                           label=f"EMA{self.cfg.EMA_LONG}"),
                ]

        ax_m.legend(handles=handles, loc="upper left",
                    facecolor="#161b22", edgecolor=C_GRID,
                    labelcolor=C_TEXT, fontsize=8, framealpha=0.9)

        # ── Titel ─────────────────────────────────────────────────
        if is_guss:
            direction_str = "⬆ LONG — Guss zum EMA50" if signal.direction == "long" \
                            else "⬇ SHORT — Reverse Guss zum EMA50"
            guss_len = ex.get("guss_len", "?")
            swing_val = ex.get("swing_high", ex.get("swing_low", 0))
            subtitle = (f"{guss_len} {'bärische' if signal.direction == 'long' else 'bullische'} "
                        f"Kerzen | Swing: {swing_val:,.0f} | "
                        f"EMA50: {ex.get('ema50', 0):,.0f} | "
                        f"EMA21: {ex.get('ema21', 0):,.0f}")
        else:
            direction_str = "⬆ LONG" if signal.direction == "long" else "⬇ SHORT"
            subtitle = f"{signal.description} | Stärke: {signal.strength:.0%}"

        # Quality Score Badge
        if quality is not None:
            grade_colors = {"S": "#f0c040", "A": "#26a69a", "B": "#4fc3f7",
                            "C": "#ff9800", "F": "#ef5350"}
            gc = grade_colors.get(quality.grade, "#7d8590")
            badge_text = (f"QF Grade: {quality.grade}  "
                          f"{quality.total_score:.0f}/{quality.max_score:.0f} pts  "
                          f"({quality.pct:.0f}%)")
            fig.text(0.99, 0.005, badge_text,
                     ha="right", va="bottom", fontsize=8,
                     fontfamily="monospace", color=gc,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                               edgecolor=gc, alpha=0.85))

            # RR-Labels in Legende ergänzen
            rr_summary = "  ".join(f"RR{i+1}={rr:.1f}R" for i, rr in enumerate(trade.rr_ratios))
            ax_m.text(0.01, 0.01, rr_summary,
                      transform=ax_m.transAxes,
                      fontsize=8, fontfamily="monospace",
                      color=C_ENTRY, alpha=0.85,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                                edgecolor=C_GRID, alpha=0.8))

        fig.suptitle(
            f"BTC/USDT — {timeframe}  |  {direction_str}  |  {signal.type}\n"
            f"{subtitle}  |  Risiko: ${trade.risk_amount:.0f}  |  "
            f"Größe: {trade.position_size:.4f} BTC",
            color=C_TEXT, fontsize=10, fontweight="bold", y=0.97,
        )

        # ── Speichern PNG ─────────────────────────────────────────
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{timeframe}_{signal.direction.upper()}_{signal.type}.png"
        out      = Path(self.cfg.CHART_OUTPUT_DIR) / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)

        # ── Score JSON Sidecar ────────────────────────────────────
        if quality is not None:
            import json
            score_data = {
                "grade":       quality.grade,
                "total_score": quality.total_score,
                "max_score":   quality.max_score,
                "pct":         round(quality.pct, 1),
                "scores": {
                    dim: {"score": s, "max": m, "note": n}
                    for dim, (s, m, n) in quality.scores.items()
                },
                "reasons_pass": quality.reasons_pass,
                "reasons_fail": quality.reasons_fail,
                "trade": {
                    "entry":        trade.entry,
                    "stop_loss":    trade.stop_loss,
                    "take_profits": trade.take_profits,
                    "rr_ratios":    trade.rr_ratios,
                    "risk_amount":  trade.risk_amount,
                    "position_size": trade.position_size,
                },
            }
            json_out = out.with_suffix(".json")
            with open(json_out, "w") as jf:
                json.dump(score_data, jf, indent=2)

        logger.info(f"Chart gespeichert: {out}")
        return str(out)
