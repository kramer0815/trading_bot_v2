import logging
import time
from datetime import datetime
from pathlib import Path

from config import Config
from data.fetcher import DataFetcher
from signals.divergence import DivergenceDetector
from signals.lsob import LSOBDetector
from signals.guss import GUSSDetector
from signals.ema_cross import EMACrossDetector
from signals.trendline import TrendlineDetector
from signals.signal_aggregator import SignalAggregator
from charts.chart_generator import ChartGenerator
from utils.risk_manager import RiskManager
from utils.quality_filter import TradeQualityFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SEPARATOR = "─" * 60


def run_analysis(cfg, fetcher, aggregator, chart_gen, risk_mgr, quality_filter):
    logger.info("=" * 60)
    logger.info(f"Analyse — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Symbol: {cfg.SYMBOL} | TFs: {cfg.TIMEFRAMES}")
    logger.info(f"Quality Filter: Min Score {cfg.QF_MIN_SCORE}/100 ({cfg.QF_MIN_SCORE}%) | Min RR {cfg.QF_MIN_RR}")
    logger.info("=" * 60)

    all_signals   = []
    charts_made   = 0
    charts_blocked = 0

    for tf in cfg.TIMEFRAMES:
        logger.info(f"\n[{tf}] Lade Daten …")
        df = fetcher.get_ohlcv(tf, limit=cfg.CANDLE_LIMIT)

        if df is None or df.empty:
            logger.warning(f"[{tf}] Keine Daten — überspringe.")
            continue

        logger.info(f"[{tf}] {len(df)} Kerzen | Letzte: {df.index[-1]}")

        detectors = [
            DivergenceDetector(cfg),
            LSOBDetector(cfg),
            GUSSDetector(cfg),
            EMACrossDetector(cfg),
            TrendlineDetector(cfg),
        ]

        tf_signals = []
        for det in detectors:
            tf_signals.extend(det.detect(df, tf))

        if not tf_signals:
            logger.info(f"[{tf}] Keine Rohsignale.")
            continue

        logger.info(f"[{tf}] {len(tf_signals)} Rohsignal(e) — starte Qualitätsprüfung:")
        logger.info(SEPARATOR)

        for sig in tf_signals:
            all_signals.append(sig)

            # Trade berechnen
            trade = risk_mgr.calculate_trade(sig, df)
            if trade is None:
                logger.warning(f"  [{sig.type}] Trade-Berechnung fehlgeschlagen — überspringe.")
                continue

            # ── Qualitätsprüfung ──────────────────────────────────
            qr = quality_filter.evaluate(sig, trade, df)

            logger.info(f"\n  Signal: {sig}")
            logger.info(f"  Trade:  Entry {trade.entry:,.0f} | SL {trade.stop_loss:,.0f} | "
                        f"TPs: {' / '.join(f'{tp:,.0f}' for tp in trade.take_profits)}")
            logger.info(f"  RR:     {' / '.join(f'{rr:.1f}R' for rr in trade.rr_ratios)}")
            logger.info(f"\n{qr}")

            if qr.passed:
                charts_made += 1
                path = chart_gen.create_trade_chart(df, sig, trade, tf, qr)
                logger.info(f"  → Chart [{qr.grade}]: {path}")
            else:
                charts_blocked += 1
                logger.info(f"  → Chart GEBLOCKT (Score {qr.total_score:.0f}/{qr.max_score:.0f})")

            logger.info(SEPARATOR)

    # ── Zusammenfassung ───────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Durchlauf abgeschlossen:")
    logger.info(f"  Rohsignale:     {len(all_signals)}")
    logger.info(f"  Charts erzeugt: {charts_made}  ✅")
    logger.info(f"  Geblockt:       {charts_blocked}  ❌")

    best = aggregator.get_best_signal(all_signals)
    if best:
        logger.info(f"  Bestes Signal:  {best}")
    logger.info("=" * 60)

    return all_signals


def main():
    cfg = Config()
    Path(cfg.CHART_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    fetcher        = DataFetcher(cfg)
    aggregator     = SignalAggregator(cfg)
    chart_gen      = ChartGenerator(cfg)
    risk_mgr       = RiskManager(cfg)
    quality_filter = TradeQualityFilter(cfg)

    if cfg.RUN_ONCE:
        run_analysis(cfg, fetcher, aggregator, chart_gen, risk_mgr, quality_filter)
    else:
        logger.info(f"Loop-Modus — Analyse alle {cfg.LOOP_INTERVAL_MINUTES} Min.")
        while True:
            try:
                run_analysis(cfg, fetcher, aggregator, chart_gen, risk_mgr, quality_filter)
            except Exception as e:
                logger.error(f"Fehler: {e}", exc_info=True)
            logger.info(f"Warte {cfg.LOOP_INTERVAL_MINUTES} Minuten …\n")
            time.sleep(cfg.LOOP_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main()
