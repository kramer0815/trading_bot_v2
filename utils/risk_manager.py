import logging
import pandas as pd
from dataclasses import dataclass, field
from signals.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    signal: Signal
    entry: float
    stop_loss: float
    take_profits: list
    risk_amount: float
    position_size: float
    rr_ratios: list

    def __str__(self):
        tps = " | ".join(f"TP{i+1}: {tp:,.2f}" for i, tp in enumerate(self.take_profits))
        return (
            f"Entry: {self.entry:,.2f} | SL: {self.stop_loss:,.2f} | {tps}\n"
            f"  Risiko: ${self.risk_amount:.2f} | Größe: {self.position_size:.4f} BTC"
        )


class RiskManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate_trade(self, sig: Signal, df: pd.DataFrame):
        try:
            # ── GUSS liefert Entry/SL/TPs bereits fertig berechnet ──────────
            if sig.type in ("GUSS_LONG", "GUSS_SHORT") and "entry" in sig.extra:
                entry        = sig.extra["entry"]
                stop_loss    = sig.extra["stop_loss"]
                take_profits = sig.extra["take_profits"]
                rr_ratios    = sig.extra["rr_ratios"]

                risk = abs(entry - stop_loss)
                if risk <= 0:
                    return None

                risk_amount   = self.cfg.ACCOUNT_BALANCE * (self.cfg.RISK_PERCENT / 100)
                position_size = risk_amount / risk

                return Trade(
                    signal=sig,
                    entry=entry,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    risk_amount=risk_amount,
                    position_size=position_size,
                    rr_ratios=rr_ratios,
                )

            # ── Alle anderen Signale: ATR-basierte Berechnung ───────────────
            entry = sig.price
            atr   = float(df["atr"].iloc[-1])

            if sig.direction == "long":
                stop_loss    = entry - atr * self.cfg.ATR_SL_MULT
                risk         = entry - stop_loss
                take_profits = [entry + risk * m for m in self.cfg.TP_LEVELS]
            else:
                stop_loss    = entry + atr * self.cfg.ATR_SL_MULT
                risk         = stop_loss - entry
                take_profits = [entry - risk * m for m in self.cfg.TP_LEVELS]

            if risk <= 0:
                return None

            risk_amount   = self.cfg.ACCOUNT_BALANCE * (self.cfg.RISK_PERCENT / 100)
            position_size = risk_amount / risk
            rr_ratios     = [abs(tp - entry) / risk for tp in take_profits]

            return Trade(
                signal=sig,
                entry=entry,
                stop_loss=stop_loss,
                take_profits=take_profits,
                risk_amount=risk_amount,
                position_size=position_size,
                rr_ratios=rr_ratios,
            )

        except Exception as e:
            logger.error(f"Trade-Berechnung fehlgeschlagen: {e}")
            return None
