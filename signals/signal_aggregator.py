import logging
from .base import Signal

logger = logging.getLogger(__name__)

TF_WEIGHT = {
    "1m": 0.3,  "3m": 0.35, "5m": 0.4,  "15m": 0.5,
    "30m": 0.6, "1h": 0.7,  "2h": 0.75, "4h": 0.85,
    "8h": 0.9,  "1d": 1.0,  "3d": 1.05, "1w": 1.1,
}

TYPE_WEIGHT = {
    "LSOB":     1.00,
    "DIV_BULL": 0.90,
    "DIV_BEAR": 0.90,
    "GUSS":     0.80,
    "EMA":      0.65,
    "TL":       0.85,   # Trendline Breakout
}


class SignalAggregator:
    def __init__(self, cfg):
        self.cfg = cfg

    def score(self, sig: Signal) -> float:
        tf_w   = TF_WEIGHT.get(sig.timeframe, 0.5)
        type_w = next((v for k, v in TYPE_WEIGHT.items() if k in sig.type), 0.5)
        return sig.strength * tf_w * type_w

    def get_best_signal(self, signals: list):
        if not signals:
            return None
        return max(signals, key=self.score)

    def get_consensus(self, signals: list) -> dict:
        long_score  = sum(self.score(s) for s in signals if s.direction == "long")
        short_score = sum(self.score(s) for s in signals if s.direction == "short")
        total = long_score + short_score + 1e-9
        return {
            "long_pct":  long_score / total,
            "short_pct": short_score / total,
            "bias":      "long" if long_score > short_score else "short",
        }
