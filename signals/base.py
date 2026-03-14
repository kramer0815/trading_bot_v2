from dataclasses import dataclass, field


@dataclass
class Signal:
    type: str
    direction: str       # "long" | "short"
    timeframe: str
    price: float
    strength: float      # 0.0 – 1.0
    description: str = ""
    extra: dict = field(default_factory=dict)

    def __str__(self):
        arrow = "▲ LONG" if self.direction == "long" else "▼ SHORT"
        return (
            f"[{self.timeframe}] {arrow} | {self.type} "
            f"@ {self.price:,.2f} | Stärke: {self.strength:.0%}"
            + (f" — {self.description}" if self.description else "")
        )
