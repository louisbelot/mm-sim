from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .accounting import Fill
from .book import Side
from .markout import Tape


@dataclass(frozen=True)
class Attribution:
    spread: float
    adverse: float
    inventory: float
    fees: float

    @property
    def total(self) -> float:
        return self.spread + self.adverse + self.inventory - self.fees


def attribute(fills: Sequence[Fill], tape: Tape, horizon: float, end: float) -> Attribution:
    if not fills:
        return Attribution(0.0, 0.0, 0.0, 0.0)
    t = np.array([f.t for f in fills])
    px = np.array([f.price for f in fills], dtype=float)
    pre = np.array([f.price if f.mid is None else f.mid for f in fills], dtype=float)
    q = np.array([f.qty if f.side == Side.BUY else -f.qty for f in fills], dtype=float)
    later = tape.at(np.minimum(t + horizon, end))
    close = float(tape.at(end))
    return Attribution(
        float(q @ (pre - px)),
        float(q @ (later - pre)),
        float(q @ (close - later)),
        sum(f.fee for f in fills),
    )
