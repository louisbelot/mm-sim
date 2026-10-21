from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .accounting import Fill
from .book import Side


class Tape:
    def __init__(self) -> None:
        self.t: list[float] = []
        self.mid: list[float] = []

    def record(self, t: float, mid: float | None) -> None:
        if mid is None or (self.mid and mid == self.mid[-1]):
            return
        if self.t and self.t[-1] == t:
            self.mid[-1] = mid
            return
        self.t.append(t)
        self.mid.append(mid)

    def at(self, t) -> np.ndarray:
        i = np.searchsorted(self.t, t, side="right") - 1
        if np.any(i < 0):
            raise ValueError("no mid on the tape that early")
        return np.asarray(self.mid)[i]


def markouts(
    fills: Sequence[Fill], tape: Tape, horizons: Sequence[float], end: float = np.inf
) -> np.ndarray:
    t = np.array([f.t for f in fills])
    px = np.array([f.price for f in fills], dtype=float)
    sign = np.array([1.0 if f.side == Side.BUY else -1.0 for f in fills])
    at = t[:, None] + np.asarray(horizons, dtype=float)[None, :]
    out = sign[:, None] * (tape.at(np.minimum(at, end)) - px[:, None])
    out[at > end] = np.nan
    return out


def curve(fills: Sequence[Fill], m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qty = np.array([f.qty for f in fills], dtype=float)[:, None]
    ok = ~np.isnan(m)
    w = np.where(ok, qty, 0.0)
    n = w.sum(axis=0)
    mean = np.where(ok, m, 0.0).T @ qty[:, 0] / n
    var = (w * (np.where(ok, m, 0.0) - mean) ** 2).sum(axis=0) / n
    return mean, np.sqrt(var / ok.sum(axis=0))
