from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import hawkes
from .book import Side
from .flow import CSTParams
from .itch import NS_PER_MINUTE, Msg
from .replay import ReplayBook, batches, on_grid


@dataclass(frozen=True)
class Calibration:
    T: float
    lam: np.ndarray
    theta: np.ndarray
    queue: np.ndarray
    limit_sizes: np.ndarray
    cancel_shares: float
    market_times: np.ndarray
    market_sides: np.ndarray
    market_sizes: np.ndarray
    hidden_shares: int

    @property
    def limit_size(self) -> float:
        return float(self.limit_sizes.mean())

    @property
    def mu(self) -> float:
        return float(self.market_sizes.sum() / 2 / self.T / self.limit_size)

    def cst(self) -> CSTParams:
        return CSTParams(
            lam=tuple(self.lam.mean(axis=0)), theta=tuple(self.theta.mean(axis=0)), mu=self.mu
        )

    def hawkes(self, unit: float = 1 / 60) -> hawkes.Hawkes:
        t = self.market_times / unit
        h = hawkes.fit(t, self.market_sides, self.T / unit, 2)
        return hawkes.Hawkes(h.mu / unit, h.alpha / unit, h.beta / unit)


def calibrate(
    msgs: Sequence[Msg], start: float, end: float, depth: int = 10, tick: int = 100
) -> Calibration:
    book = ReplayBook()
    lam = np.zeros((2, depth))
    canceled = np.zeros((2, depth))
    queue = np.zeros((2, depth))
    grid = start + np.arange(round((end - start) * 60)) / 60
    k = 0
    sizes, cancels, markets = [], [], []
    hidden = 0
    lo, hi = start * NS_PER_MINUTE, end * NS_PER_MINUTE

    def where(side: Side, px: int) -> int | None:
        opp = book.best(side.opposite)
        if opp is None:
            return None
        i = opp - px if side == Side.BUY else px - opp
        return i - 1 if 1 <= i <= depth else None

    def cancel(o, qty: int) -> None:
        cancels.append(qty)
        i = where(o.side, o.price)
        if i is not None:
            canceled[o.side, i] += qty

    def sample(t: float) -> None:
        nonlocal k
        while k < len(grid) and grid[k] * NS_PER_MINUTE <= t:
            for side in Side:
                for lvl in book.levels(side):
                    i = where(side, lvl.price)
                    if i is None:
                        break
                    queue[side, i] += lvl.qty
            k += 1

    for batch in batches(list(on_grid(msgs, tick))):
        t = batch[0].t
        sample(t)
        if t >= hi:
            break
        if t >= lo:
            hit = [0, 0]
            for m in batch:
                match m.kind:
                    case "A" | "F":
                        sizes.append(m.qty)
                        if (i := where(m.side, m.px)) is not None:
                            lam[m.side, i] += 1
                    case "E" | "C":
                        hit[book.find(-m.ref).side.opposite] += m.qty
                    case "X":
                        cancel(book.find(-m.ref), m.qty)
                    case "D":
                        o = book.find(-m.ref)
                        cancel(o, o.qty)
                    case "U":
                        o = book.find(-m.ref)
                        cancel(o, o.qty)
                        sizes.append(m.qty)
                        if (i := where(o.side, m.px)) is not None:
                            lam[o.side, i] += 1
                    case "P":
                        hidden += m.qty
            for side in Side:
                if hit[side]:
                    markets.append((t / NS_PER_MINUTE - start, int(side), hit[side]))
        for m in batch:
            book.apply(m, (0, 0))

    sample(hi)
    T = end - start
    mean_queue = queue / len(grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(mean_queue > 0, canceled / T / mean_queue, 0.0)
    mk = np.array(markets, dtype=float).reshape(-1, 3)
    return Calibration(
        T=T,
        lam=lam / T,
        theta=theta,
        queue=mean_queue,
        limit_sizes=np.array(sizes),
        cancel_shares=float(np.mean(cancels)) if cancels else 0.0,
        market_times=mk[:, 0],
        market_sides=mk[:, 1].astype(int),
        market_sizes=mk[:, 2],
        hidden_shares=hidden,
    )
