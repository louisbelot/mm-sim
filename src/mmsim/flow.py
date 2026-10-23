from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .book import Book, Event, OrderType, Side

_LIMIT, _MARKET, _CANCEL = range(3)


@dataclass(frozen=True)
class CSTParams:
    lam: tuple[float, ...]
    theta: tuple[float, ...]
    mu: float

    @classmethod
    def sky_perfect(cls, depth: int = 10) -> CSTParams:
        measured = (1.85, 1.51, 1.09, 0.88, 0.77)
        lam = measured + tuple(1.92 / i**0.52 for i in range(6, depth + 1))
        return cls(lam=lam[:depth], theta=(0.71, 0.81, 0.68, 0.56, 0.47), mu=0.94)

    def theta_at(self, i: int) -> float:
        return self.theta[min(i, len(self.theta)) - 1]


class ZeroIntelligence:
    def __init__(self, params: CSTParams, owner: int = 0) -> None:
        self.params = params
        self.owner = owner
        self._cum_lam = list(accumulate(params.lam))
        self._bid: int | None = None
        self._ask: int | None = None

    def seed(self, book: Book, bid: int, ask: int, ids: Iterator[int]) -> None:
        p = self.params
        for i, lam in enumerate(p.lam, start=1):
            n = max(1, round(lam / p.theta_at(i)))
            for _ in range(n):
                book.submit(next(ids), Side.BUY, bid - i + 1, 1, owner=self.owner)
                book.submit(next(ids), Side.SELL, ask + i - 1, 1, owner=self.owner)

    def step(
        self, book: Book, rng: random.Random, horizon: float, ids: Iterator[int]
    ) -> tuple[float, Callable[[Book], list[Event]]] | None:
        p = self.params
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None:
            self._bid = bid
        if ask is not None:
            self._ask = ask
        if self._bid is None or self._ask is None:
            raise ValueError("the book needs both a bid and an ask once before flow can start")

        lam = self._cum_lam[-1]
        actions: list[tuple] = [(_LIMIT, Side.BUY), (_LIMIT, Side.SELL)]
        weights = [lam, lam]
        if ask is not None:
            actions.append((_MARKET, Side.BUY))
            weights.append(p.mu)
        if bid is not None:
            actions.append((_MARKET, Side.SELL))
            weights.append(p.mu)
        for side in Side:
            for lvl in book.levels(side):
                mine = [o.id for o in lvl.orders.values() if o.owner == self.owner]
                if mine:
                    i = self._ask - lvl.price if side == Side.BUY else lvl.price - self._bid
                    actions.append((_CANCEL, mine))
                    weights.append(p.theta_at(i) * len(mine))

        cum = list(accumulate(weights))
        dt = rng.expovariate(cum[-1])
        if dt > horizon:
            return None
        kind, arg = actions[bisect_right(cum, rng.random() * cum[-1], 0, len(cum) - 1)]
        if kind == _CANCEL:
            oid = rng.choice(arg)
            return dt, lambda b: b.cancel(oid)
        oid = next(ids)
        if kind == _MARKET:
            return dt, lambda b: b.submit(oid, arg, 0, 1, owner=self.owner, kind=OrderType.MARKET)
        i = 1 + bisect_right(self._cum_lam, rng.random() * lam, 0, len(self._cum_lam) - 1)
        px = self._ask - i if arg == Side.BUY else self._bid + i
        return dt, lambda b: b.submit(oid, arg, px, 1, owner=self.owner)


def prob_mid_up(a: int, b: int, lam: float, mu: float, theta: float, n_max: int = 100) -> float:
    n = np.arange(1, n_max + 1)
    h = _first_to_empty(lam, mu + theta * n, lam, mu + theta * n)
    return float(h[a - 1, b - 1])


def prob_fill_first(a: int, b: int, lam: float, mu: float, theta: float, n_max: int = 100) -> float:
    n = np.arange(1, n_max + 1)
    h = _first_to_empty(lam, mu + theta * n, 0.0, mu + theta * (n - 1))
    return float(1 - h[a - 1, b - 1])


def _first_to_empty(lam_x: float, death_x: np.ndarray, lam_y: float, death_y: np.ndarray):
    n = len(death_x)
    x, y = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1), indexing="ij")
    idx = (x - 1) * n + (y - 1)
    bx = np.where(x < n, lam_x, 0.0)
    by = np.where(y < n, lam_y, 0.0)
    dx, dy = death_x[x - 1], death_y[y - 1]

    rows, cols, vals = [idx.ravel()], [idx.ravel()], [(bx + dx + by + dy).ravel()]
    for mask, step, rate in ((x < n, n, bx), (x > 1, -n, dx), (y < n, 1, by), (y > 1, -1, dy)):
        rows.append(idx[mask])
        cols.append(idx[mask] + step)
        vals.append(-rate[mask])
    m = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n)
    )
    rhs = np.where(x == 1, dx, 0.0).ravel()
    return spsolve(m, rhs).reshape(n, n)
