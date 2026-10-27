from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .book import Book, Event, OrderType, Side
from .hawkes import Hawkes

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
        actions, weights = self._menu(book)
        cum = list(accumulate(weights))
        dt = rng.expovariate(cum[-1])
        if dt > horizon:
            return None
        k = bisect_right(cum, rng.random() * cum[-1], 0, len(cum) - 1)
        return dt, self._fire(actions[k], rng, ids)

    def _market_rate(self, side: Side) -> float:
        return self.params.mu

    def _menu(self, book: Book) -> tuple[list[tuple], list[float]]:
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
            weights.append(self._market_rate(Side.BUY))
        if bid is not None:
            actions.append((_MARKET, Side.SELL))
            weights.append(self._market_rate(Side.SELL))
        for side in Side:
            for lvl in book.levels(side):
                mine = [o.id for o in lvl.orders.values() if o.owner == self.owner]
                if mine:
                    i = self._ask - lvl.price if side == Side.BUY else lvl.price - self._bid
                    actions.append((_CANCEL, mine))
                    weights.append(p.theta_at(i) * len(mine))
        return actions, weights

    def _fire(self, action: tuple, rng: random.Random, ids: Iterator[int]):
        kind, arg = action
        if kind == _CANCEL:
            oid = rng.choice(arg)
            return lambda b: b.cancel(oid)
        oid = next(ids)
        if kind == _MARKET:
            return lambda b: b.submit(oid, arg, 0, 1, owner=self.owner, kind=OrderType.MARKET)
        lam = self._cum_lam[-1]
        i = 1 + bisect_right(self._cum_lam, rng.random() * lam, 0, len(self._cum_lam) - 1)
        px = self._ask - i if arg == Side.BUY else self._bid + i
        return lambda b: b.submit(oid, arg, px, 1, owner=self.owner)


class HawkesFlow(ZeroIntelligence):
    def __init__(self, params: CSTParams, hawkes: Hawkes, owner: int = 0) -> None:
        if hawkes.d != 2:
            raise ValueError("market orders need two dimensions: buys and sells")
        super().__init__(params, owner)
        self.hawkes = hawkes
        self._s = np.zeros((2, 2))

    def _market_rate(self, side: Side) -> float:
        return self.hawkes.mu[side] + self._s[side].sum()

    def _advance(self, dt: float) -> None:
        self._s *= np.exp(-self.hawkes.beta * dt)

    def step(
        self, book: Book, rng: random.Random, horizon: float, ids: Iterator[int]
    ) -> tuple[float, Callable[[Book], list[Event]]] | None:
        actions, weights = self._menu(book)
        markets = [k for k, (kind, _) in enumerate(actions) if kind == _MARKET]
        elapsed = 0.0
        while True:
            bound = sum(weights)
            dt = rng.expovariate(bound)
            if elapsed + dt > horizon:
                self._advance(horizon - elapsed)
                return None
            self._advance(dt)
            elapsed += dt
            for k in markets:
                weights[k] = self._market_rate(actions[k][1])
            cum = list(accumulate(weights))
            u = rng.random() * bound
            if u >= cum[-1]:
                continue
            k = bisect_right(cum, u, 0, len(cum) - 1)
            kind, side = actions[k]
            if kind == _MARKET:
                self._s[:, side] += self.hawkes.alpha[:, side]
            return elapsed, self._fire(actions[k], rng, ids)


class PairedFlow(ZeroIntelligence):
    def __init__(self, params: CSTParams, hawkes: Hawkes | None = None, owner: int = 0) -> None:
        if hawkes is not None and hawkes.d != 2:
            raise ValueError("market orders need two dimensions: buys and sells")
        super().__init__(params, owner)
        self.hawkes = hawkes
        self._market: random.Random | None = None
        self._limit: random.Random | None = None
        # the flow's own time, only to order its pending arrivals; prints use the simulator's
        self._t = 0.0
        self._next_market = self._next_limit = math.inf
        self._s = np.zeros((2, 2))
        self._s_t = 0.0
        self._bound = 0.0
        self._left: dict[int, float] = {}
        self._rate: dict[int, float] = {}

    def step(
        self, book: Book, rng: random.Random, horizon: float, ids: Iterator[int]
    ) -> tuple[float, Callable[[Book], list[Event]]] | None:
        if self._market is None:
            self._start(book, rng)
        elapsed = 0.0
        while True:
            self._rates(book)
            victim, t_cancel = None, math.inf
            for oid, r in self._rate.items():
                t = self._t + max(self._left[oid], 0.0) / r
                if t < t_cancel:
                    victim, t_cancel = oid, t
            t = min(self._next_market, self._next_limit, t_cancel)
            if t - self._t > horizon - elapsed:
                self._wait(horizon - elapsed)
                return None
            elapsed += t - self._t
            self._wait(t - self._t)
            self._t = t
            if t == t_cancel:
                del self._left[victim]
                return elapsed, _cancel(victim)
            if t == self._next_limit:
                return elapsed, self._new_limit(ids)
            side = self._market_arrival()
            if side is not None and book.best(side.opposite) is not None:
                return elapsed, _market(next(ids), side, self.owner)

    def _start(self, book: Book, rng: random.Random) -> None:
        self._market = random.Random(rng.getrandbits(64))
        self._limit = random.Random(rng.getrandbits(64))
        self._rates(book)
        for oid in self._rate:
            self._left[oid] = self._limit.expovariate(1.0)
        self._next_limit = self._limit.expovariate(2 * self._cum_lam[-1])
        if self.hawkes is not None:
            self._bound = float(self.hawkes.mu.sum())
        self._next_market = self._market.expovariate(self._bound or 2 * self.params.mu)

    def _rates(self, book: Book) -> None:
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None:
            self._bid = bid
        if ask is not None:
            self._ask = ask
        if self._bid is None or self._ask is None:
            raise ValueError("the book needs both a bid and an ask once before flow can start")
        self._rate = {}
        for side in Side:
            for lvl in book.levels(side):
                i = self._ask - lvl.price if side == Side.BUY else lvl.price - self._bid
                for o in lvl.orders.values():
                    if o.owner == self.owner:
                        self._rate[o.id] = self.params.theta_at(i)
        for oid in self._left.keys() - self._rate.keys():
            del self._left[oid]

    def _wait(self, dt: float) -> None:
        for oid, r in self._rate.items():
            self._left[oid] -= r * dt
        self._t += dt

    def _new_limit(self, ids: Iterator[int]) -> Callable[[Book], list[Event]]:
        g, lam = self._limit, self._cum_lam
        self._next_limit = self._t + g.expovariate(2 * lam[-1])
        side = Side.BUY if g.random() < 0.5 else Side.SELL
        i = 1 + bisect_right(lam, g.random() * lam[-1], 0, len(lam) - 1)
        oid = next(ids)
        self._left[oid] = g.expovariate(1.0)
        return _limit(oid, side, self._ask - i if side == Side.BUY else self._bid + i, self.owner)

    def _market_arrival(self) -> Side | None:
        u = self._market.random()
        if self.hawkes is None:
            self._next_market = self._t + self._market.expovariate(2 * self.params.mu)
            return Side.BUY if u < 0.5 else Side.SELL
        self._s *= np.exp(-self.hawkes.beta * (self._t - self._s_t))
        self._s_t = self._t
        lam = self.hawkes.mu + self._s.sum(axis=1)
        u *= self._bound
        side = Side.BUY if u < lam[0] else Side.SELL if u < lam.sum() else None
        if side is not None:
            self._s[:, side] += self.hawkes.alpha[:, side]
        self._bound = float(self.hawkes.mu.sum() + self._s.sum())
        self._next_market = self._t + self._market.expovariate(self._bound)
        return side


def _cancel(oid: int) -> Callable[[Book], list[Event]]:
    return lambda b: b.cancel(oid)


def _limit(oid: int, side: Side, px: int, owner: int) -> Callable[[Book], list[Event]]:
    return lambda b: b.submit(oid, side, px, 1, owner=owner)


def _market(oid: int, side: Side, owner: int) -> Callable[[Book], list[Event]]:
    return lambda b: b.submit(oid, side, 0, 1, owner=owner, kind=OrderType.MARKET)


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
