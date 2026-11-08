from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .book import Book, Side
from .itch import NS_PER_MINUTE, Msg
from .replay import ReplayBook, batches, on_grid
from .sim import Context, Print

K = np.array([-1.0, -0.5, 0.5, 1.0])


class Tops(NamedTuple):
    t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bq: np.ndarray
    aq: np.ndarray

    @property
    def spread(self) -> np.ndarray:
        return self.ask - self.bid

    @property
    def mid(self) -> np.ndarray:
        return (self.bid + self.ask) / 2

    @property
    def imbalance(self) -> np.ndarray:
        return self.bq / (self.bq + self.aq)

    def index(self, t) -> np.ndarray:
        i = np.searchsorted(self.t, t, side="right") - 1
        if np.any(i < 0):
            raise ValueError("no quote that early")
        return i

    def mid_at(self, t) -> np.ndarray:
        return self.mid[self.index(t)]

    def between(self, start: float, end: float) -> Tops:
        lo, hi = np.searchsorted(self.t, [start, end])
        return Tops(*(a[lo:hi] for a in self))


def tops(msgs: Sequence[Msg], tick: int = 100) -> Tops:
    book = ReplayBook()
    rows, last = [], None
    for batch in batches(list(on_grid(msgs, tick))):
        for m in batch:
            book.apply(m, (0, 0))
        bid, ask = book.best_bid(), book.best_ask()
        if bid is None or ask is None:
            continue
        row = (bid, ask, book.level(Side.BUY, bid).qty, book.level(Side.SELL, ask).qty)
        if row != last:
            rows.append((batch[0].t / NS_PER_MINUTE, *row))
            last = row
    a = np.array(rows, dtype=float).reshape(-1, 5)
    return Tops(a[:, 0], *(a[:, j].astype(np.int64) for j in range(1, 5)))


def spreads(top: Tops, end: float, share: float = 0.95) -> int:
    w = np.diff(top.t, append=end)
    s = np.unique(top.spread[top.spread >= 1])
    covered = np.array([w[(top.spread >= 1) & (top.spread <= k)].sum() for k in s]) / w.sum()
    return max(2, int(s[np.argmax(covered >= share)]))


def bucket(imb, n: int) -> np.ndarray:
    imb = np.asarray(imb, dtype=float)
    lo = np.floor(imb * n)
    hi = n - 1 - np.floor((1 - imb) * n)
    return np.clip(np.where(imb <= 0.5, lo, hi), 0, n - 1).astype(np.int64)


@dataclass(frozen=True)
class Chain:
    states: np.ndarray
    Q: np.ndarray
    T: np.ndarray
    R: np.ndarray

    @property
    def G1(self) -> np.ndarray:
        return np.linalg.solve(np.eye(len(self.Q)) - self.Q, self.T @ K)

    @property
    def B(self) -> np.ndarray:
        return np.linalg.solve(np.eye(len(self.Q)) - self.Q, self.R)

    def stationary(self) -> np.ndarray:
        B = self.B
        n = len(B)
        A = np.vstack([(np.eye(n) - B).T, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def gstar(self) -> np.ndarray:
        B, G1 = self.B, self.G1
        pi = self.stationary()
        Z = np.eye(len(B)) - B + np.outer(np.ones(len(B)), pi)
        return np.linalg.solve(Z, G1)

    def series(self, terms: int) -> np.ndarray:
        B, g = self.B, self.G1
        out = g.copy()
        for _ in range(terms):
            g = B @ g
            out += g
        return out


def chain(cur: np.ndarray, nxt: np.ndarray, dm: np.ndarray, size: int) -> Chain:
    keep = np.ones(len(cur), dtype=bool)
    while True:
        seen = np.zeros(size, dtype=bool)
        seen[cur[keep]] = True
        drop = keep & ~seen[nxt]
        if not drop.any():
            break
        keep &= ~drop
    states = np.flatnonzero(seen)
    pos = np.full(size, -1)
    pos[states] = np.arange(len(states))
    c, x, d = pos[cur[keep]], pos[nxt[keep]], dm[keep]
    n = len(states)
    Q, R, T = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, len(K)))
    move = d != 0
    np.add.at(Q, (c[~move], x[~move]), 1.0)
    np.add.at(R, (c[move], x[move]), 1.0)
    np.add.at(T, (c[move], np.searchsorted(K, d[move] / 2)), 1.0)
    total = Q.sum(axis=1) + R.sum(axis=1)
    return Chain(states, Q / total[:, None], T / total[:, None], R / total[:, None])


@dataclass(frozen=True)
class Microprice:
    g: np.ndarray
    G1: np.ndarray
    counts: np.ndarray

    @property
    def n_spread(self) -> int:
        return self.g.shape[0]

    @property
    def n_imb(self) -> int:
        return self.g.shape[1]

    def adjust(self, spread, bq, aq) -> np.ndarray:
        s = np.asarray(spread)
        b = bucket(np.asarray(bq) / (np.asarray(bq) + np.asarray(aq)), self.n_imb)
        ok = (s >= 1) & (s <= self.n_spread)
        return np.where(ok, self.g[np.clip(s - 1, 0, self.n_spread - 1), b], np.nan)

    def price(self, bid, ask, bq, aq) -> np.ndarray:
        return (np.asarray(bid) + np.asarray(ask)) / 2 + self.adjust(
            np.asarray(ask) - np.asarray(bid), bq, aq
        )

    def lean(self, ctx: Context) -> float:
        bid, ask = ctx.bid, ctx.ask
        if bid is None or ask is None:
            return 0.0
        bq = _others(ctx.book, Side.BUY, bid, ctx.owner)
        aq = _others(ctx.book, Side.SELL, ask, ctx.owner)
        g = float(self.adjust(ask - bid, bq, aq))
        return 0.0 if np.isnan(g) else g

    def observe(self, p: Print) -> None: ...


def fit(top: Tops, n_imb: int = 11, n_spread: int = 2, step: int = 1) -> Microprice:
    if n_imb % 2 == 0:
        raise ValueError("n_imb must be odd so a balanced book is its own mirror")
    s = top.spread
    x = (s - 1) * n_imb + bucket(top.imbalance, n_imb)
    valid = (s >= 1) & (s <= n_spread)
    dm = (top.bid + top.ask)[step:] - (top.bid + top.ask)[:-step]
    ok = valid[:-step] & valid[step:] & (np.abs(dm) <= 2)
    cur, nxt, dm = x[:-step][ok], x[step:][ok], dm[ok]
    size = n_imb * n_spread
    mirror = _mirror(n_imb, n_spread)
    c = chain(
        np.concatenate([cur, mirror[cur]]),
        np.concatenate([nxt, mirror[nxt]]),
        np.concatenate([dm, -dm]),
        size,
    )
    g, G1 = np.full(size, np.nan), np.full(size, np.nan)
    g[c.states] = c.gstar()
    G1[c.states] = c.G1
    counts = np.bincount(cur, minlength=size)
    shape = (n_spread, n_imb)
    return Microprice(g.reshape(shape), G1.reshape(shape), counts.reshape(shape))


def _mirror(n_imb: int, n_spread: int) -> np.ndarray:
    x = np.arange(n_imb * n_spread)
    s, b = divmod(x, n_imb)
    return s * n_imb + (n_imb - 1 - b)


def _others(book: Book, side: Side, px: int, owner: int) -> int:
    lvl = book.level(side, px)
    if lvl is None:
        return 0
    return sum(o.qty for o in lvl.orders.values() if o.owner != owner)
