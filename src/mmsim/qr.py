from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .book import Book, Order, Side
from .itch import NS_PER_MINUTE, Msg
from .replay import ReplayBook, batches, on_grid

LIMIT, CANCEL, MARKET = range(3)
THIN, USUAL, DEEP = range(3)
STAYED_CANCEL, STAYED_MARKET, LEFT_CANCEL, LEFT_MARKET = range(3, 7)
BASE = 7
REGIMES = 2 * BASE
POSITIONS = 1000
UNIT, STATIONARY, QUEUE = "unit", "stationary", "queue"


class Gap:
    __slots__ = ("ask", "wide", "side", "market", "mid", "up")

    def __init__(self) -> None:
        self.ask = math.inf
        self.wide = False
        self.side = Side.BUY
        self.market = False
        self.mid = 0
        self.up = False

    def update(self, bid: int, ask: int, market: bool) -> None:
        mid = bid + ask
        if mid != self.mid:
            self.up = mid > self.mid
            self.mid = mid
        wide = ask - bid > 1
        if wide and not self.wide:
            self.side = Side.SELL if ask > self.ask else Side.BUY
            self.market = market
        elif not wide:
            self.ask = ask
        self.wide = wide

    def regime(self, side: Side) -> int:
        return (LEFT_CANCEL if side == self.side else STAYED_CANCEL) + self.market


@dataclass(frozen=True)
class Params:
    rates: tuple
    sizes: dict
    aes: tuple[int, ...]
    edges: tuple[int, int]
    queues: tuple[tuple[int, ...], ...]
    conditioned: bool
    back: tuple[tuple[float, ...], ...] | None = None
    memory: bool = False

    @property
    def K(self) -> int:
        return len(self.aes)

    @property
    def N(self) -> int:
        return len(self.rates[0][0]) - 1


@dataclass(frozen=True)
class Fit:
    T: float
    counts: np.ndarray
    tau: np.ndarray
    aes: np.ndarray
    edges: tuple[int, int]
    events: np.ndarray
    queues: np.ndarray

    def params(
        self,
        sizes: str = QUEUE,
        regimes: bool = True,
        back: bool = True,
        memory: bool = True,
        min_events: int = 30,
    ) -> Params:
        pooled_n, pooled_t = self.counts.sum(axis=1), self.tau.sum(axis=1)
        out = []
        for i in range(self.aes.size):
            levels = []
            for r in range(REGIMES):
                if not regimes:
                    n, t = pooled_n[i], pooled_t[i]
                elif memory:
                    n, t = self.counts[i, r], self.tau[i, r]
                else:
                    b = r % BASE
                    n = self.counts[i, b] + self.counts[i, b + BASE]
                    t = self.tau[i, b] + self.tau[i, b + BASE]
                levels.append(_filled(n, t, pooled_n[i], pooled_t[i], min_events))
            out.append(tuple(levels))
        return Params(
            tuple(out),
            self._sizes(sizes, regimes, memory, min_events),
            tuple(int(a) for a in self.aes),
            self.edges,
            tuple(tuple(int(v) for v in row) for row in self.queues.T),
            sizes == QUEUE,
            self._back() if back else None,
            memory and regimes,
        )

    def _back(self) -> tuple[tuple[float, ...], ...]:
        lvl, _, kind, _, _, pos = self.events.T
        out = []
        for i in range(self.aes.size):
            u = pos[(lvl == i) & (kind == CANCEL) & (pos >= 0)] / POSITIONS
            out.append(tuple(float(x) for x in u) or (0.5,))
        return tuple(out)

    def _sizes(self, mode: str, regimes: bool, memory: bool, min_events: int) -> dict:
        lvl, n, kind, qty, regime, _ = self.events.T
        if not memory:
            regime = regime % BASE
        out = {}
        for i in range(self.aes.size):
            for k in (LIMIT, CANCEL, MARKET):
                mine = (lvl == i) & (kind == k)
                if mode == UNIT:
                    out[k, i] = (int(self.aes[i]),)
                    continue
                signed = qty[mine]
                out[k, i] = tuple(int(q) for q in (signed if mode == QUEUE else np.abs(signed)))
                if mode == QUEUE:
                    for b in np.unique(n[mine]):
                        cell = signed[n[mine] == b]
                        if cell.size >= min_events:
                            out[k, i, int(b)] = tuple(int(q) for q in cell)
                        for r in np.unique(regime[mine]) if regimes else ():
                            cell = signed[(n[mine] == b) & (regime[mine] == r)]
                            if cell.size >= min_events:
                                out[k, i, int(b), int(r)] = tuple(int(q) for q in cell)
        return out

    def invariant(self, i: int) -> np.ndarray:
        c, t = self.counts[i].sum(axis=0), self.tau[i].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(t[:, None] > 0, c / t[:, None], 0.0)
        return invariant(r[:, LIMIT], r[:, CANCEL] + r[:, MARKET])


def _filled(n, t, pooled_n, pooled_t, min_events):
    rate = np.zeros_like(n, dtype=float)
    last = None
    for b in range(n.shape[0]):
        if n[b].sum() >= min_events:
            last = n[b] / t[b]
        elif pooled_n[b].sum() >= min_events:
            last = pooled_n[b] / pooled_t[b]
        if last is not None:
            rate[b] = last
    rate[-1, LIMIT] = 0.0
    return tuple(tuple(float(x) for x in row) for row in rate)


def invariant(birth: np.ndarray, death: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(death[1:] > 0, birth[:-1] / death[1:], 0.0)
    pi = np.concatenate(([1.0], np.cumprod(ratio)))
    return pi / pi.sum()


def fit(
    msgs: Sequence[Msg],
    start: float,
    end: float,
    *,
    depth: int = 5,
    bins: int = 30,
    tick: int = 100,
) -> Fit:
    book = ReplayBook()
    gap = Gap()
    lo, hi = start * NS_PER_MINUTE, end * NS_PER_MINUTE
    rows, codes, ups, ts, events = [], [], [], [], []

    for batch in batches(list(on_grid(msgs, tick))):
        t = batch[0].t
        if t >= hi:
            break
        bid, ask = book.best_bid(), book.best_ask()
        market = batch[0].kind in "EC"
        if t >= lo and bid is not None and ask is not None:
            row = len(rows)
            rows.append(_ladder(book, bid, ask, depth))
            codes.append(
                (gap.regime(Side.BUY), gap.regime(Side.SELL)) if ask - bid > 1 else (-1, -1)
            )
            ups.append(gap.up)
            ts.append(t)
            events.extend(_events(book, batch, bid, ask, row, depth))
        for m in batch:
            book.apply(m, (0, 0))
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and ask is not None:
            gap.update(bid, ask, market)

    q = np.array(rows, dtype=np.int64)
    ev = np.array(events, dtype=np.int64).reshape(-1, 6)
    aes = np.array(
        [max(1, round(np.abs(ev[ev[:, 2] == i, 4]).mean())) for i in range(depth)], dtype=np.int64
    )
    dt = np.diff(np.array([lo, *ts], dtype=float)) / NS_PER_MINUTE
    touch = q[:, :, 0].ravel()
    low, high = (int(v) for v in np.quantile(touch[touch > 0], [1 / 3, 2 / 3]))
    n = np.minimum(-(-q // aes), bins)
    regime = _regimes(q, np.array(codes, dtype=np.int64), np.array(ups), low, high)

    counts = np.zeros((depth, REGIMES, bins + 1, 3), dtype=np.int64)
    tau = np.zeros((depth, REGIMES, bins + 1))
    for s in Side:
        for i in range(depth):
            np.add.at(tau, (i, regime[:, s, i], n[:, s, i]), dt)
    row, side, lvl, kind, qty, pos = ev.T
    b = n[row, side, lvl]
    np.add.at(counts, (lvl, regime[row, side, lvl], b, kind), 1)
    grid = np.searchsorted(np.cumsum(dt), np.arange(1, int((end - start) * 60)) / 60)
    sample = q[np.minimum(grid, len(q) - 1)]
    return Fit(
        end - start,
        counts,
        tau,
        aes,
        (low, high),
        np.column_stack((lvl, b, kind, qty, regime[row, side, lvl], pos)),
        np.concatenate((sample[:, 0], sample[:, 1])),
    )


def _ladder(book: Book, bid: int, ask: int, depth: int) -> tuple:
    out = []
    for side, px, step in ((Side.BUY, ask, -1), (Side.SELL, bid, 1)):
        out.append(
            [
                lvl.qty if (lvl := book.level(side, px + step * i)) is not None else 0
                for i in range(1, depth + 1)
            ]
        )
    return out


def _events(book: Book, batch, bid: int, ask: int, row: int, depth: int) -> Iterator[tuple]:
    def level(side: Side, px: int) -> int | None:
        i = ask - px if side == Side.BUY else px - bid
        return i - 1 if 1 <= i <= depth else None

    hit = [0, 0]
    for m in batch:
        match m.kind:
            case "A" | "F":
                if (i := level(m.side, m.px)) is not None:
                    yield row, m.side, i, LIMIT, m.qty, -1
            case "E" | "C":
                hit[book.find(-m.ref).side] += m.qty
            case "X" | "D":
                o = book.find(-m.ref)
                if (i := level(o.side, o.price)) is not None:
                    qty = m.qty if m.kind == "X" else o.qty
                    yield row, o.side, i, CANCEL, *_cancel(book, o, qty)
            case "U":
                o = book.find(-m.ref)
                if (i := level(o.side, o.price)) is not None:
                    yield row, o.side, i, CANCEL, *_cancel(book, o, o.qty)
                if (i := level(o.side, m.px)) is not None:
                    yield row, o.side, i, LIMIT, m.qty, -1
    for side in Side:
        if hit[side]:
            px = bid if side == Side.BUY else ask
            if (i := level(side, px)) is not None:
                gone = hit[side] >= book.level(side, px).qty
                yield row, side, i, MARKET, -hit[side] if gone else hit[side], -1


def _cancel(book: Book, o: Order, qty: int) -> tuple[int, int]:
    q = book.level(o.side, o.price).qty
    pos = -1 if q == o.qty else POSITIONS * book.ahead(o.id) // (q - o.qty)
    return -qty if qty >= q else qty, pos


def _regimes(
    q: np.ndarray, codes: np.ndarray, ups: np.ndarray, low: int, high: int
) -> np.ndarray:
    out = np.zeros(q.shape, dtype=np.int64)
    for s in Side:
        opp = q[:, 1 - s, 0]
        out[:, s, 0] = np.select([opp <= low, opp <= high], [THIN, USUAL], DEEP)
        wide = codes[:, s] >= 0
        out[wide, s, :] = codes[wide, s, None]
        out[ups == (s == Side.BUY), s, :] += BASE
    return out
