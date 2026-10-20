"""Zero-intelligence order flow from Cont, Stoikov and Talreja (2010).

Every event is an independent Poisson arrival whose rate depends only on its
distance i, in ticks, from the opposite best quote:

    limit buy at i ticks below the best ask (sell: above the best bid)   lam(i)
    market buy or sell for one lot                                       mu
    cancel of each resting order i ticks from the opposite quote         theta(i)

Orders are one lot each. The book is then a continuous-time Markov chain, and
`ZeroIntelligence.step` simulates it exactly (Gillespie): the time to the next
event is exponential with the total rate, and the event is picked with
probability proportional to its own rate.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .book import Book, Event, OrderType, Side

_LIMIT, _MARKET, _CANCEL = range(3)


@dataclass(frozen=True)
class CSTParams:
    lam: tuple[float, ...]  # limit order rate at distance 1, 2, ... ticks
    theta: tuple[float, ...]  # cancel rate per order; the last entry covers all larger distances
    mu: float  # market order rate, per side

    @classmethod
    def sky_perfect(cls, depth: int = 10) -> CSTParams:
        """Table 2 of the paper: Sky Perfect Communications on the Tokyo Stock Exchange.

        Rates are per minute, in units of the average limit order size. lam was
        measured for i <= 5 and is extended with the paper's power-law fit
        1.92 / i^0.52.
        """
        measured = (1.85, 1.51, 1.09, 0.88, 0.77)
        lam = measured + tuple(1.92 / i**0.52 for i in range(6, depth + 1))
        return cls(lam=lam[:depth], theta=(0.71, 0.81, 0.68, 0.56, 0.47), mu=0.94)

    def theta_at(self, i: int) -> float:
        return self.theta[min(i, len(self.theta)) - 1]


class ZeroIntelligence:
    """Background trader that sends CST order flow into a book."""

    def __init__(self, params: CSTParams, owner: int = 0) -> None:
        self.params = params
        self.owner = owner
        self._cum_lam = list(accumulate(params.lam))
        # Last seen best quotes. CST pins an empty side to the edge of its price
        # grid; here limit orders keep using the last quote that side had.
        self._bid: int | None = None
        self._ask: int | None = None

    def seed(self, book: Book, bid: int, ask: int, ids: Iterator[int]) -> None:
        """Puts max(1, round(lam(i) / theta(i))) lots on the i-th level of each side.

        With a one-tick spread that is the queue size a level would settle at if it
        stayed i ticks from the opposite quote. It is only a starting point: run a
        warm-up before measuring anything.
        """
        p = self.params
        for i, lam in enumerate(p.lam, start=1):
            n = max(1, round(lam / p.theta_at(i)))
            for _ in range(n):
                book.submit(next(ids), Side.BUY, bid - i + 1, 1, owner=self.owner)
                book.submit(next(ids), Side.SELL, ask + i - 1, 1, owner=self.owner)

    def step(
        self, book: Book, rng: random.Random, horizon: float, ids: Iterator[int]
    ) -> tuple[float, list[Event]] | None:
        """Draws the next background event.

        Returns (time until the event, its events), or None if nothing happens
        within `horizon`. In that case the caller moves time forward by `horizon`
        and asks again: the chain is memoryless, so throwing the draw away is exact.
        """
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
            return dt, book.cancel(rng.choice(arg))
        if kind == _MARKET:
            return dt, book.submit(next(ids), arg, 0, 1, owner=self.owner, kind=OrderType.MARKET)
        i = 1 + bisect_right(self._cum_lam, rng.random() * lam, 0, len(self._cum_lam) - 1)
        px = self._ask - i if arg == Side.BUY else self._bid + i
        return dt, book.submit(next(ids), arg, px, 1, owner=self.owner)


def prob_mid_up(a: int, b: int, lam: float, mu: float, theta: float, n_max: int = 100) -> float:
    """P(next mid move is up), with a lots at the ask, b at the bid and a one-tick spread.

    With a one-tick spread the mid moves when one of the two best queues empties.
    Until then the queues are independent birth-death chains with births lam(1)
    and deaths mu + theta(1) * n (CST, Lemma 2). The paper inverts a Laplace
    transform; here the same probability is the solution of a linear system on
    the two-dimensional chain, truncated at n_max lots per queue.
    """
    n = np.arange(1, n_max + 1)
    h = _first_to_empty(lam, mu + theta * n, lam, mu + theta * n)
    return float(h[a - 1, b - 1])


def prob_fill_first(a: int, b: int, lam: float, mu: float, theta: float, n_max: int = 100) -> float:
    """P(a bid at the back of the queue fills before the ask queue empties), one-tick spread.

    b counts the order itself, so b - 1 lots are ahead of it. The order never
    cancels, so its queue shrinks at rate mu + theta * (lots ahead) and nothing
    joins in front of it. The ask queue is the same chain as in `prob_mid_up`.
    """
    n = np.arange(1, n_max + 1)
    h = _first_to_empty(lam, mu + theta * n, 0.0, mu + theta * (n - 1))
    return float(1 - h[a - 1, b - 1])


def _first_to_empty(lam_x: float, death_x: np.ndarray, lam_y: float, death_y: np.ndarray):
    """h[x-1, y-1] = P(X empties before Y) for independent birth-death chains.

    Births are lam_x and lam_y (blocked at the truncation level), deaths in state
    k are death_x[k-1] and death_y[k-1]. h solves the first-step equations
    rate_out * h(x, y) = sum of rate * h(neighbour), with h = 1 once X is empty
    and h = 0 once Y is.
    """
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
