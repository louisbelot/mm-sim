from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np

OUTCOMES = ((0, 0), (1, 0), (0, 1), (1, 1))


@dataclass(frozen=True)
class World:
    T: float = 1.0
    dt: float = 0.005
    Q: int = 10
    A: tuple[float, float] = (140.0, 40.0)
    k: tuple[float, float] = (1.5, 0.75)
    sigma: float = 2.0
    rate: tuple[float, float] = (2.0, 2.0)
    phi: float = 0.0
    eta: float = 0.0
    pi0: float = 0.5

    def __post_init__(self) -> None:
        if max(self.A) * self.dt > 1 or min(self.A) <= 0:
            raise ValueError("A must be positive with A dt below 1")
        if min(self.k) <= 0 or self.sigma <= 0:
            raise ValueError("k and sigma must be positive")
        if min(self.rate) < 0 or not 0 <= self.pi0 <= 1:
            raise ValueError("rates must be nonnegative and pi0 a probability")

    @property
    def N(self) -> int:
        return round(self.T / self.dt)

    @property
    def q(self) -> np.ndarray:
        return np.arange(-self.Q, self.Q + 1)

    @cached_property
    def P(self) -> np.ndarray:
        p0, p1 = -np.expm1(-np.array(self.rate) * self.dt)
        return np.array([[1 - p0, p0], [p1, 1 - p1]])

    @property
    def crossing(self) -> float:
        if self.k[0] == self.k[1]:
            return np.inf
        d = np.log(self.A[0] / self.A[1]) / (self.k[0] - self.k[1])
        return d if d > 0 else np.inf

    def penalty(self) -> np.ndarray:
        return self.phi * self.sigma**2 * self.dt * self.q.astype(float) ** 2

    def terminal(self) -> np.ndarray:
        return -self.eta * self.q.astype(float) ** 2


def predict(w: World, pi):
    return pi * w.P[1, 1] + (1 - pi) * w.P[0, 1]


def fills(w: World, bid, ask, top, bot) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(w.A) * w.dt
    k = np.array(w.k)
    shape = (2,) + (1,) * max(np.ndim(x) for x in (bid, ask, top, bot))
    a, k = a.reshape(shape), k.reshape(shape)
    pb = np.where(top, 0.0, a * np.exp(-k * np.where(top, 0.0, bid)))
    pa = np.where(bot, 0.0, a * np.exp(-k * np.where(bot, 0.0, ask)))
    return pb, pa


def outcomes(w: World, pi, bid, ask, top, bot):
    pb, pa = fills(w, bid, ask, top, bot)
    for eb, ea in OUTCOMES:
        L = (pb if eb else 1 - pb) * (pa if ea else 1 - pa)
        m = (1 - pi) * L[0] + pi * L[1]
        post = np.divide(pi * L[1], m, out=np.broadcast_to(pi, m.shape).astype(float), where=m > 0)
        yield eb, ea, m, post


def update(w: World, pi, bid, ask, bought, sold, top, bot):
    pb, pa = fills(w, bid, ask, top, bot)
    L = np.where(bought, pb, 1 - pb) * np.where(sold, pa, 1 - pa)
    post = pi * L[1] / ((1 - pi) * L[0] + pi * L[1])
    return predict(w, post)


def information(w: World, pi, bid, ask, top=False, bot=False):
    mi = 0.0
    for _, _, m, post in outcomes(w, pi, bid, ask, top, bot):
        for p, prior in ((post, pi), (1 - post, 1 - pi)):
            ratio = np.divide(p, prior, out=np.ones_like(p), where=(p > 0) & (prior > 0))
            mi = mi + m * p * np.log(ratio)
    return mi


def interp(W: np.ndarray, pi, i):
    G, M = W.shape
    pos = pi * (G - 1)
    j = np.minimum(pos.astype(int), G - 2)
    t = pos - j
    flat = W.ravel()
    at = j * M + i
    p1, p2 = flat[at], flat[at + M]
    first, last = j == 0, j == G - 2
    p0 = flat[np.where(first, at, at - M)]
    p3 = flat[np.where(last, at + M, at + 2 * M)]
    m1 = np.where(first, p2 - p1, (p2 - p0) / 2)
    m2 = np.where(last, p2 - p1, (p3 - p1) / 2)
    t2, t3 = t * t, t * t * t
    h01 = 3 * t2 - 2 * t3
    return (1 - h01) * p1 + (t3 - 2 * t2 + t) * m1 + h01 * p2 + (t3 - t2) * m2


class Cells(NamedTuple):
    pi: np.ndarray
    i: np.ndarray
    top: np.ndarray
    bot: np.ndarray


def cells(w: World, grid: int) -> Cells:
    q = w.q[None, :]
    return Cells(np.linspace(0.0, 1.0, grid)[:, None], q + w.Q, q == w.Q, q == -w.Q)


def objective(w: World, W: np.ndarray, c: Cells, bid, ask, learn: bool = True):
    nxt = predict(w, c.pi)
    total = 0.0
    for eb, ea, m, post in outcomes(w, c.pi, bid, ask, c.top, c.bot):
        gain = eb * np.where(c.top, 0.0, bid) + ea * np.where(c.bot, 0.0, ask)
        j = np.clip(c.i + eb - ea, 0, 2 * w.Q)
        total = total + m * (gain + interp(W, predict(w, post) if learn else nxt, j))
    return total


def line(g: Callable, d: np.ndarray, steps: int = 6, h: float = 1e-4) -> np.ndarray:
    vals = g(d)
    d = np.broadcast_to(d, vals.shape)
    best = vals.argmax(axis=0)[None]

    def pick(k):
        return np.take_along_axis(d, np.clip(best + k, 0, len(d) - 1), axis=0)[0]

    lo, x, hi = pick(-1), pick(0), pick(1)
    top = vals.max(axis=0)
    start = x
    for _ in range(steps):
        up, mid, dn = g(x + h), g(x), g(x - h)
        g1 = (up - dn) / (2 * h)
        g2 = (up - 2 * mid + dn) / (h * h)
        newton = np.where(g2 < 0, -g1 / np.where(g2 < 0, g2, -1.0), np.sign(g1) * (hi - lo) / 4)
        x = np.clip(x + newton, lo, hi)
    return np.where(top > g(x), start, x)


def best(
    w: World,
    f: Callable,
    c: Cells,
    warm: tuple | None = None,
    sweeps: int = 8,
    grid: int = 81,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    dmax = 5 / min(w.k)
    d = np.linspace(0.0, dmax, grid)[:, None, None]
    near = np.linspace(-2, 2, 5)[:, None, None] * dmax / (grid - 1)
    shape = np.broadcast_shapes(c.pi.shape, c.i.shape)
    if warm is None:
        bid = ask = np.full(shape, 1 / np.mean(w.k))
    else:
        bid, ask = (np.where(np.isfinite(x), x, 1 / np.mean(w.k)) for x in warm)
    for sweep in range(sweeps):
        db = d if sweep == 0 else np.clip(bid + near, 0.0, dmax)
        nb = line(lambda x, a=ask: f(x, a), db)
        da = d if sweep == 0 else np.clip(ask + near, 0.0, dmax)
        na = line(lambda x, b=nb: f(b, x), da)
        moved = max(np.abs(nb - bid).max(), np.abs(na - ask).max())
        bid, ask = nb, na
        if moved < tol:
            break
    return np.where(c.top, np.inf, bid), np.where(c.bot, np.inf, ask)


class Plan(NamedTuple):
    V: np.ndarray
    bid: np.ndarray
    ask: np.ndarray


def solve(w: World, grid: int = 101, learn: bool = True) -> Plan:
    c = cells(w, grid)
    N, M = w.N, len(w.q)
    V = np.empty((N + 1, grid, M))
    V[N] = w.terminal()
    bid, ask = np.empty((N, grid, M)), np.empty((N, grid, M))
    pen = w.penalty()
    warm = None
    for n in range(N - 1, -1, -1):
        W = V[n + 1] - pen

        def f(b, a, W=W):
            return objective(w, W, c, b, a, learn)

        bid[n], ask[n] = warm = best(w, f, c, warm)
        V[n] = f(bid[n], ask[n])
    return Plan(V, bid, ask)


def bayes(w: World, grid: int = 101) -> Plan:
    return solve(w, grid, learn=True)


def ce(w: World, grid: int = 101) -> Plan:
    return solve(w, grid, learn=False)
