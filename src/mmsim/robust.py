from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from . import avellaneda as av


class Market(NamedTuple):
    b: np.ndarray | float
    A: np.ndarray | float
    k: np.ndarray | float


FIXED = Market(0.0, 140.0, 1.5)


@dataclass(frozen=True)
class Game:
    T: float = 1.0
    dt: float = 0.005
    sigma: float = 2.0
    Q: int = 50
    b: tuple[float, float] = (-5.0, 5.0)
    A: tuple[float, float] = (105.0, 175.0)
    k: tuple[float, float] = (1.125, 1.875)
    zeta: float = 0.0
    nu: float = 0.0
    eta: float = 0.0

    def __post_init__(self) -> None:
        if self.A[1] * self.dt > 1:
            raise ValueError("A dt must stay below 1 so a fill is a probability")
        if self.k[0] <= 0:
            raise ValueError("k must be positive")

    @property
    def N(self) -> int:
        return round(self.T / self.dt)

    @property
    def q(self) -> np.ndarray:
        return np.arange(-self.Q, self.Q + 1)

    def penalty(self, q: np.ndarray) -> np.ndarray:
        return self.zeta * q**2 + self.nu * np.abs(q)

    def terminal(self) -> np.ndarray:
        return -self.eta * self.q.astype(float) ** 2


class Stage(NamedTuple):
    q: np.ndarray
    W: np.ndarray
    up: np.ndarray
    dn: np.ndarray
    top: np.ndarray
    bot: np.ndarray
    dt: float

    def at(self, i: np.ndarray) -> Stage:
        return Stage(*(x[i] for x in self[:-1]), self.dt)


def stage(game: Game, V: np.ndarray) -> Stage:
    q = game.q
    W = V - game.penalty(q)
    pad = np.pad(W, 1, mode="edge")
    return Stage(q, W, pad[2:], pad[:-2], q == game.Q, q == -game.Q, game.dt)


def fills(st: Stage, bid, ask, A, k) -> tuple[np.ndarray, np.ndarray]:
    a = A * st.dt
    pb = np.where(st.top, 0.0, a * np.exp(-k * np.where(st.top, 0.0, bid)))
    pa = np.where(st.bot, 0.0, a * np.exp(-k * np.where(st.bot, 0.0, ask)))
    return pb, pa


def payoff(st: Stage, bid, ask, b, A, k) -> np.ndarray:
    pb, pa = fills(st, bid, ask, A, k)
    edge = pb * np.where(st.top, 0.0, bid) + pa * np.where(st.bot, 0.0, ask)
    drift = b * st.dt * (st.q + pb - pa)
    stay = 1 - pb - pa + 2 * pb * pa
    return edge + drift + pb * (1 - pa) * st.up + pa * (1 - pb) * st.dn + stay * st.W


def respond(
    st: Stage, b, A, k, warm: tuple[np.ndarray, np.ndarray] | None = None, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    shape = st.q.shape
    a = np.broadcast_to(A * st.dt, shape)
    k = np.broadcast_to(k, shape)
    bdt = np.broadcast_to(b * st.dt, shape)
    bid, ask = (1 / k, 1 / k) if warm is None else warm
    bid = np.where(np.isfinite(bid), bid, 1 / k)
    ask = np.where(np.isfinite(ask), ask, 1 / k)
    gain_up, gain_dn = st.up - st.W, st.dn - st.W
    for _ in range(10_000):
        pa = np.where(st.bot, 0.0, a * np.exp(-k * np.where(st.bot, 0.0, ask)))
        nb = np.maximum(0.0, 1 / k - bdt - (1 - pa) * gain_up + pa * gain_dn)
        pb = np.where(st.top, 0.0, a * np.exp(-k * nb))
        na = np.maximum(0.0, 1 / k + bdt - (1 - pb) * gain_dn + pb * gain_up)
        moved = max(np.abs(nb - bid).max(), np.abs(na - ask).max())
        bid, ask = nb, na
        if moved < tol:
            return np.where(st.top, np.inf, bid), np.where(st.bot, np.inf, ask)
    raise RuntimeError("quote iteration did not converge")


class Solution(NamedTuple):
    V: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    market: Market


Step = Callable[[int, Stage, tuple | None], tuple[np.ndarray, np.ndarray, Market]]


def backward(game: Game, step: Step) -> Solution:
    N, M = game.N, len(game.q)
    V = np.empty((N + 1, M))
    V[N] = game.terminal()
    bid, ask = np.empty((N, M)), np.empty((N, M))
    b, A, k = (np.empty((N, M)) for _ in range(3))
    warm = None
    for n in range(N - 1, -1, -1):
        st = stage(game, V[n + 1])
        qb, qa, mkt = step(n, st, warm)
        warm = qb, qa
        bid[n], ask[n] = qb, qa
        b[n], A[n], k[n] = (np.broadcast_to(x, (M,)) for x in mkt)
        V[n] = payoff(st, qb, qa, *mkt)
    return Solution(V, bid, ask, Market(b, A, k))


def at(x, n: int):
    return x[n] if np.ndim(x) == 2 else x


def best_response(game: Game, market: Market = FIXED) -> Solution:
    def step(n, st, warm):
        mkt = Market(*(at(x, n) for x in market))
        return (*respond(st, *mkt, warm), mkt)

    return backward(game, step)


def drift(st: Stage, A, k, lo: float, hi: float, warm=None, tol: float = 1e-11):
    def side(b, w=None):
        bid, ask = respond(st, b, A, k, w)
        pb, pa = fills(st, bid, ask, A, k)
        return st.q + pb - pa, (bid, ask)

    x_lo, q_lo = side(lo, warm)
    x_hi, q_hi = side(hi, q_lo)
    b = np.where(x_lo >= 0, lo, hi)
    bid = np.where(x_lo >= 0, q_lo[0], q_hi[0])
    ask = np.where(x_lo >= 0, q_lo[1], q_hi[1])
    inner = np.flatnonzero((x_lo < 0) & (x_hi > 0))
    if inner.size:
        sub = st.at(inner)
        Ai, ki = (np.broadcast_to(x, st.q.shape)[inner] for x in (A, k))
        left, right = np.full(inner.size, lo), np.full(inner.size, hi)
        w = None
        while (right - left).max() > tol:
            mid = (left + right) / 2
            qb, qa = respond(sub, mid, Ai, ki, w)
            w = qb, qa
            pb, pa = fills(sub, qb, qa, Ai, ki)
            up = sub.q + pb - pa > 0
            right, left = np.where(up, mid, right), np.where(up, left, mid)
        b[inner] = (left + right) / 2
        bid[inner], ask[inner] = respond(sub, b[inner], Ai, ki, w)
    return b, bid, ask


def equilibrium(game: Game, controls: str = "bAk") -> Solution:
    A = game.A[0] if "A" in controls else FIXED.A
    k = game.k[1] if "k" in controls else FIXED.k

    def step(n, st, warm):
        if "b" in controls:
            b, bid, ask = drift(st, A, k, *game.b, warm)
        else:
            b = FIXED.b
            bid, ask = respond(st, b, A, k, warm)
        return bid, ask, Market(b, A, k)

    return backward(game, step)


def choices(game: Game, controls: str, grid: int = 9) -> Market:
    b = np.array(game.b) if "b" in controls else np.array([FIXED.b])
    A = np.linspace(*game.A, grid) if "A" in controls else np.array([FIXED.A])
    k = np.linspace(*game.k, grid) if "k" in controls else np.array([FIXED.k])
    bb, AA, kk = np.meshgrid(b, A, k, indexing="ij")
    return Market(bb.ravel(), AA.ravel(), kk.ravel())


def worst_case(game: Game, bid: np.ndarray, ask: np.ndarray, controls: str = "bAk", grid: int = 9):
    opts = choices(game, controls, grid)
    col = [x[:, None] for x in opts]

    def step(n, st, warm):
        g = payoff(st, bid[n], ask[n], *col)
        i = g.argmin(axis=0)
        return bid[n], ask[n], Market(*(x[i] for x in opts))

    return backward(game, step)


def evaluate(game: Game, bid: np.ndarray, ask: np.ndarray, market: Market = FIXED) -> np.ndarray:
    def step(n, st, warm):
        return bid[n], ask[n], Market(*(at(x, n) for x in market))

    return backward(game, step).V


def symmetric(game: Game, k: float = FIXED.k) -> tuple[np.ndarray, np.ndarray]:
    d = np.full((game.N, len(game.q)), 1 / k)
    return d, d.copy()


def avellaneda(game: Game, gamma: float, k: float = FIXED.k) -> tuple[np.ndarray, np.ndarray]:
    tau = game.T - game.dt * np.arange(game.N)
    bid, ask = av.quotes(0.0, game.q[None, :], tau[:, None], gamma, game.sigma, k)
    return np.maximum(-bid, 0.0), np.maximum(ask, 0.0)


class Run(NamedTuple):
    wealth: np.ndarray
    q: np.ndarray
    spread: np.ndarray
    reward: np.ndarray
    visits: np.ndarray


def simulate(
    game: Game,
    bid: np.ndarray,
    ask: np.ndarray,
    market: Market = FIXED,
    paths: int = 10_000,
    seed: int = 0,
    q0: int = 0,
) -> Run:
    rng = np.random.default_rng(seed)
    q = np.full(paths, q0)
    x = np.zeros(paths)
    z = np.full(paths, 100.0)
    reward = np.zeros(paths)
    quoted = np.zeros(paths)
    sides = np.zeros(paths)
    visits = np.zeros((game.N, len(game.q)))
    for n in range(game.N):
        i = q + game.Q
        visits[n] = np.bincount(i, minlength=len(game.q))
        db, da = bid[n, i], ask[n, i]
        b, A, k = (at(v, n)[i] if np.ndim(v) == 2 else v for v in market)
        a = A * game.dt
        top, bot = q == game.Q, q == -game.Q
        pb = np.where(top, 0.0, a * np.exp(-k * np.where(top, 0.0, db)))
        pa = np.where(bot, 0.0, a * np.exp(-k * np.where(bot, 0.0, da)))
        bought = rng.random(paths) < pb
        sold = rng.random(paths) < pa
        both = ~(top | bot)
        quoted += np.where(both, db + da, 0.0)
        sides += both
        before = x + q * z
        x = x + np.where(sold, z + da, 0.0) - np.where(bought, z - db, 0.0)
        q = q + bought - sold
        z = z + b * game.dt + game.sigma * np.sqrt(game.dt) * rng.standard_normal(paths)
        reward += x + q * z - before - game.penalty(q)
    reward -= game.eta * q.astype(float) ** 2
    wealth = x + q * z
    spread = np.divide(quoted, sides, out=np.full(paths, np.nan), where=sides > 0)
    return Run(wealth, q, spread, reward, visits)
