from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class Game:
    T: float = 1.0
    dt: float = 0.0025
    A: float = 140.0
    k: float = 1.5
    sigma: float = 2.0
    Q: tuple[int, int] = (20, 20)
    tick: float = 0.05
    phi: float = 0.0
    eta: float = 0.0
    beta: float = 0.0
    share: float = 0.5
    reach: float = 5.0

    def __post_init__(self) -> None:
        if 2 * self.A * self.dt > 1:
            raise ValueError("2 A dt must stay below 1: at most one order per step")
        if self.k <= 0 or self.tick < 0:
            raise ValueError("k must be positive and the tick non-negative")
        if not 0 <= self.share <= 1:
            raise ValueError("share is a probability")

    @property
    def N(self) -> int:
        return round(self.T / self.dt)

    @property
    def J(self) -> int:
        return max(0, math.ceil(self.reach / self.tick - 0.5))

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        Q1, Q2 = self.Q
        return np.arange(-Q1, Q1 + 1)[:, None], np.arange(-Q2, Q2 + 1)[None, :]

    def drift(self, q) -> np.ndarray:
        return -self.beta * np.sign(q)


class Solution(NamedTuple):
    V1: np.ndarray
    V2: np.ndarray
    bid1: np.ndarray
    ask1: np.ndarray
    bid2: np.ndarray
    ask2: np.ndarray
    hard: int = 0

    @property
    def quotes(self) -> tuple[np.ndarray, ...]:
        return self.bid1, self.ask1, self.bid2, self.ask2


def fill(game: Game, d):
    return game.A * game.dt * np.exp(-game.k * d)


def edge(game: Game, d, g):
    x = np.where(np.isfinite(d), d, 0.0)
    return fill(game, d) * (x + g)


def depth(game: Game, j):
    return np.where(j > game.J, np.inf, (np.minimum(j, game.J) + 0.5) * game.tick)


def peak(game: Game, g) -> np.ndarray:
    x = np.clip((1 / game.k - g) / game.tick - 0.5, 0, game.J)
    lo = np.floor(x).astype(np.int64)
    hi = np.minimum(lo + 1, game.J)
    return np.where(edge(game, depth(game, hi), g) > edge(game, depth(game, lo), g), hi, lo)


def best(game: Game, g) -> np.ndarray:
    if game.tick == 0:
        return np.maximum(0.0, 1 / game.k - g)
    return depth(game, peak(game, g))


def last(cond: np.ndarray) -> np.ndarray:
    J = cond.shape[0] - 1
    return np.where(cond.any(axis=0), J - np.argmax(cond[::-1], axis=0), -1)


def pull(x: np.ndarray, i: np.ndarray) -> np.ndarray:
    return np.take_along_axis(x, np.maximum(i, 0)[None], 0)[0]


class Player(NamedTuple):
    win: np.ndarray
    lose: np.ndarray
    tie: np.ndarray
    rest: np.ndarray
    stay: np.ndarray
    hold: np.ndarray
    alone: np.ndarray
    can: np.ndarray


def player(game: Game, g, h, s: float, can) -> Player:
    J = game.J
    e = np.arange(J + 1).reshape(-1, *np.ones(np.ndim(g), dtype=int))
    shape = (J + 1, *np.shape(g))
    d = depth(game, e)
    p = fill(game, d)
    win = np.broadcast_to(p * (d + g), shape)
    lose = np.broadcast_to(p * h, shape)
    tie = np.broadcast_to(p * (s * (d + g) + (1 - s) * h), shape)
    pk = peak(game, g)
    cut = np.broadcast_to(np.clip(np.minimum(pk, e - 1), 0, J), shape)
    under = np.where(e > 0, np.take_along_axis(win, cut, 0), -np.inf)
    top = e == J
    later = np.where(top, -np.inf, np.roll(tie, -1, axis=0))
    hold = (e <= pk) & ~top & (win >= np.maximum(later, np.roll(lose, -1, axis=0)))
    can = np.broadcast_to(can, np.shape(g))
    return Player(
        win,
        lose,
        tie,
        ~can | (lose >= np.maximum(tie, under)),
        can & (tie >= np.maximum(lose, under)),
        hold,
        (e == pk) & (win >= 0),
        can,
    )


def stage(game: Game, g1, h1, g2, h2, can1, can2) -> tuple[np.ndarray, np.ndarray, int]:
    none = game.J + 1
    a = player(game, g1, h1, game.share, can1)
    b = player(game, g2, h2, 1 - game.share, can2)

    def lead(x: Player, y: Player):
        e = last(x.can & np.where(y.can, x.hold | x.alone, x.alone) & y.rest)
        back = np.where(pull(x.hold & y.can, e), e + 1, none)
        return e, back, pull(x.win, e) + pull(y.lose, e)

    e1, back1, s1 = lead(a, b)
    e2, back2, s2 = lead(b, a)
    et = last(a.stay & b.stay)
    st = pull(a.tie, et) + pull(b.tie, et)
    top = np.maximum(np.maximum(e1, e2), et)
    cand = np.stack([np.where(e == top, x, -np.inf) for e, x in ((et, st), (e1, s1), (e2, s2))])
    kind = np.argmax(cand, axis=0)
    found = top >= 0
    j1 = np.where(found, np.choose(kind, [et, e1, back2]), none)
    j2 = np.where(found, np.choose(kind, [et, back1, e2]), none)
    quiet = ~(a.can & (pull(a.win, peak(game, g1)) > 0)) & ~(
        b.can & (pull(b.win, peak(game, g2)) > 0)
    )
    hard = np.argwhere(~found & ~quiet)
    for i in hard:
        i = tuple(i)
        args = (x[i] for x in np.broadcast_arrays(g1, h1, g2, h2, a.can, b.can))
        j1[i], j2[i] = exhaust(game, *args)
    return j1, j2, len(hard)


def exhaust(game: Game, g1, h1, g2, h2, can1, can2) -> tuple[int, int]:
    none = game.J + 1
    j = np.arange(none + 1)
    d = depth(game, j)
    u1, u2 = side(game, d[:, None], d[None, :], g1, h1, g2, h2)
    u1 = np.where((j[:, None] < none) & ~can1, -np.inf, u1)
    u2 = np.where((j[None, :] < none) & ~can2, -np.inf, u2)
    ne = (u1 >= u1.max(axis=0, keepdims=True)) & (u2 >= u2.max(axis=1, keepdims=True))
    traded = np.minimum(d[:, None], d[None, :])
    traded = np.where(np.isfinite(u1 + u2), np.where(np.isinf(traded), -1.0, traded), -np.inf)
    if ne.any():
        score = np.where(ne, traded, -np.inf)
        top = ne & (score == score.max())
        pick = np.argmax(np.where(top, u1 + u2, -np.inf))
        return np.unravel_index(pick, u1.shape)
    lead = []
    for u, v in ((u1, u2), (u2.T, u1.T)):
        follow = np.where(v >= v.max(axis=1, keepdims=True), u, -np.inf).max(axis=1)
        x = int(np.argmax(follow))
        y = int(np.argmax(np.where(v[x] >= v[x].max(), u[x], -np.inf)))
        lead.append((u[x, y] + v[x, y], x, y))
    (s1, x1, y1), (s2, x2, y2) = lead
    return (x1, y1) if s1 >= s2 else (y2, x2)


def side(game: Game, d1, d2, g1, h1, g2, h2) -> tuple[np.ndarray, np.ndarray]:
    p1, p2 = fill(game, d1), fill(game, d2)
    win1, win2 = edge(game, d1, g1), edge(game, d2, g2)
    tie = (d1 == d2) & np.isfinite(d1)
    s = game.share
    u1 = np.where(tie, s * win1 + (1 - s) * p1 * h1, np.where(d1 < d2, win1, p2 * h1))
    u2 = np.where(tie, (1 - s) * win2 + s * p2 * h2, np.where(d2 < d1, win2, p1 * h2))
    return u1, u2


def gains(V: np.ndarray) -> tuple[np.ndarray, ...]:
    p = np.pad(V, 1, mode="edge")
    return p[2:, 1:-1] - V, p[1:-1, 2:] - V, p[:-2, 1:-1] - V, p[1:-1, :-2] - V


def advance(game: Game, V1, V2, quotes) -> tuple[np.ndarray, np.ndarray]:
    q1, q2 = game.grid()
    b1, a1, b2, a2 = quotes
    up1, up2, dn1, dn2 = gains(V1)
    vp1, vp2, vd1, vd2 = gains(V2)
    ub1, ub2 = side(game, b1, b2, up1, up2, vp2, vp1)
    ua1, ua2 = side(game, a1, a2, dn1, dn2, vd2, vd1)
    carry = game.drift(q1 + q2) * game.dt
    W1 = V1 + carry * q1 - game.phi * q1**2 * game.dt + ub1 + ua1
    W2 = V2 + carry * q2 - game.phi * q2**2 * game.dt + ub2 + ua2
    return W1, W2


def backward(game: Game, policy) -> Solution:
    q1, q2 = game.grid()
    shape = (q1.size, q2.size)
    V1, V2 = np.empty((game.N + 1, *shape)), np.empty((game.N + 1, *shape))
    V1[-1] = np.broadcast_to(-game.eta * q1**2, shape)
    V2[-1] = np.broadcast_to(-game.eta * q2**2, shape)
    quotes = np.empty((4, game.N, *shape))
    for n in range(game.N - 1, -1, -1):
        quotes[:, n] = policy(n, V1[n + 1], V2[n + 1])
        V1[n], V2[n] = advance(game, V1[n + 1], V2[n + 1], quotes[:, n])
    return Solution(V1, V2, *quotes)


def equilibrium(game: Game, V1, V2) -> tuple[tuple[np.ndarray, ...], int]:
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    up1, up2, dn1, dn2 = gains(V1)
    vp1, vp2, vd1, vd2 = gains(V2)
    b1, b2, nb = stage(game, up1, up2, vp2, vp1, q1 < Q1, q2 < Q2)
    a1, a2, na = stage(game, dn1, dn2, vd2, vd1, q1 > -Q1, q2 > -Q2)
    return tuple(depth(game, j) for j in (b1, a1, b2, a2)), nb + na


def solve(game: Game) -> Solution:
    if game.tick <= 0:
        raise ValueError("competition needs a tick: Bertrand has no pure equilibrium on a line")

    def policy(n, V1, V2):
        nonlocal hard
        quotes, h = equilibrium(game, V1, V2)
        hard += h
        return quotes

    hard = 0
    return backward(game, policy)._replace(hard=hard)


def pick(game: Game, G1, G2, can1, can2):
    d1, d2 = best(game, G1), best(game, G2)
    f1 = np.where(can1, edge(game, d1, G1), -np.inf)
    f2 = np.where(can2, edge(game, d2, G2), -np.inf)
    one = (f1 >= f2) & (f1 > 0)
    two = ~one & (f2 > 0)
    gain = np.maximum(0.0, np.maximum(f1, f2))
    return np.where(one, d1, np.inf), np.where(two, d2, np.inf), gain


def joint(game: Game, S) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    total = q1 + q2
    up1, up2, dn1, dn2 = gains(S)
    b1, b2, gb = pick(game, up1, up2, q1 < Q1, q2 < Q2)
    a1, a2, ga = pick(game, dn1, dn2, q1 > -Q1, q2 > -Q2)
    S = S + game.drift(total) * game.dt * total - game.phi * (q1**2 + q2**2) * game.dt + gb + ga
    return (b1, a1, b2, a2), S


def cartel(game: Game) -> Solution:
    q1, q2 = game.grid()
    S = np.broadcast_to(-game.eta * (q1**2 + q2**2), (q1.size, q2.size)).astype(float)

    def policy(n, V1, V2):
        nonlocal S
        quotes, S = joint(game, S)
        return quotes

    return backward(game, policy)


def monopoly(game: Game) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = game.Q[0]
    q = np.arange(-Q, Q + 1)
    V = np.empty((game.N + 1, q.size))
    V[-1] = -game.eta * q**2
    bid, ask = np.empty((game.N, q.size)), np.empty((game.N, q.size))
    for n in range(game.N - 1, -1, -1):
        W = V[n + 1]
        p = np.pad(W, 1, mode="edge")
        gain = 0.0
        for out, g, can in ((bid, p[2:] - W, q < Q), (ask, p[:-2] - W, q > -Q)):
            d = best(game, g)
            f = edge(game, d, g)
            ok = can & (f > 0)
            out[n] = np.where(ok, d, np.inf)
            gain = gain + np.where(ok, f, 0.0)
        V[n] = W + game.drift(q) * game.dt * q - game.phi * q**2 * game.dt + gain
    return V, bid, ask


def evaluate(game: Game, quotes) -> tuple[np.ndarray, np.ndarray]:
    sol = backward(game, lambda n, V1, V2: tuple(x[n] for x in quotes))
    return sol.V1, sol.V2


class Run(NamedTuple):
    reward1: np.ndarray
    reward2: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    trades1: np.ndarray
    trades2: np.ndarray


def simulate(game: Game, quotes, paths: int = 10_000, seed: int = 0) -> Run:
    rng = np.random.default_rng(seed)
    Q1, Q2 = game.Q
    a = game.A * game.dt
    q1, q2 = np.zeros(paths, dtype=np.int64), np.zeros(paths, dtype=np.int64)
    r1, r2 = np.zeros(paths), np.zeros(paths)
    n1, n2 = np.zeros(paths, dtype=np.int64), np.zeros(paths, dtype=np.int64)
    for n in range(game.N):
        b1, a1, b2, a2 = (x[n, q1 + Q1, q2 + Q2] for x in quotes)
        dz = game.drift(q1 + q2) * game.dt + game.sigma * np.sqrt(game.dt) * rng.standard_normal(
            paths
        )
        r1 += q1 * dz - game.phi * q1**2 * game.dt
        r2 += q2 * dz - game.phi * q2**2 * game.dt
        u = rng.random(paths)
        reach = rng.exponential(1 / game.k, paths)
        first = rng.random(paths) < game.share
        for hit, d1, d2, sign in ((u < a, b1, b2, 1), ((u >= a) & (u < 2 * a), a1, a2, -1)):
            hit = hit & (reach >= np.minimum(d1, d2))
            w1 = hit & ((d1 < d2) | ((d1 == d2) & first))
            w2 = hit & ~w1
            r1 += np.where(w1, d1, 0.0)
            r2 += np.where(w2, d2, 0.0)
            q1 += sign * w1
            q2 += sign * w2
            n1 += w1
            n2 += w2
    r1 -= game.eta * q1**2
    r2 -= game.eta * q2**2
    return Run(r1, r2, q1, q2, n1, n2)
