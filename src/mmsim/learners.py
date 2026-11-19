from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import numpy as np

from .duopoly import Game, advance, depth, equilibrium, gains, joint, side


class Stationary(NamedTuple):
    V1: np.ndarray
    V2: np.ndarray
    quotes: tuple[np.ndarray, ...]
    eps: float = 0.0


def levels(game: Game) -> np.ndarray:
    return depth(game, np.arange(game.J + 1))


def shape(game: Game) -> tuple[int, int]:
    return 2 * game.Q[0] + 1, 2 * game.Q[1] + 1


def value(game: Game, gamma: float, quotes) -> tuple[np.ndarray, np.ndarray]:
    m = shape(game)
    n = m[0] * m[1]
    zero = np.zeros(m)
    r1, r2 = advance(game, zero, zero, quotes)
    M1, M2 = np.empty((n, n)), np.empty((n, n))
    for i, e in enumerate(np.eye(n).reshape(n, *m)):
        W1, W2 = advance(game, gamma * e, gamma * e, quotes)
        M1[:, i], M2[:, i] = (W1 - r1).ravel(), (W2 - r2).ravel()
    eye = np.eye(n)
    V1 = np.linalg.solve(eye - M1, r1.ravel())
    V2 = np.linalg.solve(eye - M2, r2.ravel())
    return V1.reshape(m), V2.reshape(m)


def deviation(game: Game, V1, V2, quotes) -> tuple[np.ndarray, ...]:
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    d = depth(game, np.arange(game.J + 2))[:, None, None]
    off = np.isinf(d)
    b1, a1, b2, a2 = quotes
    up1, up2, dn1, dn2 = gains(V1)
    vp1, vp2, vd1, vd2 = gains(V2)
    out = []
    for x1, x2, g, c1, c2 in (
        (b1, b2, (up1, up2, vp2, vp1), q1 < Q1, q2 < Q2),
        (a1, a2, (dn1, dn2, vd2, vd1), q1 > -Q1, q2 > -Q2),
    ):
        u1, u2 = side(game, x1, x2, *g)
        best1 = np.where(off | c1, side(game, d, x2, *g)[0], -np.inf).max(axis=0)
        best2 = np.where(off | c2, side(game, x1, d, *g)[1], -np.inf).max(axis=0)
        out += [best1 - u1, best2 - u2]
    return out[0], out[2], out[1], out[3]


def gap(game: Game, gamma: float, quotes, V1, V2) -> float:
    return max(x.max() for x in deviation(game, gamma * V1, gamma * V2, quotes))


def exploit(game: Game, gamma: float, quotes, V1, V2) -> float:
    B1 = respond(game, gamma, quotes, 1)[0]
    B2 = respond(game, gamma, quotes, 2)[0]
    return max((B1 - V1).max(), (B2 - V2).max())


def polish(game: Game, gamma: float, quotes, tol: float, rounds: int) -> Stationary | None:
    seen = set()
    for _ in range(rounds):
        V1, V2 = value(game, gamma, quotes)
        W1, W2 = gamma * V1, gamma * V2
        gb1, ga1, gb2, ga2 = deviation(game, W1, W2, quotes)
        if max(gb1.max(), ga1.max(), gb2.max(), ga2.max()) <= tol:
            return Stationary(V1, V2, quotes)
        key = np.array(quotes).tobytes()
        if key in seen:
            return None
        seen.add(key)
        fresh, _ = equilibrium(game, W1, W2)
        keep_b = (gb1 <= tol) & (gb2 <= tol)
        keep_a = (ga1 <= tol) & (ga2 <= tol)
        quotes = tuple(
            np.where(keep, old, new)
            for old, new, keep in zip(quotes, fresh, (keep_b, keep_a, keep_b, keep_a), strict=True)
        )
    return None


def nearest(game: Game, gamma: float, tol: float, rounds: int, horizon: int, tries: int):
    V1 = V2 = np.zeros(shape(game))
    best = None
    for i in range(horizon):
        quotes, _ = equilibrium(game, gamma * V1, gamma * V2)
        V1, V2 = advance(game, gamma * V1, gamma * V2, quotes)
        if (horizon - 1 - i) % 25 or i < horizon - 25 * tries:
            continue
        sol = polish(game, gamma, quotes, tol, rounds)
        if sol is not None:
            return sol
        E1, E2 = value(game, gamma, quotes)
        g = gap(game, gamma, quotes, E1, E2)
        if best is None or g < best[0]:
            best = g, E1, E2, quotes
    _, E1, E2, quotes = best
    return Stationary(E1, E2, quotes, exploit(game, gamma, quotes, E1, E2))


def markov(
    game: Game,
    gamma: float,
    steps: int = 20,
    tol: float = 1e-10,
    rounds: int = 200,
    horizon: int = 1500,
    tries: int = 20,
) -> Stationary:
    if game.tick <= 0:
        raise ValueError("competition needs a tick")
    zero = np.zeros(shape(game))
    quotes, _ = equilibrium(game, zero, zero)
    for g in np.linspace(0, gamma, steps + 1)[1:]:
        sol = polish(game, g, quotes, tol, rounds)
        if sol is None:
            return nearest(game, gamma, tol, rounds // 4, horizon, tries)
        quotes = sol.quotes
    return sol


def collusive(game: Game, gamma: float, tol: float = 1e-11, cap: int = 200_000) -> Stationary:
    S = np.zeros(shape(game))
    for _ in range(cap):
        quotes, W = joint(game, gamma * S)
        if np.abs(W - S).max() < tol:
            return Stationary(*value(game, gamma, quotes), quotes)
        S = W
    raise RuntimeError(f"no fixed point in {cap} iterations")


def respond(game: Game, gamma: float, quotes, who: int = 1, tol: float = 1e-11, cap: int = 200_000):
    b1, a1, b2, a2 = quotes
    if who == 2:
        game = replace(game, Q=game.Q[::-1], share=1 - game.share)
        b2, a2 = b1.T, a1.T
    q1, q2 = game.grid()
    Q1 = game.Q[0]
    d = depth(game, np.arange(game.J + 2))[:, None, None]
    off = np.isinf(d)
    carry = game.drift(q1 + q2) * game.dt * q1 - game.phi * q1**2 * game.dt
    V = np.zeros(shape(game))
    for _ in range(cap):
        up1, up2, dn1, dn2 = gains(gamma * V)
        ub = np.where(off | (q1 < Q1), side(game, d, b2, up1, up2, 0.0, 0.0)[0], -np.inf)
        ua = np.where(off | (q1 > -Q1), side(game, d, a2, dn1, dn2, 0.0, 0.0)[0], -np.inf)
        W = gamma * V + carry + ub.max(axis=0) + ua.max(axis=0)
        if np.abs(W - V).max() < tol:
            break
        V = W
    else:
        raise RuntimeError(f"no fixed point in {cap} iterations")
    bid, ask = d[ub.argmax(axis=0), 0, 0], d[ua.argmax(axis=0), 0, 0]
    return (W, bid, ask) if who == 1 else (W.T, bid.T, ask.T)
