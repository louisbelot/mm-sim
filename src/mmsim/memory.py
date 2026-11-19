from __future__ import annotations

from dataclasses import replace

import numpy as np

from .duopoly import Game, depth, fill, gains, side
from .learners import digit, greedy


def code(b1, a1, b2, a2) -> np.ndarray:
    return 3 * digit(b2, b1) + digit(a2, a1)


def shares(game: Game, d1, d2) -> tuple[np.ndarray, np.ndarray]:
    p = fill(game, np.minimum(d1, d2))
    tie = d1 == d2
    s = game.share
    return p * np.where(tie, s, d1 < d2), p * np.where(tie, 1 - s, d2 < d1)


def carry(game: Game) -> tuple[np.ndarray, np.ndarray]:
    q1, q2 = game.grid()
    drift = game.drift(q1 + q2) * game.dt
    return drift * q1 - game.phi * q1**2 * game.dt, drift * q2 - game.phi * q2**2 * game.dt


def chain(game: Game, quotes, nxt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b1, a1, b2, a2, nxt = np.broadcast_arrays(*quotes, nxt)
    C, m1, m2 = b1.shape
    n = C * m1 * m2
    i1, i2 = np.indices((m1, m2))
    here = np.arange(n).reshape(C, m1, m2)
    base = nxt * m1 * m2
    c1, c2 = carry(game)
    r1 = np.broadcast_to(c1, b1.shape).astype(float)
    r2 = np.broadcast_to(c2, b1.shape).astype(float)
    P = np.zeros((n, n))
    stay = np.ones(b1.shape)
    for d1, d2, step in ((b1, b2, 1), (a1, a2, -1)):
        p1, p2 = shares(game, d1, d2)
        r1 = r1 + p1 * np.where(np.isfinite(d1), d1, 0.0)
        r2 = r2 + p2 * np.where(np.isfinite(d2), d2, 0.0)
        j1 = np.clip(i1 + step, 0, m1 - 1)
        j2 = np.clip(i2 + step, 0, m2 - 1)
        np.add.at(P, (here, base + j1 * m2 + i2), p1)
        np.add.at(P, (here, base + i1 * m2 + j2), p2)
        stay -= p1 + p2
    np.add.at(P, (here, base + i1 * m2 + i2), stay)
    return P, r1.ravel(), r2.ravel()


def evaluate(game: Game, gamma: float, quotes, nxt) -> tuple[np.ndarray, np.ndarray]:
    P, r1, r2 = chain(game, quotes, nxt)
    M = np.eye(P.shape[0]) - gamma * P
    V = np.linalg.solve(M, np.stack([r1, r2], axis=1))
    m = np.broadcast_shapes(*(np.shape(x) for x in quotes), np.shape(nxt))
    return V[:, 0].reshape(m), V[:, 1].reshape(m)


def offers(game: Game, W, rival_b, rival_a):
    q1, _ = game.grid()
    Q1 = game.Q[0]
    d = depth(game, np.arange(game.J + 2)).reshape(-1, 1, 1, 1)
    off = np.isinf(d)
    up1, up2, dn1, dn2 = gains(W)
    ub = np.where(off | (q1 < Q1), side(game, d, rival_b, up1, up2, 0.0, 0.0)[0], -np.inf)
    ua = np.where(off | (q1 > -Q1), side(game, d, rival_a, dn1, dn2, 0.0, 0.0)[0], -np.inf)
    return d, ub, ua


def improve(game: Game, gamma: float, V, quotes):
    _, _, b2, a2 = quotes
    c1, _ = carry(game)
    best = np.full(np.shape(V), -np.inf)
    bid, ask = np.full(np.shape(V), np.inf), np.full(np.shape(V), np.inf)
    for c in range(9):
        W = gamma * V[c]
        d, ub, ua = offers(game, W, b2, a2)
        ub = np.where(digit(b2, d) == c // 3, ub, -np.inf)
        ua = np.where(digit(a2, d) == c % 3, ua, -np.inf)
        total = W + c1 + ub.max(axis=0) + ua.max(axis=0)
        better = total > best
        best = np.where(better, total, best)
        bid = np.where(better, d[ub.argmax(axis=0), 0, 0, 0], bid)
        ask = np.where(better, d[ua.argmax(axis=0), 0, 0, 0], ask)
    return best, bid, ask


def mirror(game: Game, quotes, flip: bool = True):
    shape = np.broadcast_shapes(*map(np.shape, quotes))

    def turn(x):
        x = np.swapaxes(np.asarray(x), -1, -2)
        return x[::-1] if flip else x

    b1, a1, b2, a2 = (np.broadcast_to(x, shape) for x in quotes)
    other = replace(game, Q=game.Q[::-1], share=1 - game.share)
    return other, tuple(turn(x) for x in (b2, a2, b1, a1)), turn


def deviation(game: Game, gamma: float, quotes, V1, V2) -> tuple[np.ndarray, np.ndarray]:
    g1 = improve(game, gamma, V1, quotes)[0] - V1
    other, flipped, turn = mirror(game, quotes)
    g2 = turn(improve(other, gamma, turn(V2), flipped)[0] - turn(V2))
    return g1, g2


def best_reply(game: Game, gamma: float, quotes, tol: float = 1e-10, cap: int = 100):
    b1, a1, b2, a2 = np.broadcast_arrays(*quotes)
    for _ in range(cap):
        mine = (b1, a1, b2, a2)
        V = evaluate(game, gamma, mine, code(*mine))[0]
        best, bid, ask = improve(game, gamma, V, mine)
        move = best - V > tol
        if not move.any():
            return V, b1, a1
        b1, a1 = np.where(move, bid, b1), np.where(move, ask, a1)
    raise RuntimeError(f"policy iteration did not settle in {cap} rounds")


def respond(game: Game, gamma: float, quotes, who: int = 1) -> np.ndarray:
    if who == 1:
        return best_reply(game, gamma, quotes)[0]
    other, flipped, turn = mirror(game, quotes)
    return turn(best_reply(other, gamma, flipped)[0])


def profile(game: Game, table1: np.ndarray, table2: np.ndarray) -> tuple[np.ndarray, ...]:
    B1, A1 = greedy(game, table1)
    B2, A2 = greedy(game, table2)
    m = (9, B1.shape[0], B2.shape[0])
    one = (np.broadcast_to(x.T[:, :, None], m) for x in (B1, A1))
    two = (np.broadcast_to(x.T[::-1][:, None, :], m) for x in (B2, A2))
    return (*one, *two)
