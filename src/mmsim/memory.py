from __future__ import annotations

import numpy as np

from .duopoly import Game, fill
from .learners import digit


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
