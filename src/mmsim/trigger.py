from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .duopoly import Game, advance, depth
from .learners import value
from .memory import carry, mirror, offers


class Trigger(NamedTuple):
    V1: np.ndarray
    V2: np.ndarray
    gain1: np.ndarray
    gain2: np.ndarray

    @property
    def gain(self) -> float:
        return float(max(self.gain1.max(), self.gain2.max()))


def phases(coop, punish, length: int | None):
    if length is not None and length < 1:
        raise ValueError("a punishment lasts at least one step")
    nxt = [0, 1] if length is None else [0, *range(2, length + 1), 0]
    C = len(nxt)
    shape = np.broadcast_shapes(*map(np.shape, coop), *map(np.shape, punish))
    quotes = tuple(
        np.stack([np.broadcast_to(x, shape)] + [np.broadcast_to(y, shape)] * (C - 1))
        for x, y in zip(coop, punish, strict=True)
    )
    return quotes, np.broadcast_to(np.array(nxt)[:, None, None], quotes[0].shape)


def war(game: Game) -> tuple[np.ndarray, ...]:
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    d = depth(game, 0)
    shape = (q1.size, q2.size)
    return tuple(
        np.broadcast_to(np.where(can, d, np.inf), shape)
        for can in (q1 < Q1, q1 > -Q1, q2 < Q2, q2 > -Q2)
    )


def cheat(game: Game, gamma: float, V, quotes) -> np.ndarray:
    b1, a1, b2, a2 = quotes
    W = gamma * V[1]
    d, ub, ua = offers(game, W, b2, a2)
    bex = np.where(d == b1, -np.inf, ub).max(axis=0)
    aex = np.where(d == a1, -np.inf, ua).max(axis=0)
    best = np.maximum(bex + ua.max(axis=0), ub.max(axis=0) + aex)
    return W + carry(game)[0] + best - V


def values(game: Game, gamma: float, coop, punish, length: int | None):
    C1, C2 = value(game, gamma, coop)
    if length is None:
        P1, P2 = value(game, gamma, punish)
        return np.stack([C1, P1]), np.stack([C2, P2])
    V1, V2 = [C1], [C2]
    for _ in range(length):
        W1, W2 = advance(game, gamma * V1[-1], gamma * V2[-1], punish)
        V1.append(W1)
        V2.append(W2)
    return np.stack([C1, *V1[:0:-1]]), np.stack([C2, *V2[:0:-1]])


def check(game: Game, gamma: float, coop, punish, length: int | None) -> Trigger:
    quotes, _ = phases(coop, punish, length)
    V1, V2 = values(game, gamma, coop, punish, length)
    g1 = cheat(game, gamma, V1, quotes)
    other, flipped, turn = mirror(game, quotes, flip=False)
    g2 = turn(cheat(other, gamma, turn(V2), flipped))
    return Trigger(V1, V2, g1, g2)


def widen(game: Game, quotes, m: int) -> tuple[np.ndarray, ...]:
    out = []
    for x in quotes:
        j = np.round(np.where(np.isfinite(x), x, 0.0) / game.tick - 0.5).astype(np.int64)
        out.append(np.where(np.isfinite(x), depth(game, np.minimum(j + m, game.J)), np.inf))
    return tuple(out)


def shared(game: Game, quotes) -> tuple[np.ndarray, ...]:
    q1, q2 = game.grid()
    Q1, Q2 = game.Q
    b1, a1, b2, a2 = np.broadcast_arrays(*quotes)
    b, a = np.minimum(b1, b2), np.minimum(a1, a2)
    return (
        np.where(q1 < Q1, b, np.inf),
        np.where(q1 > -Q1, a, np.inf),
        np.where(q2 < Q2, b, np.inf),
        np.where(q2 > -Q2, a, np.inf),
    )
