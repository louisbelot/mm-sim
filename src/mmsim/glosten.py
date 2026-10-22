from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

BUY, NONE, SELL = 1, 0, -1


def binary_quotes(
    mu: float, alpha: float, lo: float, hi: float, eta: float = 0.5
) -> tuple[float, float]:
    u = (1 - alpha) * eta
    bid = lo + (hi - lo) * mu * u / ((1 - mu) * alpha + u)
    ask = lo + (hi - lo) * mu * (alpha + u) / (mu * alpha + u)
    return bid, ask


class Model(NamedTuple):
    alpha: float
    noise: float = 0.0
    eta: float = 0.5


def _informed(x: np.ndarray, noise: float) -> np.ndarray:
    return ndtr(x / noise) if noise > 0 else (x > 0).astype(float)


def likelihood(m: Model, v: np.ndarray, side: int, bid: float, ask: float) -> np.ndarray:
    buy = m.alpha * _informed(v - ask, m.noise) + (1 - m.alpha) * m.eta
    sell = m.alpha * _informed(bid - v, m.noise) + (1 - m.alpha) * m.eta
    if side == BUY:
        return buy
    if side == SELL:
        return sell
    return 1 - buy - sell


def quotes(m: Model, v: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    on = p > 0
    v, p = v[on], p[on]
    lo, hi = v.min(), v.max()
    if lo == hi:
        return lo, hi

    def ask_gap(a):
        w = p * likelihood(m, v, BUY, lo, a)
        return w @ v / w.sum() - a

    def bid_gap(b):
        w = p * likelihood(m, v, SELL, b, hi)
        return w @ v / w.sum() - b

    tol = 1e-12 * (hi - lo)
    return brentq(bid_gap, lo, hi, xtol=tol), brentq(ask_gap, lo, hi, xtol=tol)


def update(m: Model, v: np.ndarray, p: np.ndarray, side: int, bid: float, ask: float):
    w = p * likelihood(m, v, side, bid, ask)
    return w / w.sum()


class Path(NamedTuple):
    bid: np.ndarray
    ask: np.ndarray
    mean: np.ndarray
    side: np.ndarray
    informed: np.ndarray

    def mm_pnl(self, value: float) -> np.ndarray:
        px = np.where(self.side == BUY, self.ask, self.bid)
        return self.side * (px - value)


def simulate(
    m: Model, v: np.ndarray, prior: np.ndarray, value: float, n: int, seed: int = 0
) -> Path:
    rng = np.random.default_rng(seed)
    p = prior
    bid, ask, mean = np.empty(n), np.empty(n), np.empty(n + 1)
    side = np.zeros(n, dtype=int)
    informed = rng.random(n) < m.alpha
    mean[0] = p @ v
    for i in range(n):
        bid[i], ask[i] = quotes(m, v, p)
        if informed[i]:
            w = value + m.noise * rng.standard_normal()
            side[i] = BUY if w > ask[i] else SELL if w < bid[i] else NONE
        else:
            u = rng.random()
            side[i] = BUY if u < m.eta else SELL if u < 2 * m.eta else NONE
        p = update(m, v, p, side[i], bid[i], ask[i])
        mean[i + 1] = p @ v
    return Path(bid, ask, mean, side, informed)
