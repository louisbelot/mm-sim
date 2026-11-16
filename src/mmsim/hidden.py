from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

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
