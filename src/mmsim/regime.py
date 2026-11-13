from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from . import robust as r


@dataclass(frozen=True)
class World:
    T: float = 1.0
    dt: float = 0.005
    Q: int = 50
    A: float = 140.0
    k: float = 1.5
    b: tuple[float, float] = (-5.0, 5.0)
    sigma: tuple[float, float] = (2.0, 2.0)
    rate: tuple[float, float] = (2.0, 2.0)
    phi: float = 0.0
    eta: float = 0.0
    pi0: float = 0.5

    def __post_init__(self) -> None:
        if self.A * self.dt > 1:
            raise ValueError("A dt must stay below 1 so a fill is a probability")
        if min(self.sigma) <= 0 or self.k <= 0:
            raise ValueError("sigma and k must be positive")
        if min(self.rate) < 0 or not 0 <= self.pi0 <= 1:
            raise ValueError("rates must be nonnegative and pi0 a probability")

    @property
    def N(self) -> int:
        return round(self.T / self.dt)

    @property
    def q(self) -> np.ndarray:
        return np.arange(-self.Q, self.Q + 1)

    @property
    def P(self) -> np.ndarray:
        p0, p1 = -np.expm1(-np.array(self.rate) * self.dt)
        return np.array([[1 - p0, p0], [p1, 1 - p1]])

    @property
    def stationary(self) -> float:
        p0, p1 = self.P[0, 1], self.P[1, 0]
        return self.pi0 if p0 + p1 == 0 else p0 / (p0 + p1)

    def penalty(self) -> np.ndarray:
        s2 = np.array(self.sigma)[:, None] ** 2
        return self.phi * s2 * self.dt * self.q[None, :].astype(float) ** 2

    def terminal(self) -> np.ndarray:
        return -self.eta * self.q.astype(float) ** 2

    def game(self, b: tuple[float, float], sigma2: float) -> r.Game:
        return r.Game(
            T=self.T,
            dt=self.dt,
            sigma=float(np.sqrt(sigma2)),
            Q=self.Q,
            b=b,
            A=(self.A, self.A),
            k=(self.k, self.k),
            zeta=self.phi * sigma2 * self.dt,
            eta=self.eta,
        )


def loglik(w: World, dz, j: int):
    s = w.sigma[j]
    return -0.5 * (dz - w.b[j] * w.dt) ** 2 / (s * s * w.dt) - np.log(s)


def update(w: World, pi, dz):
    post = expit(logit(pi) + loglik(w, dz, 1) - loglik(w, dz, 0))
    return post * w.P[1, 1] + (1 - post) * w.P[0, 1]


def priors(w: World) -> np.ndarray:
    pi = np.empty(w.N + 1)
    pi[0] = w.pi0
    for n in range(w.N):
        pi[n + 1] = pi[n] * w.P[1, 1] + (1 - pi[n]) * w.P[0, 1]
    return pi
