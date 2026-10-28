from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal


@dataclass(frozen=True)
class Model:
    sigma: float
    A: float
    k: float
    gamma: float
    Q: int
    mu: float = 0.0
    xi: float = 0.0
    skew: float = 0.0

    @property
    def q(self) -> np.ndarray:
        return np.arange(-self.Q, self.Q + 1)

    @property
    def alpha(self) -> float:
        return self.k * self.gamma * self.sigma**2 / 2

    @property
    def eta(self) -> float:
        return self.A * (1 + self.gamma / self.k) ** -(1 + self.k / self.gamma)

    @property
    def base(self) -> float:
        return math.log1p(self.gamma / self.k) / self.gamma + self.xi / 2

    @cached_property
    def _bands(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.q
        diag = self.alpha * q**2 - self.k * self.mu * q
        off = np.full(2 * self.Q, -self.eta * math.exp(-self.k * self.xi / 2))
        return diag, off

    def matrix(self) -> np.ndarray:
        diag, off = self._bands
        tilt = math.exp(self.skew / 2)
        return np.diag(diag) + np.diag(off * tilt, 1) + np.diag(off / tilt, -1)

    @cached_property
    def settle(self) -> float:
        lam = eigvalsh_tridiagonal(*self._bands, select="i", select_range=(0, 1))
        return 80 / (lam[1] - lam[0])

    def v(self, tau: float) -> np.ndarray:
        if math.isinf(tau):
            tau = self.settle
        terminal = np.exp(-self.k * self.xi * self.q**2 / 2)
        v = _expm_metzler(-tau * self.matrix()) @ terminal
        return v / v.max()

    def quotes(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        d = np.diff(np.log(self.v(tau))) / self.k
        return np.append(self.base - d, np.nan), np.insert(self.base + d, 0, np.nan)

    def approx(self) -> tuple[np.ndarray, np.ndarray]:
        g, k = self.gamma, self.k
        c = math.sqrt(self.sigma**2 * g / (2 * k * self.A) * (1 + g / k) ** (1 + k / g))
        c *= math.exp(k * self.xi / 4)
        lean = (self.q + 0.5) - self.mu / (g * self.sigma**2)
        tilt = self.skew / (2 * k)
        return self.base + lean * c + tilt, self.base - (lean - 1) * c - tilt


def _expm_metzler(B: np.ndarray) -> np.ndarray:
    n = len(B)
    P = B - B.diagonal().min() * np.eye(n)
    s = max(0, math.frexp(P.sum(axis=1).max())[1]) + 1
    P /= 2**s
    E = term = np.eye(n)
    for j in range(1, n + 20):
        term = term @ P / j
        E = E + term
    for _ in range(s):
        E = E @ E
        E /= E.max()
    return E
