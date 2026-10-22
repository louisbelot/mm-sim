from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq


class Equilibrium(NamedTuple):
    dt: np.ndarray
    lam: np.ndarray
    beta: np.ndarray
    Sigma: np.ndarray
    alpha: np.ndarray
    delta: np.ndarray
    sigma_u: float

    @property
    def t(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.dt)])

    @property
    def profit(self) -> float:
        return float(self.alpha[0] * self.Sigma[0] + self.delta[0])


def solve(N: int, Sigma0: float, sigma_u: float, T: float = 1.0) -> Equilibrium:
    dt = np.full(N, T / N)
    unit = _backward(dt, 1.0, sigma_u)
    return _backward(dt, Sigma0 / unit.Sigma[0], sigma_u)


def _backward(dt: np.ndarray, Sigma_N: float, sigma_u: float) -> Equilibrium:
    N = len(dt)
    lam, beta = np.empty(N), np.empty(N)
    Sigma, alpha, delta = np.empty(N + 1), np.zeros(N + 1), np.zeros(N + 1)
    Sigma[N] = Sigma_N
    s2 = sigma_u**2
    for n in range(N, 0, -1):
        a, h = alpha[n], dt[n - 1]
        lm = _lam(a, s2 * h / Sigma[n])
        b = lm * s2 / Sigma[n]
        lam[n - 1], beta[n - 1] = lm, b
        Sigma[n - 1] = Sigma[n] / (1 - b * lm * h)
        alpha[n - 1] = 1 / (4 * lm * (1 - a * lm))
        delta[n - 1] = delta[n] + a * lm**2 * s2 * h
    return Equilibrium(dt, lam, beta, Sigma, alpha, delta, sigma_u)


def _lam(a: float, c: float) -> float:
    if a == 0:
        return 1 / math.sqrt(2 * c)

    def f(lm):
        return 2 * c * lm * lm * (1 - a * lm) - 1 + 2 * a * lm

    return brentq(f, 0, 1 / (2 * a), xtol=1e-300)


def insider_profit(eq: Equilibrium, beta: np.ndarray) -> float:
    e2, total = eq.Sigma[0], 0.0
    for lm, b, h in zip(eq.lam, beta, eq.dt, strict=True):
        keep = 1 - lm * b * h
        total += b * h * keep * e2
        e2 = keep**2 * e2 + lm**2 * eq.sigma_u**2 * h
    return total


class Paths(NamedTuple):
    v: np.ndarray
    p: np.ndarray
    x: np.ndarray
    u: np.ndarray
    profit: np.ndarray


def simulate(eq: Equilibrium, p0: float, n_paths: int, seed: int = 0) -> Paths:
    rng = np.random.default_rng(seed)
    N = len(eq.dt)
    v = p0 + math.sqrt(eq.Sigma[0]) * rng.standard_normal(n_paths)
    p = np.empty((n_paths, N + 1))
    x = np.empty((n_paths, N))
    u = eq.sigma_u * np.sqrt(eq.dt) * rng.standard_normal((n_paths, N))
    p[:, 0] = p0
    for n in range(N):
        x[:, n] = eq.beta[n] * (v - p[:, n]) * eq.dt[n]
        p[:, n + 1] = p[:, n] + eq.lam[n] * (x[:, n] + u[:, n])
    profit = ((v[:, None] - p[:, 1:]) * x).sum(axis=1)
    return Paths(v, p, x, u, profit)
