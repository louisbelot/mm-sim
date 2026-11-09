from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .multi import Flow


def hedge(C, idx) -> tuple[np.ndarray, float]:
    C = np.asarray(C, dtype=float)
    idx = np.asarray(idx, dtype=int) + 1
    if len(idx) == 0:
        return np.zeros(0), float(C[0, 0])
    c = C[idx, 0]
    h = -np.linalg.solve(C[np.ix_(idx, idx)], c)
    return h, float(C[0, 0] + c @ h)


def stepwise(C, k: int) -> tuple[list[int], np.ndarray]:
    pool = list(range(len(C) - 1))
    chosen, var = [], [hedge(C, [])[1]]
    for _ in range(k):
        best = min(pool, key=lambda j: hedge(C, [*chosen, j])[1])
        chosen.append(best)
        pool.remove(best)
        var.append(hedge(C, chosen)[1])
    return chosen, np.array(var)


@dataclass(frozen=True)
class Fund:
    w: np.ndarray
    Sigma: np.ndarray
    resid: float = 0.0

    @property
    def d(self) -> int:
        return len(self.w)

    @cached_property
    def joint(self) -> np.ndarray:
        w, S = np.asarray(self.w, dtype=float), np.asarray(self.Sigma, dtype=float)
        C = np.empty((self.d + 1, self.d + 1))
        C[0, 0] = w @ S @ w + self.resid
        C[0, 1:] = C[1:, 0] = S @ w
        C[1:, 1:] = S
        return C


@dataclass(frozen=True)
class Dealer:
    fund: Fund
    flow: Flow
    gamma: float
    z: float
    names: tuple[int, ...] = ()
    eta: float | tuple[float, ...] = math.inf
    kappa_etf: float = 0.0
    kappa: float | tuple[float, ...] = 0.0
    redeem: bool = False

    @property
    def m(self) -> int:
        return len(self.names)

    @cached_property
    def etas(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.eta, dtype=float), (self.m,)).copy()

    @cached_property
    def M(self) -> np.ndarray:
        f = self.fund
        M = np.zeros((f.d, self.m + 1))
        M[:, 0] = f.w
        M[list(self.names), np.arange(1, self.m + 1)] = 1.0
        return M

    @cached_property
    def Gamma(self) -> np.ndarray:
        G = self.M.T @ np.asarray(self.fund.Sigma, dtype=float) @ self.M
        G[0, 0] += self.fund.resid
        return G

    @cached_property
    def D(self) -> np.ndarray:
        return np.concatenate([[2 * self.z * self.flow.d2H(0.0)], 1 / (2 * self.etas)])

    @cached_property
    def terminal(self) -> np.ndarray:
        k = np.broadcast_to(np.asarray(self.kappa, dtype=float), (self.fund.d,))
        if self.redeem:
            return self.M.T @ (k[:, None] * self.M)
        return np.diag(np.concatenate([[self.kappa_etf], k[list(self.names)]]))

    @cached_property
    def _spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(self.D)
        mu, U = np.linalg.eigh(self.gamma / 2 * np.outer(r, r) * self.Gamma)
        return np.sqrt(np.clip(mu, 0, None) / 2), U

    def riccati(self, tau: float) -> np.ndarray:
        k, U = self._spectrum
        r = np.sqrt(self.D)
        At = U * k @ U.T
        if not math.isinf(tau):
            d0 = U.T @ (np.outer(r, r) * self.terminal) @ U - np.diag(k)
            E = np.exp(-2 * k * tau)
            G = np.where(k * tau > 1e-12, -np.expm1(-4 * k * tau) / np.where(k > 0, 4 * k, 1), tau)
            F = np.linalg.solve((np.eye(len(k)) + 2 * G[:, None] * d0).T, d0.T).T
            At = At + U @ (E[:, None] * F * E) @ U.T
        return At / np.outer(r, r)

    def stationary_hedge(self) -> np.ndarray:
        A = self.riccati(math.inf)
        return -np.linalg.solve(A[1:, 1:], A[1:, 0])

    def controls(self, tau: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = self.riccati(tau)
        Ax = x @ A
        own = self.z * A[0, 0]
        bid = self.flow.delta(2 * Ax[..., 0] + own)
        ask = self.flow.delta(-2 * Ax[..., 0] + own)
        return bid, ask, -Ax[..., 1:] / self.etas


Controls = Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class Run:
    spread: np.ndarray
    carry: np.ndarray
    hedging: np.ndarray
    close: np.ndarray
    penalty: np.ndarray
    fills: np.ndarray
    x: np.ndarray

    @property
    def pnl(self) -> np.ndarray:
        return self.spread + self.carry - self.hedging - self.close

    @property
    def objective(self) -> np.ndarray:
        return self.pnl - self.penalty


def simulate(
    dealer: Dealer,
    T: float,
    *,
    dt: float,
    paths: int,
    seed: int = 0,
    x0=None,
    controls: Controls | None = None,
    fee: float = 0.0,
) -> Run:
    rng = np.random.default_rng(seed)
    controls = controls or dealer.controls
    n = dealer.m + 1
    G = dealer.Gamma
    lam, V = np.linalg.eigh(G)
    L = V * np.sqrt(np.clip(lam, 0, None))
    x = np.zeros((paths, n)) if x0 is None else np.tile(np.asarray(x0, dtype=float), (paths, 1))
    spread, carry, hedging, penalty = (np.zeros(paths) for _ in range(4))
    fills = np.zeros(paths, dtype=int)
    z, etas = dealer.z, dealer.etas
    for s in range(round(T / dt)):
        bid, ask, xi = controls(T - s * dt, x)
        penalty += dealer.gamma / 2 * np.einsum("pi,ij,pj->p", x, G, x) * dt
        carry += (x * (math.sqrt(dt) * rng.standard_normal((paths, n)) @ L.T)).sum(axis=1)
        hedging += (etas * xi**2).sum(axis=1) * dt
        x[:, 1:] += xi * dt
        u = rng.random((2, paths))
        for side, quote, v in ((1, bid, u[0]), (-1, ask, u[1])):
            hit = v < -np.expm1(-dealer.flow.rate(quote) * dt)
            spread += np.where(hit, z * quote, 0.0)
            x[:, 0] += side * z * hit
            fills += hit
    close = np.einsum("pi,ij,pj->p", x, dealer.terminal, x)
    if dealer.redeem:
        close += fee * np.abs(x[:, 0])
    return Run(spread, carry, hedging, close, penalty, fills, x)

