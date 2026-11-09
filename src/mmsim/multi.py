from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.special import lambertw


@dataclass(frozen=True)
class Exponential:
    A: float
    k: float

    def rate(self, delta):
        return self.A * np.exp(-self.k * delta)

    def delta(self, p):
        return p + 1 / self.k

    def H(self, p):
        return self.A / (self.k * math.e) * np.exp(-self.k * p)

    def d2H(self, p):
        return self.k**2 * self.H(p)


@dataclass(frozen=True)
class Logistic:
    lam: float
    alpha: float
    beta: float

    def rate(self, delta):
        return self.lam / (1 + np.exp(self.alpha + self.beta * delta))

    def _w(self, p):
        return lambertw(np.exp(-(1 + self.alpha + self.beta * np.asarray(p)))).real

    def delta(self, p):
        return p + (1 + self._w(p)) / self.beta

    def H(self, p):
        return self.lam / self.beta * self._w(p)

    def d2H(self, p):
        w = self._w(p)
        return self.lam * self.beta * w / (1 + w) ** 3


Flow = Exponential | Logistic


def covariance(sigma, rho) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    R = np.asarray(rho, dtype=float)
    if R.ndim == 0:
        R = np.full((len(sigma), len(sigma)), float(rho))
        np.fill_diagonal(R, 1.0)
    return R * np.outer(sigma, sigma)


@dataclass(frozen=True)
class Market:
    Sigma: np.ndarray
    flow: tuple[Flow, ...]
    gamma: float
    z: np.ndarray
    mu: np.ndarray

    @classmethod
    def of(cls, sigma, rho, flow, gamma: float, z=1.0, mu=0.0) -> Market:
        Sigma = covariance(sigma, rho)
        d = len(Sigma)
        flow = tuple(flow) if isinstance(flow, list | tuple) else (flow,) * d
        z = np.broadcast_to(np.asarray(z, dtype=float), (d,)).copy()
        mu = np.broadcast_to(np.asarray(mu, dtype=float), (d,)).copy()
        return cls(Sigma, flow, gamma, z, mu)

    @property
    def d(self) -> int:
        return len(self.Sigma)

    def independent(self) -> Market:
        return Market(np.diag(np.diag(self.Sigma)), self.flow, self.gamma, self.z, self.mu)

    @cached_property
    def Dp(self) -> np.ndarray:
        return 2 * self.z * np.array([f.d2H(0.0) for f in self.flow])

    @cached_property
    def _spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.sqrt(self.Dp)
        lam2, P = np.linalg.eigh(self.gamma * np.outer(r, r) * self.Sigma)
        return np.sqrt(np.clip(lam2, 0, None)), P

    def riccati(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        lam, P = self._spectrum
        r = np.sqrt(self.Dp)
        if math.isinf(tau):
            a, f = lam, 1 / lam
        else:
            x = lam * tau
            a = lam * np.tanh(x)
            f = np.where(x < 1e-8, tau, np.tanh(x) / np.where(lam > 0, lam, 1.0))
        A = (P * a) @ P.T / (2 * np.outer(r, r))
        B = -(P @ (f * (P.T @ (r * self.mu)))) / r
        return A, B

    def stationary(self) -> np.ndarray:
        return self.riccati(math.inf)[0]

    def target(self) -> np.ndarray:
        A, B = self.riccati(math.inf)
        return -np.linalg.solve(A, B) / 2

    def quotes(self, q, tau: float = math.inf) -> tuple[np.ndarray, np.ndarray]:
        A, B = self.riccati(tau)
        skew = 2 * np.asarray(q, dtype=float) @ A
        own = self.z * np.diag(A)
        return self._delta(skew + own + B), self._delta(-skew + own - B)

    def policy(self, tau: float, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.quotes(n * self.z, tau)

    def _delta(self, p: np.ndarray) -> np.ndarray:
        return np.stack([f.delta(p[..., i]) for i, f in enumerate(self.flow)], axis=-1)
