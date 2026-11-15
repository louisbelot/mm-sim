from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from mmsim import options as o


@dataclass(frozen=True)
class Heston:
    kappa: float
    theta: float
    xi: float
    rho: float

    def drift(self, nu):
        return self.kappa * (self.theta - nu)

    def cf(self, u, T, nu):
        C, D = self.cd(u, T)
        return np.exp(C + D * nu)

    def cd(self, u, T) -> tuple[np.ndarray, np.ndarray]:
        u, T = np.asarray(u, complex), np.asarray(T, float)
        k, x = self.kappa, self.xi
        b = k - self.rho * x * 1j * u
        d = np.sqrt(b * b + x * x * (1j * u + u * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = k * self.theta / x**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / x**2 * (1 - e) / (1 - g * e)
        return C, D


class Greeks(NamedTuple):
    price: np.ndarray
    delta: np.ndarray
    dnu: np.ndarray


@dataclass(frozen=True)
class Pricer:
    model: Heston
    K: np.ndarray
    call: np.ndarray
    nodes: int = 192
    nu_lo: float = 0.0
    tol: float = 1e-13

    @classmethod
    def of(cls, model: Heston, K, call=True, **kw) -> Pricer:
        K, call = np.broadcast_arrays(np.atleast_1d(np.asarray(K, float)), np.atleast_1d(call))
        return cls(model, K.copy(), call.astype(bool), **kw)

    @cached_property
    def _gl(self) -> tuple[np.ndarray, np.ndarray]:
        return np.polynomial.legendre.leggauss(self.nodes)

    def umax(self, T: float) -> float:
        u = np.geomspace(1.0, 1e5, 400)
        C, D = self.model.cd(u - 0.5j, T)
        small = np.abs(np.exp(C + D * self.nu_lo)) / (u * u + 0.25) < self.tol
        return float(u[np.argmax(small)] if small.any() else u[-1])

    def _grid(self, T: np.ndarray):
        x, w = self._gl
        um = np.array([self.umax(t) for t in T])[:, None] / 2
        u = (x + 1) * um
        C, D = self.model.cd(u - 0.5j, T[:, None])
        return u, w * um / (u * u + 0.25), C, D

    def greeks(self, S, nu, T) -> Greeks:
        S = np.asarray(S, float)[..., None]
        nu = np.asarray(nu, float)[..., None, None]
        u, w, C, D = self._grid(np.broadcast_to(np.asarray(T, float), self.K.shape))
        f = np.exp(C + D * nu + 1j * u * np.log(S / self.K)[..., None])
        root = np.sqrt(S * self.K) / math.pi
        call = S - root * (f.real * w).sum(-1)
        dc = 1 - np.sqrt(self.K / S) / math.pi * ((f * (0.5 + 1j * u)).real * w).sum(-1)
        put = ~self.call
        return Greeks(
            price=np.where(put, call - S + self.K, call),
            delta=np.where(put, dc - 1, dc),
            dnu=-root * ((f * D).real * w).sum(-1),
        )

    def vega(self, S, nu, T) -> np.ndarray:
        return 2 * np.sqrt(np.asarray(nu, float))[..., None] * self.greeks(S, nu, T).dnu


def iv(price: float, S: float, K: float, T: float, call: bool = True) -> float:
    def gap(s):
        return float(o.greeks(S, K, T, s, call).price) - price

    return brentq(gap, 1e-4, 4.0, xtol=1e-12)
