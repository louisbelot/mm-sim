from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from mmsim import options as o
from mmsim.multi import Market


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


def _lerp(lo: float, h: float, n: int, x):
    t = np.clip((np.asarray(x, float) - lo) / h, 0, n - 1)
    i = np.minimum(t.astype(int), n - 2)
    return i, t - i


@dataclass(frozen=True)
class Reduced:
    m: Market
    vega: np.ndarray
    P: Heston
    Q: Heston
    nu_lo: float
    nu_hi: float
    vbar: float
    T: float
    rows: int = 31
    points: int = 401
    mv: bool = False
    rtol: float = 1e-8

    @property
    def eta(self) -> float:
        return self.P.xi / 2 * math.sqrt(1 - self.P.rho**2 * self.mv)

    @cached_property
    def nu(self) -> np.ndarray:
        return np.linspace(self.nu_lo, self.nu_hi, self.rows)

    @cached_property
    def v(self) -> np.ndarray:
        return np.linspace(-self.vbar, self.vbar, self.points)

    def edge(self, nu):
        return (self.P.drift(nu) - self.Q.drift(nu)) / (2 * np.sqrt(nu))

    @cached_property
    def _legs(self):
        h, tol = self.v[1] - self.v[0], 1e-9 * self.vbar
        legs = []
        for f, z, V in zip(self.m.flow, self.m.z, self.vega, strict=True):
            for side in (1, -1):
                nxt = self.v + side * z * V
                i, t = _lerp(-self.vbar, h, self.points, nxt)
                legs.append((f, z, i, t, np.abs(nxt) <= self.vbar + tol))
        return legs

    @cached_property
    def _static(self) -> np.ndarray:
        nu, v = self.nu[:, None], self.v[None, :]
        return v * self.edge(nu) - self.m.gamma / 2 * (self.eta * v) ** 2

    def rhs(self, w: np.ndarray) -> np.ndarray:
        W = w.reshape(self.rows, self.points)
        out = self._static.copy()
        if self.rows > 1:
            h, nu = self.nu[1] - self.nu[0], self.nu[:, None]
            a = self.P.drift(nu)
            E = np.pad(W, ((1, 1), (0, 0)), mode="edge")
            up = np.where(a > 0, E[2:] - W, W - E[:-2]) / h
            R = np.pad(W, ((1, 1), (0, 0)), mode="reflect")
            out += a * up + self.P.xi**2 * nu / 2 * (R[2:] - 2 * W + R[:-2]) / h**2
        for f, z, i, t, ok in self._legs:
            nxt = W[:, i] * (1 - t) + W[:, i + 1] * t
            out += np.where(ok, z * f.H((W - nxt) / z), 0.0)
        return out.ravel()

    @cached_property
    def _sol(self):
        return solve_ivp(
            lambda _, w: self.rhs(w),
            (0, self.T),
            np.zeros(self.rows * self.points),
            method="DOP853",
            rtol=self.rtol,
            atol=self.rtol * self.vbar,
            dense_output=True,
        )

    def w(self, tau: float) -> np.ndarray:
        return self._sol.sol(min(tau, self.T)).reshape(self.rows, self.points)

    def _at(self, W: np.ndarray, nu, v):
        if self.rows == 1:
            row = np.broadcast_to(W[0], np.shape(nu) + (self.points,))
        else:
            j, s = _lerp(self.nu_lo, self.nu[1] - self.nu[0], self.rows, nu)
            row = W[j] * (1 - s[..., None]) + W[j + 1] * s[..., None]
        i, t = _lerp(-self.vbar, self.v[1] - self.v[0], self.points, v)
        pick = np.take_along_axis
        return pick(row, i, -1) * (1 - t) + pick(row, i + 1, -1) * t

    def value(self, nu: float) -> float:
        return float(self._at(self.w(self.T), np.array([nu]), np.zeros((1, 1)))[0, 0])

    def quotes(self, tau: float, nu, v) -> tuple[np.ndarray, np.ndarray]:
        W = self.w(tau)
        nu = np.atleast_1d(np.asarray(nu, float))
        v = np.broadcast_to(np.asarray(v, float), nu.shape)[:, None]
        here = self._at(W, nu, v)
        out = []
        for side in (1, -1):
            nxt = v + side * self.m.z * self.vega
            gap = (here - self._at(W, nu, nxt)) / self.m.z
            q = np.stack([f.delta(gap[:, i]) for i, f in enumerate(self.m.flow)], axis=-1)
            out.append(np.where(np.abs(nxt) <= self.vbar * (1 + 1e-9), q, np.inf))
        return out[0], out[1]

    def policy(self, tau: float, nu: np.ndarray, n: np.ndarray, vega: np.ndarray):
        return self.quotes(tau, nu, (n * self.m.z * vega).sum(axis=-1))


Quotes = Callable[[float, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def own(r: Reduced, quotes: Quotes, *, nu0: float, dt: float, paths: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nu, v = np.full(paths, nu0), np.zeros(paths)
    zV = r.m.z * r.vega
    out = np.zeros(paths)
    for s in range(round(r.T / dt)):
        bid, ask = quotes(r.T - s * dt, nu, v)
        u = rng.random((paths, r.m.d, 2))
        for i, f in enumerate(r.m.flow):
            for side, q, k in ((1.0, bid[:, i], 0), (-1.0, ask[:, i], 1)):
                ok = np.isfinite(q) & (np.abs(v + side * zV[i]) <= r.vbar * (1 + 1e-9))
                q = np.where(ok, q, 0.0)
                hit = ok & (u[:, i, k] < -np.expm1(-f.rate(q) * dt))
                out += np.where(hit, r.m.z[i] * q, 0.0)
                v += side * hit * zV[i]
        zn, zs = rng.standard_normal((2, paths))
        out += (v * r.edge(nu) - r.m.gamma / 2 * (r.eta * v) ** 2) * dt
        out += r.eta * v * math.sqrt(dt) * zs
        nu = np.maximum(nu + r.P.drift(nu) * dt + r.P.xi * np.sqrt(nu * dt) * zn, 1e-8)
    return out


Policy = Callable[[float, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
TERMS = ("spread", "delta", "vega", "cost", "other")


@dataclass(frozen=True)
class Session:
    pnl: np.ndarray
    penalty: np.ndarray
    terms: dict[str, np.ndarray]
    fills: np.ndarray
    vega: np.ndarray
    peak: np.ndarray

    @property
    def objective(self) -> np.ndarray:
        return self.pnl - self.penalty


def simulate(
    pr: Pricer,
    expiry: np.ndarray,
    m: Market,
    policy: Policy,
    *,
    P: Heston,
    S0: float,
    nu0: float,
    T: float,
    dt: float,
    paths: int,
    mv: bool = False,
    h: float = 0.0,
    every: int = 1,
    vbar: float = math.inf,
    n0=0.0,
    seed: int = 0,
) -> Session:
    if T > np.min(expiry):
        raise ValueError("horizon runs past the first expiry")
    rng = np.random.default_rng(seed)
    d, z = len(pr.K), m.z
    eta = P.xi / 2 * math.sqrt(1 - P.rho**2 * mv)
    n = np.broadcast_to(np.asarray(n0, float), (paths, d)).copy()
    S, nu = np.full(paths, S0), np.full(paths, nu0)
    g = pr.greeks(S, nu, expiry)
    wealth0 = (n * z * g.price).sum(axis=1)
    hedge, cash = np.zeros(paths), np.zeros(paths)
    terms = {k: np.zeros(paths) for k in TERMS}
    penalty, peak = np.zeros(paths), np.zeros(paths)
    fills = np.zeros((paths, d), dtype=int)
    for s in range(round(T / dt)):
        vegas = 2 * np.sqrt(nu)[:, None] * g.dnu
        bid, ask = policy(T - s * dt, nu, n, vegas)
        u = rng.random((paths, d, 2))
        vega = (n * z * vegas).sum(axis=1)
        for i in range(d):
            dv, f = z[i] * vegas[:, i], m.flow[i]
            for side, quote, k in ((1.0, bid[:, i], 0), (-1.0, ask[:, i], 1)):
                ok = np.isfinite(quote) & (np.abs(vega + side * dv) <= vbar)
                q = np.where(ok, quote, 0.0)
                hit = ok & (u[:, i, k] < -np.expm1(-f.rate(q) * dt))
                n[:, i] += side * hit
                fills[:, i] += hit
                terms["spread"] += np.where(hit, z[i] * q, 0.0)
                cash += np.where(hit, z[i] * (q - side * g.price[:, i]), 0.0)
                vega += side * hit * dv
        np.maximum(peak, np.abs(vega), out=peak)
        penalty += m.gamma / 2 * (eta * vega) ** 2 * dt
        nz = n * z
        held = (nz * g.delta).sum(axis=1)
        if s % every == 0:
            target = -held - mv * P.rho * P.xi * (nz * g.dnu).sum(axis=1) / S
            trade = target - hedge
            cost = h * np.abs(trade)
            cash -= trade * S + cost
            terms["cost"] -= cost
            hedge = target
        zs, zn = rng.standard_normal((2, paths))
        zn = P.rho * zs + math.sqrt(1 - P.rho**2) * zn
        S1 = S * np.exp(np.sqrt(nu * dt) * zs - nu * dt / 2)
        nu1 = np.maximum(nu + P.drift(nu) * dt + P.xi * np.sqrt(nu * dt) * zn, 1e-8)
        g1 = pr.greeks(S1, nu1, expiry - (s + 1) * dt)
        dS, dnu = S1 - S, nu1 - nu
        parts = {"delta": (held + hedge) * dS, "vega": (nz * g.dnu).sum(axis=1) * dnu}
        for k, p in parts.items():
            terms[k] += p
        exact = (nz * (g1.price - g.price)).sum(axis=1) + hedge * dS
        terms["other"] += exact - sum(parts.values())
        S, nu, g = S1, nu1, g1
    wealth = cash + hedge * S + (n * z * g.price).sum(axis=1)
    return Session(
        pnl=wealth - wealth0,
        penalty=penalty,
        terms=terms,
        fills=fills,
        vega=(n * z * 2 * np.sqrt(nu)[:, None] * g.dnu).sum(axis=1),
        peak=peak,
    )
