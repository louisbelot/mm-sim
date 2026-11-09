from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ndtr

from mmsim.multi import Logistic, Market

YEAR_DAYS = 252
Policy = Callable[[float, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class Greeks(NamedTuple):
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray


def greeks(S, K, T, sigma, call=True) -> Greeks:
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    w = np.where(call, 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = sigma * np.sqrt(T)
        d1 = np.log(S / K) / sd + sd / 2
        d2 = d1 - sd
        pdf = np.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
        vega = S * pdf * np.sqrt(T)
        return Greeks(
            price=w * (S * ndtr(w * d1) - K * ndtr(w * d2)),
            delta=ndtr(d1) - (w < 0),
            gamma=pdf / (S * sd),
            vega=vega,
            theta=-S * pdf * sigma / (2 * np.sqrt(T)),
            vanna=-pdf * d2 / sigma,
            volga=vega * d1 * d2 / sigma,
        )


@dataclass(frozen=True)
class Chain:
    S: float
    K: np.ndarray
    T: np.ndarray
    iv: np.ndarray
    call: np.ndarray

    @classmethod
    def of(cls, S: float, K, T, iv, call=True) -> Chain:
        K, T, iv, call = np.broadcast_arrays(
            *np.atleast_1d(np.asarray(K, float), np.asarray(T, float), np.asarray(iv, float)),
            np.atleast_1d(call),
        )
        return cls(float(S), K.copy(), T.copy(), iv.copy(), call.astype(bool))

    @classmethod
    def otm(cls, S: float, K, T, iv) -> Chain:
        return cls.of(S, K, T, iv, call=np.asarray(K, float) >= S)

    @property
    def d(self) -> int:
        return len(self.K)

    @cached_property
    def g(self) -> Greeks:
        return greeks(self.S, self.K, self.T, self.iv, self.call)


def requests(c: Chain, per_day: float = 30.0, decay: float = 0.7, width: float = 0.1):
    return YEAR_DAYS * per_day / (1 + decay * np.abs(c.K - c.S) / (width * c.S))


def market(
    c: Chain,
    *,
    lam,
    alpha: float,
    beta: float,
    z,
    eta: float,
    gamma: float,
    eps: float = 0.0,
) -> Market:
    V = c.g.vega
    Sigma = eta**2 * np.outer(V, V) + eps**2 * np.diag(V**2)
    lam = np.broadcast_to(np.asarray(lam, float), (c.d,))
    flow = tuple(Logistic(float(a), alpha, beta / v) for a, v in zip(lam, V, strict=True))
    z = np.broadcast_to(np.asarray(z, float), (c.d,)).copy()
    return Market(Sigma, flow, gamma, z, np.zeros(c.d))


def flat(m: Market) -> Policy:
    def policy(tau: float, n: np.ndarray, vega: np.ndarray):
        bid, ask = m.quotes(np.zeros(m.d), tau)
        return np.broadcast_to(bid, n.shape), np.broadcast_to(ask, n.shape)

    return policy


def closed(m: Market, c: Chain) -> Policy:
    V0 = c.g.vega

    def policy(tau: float, n: np.ndarray, vega: np.ndarray):
        return m.quotes(n * m.z * vega / V0, tau)

    return policy


@dataclass(frozen=True)
class Reduced:
    m: Market
    vega: np.ndarray
    eta: float
    vbar: float
    T: float
    points: int = 801
    rtol: float = 1e-8

    @cached_property
    def v(self) -> np.ndarray:
        return np.linspace(-self.vbar, self.vbar, self.points)

    @cached_property
    def _legs(self):
        v, tol = self.v, 1e-9 * self.vbar
        legs = []
        for f, z, V in zip(self.m.flow, self.m.z, self.vega, strict=True):
            for side in (1, -1):
                nxt = v + side * z * V
                legs.append((f, z, nxt, np.abs(nxt) <= self.vbar + tol))
        return legs

    def rhs(self, w: np.ndarray) -> np.ndarray:
        out = -self.m.gamma / 2 * (self.eta * self.v) ** 2
        for f, z, nxt, ok in self._legs:
            gap = (w - np.interp(nxt, self.v, w)) / z
            out += np.where(ok, z * f.H(gap), 0.0)
        return out

    @cached_property
    def _sol(self):
        return solve_ivp(
            lambda _, w: self.rhs(w),
            (0, self.T),
            np.zeros(self.points),
            method="DOP853",
            rtol=self.rtol,
            atol=self.rtol * self.vbar,
            dense_output=True,
        )

    def w(self, tau: float) -> np.ndarray:
        return self._sol.sol(min(tau, self.T))

    def value(self) -> float:
        return float(np.interp(0.0, self.v, self.w(self.T)))

    def quotes(self, tau: float, v) -> tuple[np.ndarray, np.ndarray]:
        w = self.w(tau)
        v = np.asarray(v, float)[..., None]
        here = np.interp(v, self.v, w)
        out = []
        for side in (1, -1):
            nxt = v + side * self.m.z * self.vega
            gap = (here - np.interp(nxt, self.v, w)) / self.m.z
            q = np.stack([f.delta(gap[..., i]) for i, f in enumerate(self.m.flow)], axis=-1)
            out.append(np.where(np.abs(nxt) <= self.vbar, q, np.inf))
        return out[0], out[1]

    def policy(self, tau: float, n: np.ndarray, vega: np.ndarray):
        return self.quotes(tau, (n * self.m.z * vega).sum(axis=-1))


TERMS = ("spread", "delta", "gamma", "vega", "cross", "cost", "other")


@dataclass(frozen=True)
class Session:
    pnl: np.ndarray
    penalty: np.ndarray
    terms: dict[str, np.ndarray]
    fills: np.ndarray
    n: np.ndarray
    vega: np.ndarray
    peak: np.ndarray
    peak_gamma: np.ndarray

    @property
    def objective(self) -> np.ndarray:
        return self.pnl - self.penalty


def simulate(
    c: Chain,
    m: Market,
    policy: Policy,
    *,
    T: float,
    dt: float,
    paths: int,
    vol: float,
    eta: float,
    eps: float = 0.0,
    rho: float = 0.0,
    h: float = 0.0,
    every: int = 1,
    vbar: float = math.inf,
    gbar: float = math.inf,
    n0=0.0,
    seed: int = 0,
) -> Session:
    if T > c.T.min():
        raise ValueError("horizon runs past the first expiry")
    rng = np.random.default_rng(seed)
    d, z = c.d, m.z
    n = np.broadcast_to(np.asarray(n0, float), (paths, d)).copy()
    S = np.full((paths, 1), c.S)
    x = np.zeros((paths, 1))
    e = np.zeros((paths, d))
    sigma = c.iv + x + e
    g = greeks(S, c.K, c.T, sigma, c.call)
    hedge = np.zeros(paths)
    cash = np.zeros(paths)
    wealth0 = (n * z * g.price).sum(axis=1)
    terms = {k: np.zeros(paths) for k in TERMS}
    penalty = np.zeros(paths)
    fills = np.zeros((paths, d), dtype=int)
    peak, peak_gamma = np.zeros(paths), np.zeros(paths)
    steps = round(T / dt)
    for s in range(steps):
        tau = T - s * dt
        bid, ask = policy(tau, n, g.vega)
        u = rng.random((paths, d, 2))
        vega = (n * z * g.vega).sum(axis=1)
        gam = (n * z * g.gamma).sum(axis=1) * S[:, 0] ** 2
        for i in range(d):
            dv, dg = z[i] * g.vega[:, i], z[i] * g.gamma[:, i] * S[:, 0] ** 2
            f = m.flow[i]
            for side, quote, k in ((1.0, bid[:, i], 0), (-1.0, ask[:, i], 1)):
                ok = (np.abs(vega + side * dv) <= vbar) & (np.abs(gam + side * dg) <= gbar)
                ok &= np.isfinite(quote)
                q = np.where(ok, quote, 0.0)
                hit = ok & (u[:, i, k] < -np.expm1(-f.rate(q) * dt))
                n[:, i] += side * hit
                fills[:, i] += hit
                terms["spread"] += np.where(hit, z[i] * q, 0.0)
                cash += np.where(hit, z[i] * (q - side * g.price[:, i]), 0.0)
                vega += side * hit * dv
                gam += side * hit * dg
        np.maximum(peak, np.abs(vega), out=peak)
        np.maximum(peak_gamma, np.abs(gam), out=peak_gamma)
        penalty += m.gamma / 2 * (eta * vega) ** 2 * dt
        held = (n * z * g.delta).sum(axis=1)
        if s % every == 0:
            trade = -held - hedge
            cost = h * np.abs(trade)
            cash -= trade * S[:, 0] + cost
            terms["cost"] -= cost
            hedge = -held
        zs, zx = rng.standard_normal((2, paths, 1))
        zx = rho * zs + math.sqrt(1 - rho * rho) * zx
        dx = eta * math.sqrt(dt) * zx
        de = eps * math.sqrt(dt) * rng.standard_normal((paths, d))
        sv = np.maximum(vol + x, 1e-4)
        S1 = S * np.exp(sv * math.sqrt(dt) * zs - sv * sv * dt / 2)
        x, e = x + dx, e + de
        sigma1 = np.maximum(c.iv + x + e, 1e-4)
        g1 = greeks(S1, c.K, c.T - (s + 1) * dt, sigma1, c.call)
        dS, ds = S1 - S, sigma1 - sigma
        nz = n * z
        exact = (nz * (g1.price - g.price)).sum(axis=1)
        parts = {
            "delta": (held + hedge) * dS[:, 0],
            "gamma": (nz * (g.theta * dt + g.gamma * dS**2 / 2)).sum(axis=1),
            "vega": (nz * g.vega * ds).sum(axis=1),
            "cross": (nz * (g.volga * ds**2 / 2 + g.vanna * dS * ds)).sum(axis=1),
        }
        for k, p in parts.items():
            terms[k] += p
        terms["other"] += exact + hedge * dS[:, 0] - sum(parts.values())
        S, sigma, g = S1, sigma1, g1
    wealth = cash + hedge * S[:, 0] + (n * z * g.price).sum(axis=1)
    return Session(
        pnl=wealth - wealth0,
        penalty=penalty,
        terms=terms,
        fills=fills,
        n=n,
        vega=(n * z * g.vega).sum(axis=1),
        peak=peak,
        peak_gamma=peak_gamma,
    )
