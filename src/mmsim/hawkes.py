from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_SPAN = 300.0


@dataclass(frozen=True)
class Hawkes:
    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    @classmethod
    def make(cls, mu, alpha, beta) -> Hawkes:
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        d = len(mu)
        alpha = np.asarray(alpha, dtype=float).reshape(d, d)
        beta = np.broadcast_to(np.asarray(beta, dtype=float), (d, d)).copy()
        if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
            raise ValueError("need mu > 0, alpha >= 0 and beta > 0")
        return cls(mu, alpha, beta)

    @classmethod
    def symmetric(cls, rate: float, own: float, cross: float, beta: float) -> Hawkes:
        mu = rate * (1 - own - cross)
        return cls.make([mu, mu], [[own * beta, cross * beta], [cross * beta, own * beta]], beta)

    @property
    def d(self) -> int:
        return len(self.mu)

    @property
    def branching(self) -> np.ndarray:
        return self.alpha / self.beta

    def radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.branching))))

    def mean_rate(self) -> np.ndarray:
        if self.radius() >= 1:
            raise ValueError("not stationary: the branching matrix has spectral radius >= 1")
        return np.linalg.solve(np.eye(self.d) - self.branching, self.mu)

    def simulate(self, T: float, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        d, mu = self.d, self.mu.tolist()
        alpha, beta = self.alpha.tolist(), self.beta.tolist()
        s = [[0.0] * d for _ in range(d)]
        times, marks = [], []
        t = 0.0
        while True:
            lam = [mu[i] + sum(s[i]) for i in range(d)]
            bound = sum(lam)
            dt = rng.expovariate(bound)
            t += dt
            if t > T:
                break
            for i in range(d):
                for j in range(d):
                    s[i][j] *= math.exp(-beta[i][j] * dt)
            lam = [mu[i] + sum(s[i]) for i in range(d)]
            u = rng.random() * bound
            if u >= sum(lam):
                continue
            k = 0
            while u >= lam[k]:
                u -= lam[k]
                k += 1
            times.append(t)
            marks.append(k)
            for i in range(d):
                s[i][k] += alpha[i][k]
        return np.array(times), np.array(marks, dtype=int)

    def intensity(self, times: np.ndarray, marks: np.ndarray, at: np.ndarray) -> np.ndarray:
        lam = np.tile(self.mu, (len(at), 1))
        for j in range(self.d):
            tj = times[marks == j]
            for i in range(self.d):
                lam[:, i] += self.alpha[i, j] * decayed_at(tj, at, self.beta[i, j])[0]
        return lam

    def compensator(self, times: np.ndarray, marks: np.ndarray, at: np.ndarray) -> np.ndarray:
        at = np.asarray(at, dtype=float)
        out = np.outer(at, self.mu)
        for j in range(self.d):
            tj = times[marks == j]
            n = np.searchsorted(tj, at, side="left")
            for i in range(self.d):
                b = self.beta[i, j]
                out[:, i] += self.alpha[i, j] / b * (n - decayed_at(tj, at, b)[0])
        return out

    def loglik(self, times: np.ndarray, marks: np.ndarray, T: float) -> float:
        own = [times[marks == i] for i in range(self.d)]
        return _score(self.mu, self.alpha, self.beta, own, T)[0]

    def residuals(self, times: np.ndarray, marks: np.ndarray) -> list[np.ndarray]:
        big = self.compensator(times, marks, times)
        return [np.diff(big[marks == i, i], prepend=0.0) for i in range(self.d)]

    def stationary_state(self) -> np.ndarray:
        return self.branching * self.mean_rate()[None, :]

    def forecast(self, horizon: float = math.inf) -> np.ndarray:
        d = self.d
        S = np.kron(np.eye(d), np.ones(d))
        M = np.diag(self.alpha.ravel()) @ S[np.tile(np.arange(d), d)] - np.diag(self.beta.ravel())
        if math.isinf(horizon):
            return -S @ np.linalg.inv(M)
        return S @ np.linalg.solve(M, expm(M * horizon) - np.eye(d * d))


class Tracker:
    def __init__(self, model: Hawkes, horizon: float = math.inf) -> None:
        self.model = model
        self.t: float | None = None
        self.s = model.stationary_state()
        self._rest = self.s.copy()
        self._w = model.forecast(horizon)

    def _decayed(self, t: float) -> np.ndarray:
        if self.t is None:
            return self.s
        if t < self.t:
            raise ValueError("events have to arrive in time order")
        return self.s * np.exp(-self.model.beta * (t - self.t))

    def update(self, t: float, mark: int) -> None:
        self.s = self._decayed(t)
        self.s[:, mark] += self.model.alpha[:, mark]
        self.t = t

    def intensity(self, t: float) -> np.ndarray:
        return self.model.mu + self._decayed(t).sum(axis=1)

    def excess(self, t: float) -> np.ndarray:
        return self._w @ (self._decayed(t) - self._rest).ravel()


def decayed_at(t: np.ndarray, at, beta: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    at = np.asarray(at, dtype=float)
    r, s = np.zeros(len(at)), np.zeros(len(at))
    if len(t) == 0 or len(at) == 0:
        return r, s
    t0 = min(t[0], at[0])
    width = _SPAN / beta
    tb = np.floor((t - t0) / width)
    ab = np.floor((at - t0) / width)
    carry_r = carry_s = 0.0
    prev = None
    for b in np.unique(np.concatenate([tb, ab])):
        a = t0 + b * width
        if prev is not None:
            gap = a - prev
            f = math.exp(-beta * gap)
            carry_s = f * (carry_s + gap * carry_r)
            carry_r *= f
        lo, hi = np.searchsorted(tb, [b, b + 1])
        x = t[lo:hi] - a
        g = np.exp(beta * x)
        cr = np.concatenate([[0.0], np.cumsum(g)])
        cs = np.concatenate([[0.0], np.cumsum(-x * g)])
        alo, ahi = np.searchsorted(ab, [b, b + 1])
        y = at[alo:ahi] - a
        k = np.searchsorted(x, y, side="left")
        f = np.exp(-beta * y)
        r[alo:ahi] = (carry_r + cr[k]) * f
        s[alo:ahi] = (carry_s + y * carry_r + cs[k] + y * cr[k]) * f
        carry_r += cr[-1]
        carry_s += cs[-1]
        prev = a
    return r, s


def _score(mu, alpha, beta, own, T):
    d = len(mu)
    ll = -mu.sum() * T
    g_mu = np.full(d, -T)
    g_a, g_b = np.zeros((d, d)), np.zeros((d, d))
    for i in range(d):
        rb = [decayed_at(own[j], own[i], beta[i, j]) for j in range(d)]
        lam = mu[i] + sum(alpha[i, j] * rb[j][0] for j in range(d))
        inv = 1 / lam
        ll += np.log(lam).sum()
        g_mu[i] += inv.sum()
        for j in range(d):
            b, n = beta[i, j], len(own[j])
            (e,), (h,) = decayed_at(own[j], [T], b)
            G = (n - e) / b
            ll -= alpha[i, j] * G
            g_a[i, j] += rb[j][0] @ inv - G
            g_b[i, j] += -alpha[i, j] * (rb[j][1] @ inv) - alpha[i, j] * (h - G) / b
    return float(ll), g_mu, g_a, g_b


def fit(times: np.ndarray, marks: np.ndarray, T: float, d: int) -> Hawkes:
    own = [times[marks == i] for i in range(d)]
    n = np.array([len(x) for x in own], dtype=float)
    dd = d * d
    x0 = np.concatenate([np.log(n / T / 2), np.full(dd, np.log(0.5 / d)), np.zeros(dd)])

    def unpack(x):
        beta = np.exp(x[d + dd :]).reshape(d, d)
        return np.exp(x[:d]), np.exp(x[d : d + dd]).reshape(d, d) * beta, beta

    def f(x):
        mu, alpha, beta = unpack(x)
        ll, g_mu, g_a, g_b = _score(mu, alpha, beta, own, T)
        grad = np.concatenate(
            [mu * g_mu, (alpha * g_a).ravel(), (beta * g_b + alpha * g_a).ravel()]
        )
        return -ll, -grad

    res = minimize(f, x0, jac=True, method="L-BFGS-B")
    return Hawkes(*unpack(res.x))
