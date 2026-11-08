from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class Model:
    lam: float
    zeta: float
    eta: float
    eps: float
    width: float
    spread: float = 1.0
    impact: float = 0.0
    phi: float = 0.0
    varphi: float = 0.0
    Q: int = 10
    n: int = 100

    @property
    def alpha(self) -> np.ndarray:
        return np.linspace(-self.width, self.width, 2 * self.n + 1)

    @property
    def q(self) -> np.ndarray:
        return np.arange(-self.Q, self.Q + 1)


@dataclass(frozen=True)
class Solution:
    alpha: np.ndarray
    h: np.ndarray
    bid: np.ndarray
    ask: np.ndarray

    @property
    def Q(self) -> int:
        return (len(self.h) - 1) // 2

    def node(self, a: float) -> int:
        da = self.alpha[1] - self.alpha[0]
        return min(max(round((a - self.alpha[0]) / da), 0), len(self.alpha) - 1)

    def value(self, a: float, q: int) -> float:
        return float(np.interp(a, self.alpha, self.h[q + self.Q]))

    def post(self, a: float, q: int) -> tuple[bool, bool]:
        i, j = min(max(q, -self.Q), self.Q) + self.Q, self.node(a)
        return bool(self.bid[i, j]), bool(self.ask[i, j])


def solve(m: Model, T: float, policy: tuple[np.ndarray, np.ndarray] | None = None) -> Solution:
    a = m.alpha
    da = a[1] - a[0]
    q = m.q[:, None]
    steps = max(1, math.ceil(T * (2 * m.lam + m.zeta) / 0.1))
    dt = T / steps
    s = m.eps / da
    ab = _implicit(a, m.zeta, m.eta, dt)
    run = a * q - m.phi * q**2
    h = -(m.spread / 2 * np.abs(q) + m.varphi * q**2) * np.ones_like(a)
    edge = m.spread / 2 - m.impact
    bid = ask = np.zeros(h.shape, dtype=bool)
    for _ in range(steps):
        up, dn = _shift(h, s), _shift(h, -s)
        ga = np.full(h.shape, -np.inf)
        gb = np.full(h.shape, -np.inf)
        ga[1:] = edge + up[:-1] - up[1:]
        gb[:-1] = edge + dn[1:] - dn[:-1]
        if policy is None:
            bid, ask = gb > 0, ga > 0
        else:
            bid, ask = policy[0] & np.isfinite(gb), policy[1] & np.isfinite(ga)
        jumps = m.lam * (up + np.where(ask, ga, 0.0) + dn + np.where(bid, gb, 0.0) - 2 * h)
        h = solve_banded((1, 1), ab, (h + dt * (run + jumps)).T).T
    return Solution(a, h, bid, ask)


def _implicit(a: np.ndarray, zeta: float, eta: float, dt: float) -> np.ndarray:
    da = a[1] - a[0]
    v = -zeta * a / da
    up, lo = np.maximum(v, 0.0), np.maximum(-v, 0.0)
    diff = np.full(len(a), eta**2 / (2 * da**2))
    diff[[0, -1]] = 0.0
    ab = np.zeros((3, len(a)))
    ab[0, 1:] = -dt * (up + diff)[:-1]
    ab[1] = 1 + dt * (up + lo + 2 * diff)
    ab[2, :-1] = -dt * (lo + diff)[1:]
    return ab


def _shift(h: np.ndarray, s: float) -> np.ndarray:
    n = h.shape[1]
    x = np.arange(n) + s
    lo = np.clip(np.floor(x).astype(np.int64), 0, n - 2)
    w = x - lo
    return h[:, lo] * (1 - w) + h[:, lo + 1] * w


def always(m: Model) -> tuple[np.ndarray, np.ndarray]:
    on = np.ones((2 * m.Q + 1, 2 * m.n + 1), dtype=bool)
    return on, on


def simulate(
    m: Model,
    sol: Solution,
    T: float,
    a0: float = 0.0,
    q0: int = 0,
    n_paths: int = 20_000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.full(n_paths, a0)
    q = np.full(n_paths, q0)
    t = np.zeros(n_paths)
    out = np.zeros(n_paths)
    live = np.ones(n_paths, dtype=bool)
    z = m.zeta
    da = sol.alpha[1] - sol.alpha[0]
    while live.any():
        k = np.flatnonzero(live)
        tau = np.minimum(rng.exponential(1 / (2 * m.lam), len(k)), T - t[k])
        decay = np.exp(-z * tau)
        out[k] += q[k] * a[k] * (1 - decay) / z - m.phi * q[k] ** 2 * tau
        sd = m.eta * np.sqrt((1 - decay**2) / (2 * z))
        a[k] = a[k] * decay + sd * rng.standard_normal(len(k))
        t[k] += tau
        hit = t[k] < T
        live[k[~hit]] = False
        k = k[hit]
        lifted = rng.random(len(k)) < 0.5
        j = np.clip(np.rint((a[k] - sol.alpha[0]) / da), 0, 2 * m.n).astype(np.int64)
        i = q[k] + m.Q
        fill = np.where(lifted, sol.ask[i, j], sol.bid[i, j])
        side = np.where(lifted, -1, 1)
        q[k] += np.where(fill, side, 0)
        out[k] += fill * m.spread / 2 - side * q[k] * m.impact
        a[k] += np.where(lifted, m.eps, -m.eps)
    return out - m.spread / 2 * np.abs(q) - m.varphi * q**2

