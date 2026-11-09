from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp

from mmsim.glosten import binary_quotes


class LMSR:
    def __init__(self, b: float, n: int, p0: float = 0.5):
        self.b = b
        self.q = np.full(n, b * math.log(p0 / (1 - p0)))

    @property
    def price(self) -> np.ndarray:
        return expit(self.q / self.b)

    def cost(self, q) -> np.ndarray:
        return self.b * np.logaddexp(q / self.b, 0.0)

    def quote(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cost(self.q)
        return self.cost(self.q + 1) - c, self.cost(self.q - 1) - c + 1

    def buy(self, yes, mask=True) -> np.ndarray:
        up, down = self.quote()
        self.q = self.q + np.where(mask, np.where(yes, 1.0, -1.0), 0.0)
        return np.where(mask, np.where(yes, up, down), 0.0)


class LSLMSR:
    def __init__(self, v: float, n: int, p0: float = 0.5, x: float = 1.0):
        self.alpha = v / (2 * math.log(2))
        self.x = x
        t = brentq(lambda t: self._norm(np.array([[1 + t, 1 - t]]))[0] - p0, -0.999, 0.999)
        self.q = np.tile([x * (1 + t), x * (1 - t)], (n, 1))

    def cost(self, q) -> np.ndarray:
        b = self.alpha * q.sum(axis=-1)
        return b * logsumexp(q / b[..., None], axis=-1)

    def prices(self, q=None) -> np.ndarray:
        q = self.q if q is None else q
        Q = q.sum(axis=-1, keepdims=True)
        z = q / (self.alpha * Q)
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        S = e.sum(axis=-1, keepdims=True)
        mix = (q * e).sum(axis=-1, keepdims=True)
        return self.alpha * (np.log(S) + z.max(axis=-1, keepdims=True)) + (e * Q - mix) / (Q * S)

    def _norm(self, q) -> np.ndarray:
        p = self.prices(q)
        return p[:, 0] / p.sum(axis=-1)

    @property
    def price(self) -> np.ndarray:
        return self._norm(self.q)

    def quote(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cost(self.q)
        return self.cost(self.q + [1.0, 0.0]) - c, self.cost(self.q + [0.0, 1.0]) - c

    def buy(self, yes, mask=True) -> np.ndarray:
        up, down = self.quote()
        step = np.where(np.asarray(yes)[..., None], [1.0, 0.0], [0.0, 1.0])
        self.q = self.q + np.where(np.asarray(mask)[..., None], step, 0.0)
        return np.where(mask, np.where(yes, up, down), 0.0)


class FPMM:
    def __init__(self, F: float, n: int, p0: float = 0.5):
        self.F = F
        self.y = np.full(n, F * math.sqrt((1 - p0) / p0))
        self.n = np.full(n, F * math.sqrt(p0 / (1 - p0)))

    @property
    def price(self) -> np.ndarray:
        return self.n / (self.y + self.n)

    @staticmethod
    def _cash(out, other) -> np.ndarray:
        s = out + other - 1
        return (np.sqrt(s * s + 4 * other) - s) / 2

    def quote(self) -> tuple[np.ndarray, np.ndarray]:
        return self._cash(self.y, self.n), self._cash(self.n, self.y)

    def buy(self, yes, mask=True) -> np.ndarray:
        up, down = self.quote()
        c = np.where(yes, up, down)
        y = np.where(yes, self.y + c - 1, self.y + c)
        n = np.where(yes, self.n + c, self.n + c - 1)
        self.y, self.n = np.where(mask, y, self.y), np.where(mask, n, self.n)
        return np.where(mask, c, 0.0)


class Quoter:
    def __init__(self, alpha: float, n: int, p0: float = 0.5):
        self.alpha = alpha
        self.mu = np.full(n, p0)

    @property
    def price(self) -> np.ndarray:
        return self.mu

    def quote(self) -> tuple[np.ndarray, np.ndarray]:
        bid, ask = binary_quotes(self.mu, self.alpha, 0.0, 1.0)
        return ask, 1 - bid

    def buy(self, yes, mask=True) -> np.ndarray:
        bid, ask = binary_quotes(self.mu, self.alpha, 0.0, 1.0)
        self.mu = np.where(mask, np.where(yes, ask, bid), self.mu)
        return np.where(mask, np.where(yes, ask, 1 - bid), 0.0)


def belief(p, signal, rho: float) -> np.ndarray:
    a = np.where(signal, rho, 1 - rho) * p
    return a / (a + np.where(signal, 1 - rho, rho) * (1 - p))


@dataclass(frozen=True)
class Markets:
    error: np.ndarray
    maker: np.ndarray
    informed: np.ndarray
    noise: np.ndarray
    price: np.ndarray
    outcome: np.ndarray
    bayes: np.ndarray
    volume: np.ndarray

    def brier(self, p=None) -> float:
        p = self.price if p is None else p
        return float(np.mean((p - self.outcome) ** 2))

    def log_score(self, p=None) -> float:
        p = self.price if p is None else p
        return float(np.mean(np.log(np.where(self.outcome, p, 1 - p))))


def run(
    make,
    alpha: float,
    trades: int,
    markets: int,
    *,
    rho: float = 1.0,
    p0: float = 0.5,
    seed: int = 0,
) -> Markets:
    rng = np.random.default_rng(seed)
    mm = make(markets)
    omega = rng.random(markets) < p0
    error = np.empty(trades + 1)
    error[0] = np.abs(mm.price - omega).mean()
    pnl = {"maker": np.zeros(markets), "informed": np.zeros(markets), "noise": np.zeros(markets)}
    llr = np.zeros(markets)
    volume = np.zeros(markets)
    for t in range(trades):
        informed = rng.random(markets) < alpha
        signal = omega if rho == 1 else np.where(rng.random(markets) < rho, omega, ~omega)
        llr += np.where(informed, np.where(signal, 1.0, -1.0), 0.0)
        up, down = mm.quote()
        mine = belief(mm.price, signal, rho)
        wants = np.where(signal, mine > up, 1 - mine > down)
        yes = np.where(informed, signal, rng.random(markets) < 0.5)
        go = ~informed | wants
        paid = mm.buy(yes, go)
        gain = np.where(go, (yes == omega) - paid, 0.0)
        pnl["maker"] -= gain
        pnl["informed"] += np.where(informed, gain, 0.0)
        pnl["noise"] += np.where(informed, 0.0, gain)
        volume += go
        error[t + 1] = np.abs(mm.price - omega).mean()
    if rho < 1:
        bayes = expit(math.log(p0 / (1 - p0)) + llr * math.log(rho / (1 - rho)))
    else:
        bayes = np.where(llr != 0, omega.astype(float), p0)
    return Markets(
        error, pnl["maker"], pnl["informed"], pnl["noise"], mm.price, omega, bayes, volume
    )
