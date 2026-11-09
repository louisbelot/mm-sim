from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
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
