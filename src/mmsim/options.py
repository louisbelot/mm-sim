from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from scipy.special import ndtr


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
