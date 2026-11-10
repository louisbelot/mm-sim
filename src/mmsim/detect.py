from __future__ import annotations

import numpy as np


def stationary(Q: np.ndarray) -> np.ndarray:
    n = len(Q)
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


class Filter:
    def __init__(self, Q, rates, p0=None, t: float | None = None) -> None:
        Q = np.asarray(Q, dtype=float)
        self.rates = np.asarray(rates, dtype=float)
        self._w, self._V = np.linalg.eig(Q - np.diag(self.rates.sum(axis=1)))
        self._U = np.linalg.inv(self._V)
        self._top = self._w.real.max()
        self.p = stationary(Q) if p0 is None else np.asarray(p0, dtype=float)
        self.t = t

    def at(self, t: float) -> np.ndarray:
        if self.t is None or t == self.t:
            return self.p
        p = (((self.p @ self._V) * np.exp((self._w - self._top) * (t - self.t))) @ self._U).real
        return p / p.sum()

    def observe(self, t: float, mark: int, n: int = 1) -> None:
        p = self.at(t) * self.rates[:, mark] ** n
        self.p = p / p.sum()
        self.t = t


def one_sided(mu: float, r: float, on: float, off: float, t: float | None = None) -> Filter:
    Q = [[-on, on / 2, on / 2], [off, -off, 0.0], [off, 0.0, -off]]
    return Filter(Q, [[mu, mu], [mu + r, mu], [mu, mu + r]], t=t)


def toxic(mu: float, r: float, on: float, off: float, t: float | None = None) -> Filter:
    return Filter([[-on, on], [off, -off]], [[2 * mu], [2 * mu + r]], t=t)
