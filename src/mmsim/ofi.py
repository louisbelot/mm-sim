from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .micro import Tops


def events(top: Tops) -> np.ndarray:
    pb, pa, qb, qa = top.bid, top.ask, top.bq, top.aq
    return (
        (pb[1:] >= pb[:-1]) * qb[1:]
        - (pb[1:] <= pb[:-1]) * qb[:-1]
        - (pa[1:] <= pa[:-1]) * qa[1:]
        + (pa[1:] >= pa[:-1]) * qa[:-1]
    )


def flow(top: Tops, t) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(events(top))])
    return c[np.searchsorted(top.t[1:], t, side="right")]


def integral(top: Tops, v: np.ndarray, t) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(v[:-1] * np.diff(top.t))])
    i = top.index(t)
    return c[i] + v[i] * (np.asarray(t) - top.t[i])


class Bins(NamedTuple):
    t: np.ndarray
    ofi: np.ndarray
    dp: np.ndarray
    depth: np.ndarray


def bins(top: Tops, start: float, end: float, dt: float) -> Bins:
    edges = start + dt * np.arange(round((end - start) / dt) + 1)
    d = integral(top, (top.bq + top.aq) / 2, edges)
    return Bins(
        edges[:-1],
        np.diff(flow(top, edges)),
        np.diff(top.mid_at(edges)),
        np.diff(d) / dt,
    )


@dataclass(frozen=True)
class Fit:
    alpha: float
    beta: float
    r2: float
    t: float
    n: int

    def predict(self, x) -> np.ndarray:
        return self.alpha + self.beta * np.asarray(x)


def regress(x, y, groups=None) -> Fit:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    inv = np.linalg.inv(X.T @ X)
    if groups is None:
        meat = (X.T * e**2) @ X
    else:
        _, g = np.unique(groups, return_inverse=True)
        s = np.zeros((g.max() + 1, 2))
        np.add.at(s, g, X * e[:, None])
        meat = s.T @ s
    cov = inv @ meat @ inv
    tss = ((y - y.mean()) ** 2).sum()
    t = float(coef[1] / np.sqrt(cov[1, 1]))
    return Fit(float(coef[0]), float(coef[1]), 1 - float(e @ e) / tss, t, len(x))


def oos_r2(y, pred) -> float:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    return 1 - float(((y - pred) ** 2).sum()) / float((y**2).sum())


class Window(NamedTuple):
    start: float
    fit: Fit
    depth: float


def windows(top: Tops, start: float, end: float, window: float, dt: float) -> list[Window]:
    out = []
    for s in np.arange(start, end - window + 1e-9, window):
        b = bins(top, s, s + window, dt)
        out.append(Window(float(s), regress(b.ofi, b.dp), float(b.depth.mean())))
    return out


def scaling(ws: list[Window]) -> Fit:
    return regress(np.log([w.depth for w in ws]), np.log([w.fit.beta for w in ws]))
