from __future__ import annotations

import numpy as np


def sample(t, x, grid) -> np.ndarray:
    i = np.searchsorted(t, grid, side="right") - 1
    if np.any(i < 0):
        raise ValueError("grid starts before the first observation")
    return np.asarray(x)[i]


def covariance(tx, x, ty, y, lag: float = 0.0) -> float:
    tx, ty = np.asarray(tx, dtype=float), np.asarray(ty, dtype=float) - lag
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lo = np.maximum(np.searchsorted(ty, tx[:-1], side="right"), 1)
    hi = np.minimum(np.searchsorted(ty, tx[1:], side="left"), len(ty) - 1)
    ok = hi >= lo
    dy = np.where(ok, y[hi] - y[np.maximum(lo - 1, 0)], 0.0)
    return float(np.diff(x) @ dy)


def matrix(series, window: tuple[float, float]) -> np.ndarray:
    cut = [between(t, x, *window) for t, x in series]
    n = len(cut)
    C = np.empty((n, n))
    for i in range(n):
        C[i, i] = np.sum(np.diff(cut[i][1]) ** 2)
        for j in range(i):
            C[i, j] = C[j, i] = covariance(*cut[i], *cut[j])
    return C


def lead_lag(tx, x, ty, y, lags) -> tuple[float, np.ndarray]:
    c = np.array([covariance(tx, x, ty, y, lag) for lag in lags])
    return float(np.asarray(lags)[np.argmax(np.abs(c))]), c


def between(t, x, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    lo, hi = np.searchsorted(t, [start, end], side="right")
    keep = slice(max(lo - 1, 0), hi)
    tt = t[keep].copy()
    tt[0] = max(tt[0], start)
    return tt, x[keep]
