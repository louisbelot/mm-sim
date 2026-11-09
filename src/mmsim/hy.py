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


def triangle(k: int) -> np.ndarray:
    j = np.arange(1, k)
    return np.minimum(j, k - j) / k


def width(n: int, theta: float) -> int:
    return max(2, round(theta * np.sqrt(n)))


def preaverage(x, k: int) -> np.ndarray:
    return np.convolve(np.diff(np.asarray(x, dtype=float)), triangle(k), "valid")[:-1]


def preaveraged(tx, x, ty, y, *, theta: float = 1.0, lag: float = 0.0) -> float:
    tx, ty = np.asarray(tx, dtype=float), np.asarray(ty, dtype=float) - lag
    kx, ky = width(len(tx) - 1, theta), width(len(ty) - 1, theta)
    bx, by = preaverage(x, kx), preaverage(y, ky)
    lo = np.searchsorted(ty[ky : ky + len(by)], tx[: len(bx)], side="right")
    hi = np.searchsorted(ty[: len(by)], tx[kx : kx + len(bx)], side="left")
    cs = np.concatenate([[0.0], np.cumsum(by)])
    total = bx @ (cs[np.maximum(hi, lo)] - cs[lo])
    return float(total / (triangle(kx).sum() * triangle(ky).sum()))


def preaveraged_variance(x, *, theta: float = 1.0) -> float:
    dx = np.diff(np.asarray(x, dtype=float))
    n = len(dx)
    k = width(n, theta)
    g = triangle(k)
    b = preaverage(x, k)
    psi1 = np.sum(np.diff(np.r_[0.0, g, 0.0]) ** 2)
    psi2 = g @ g
    return float(n / (len(b) * psi2) * (b @ b) - psi1 / (2 * psi2) * (dx @ dx))


def psd(C, floor: float = 0.0) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(C, dtype=float))
    return (V * np.maximum(w, floor * w.max())) @ V.T


def matrix(series, window: tuple[float, float], theta: float | None = None) -> np.ndarray:
    cut = [between(t, x, *window) for t, x in series]
    n = len(cut)
    C = np.empty((n, n))
    for i in range(n):
        if theta is None:
            C[i, i] = np.sum(np.diff(cut[i][1]) ** 2)
        else:
            C[i, i] = preaveraged_variance(cut[i][1], theta=theta)
        for j in range(i):
            if theta is None:
                C[i, j] = covariance(*cut[i], *cut[j])
            else:
                C[i, j] = preaveraged(*cut[i], *cut[j], theta=theta)
            C[j, i] = C[i, j]
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
