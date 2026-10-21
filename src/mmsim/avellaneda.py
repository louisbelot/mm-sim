from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reservation_price(s, q, tau, gamma: float, sigma: float):
    return s - q * gamma * sigma**2 * tau


def spread(tau, gamma: float, sigma: float, k: float):
    return gamma * sigma**2 * tau + 2 / gamma * np.log1p(gamma / k)


def quotes(s, q, tau, gamma: float, sigma: float, k: float):
    r = reservation_price(s, q, tau, gamma, sigma)
    half = spread(tau, gamma, sigma, k) / 2
    return r - half, r + half


@dataclass(frozen=True)
class PaperRun:
    profit: np.ndarray
    final_q: np.ndarray
    spread: float
    t: np.ndarray
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray


def simulate(
    gamma: float,
    strategy: str = "inventory",
    *,
    n_paths: int = 1000,
    s0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    dt: float = 0.005,
    k: float = 1.5,
    A: float = 140.0,
    seed: int = 0,
) -> PaperRun:
    if strategy not in ("inventory", "symmetric"):
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = np.random.default_rng(seed)
    n = round(T / dt)
    tau = T - dt * np.arange(n)
    avg = float(spread(T / 2, gamma, sigma, k))

    s = np.full(n_paths, s0)
    q = np.zeros(n_paths)
    x = np.zeros(n_paths)
    path = np.empty((3, n))
    for j in range(n):
        if strategy == "inventory":
            bid, ask = quotes(s, q, tau[j], gamma, sigma, k)
        else:
            bid, ask = s - avg / 2, s + avg / 2
        path[:, j] = s[0], bid[0], ask[0]
        bought = rng.random(n_paths) < np.minimum(1.0, A * np.exp(-k * (s - bid)) * dt)
        sold = rng.random(n_paths) < np.minimum(1.0, A * np.exp(-k * (ask - s)) * dt)
        q += bought.astype(float) - sold
        x += ask * sold - bid * bought
        s += sigma * np.sqrt(dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)

    return PaperRun(x + q * s, q, avg, dt * np.arange(n), *path)
