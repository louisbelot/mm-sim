"""Avellaneda and Stoikov (2008), "High-frequency trading in a limit order book".

A market maker with utility -exp(-gamma * wealth) at time T quotes around a mid
that follows ds = sigma dW. A quote delta away from the mid is hit by market
orders at Poisson rate A exp(-k delta). The paper's approximate solution is

    reservation price   r = s - q gamma sigma^2 (T - t)                         (29)
    total spread        gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k)    (30)

with the bid and ask at r -/+ half the spread. r is the exact indifference price
when the inventory is frozen. The spread comes from linearising the order
arrival terms of the HJB equation; its second term is the delta that maximises
the expected utility gain of one fill, A exp(-k delta) (1 - exp(-gamma delta)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reservation_price(s, q, tau, gamma: float, sigma: float):
    return s - q * gamma * sigma**2 * tau


def spread(tau, gamma: float, sigma: float, k: float):
    return gamma * sigma**2 * tau + 2 / gamma * np.log1p(gamma / k)


def quotes(s, q, tau, gamma: float, sigma: float, k: float):
    """(bid, ask) around the reservation price. Works on scalars or arrays."""
    r = reservation_price(s, q, tau, gamma, sigma)
    half = spread(tau, gamma, sigma, k) / 2
    return r - half, r + half


@dataclass(frozen=True)
class PaperRun:
    profit: np.ndarray  # x_T + q_T s_T, one per path
    final_q: np.ndarray
    spread: float  # quoted spread averaged over the run
    t: np.ndarray  # the first path, for plotting
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
    """The experiment of the paper's section 3.3, defaults included.

    Each step the agent quotes, each quote is hit with probability
    A exp(-k delta) dt (capped at 1), and then the mid moves by +-sigma sqrt(dt).
    "symmetric" quotes the inventory strategy's spread averaged over [0, T]
    around the mid. The random draws do not depend on the strategy, so two runs
    with the same seed see the same mid paths and the same order arrivals.
    """
    if strategy not in ("inventory", "symmetric"):
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = np.random.default_rng(seed)
    n = round(T / dt)
    tau = T - dt * np.arange(n)
    avg = float(spread(T / 2, gamma, sigma, k))  # the spread is linear in tau

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
