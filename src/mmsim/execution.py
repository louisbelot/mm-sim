from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .accounting import Fill
from .book import OrderType, Side
from .flow import ZeroIntelligence
from .sim import Agent, Context, Print, Simulator
from .tournament import Entrants, Result, result


@dataclass(frozen=True)
class AlmgrenChriss:
    sigma: float
    eta: float
    gamma: float = 0.0
    eps: float = 0.0

    def kappa(self, lam: float, tau: float) -> float:
        eta = self.eta - 0.5 * self.gamma * tau
        return math.acosh(1.0 + 0.5 * lam * self.sigma**2 * tau**2 / eta) / tau

    def holdings(self, X: float, T: float, N: int, lam: float) -> np.ndarray:
        t = np.linspace(0.0, T, N + 1)
        if lam == 0.0:
            return X * (1.0 - t / T)
        k = self.kappa(lam, T / N)
        return X * np.exp(-k * t) * np.expm1(-2.0 * k * (T - t)) / math.expm1(-2.0 * k * T)

    def cost(self, x: np.ndarray, tau: float) -> tuple[float, float]:
        n = -np.diff(x)
        E = self.gamma * (x[1:] @ n) + self.eps * np.abs(n).sum() + self.eta / tau * (n @ n)
        return float(E), float(self.sigma**2 * tau * (x[1:] @ x[1:]))


@dataclass(frozen=True)
class ObizhaevaWang:
    kappa: float
    rho: float
    lam: float = 0.0

    def response(self, s) -> np.ndarray:
        return self.lam + self.kappa * np.exp(-self.rho * np.asarray(s, dtype=float))

    def trades(self, X: float, T: float, N: int) -> np.ndarray:
        a = math.exp(-self.rho * T / N)
        x0 = X / (2.0 + (N - 1) * (1.0 - a))
        x = np.full(N + 1, (1.0 - a) * x0)
        x[0] = x[-1] = x0
        return x

    def cost(self, x: np.ndarray, t: np.ndarray) -> float:
        K = np.exp(-self.rho * np.abs(np.subtract.outer(t, t)))
        return float(0.5 * self.lam * x.sum() ** 2 + 0.5 * self.kappa * (x @ K @ x))

    def optimum(self, X: float, T: float) -> float:
        return 0.5 * self.lam * X**2 + self.kappa * X**2 / (self.rho * T + 2.0)

    def steady(self, X: float, T: float) -> float:
        rt = self.rho * T
        return 0.5 * self.lam * X**2 + self.kappa * X**2 * (rt + math.expm1(-rt)) / rt**2


def fit_response(lags: np.ndarray, impact: np.ndarray, se: np.ndarray) -> ObizhaevaWang:
    def resid(p):
        return (ObizhaevaWang(p[1], p[2], p[0]).response(lags) - impact) / se

    start = (impact[-1], max(impact[0] - impact[-1], 0.0), 1.0 / lags[len(lags) // 2])
    lam, kappa, rho = least_squares(resid, start, bounds=((-np.inf, 0.0, 0.0), np.inf)).x
    return ObizhaevaWang(float(kappa), float(rho), float(lam))


def targets(x: np.ndarray) -> np.ndarray:
    return np.rint(x[0] - x).astype(int)


class Executor(Agent):
    def __init__(self, side: Side, times: Sequence[float], done: Sequence[int]) -> None:
        self.side = side
        self.times = list(times)
        self.done = list(done)
        self.arrival = math.nan
        self._t0 = 0.0
        self._k = 0

    def on_start(self, ctx: Context) -> None:
        self._t0 = ctx.t
        self._arm(ctx)

    def on_timer(self, ctx: Context) -> None:
        if self._k == 0:
            self.arrival = ctx.mark()
        qty = self.done[self._k] - abs(ctx.position)
        if qty > 0:
            ctx.submit(self.side, 0, qty, kind=OrderType.MARKET)
        self._k += 1
        self._arm(ctx)

    def _arm(self, ctx: Context) -> None:
        if self._k < len(self.times):
            ctx.set_timer(max(self._t0 + self.times[self._k] - ctx.t, 0.0))


def scatter(X: int, T: float, N: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= N <= X:
        raise ValueError("need between 1 and X slices")
    times = np.sort(rng.uniform(0.0, T, N))
    done = np.append(np.sort(rng.choice(np.arange(1, X), N - 1, replace=False)), X)
    return times, done


class Pov(Agent):
    def __init__(self, side: Side, qty: int, rate: float, start: float = 0.0) -> None:
        if not 0.0 < rate < 1.0:
            raise ValueError("participation has to be between 0 and 1")
        self.side = side
        self.qty = qty
        self.rate = rate
        self.start = start
        self.arrival = math.nan
        self._seen = -1

    def on_start(self, ctx: Context) -> None:
        ctx.set_timer(self.start)

    def on_timer(self, ctx: Context) -> None:
        self.arrival = ctx.mark()
        self._seen = 0

    def on_trade(self, ctx: Context, p: Print) -> None:
        if self._seen < 0:
            return
        self._seen += p.qty
        own = abs(ctx.position)
        want = min(self.qty, math.floor(self.rate / (1.0 - self.rate) * (self._seen - own)))
        if want > own:
            ctx.submit(self.side, 0, want - own, kind=OrderType.MARKET)


class Passive(Agent):
    def __init__(
        self, side: Side, times: Sequence[float], done: Sequence[int], slack: int = 0
    ) -> None:
        self.side = side
        self.times = list(times)
        self.done = list(done)
        self.slack = slack
        self.arrival = math.nan
        self._oid: int | None = None
        self._t0 = 0.0
        self._k = 0

    def on_start(self, ctx: Context) -> None:
        self._t0 = ctx.t
        self._arm(ctx)

    def on_timer(self, ctx: Context) -> None:
        if math.isnan(self.arrival):
            self.arrival = ctx.mark()
        if self._oid is not None:
            ctx.cancel(self._oid)
            self._oid = None
        last = self._k == len(self.times) - 1
        behind = self.done[self._k] - abs(ctx.position) - (0 if last else self.slack)
        if behind > 0:
            ctx.submit(self.side, 0, behind, kind=OrderType.MARKET)
        if not last:
            self._post(ctx, self.done[self._k + 1] - abs(ctx.position))
        elif abs(ctx.position) < self.done[-1]:
            ctx.set_timer(0.1)
            return
        self._k += 1
        self._arm(ctx)

    def on_book(self, ctx: Context) -> None:
        o = None if self._oid is None else ctx.orders.get(self._oid)
        if o is None:
            return
        touch = self._touch(ctx)
        if touch is None:
            return
        if touch > o.price if self.side == Side.BUY else touch < o.price:
            qty = o.qty
            ctx.cancel(self._oid)
            self._post(ctx, qty)

    def _touch(self, ctx: Context) -> int | None:
        return ctx.bid if self.side == Side.BUY else ctx.ask

    def _post(self, ctx: Context, qty: int) -> None:
        touch = self._touch(ctx)
        self._oid = None if qty <= 0 or touch is None else ctx.submit(self.side, touch, qty)

    def _arm(self, ctx: Context) -> None:
        if self._k < len(self.times):
            ctx.set_timer(max(self._t0 + self.times[self._k] - ctx.t, 0.0))


def shortfall(fills: Sequence[Fill], arrival: float) -> float:
    return sum((f.price - arrival) * (f.qty if f.side == Side.BUY else -f.qty) for f in fills)


def trial(
    seed: int,
    makers: Entrants,
    executor: Executor | None = None,
    *,
    flow: Callable[[], ZeroIntelligence],
    warm: float = 30.0,
    minutes: float = 60.0,
    horizon: float = 1.0,
) -> tuple[dict[str, Result], float]:
    agents = [make() for make in makers.values()]
    if executor is not None:
        agents.append(executor)
    sim = Simulator(flow(), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(warm)
    start = sim.t
    sim.run(minutes)
    res = {
        name: result(ctx, sim, start, horizon)
        for name, ctx in zip(makers, sim.contexts, strict=False)
    }
    if executor is None:
        return res, 0.0
    return res, shortfall(sim.contexts[-1].account.fills, executor.arrival)


def impulse(
    flow: Callable[[], ZeroIntelligence],
    qty: int,
    lags: Sequence[float],
    seed: int,
    *,
    warm: float = 30.0,
) -> np.ndarray:
    quotes = []
    for n in (qty, 0):
        sim = Simulator(flow(), [Executor(Side.BUY, [0.0], [n])], seed=seed, record_every=None)
        sim.seed(10_000, 10_001)
        sim.warm_up(warm)
        row = []
        for dt in np.diff(lags, prepend=0.0):
            sim.run(dt)
            row.append((sim.book.best_bid(), sim.book.best_ask()))
        quotes.append(row)
    hit, base = np.array(quotes, dtype=float)
    return hit - base
