from __future__ import annotations

import math

import numpy as np

from . import avellaneda, glft
from .accounting import Fill
from .book import Side
from .hawkes import Hawkes, Tracker
from .markout import Tape
from .sim import Agent, Context, Print


class FlowLean:
    def __init__(self, model: Hawkes, kappa: float, horizon: float = math.inf) -> None:
        self.kappa = kappa
        self.tracker = Tracker(model, horizon)

    @classmethod
    def fit(
        cls, model: Hawkes, times: np.ndarray, marks: np.ndarray, tape: Tape, horizon: float
    ) -> FlowLean:
        tr = Tracker(model, horizon)
        x = np.empty(len(times))
        for n, (t, m) in enumerate(zip(times, marks, strict=True)):
            tr.update(t, m)
            e = tr.excess(t)
            x[n] = e[Side.BUY] - e[Side.SELL]
        ok = times + horizon <= tape.t[-1]
        y = tape.at(times[ok] + horizon) - tape.at(times[ok])
        return cls(model, float(x[ok] @ y / (x[ok] @ x[ok])), horizon)

    @classmethod
    def from_impact(
        cls,
        model: Hawkes,
        times: np.ndarray,
        marks: np.ndarray,
        tape: Tape,
        horizon: float = math.inf,
    ) -> FlowLean:
        sign = np.where(marks == Side.BUY, 1.0, -1.0)
        moved = tape.at(times) - tape.before(times)
        return cls(model, float(sign @ moved / len(times)), horizon)

    def observe(self, p: Print) -> None:
        self.tracker.update(p.t, p.side)

    def shift(self, t: float) -> float:
        e = self.tracker.excess(t)
        return self.kappa * float(e[Side.BUY] - e[Side.SELL])


class Quoter(Agent):
    def __init__(
        self,
        size: int = 1,
        max_position: int | None = None,
        keep: int = 0,
        signal: FlowLean | None = None,
    ) -> None:
        self.size = size
        self.max_position = max_position
        self.keep = keep
        self.signal = signal
        self._oids: dict[Side, int | None] = {Side.BUY: None, Side.SELL: None}

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        raise NotImplementedError

    def fair(self, ctx: Context) -> float | None:
        mid = ctx.mid
        if mid is None or self.signal is None:
            return mid
        return mid + self.signal.shift(ctx.t)

    def on_trade(self, ctx: Context, p: Print) -> None:
        if self.signal is not None:
            self.signal.observe(p)

    def on_start(self, ctx: Context) -> None:
        self.requote(ctx)

    def on_book(self, ctx: Context) -> None:
        self.requote(ctx)

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        self.requote(ctx)

    def requote(self, ctx: Context) -> None:
        bid, ask = self.targets(ctx)
        if self.max_position is not None:
            if ctx.position + self.size > self.max_position:
                bid = None
            if ctx.position - self.size < -self.max_position:
                ask = None
        self._quote(ctx, Side.BUY, bid)
        self._quote(ctx, Side.SELL, ask)

    def _quote(self, ctx: Context, side: Side, px: int | None) -> None:
        oid = self._oids[side]
        live = ctx.orders.get(oid)
        if live is not None and px is not None:
            behind = px - live.price if side == Side.BUY else live.price - px
            if 0 <= behind <= self.keep:
                return
        if live is not None:
            ctx.cancel(oid)
        self._oids[side] = None if px is None else ctx.submit(side, px, self.size, post_only=True)


class SymmetricQuoter(Quoter):
    def __init__(
        self,
        half_spread: float = 0.5,
        size: int = 1,
        max_position: int | None = None,
        keep: int = 0,
        signal: FlowLean | None = None,
    ) -> None:
        super().__init__(size, max_position, keep, signal)
        self.half_spread = half_spread

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        mid = self.fair(ctx)
        if mid is None:
            return None, None
        bid = min(math.floor(mid - self.half_spread), ctx.ask - 1)
        ask = max(math.ceil(mid + self.half_spread), ctx.bid + 1)
        return bid, ask


class AvellanedaStoikov(Quoter):
    def __init__(
        self,
        gamma: float,
        sigma: float,
        k: float,
        *,
        horizon: float | None = None,
        size: int = 1,
        max_position: int | None = None,
        keep: int = 0,
        signal: FlowLean | None = None,
    ) -> None:
        super().__init__(size, max_position, keep, signal)
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.horizon = horizon

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        s = self.fair(ctx)
        if s is None:
            return None, None
        tau = ctx.end - ctx.t
        if self.horizon is not None:
            tau = min(tau, self.horizon)
        bid, ask = avellaneda.quotes(s, ctx.position, tau, self.gamma, self.sigma, self.k)
        return min(math.floor(bid), ctx.ask - 1), max(math.ceil(ask), ctx.bid + 1)


class GLFT(Quoter):
    def __init__(
        self,
        model: glft.Model,
        *,
        size: int = 1,
        keep: int = 0,
        signal: FlowLean | None = None,
    ) -> None:
        super().__init__(size, model.Q * size, keep, signal)
        self.model = model
        self.bids, self.asks = model.quotes(math.inf)

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        s = self.fair(ctx)
        if s is None:
            return None, None
        Q = self.model.Q
        i = Q + min(max(round(ctx.position / self.size), -Q), Q)
        db, da = self.bids[i], self.asks[i]
        bid = None if math.isnan(db) else min(math.floor(s - db), ctx.ask - 1)
        ask = None if math.isnan(da) else max(math.ceil(s + da), ctx.bid + 1)
        return bid, ask
