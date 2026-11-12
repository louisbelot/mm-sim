from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .accounting import Fill
from .book import Book, Event, EventType, OrderType, Reason, Side, Tif
from .bots import Quoter
from .execution import Pov, shortfall
from .flow import ZeroIntelligence
from .sim import Agent, Context, Print, Simulator


@dataclass(frozen=True)
class Rule:
    band: float | None = None
    limit: float | None = None
    trip: float | None = None
    pause: float = 5.0
    window: float = 5.0

    def __post_init__(self) -> None:
        if self.limit is not None and self.band is None:
            raise ValueError("a limit state needs a band")


class Guard(Book):
    def __init__(self, rule: Rule | None = None) -> None:
        super().__init__()
        self.rule = rule or Rule()
        self.clock: Callable[[], float] = lambda: 0.0
        self.ref = math.nan
        self.trades: list[tuple[float, int, int]] = []
        self.pauses: list[tuple[float, float]] = []
        self._window: deque[tuple[float, int]] = deque()
        self._sum = 0
        self._since: float | None = None
        self._paused = False
        self._until = -math.inf

    def halted(self) -> bool:
        self._tick(self.clock())
        return self._paused

    def bands(self) -> tuple[int, int] | None:
        if self.rule.band is None or math.isnan(self.ref):
            return None
        return math.ceil(self.ref - self.rule.band), math.floor(self.ref + self.rule.band)

    def submit(
        self,
        oid: int,
        side: Side,
        price: int,
        qty: int,
        *,
        owner: int = 0,
        kind: OrderType = OrderType.LIMIT,
        tif: Tif = Tif.GTC,
        post_only: bool = False,
    ) -> list[Event]:
        now = self._enter()
        if self._paused and self._marketable(side, price, kind):
            return [Event(EventType.REJECTED, oid, side, reason=Reason.HALTED)]
        b = self.bands()
        if b is not None and not post_only:
            lo, hi = b
            if kind == OrderType.MARKET:
                kind, price = OrderType.LIMIT, lo if side == Side.SELL else hi
                tif = Tif.FOK if tif == Tif.FOK else Tif.IOC
            else:
                price = max(price, lo) if side == Side.SELL else min(price, hi)
        out = super().submit(
            oid, side, price, qty, owner=owner, kind=kind, tif=tif, post_only=post_only
        )
        self._leave(now, out)
        return out

    def modify(self, oid: int, price: int, qty: int) -> list[Event]:
        now = self._enter()
        o = self._orders.get(oid)
        if o is not None:
            if self._paused and self._marketable(o.side, price, OrderType.LIMIT):
                return [Event(EventType.REJECTED, oid, o.side, reason=Reason.HALTED)]
            b = self.bands()
            if b is not None:
                price = max(price, b[0]) if o.side == Side.SELL else min(price, b[1])
        out = super().modify(oid, price, qty)
        self._leave(now, out)
        return out

    def cancel(self, oid: int) -> list[Event]:
        now = self._enter()
        out = super().cancel(oid)
        self._leave(now, out)
        return out

    def _enter(self) -> float:
        now = self.clock()
        self._tick(now)
        if self._since is None and not self._paused:
            w = self._window
            while w and w[0][0] < now - self.rule.window:
                self._sum -= w.popleft()[1]
            if w:
                self.ref = self._sum / len(w)
            elif math.isnan(self.ref):
                mid = self.mid()
                if mid is not None:
                    self.ref = mid
        return now

    def _leave(self, now: float, out: list[Event]) -> None:
        r = self.rule
        prints = [e for e in out if e.type == EventType.TRADE]
        for e in prints:
            self.trades.append((now, e.price, e.qty))
            self._window.append((now, e.price))
            self._sum += e.price
        if self._paused or math.isnan(self.ref):
            return
        if r.trip is not None and any(abs(e.price - self.ref) >= r.trip for e in prints):
            self._halt(now)
            return
        if r.limit is not None:
            lo, hi = self.bands()
            bid, ask = self.best_bid(), self.best_ask()
            stuck = bid is None or bid < lo or ask is None or ask > hi
            if not stuck:
                self._since = None
            elif self._since is None:
                self._since = now

    def _tick(self, now: float) -> None:
        if self._since is not None and now >= self._since + self.rule.limit:
            self._halt(self._since + self.rule.limit)
        if self._paused and now >= self._until:
            self._paused = False
            self.ref = math.nan
            self._window.clear()
            self._sum = 0

    def _halt(self, t: float) -> None:
        self._since = None
        self._paused = True
        self._until = t + self.rule.pause
        self.pauses.append((t, self._until))

    def _marketable(self, side: Side, price: int, kind: OrderType) -> bool:
        if kind == OrderType.MARKET:
            return True
        if side == Side.BUY:
            ask = self.best_ask()
            return ask is not None and price >= ask
        bid = self.best_bid()
        return bid is not None and price <= bid


class Stops(Agent):
    def __init__(self, levels: Sequence[tuple[int, int]], retry: float = 0.05) -> None:
        self.levels = levels
        self.retry = retry
        self.triggered = 0
        self._pending: list[tuple[int, int]] = []
        self._armed = False

    def on_start(self, ctx: Context) -> None:
        ref = round(ctx.mark())
        self._pending = sorted((ref - d, q) for d, q in self.levels)

    def on_trade(self, ctx: Context, p: Print) -> None:
        while self._pending and self._pending[-1][0] >= p.price:
            self.triggered += self._pending.pop()[1]
        self._sell(ctx)

    def on_timer(self, ctx: Context) -> None:
        self._armed = False
        self._sell(ctx)

    def _sell(self, ctx: Context) -> None:
        owed = self.triggered + ctx.position
        if owed <= 0:
            return
        ctx.submit(Side.SELL, 0, owed, kind=OrderType.MARKET)
        if self.triggered + ctx.position > 0 and not self._armed:
            self._armed = True
            ctx.set_timer(self.retry)


class Value(Agent):
    def __init__(self, every: float, kappa: float, cap: int, seed: int = 0) -> None:
        self.every = every
        self.kappa = kappa
        self.cap = cap
        self.rng = random.Random(seed)
        self.value = math.nan

    def on_start(self, ctx: Context) -> None:
        self.value = ctx.mark()
        ctx.set_timer(self.rng.expovariate(1 / self.every))

    def on_timer(self, ctx: Context) -> None:
        ctx.set_timer(self.rng.expovariate(1 / self.every))
        gap = self.value - ctx.mark()
        q = min(self.cap, math.floor(self.kappa * abs(gap)))
        if q > 0:
            ctx.submit(Side.BUY if gap > 0 else Side.SELL, 0, q, kind=OrderType.MARKET)


class Maker(Quoter):
    def __init__(
        self,
        half: float = 1.0,
        size: int = 1,
        skew: float = 0.0,
        soft: int | None = None,
        hard: int | None = None,
        flat: int = 0,
        loss: float | None = None,
        rest: float = 5.0,
        cool: float = 0.0,
    ) -> None:
        super().__init__(size, soft)
        self.half = half
        self.skew = skew
        self.hard = hard
        self.flat = flat
        self.loss = loss
        self.rest = rest
        self.cool = cool
        self.kills: list[float] = []
        self._cool_until = -math.inf
        self._cool_side = Side.BUY
        self._peak = -math.inf
        self._out = False
        self._back = math.inf

    def targets(self, ctx: Context) -> tuple[int | None, int | None]:
        mid = ctx.mid
        if mid is None:
            return None, None
        c = mid - self.skew * ctx.position
        bid = min(math.floor(c - self.half), ctx.ask - 1)
        ask = max(math.ceil(c + self.half), ctx.bid + 1)
        if ctx.t < self._cool_until:
            if self._cool_side == Side.BUY:
                bid = None
            else:
                ask = None
        return bid, ask

    def on_book(self, ctx: Context) -> None:
        self._step(ctx)

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        self._step(ctx)

    def on_timer(self, ctx: Context) -> None:
        if self._out and ctx.t >= self._back:
            self._out = False
            self._peak = -math.inf
        self._step(ctx)

    def _step(self, ctx: Context) -> None:
        if self._out:
            self._unwind(ctx, 0)
            return
        if self.loss is not None:
            v = ctx.pnl()
            self._peak = max(self._peak, v)
            if v < self._peak - self.loss:
                self._out = True
                self.kills.append(ctx.t)
                ctx.cancel_all()
                self._unwind(ctx, 0)
                self._back = ctx.t + self.rest
                ctx.set_timer(self.rest)
                return
        q = ctx.position
        if self.hard is not None and abs(q) >= self.hard:
            self._unwind(ctx, self.flat)
            self._cool_until = ctx.t + self.cool
            self._cool_side = Side.BUY if q > 0 else Side.SELL
            if self.cool > 0:
                ctx.set_timer(self.cool)
        self.requote(ctx)

    def _unwind(self, ctx: Context, target: int) -> None:
        q = ctx.position
        if abs(q) <= target:
            return
        side = Side.SELL if q > 0 else Side.BUY
        ctx.submit(side, 0, abs(q) - target, kind=OrderType.MARKET)


@dataclass(frozen=True, slots=True)
class Crash:
    drop: float
    dip: float
    far: int
    sold: int
    cost: float
    makers: float
    dumped: int
    kills: int
    pauses: int
    halted: float
    recover: float
    volume: int


@dataclass
class Run:
    sim: Simulator
    makers: list[Maker]
    seller: Pov | None
    t0: float
    arrival: float
    start: float

    @property
    def guard(self) -> Guard:
        return self.sim.book

    def summary(self, far: float = 20.0, near: float = 2.0) -> Crash:
        sim, arrival = self.sim, self.arrival
        t1 = self.t0 + self.start
        trades = [(t, p, q) for t, p, q in self.guard.trades if t >= t1]
        low = min((p for _, p, _ in trades), default=arrival)
        tt, mids = np.asarray(sim.tape.t), np.asarray(sim.tape.mid)
        late = tt >= t1
        tt, mids = tt[late], mids[late]
        dip, recover = 0.0, math.nan
        if len(mids):
            k = int(np.argmin(mids))
            dip = arrival - float(mids[k])
            back = np.nonzero(mids[k:] >= arrival - near)[0]
            if len(back):
                recover = float(tt[k + back[0]] - tt[k])
        n = len(self.makers)
        ctxs = sim.contexts
        mark = float(sim.tape.at(sim.end))
        sold, cost = 0, math.nan
        if self.seller is not None:
            fills = ctxs[n].account.fills
            sold = sum(f.qty for f in fills)
            if sold:
                cost = shortfall(fills, self.seller.arrival) / sold
        pauses = [(a, b) for a, b in self.guard.pauses if b > self.t0]
        return Crash(
            drop=arrival - low,
            dip=dip,
            far=sum(q for _, p, q in trades if p <= arrival - far),
            sold=sold,
            cost=cost,
            makers=sum(c.account.value(mark) for c in ctxs[:n]),
            dumped=sum(f.qty for c in ctxs[:n] for f in c.account.fills if not f.maker),
            kills=sum(len(m.kills) for m in self.makers),
            pauses=sum(1 for a, _ in pauses if a >= self.t0),
            halted=sum(min(b, sim.end) - max(a, self.t0) for a, b in pauses),
            recover=recover,
            volume=sum(q for _, _, q in trades),
        )


def run(
    seed: int,
    rule: Rule,
    *,
    flow: Callable[[], ZeroIntelligence],
    makers: Callable[[], Sequence[Maker]],
    value: Callable[[int], Sequence[Value]] = lambda seed: (),
    qty: int = 0,
    rate: float = 0.2,
    stops: Sequence[tuple[int, int]] = (),
    warm: float = 30.0,
    minutes: float = 60.0,
    start: float = 5.0,
    record: float | None = None,
) -> Run:
    mm = list(makers())
    seller = Pov(Side.SELL, qty, rate, start) if qty else None
    agents: list[Agent] = [*mm]
    if seller is not None:
        agents.append(seller)
    if stops:
        agents.append(Stops(stops))
    agents.extend(value(seed))
    sim = Simulator(flow(), agents, seed=seed, book=lambda: Guard(rule), record_every=record)
    sim.book.clock = lambda: sim.t
    sim.seed(10_000, 10_001)
    sim.warm_up(warm)
    t0, arrival = sim.t, sim.tape.mid[-1]
    sim.run(minutes)
    return Run(sim, mm, seller, t0, arrival, start)


def session(seed: int, rule: Rule, *, far: float = 20.0, near: float = 2.0, **kw) -> Crash:
    return run(seed, rule, **kw).summary(far, near)
