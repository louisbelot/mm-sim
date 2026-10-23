from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .accounting import Account, Fill
from .book import Book, Event, EventType, OrderType, Side, Tif
from .flow import ZeroIntelligence
from .markout import Tape


class Agent:
    def on_start(self, ctx: Context) -> None:
        pass

    def on_book(self, ctx: Context) -> None:
        ...

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        pass

    def on_timer(self, ctx: Context) -> None:
        pass


@dataclass(slots=True)
class LiveOrder:
    side: Side
    price: int
    qty: int
    ahead: int = 0


class Sample(NamedTuple):
    t: float
    mid: float
    position: int
    pnl: float


class Context:
    def __init__(self, sim: Simulator, agent: Agent, owner: int, account: Account) -> None:
        self._sim = sim
        self.agent = agent
        self.owner = owner
        self.account = account
        self.orders: dict[int, LiveOrder] = {}
        self.history: list[Sample] = []
        self._mark = math.nan

    @property
    def t(self) -> float:
        return self._sim.t

    @property
    def end(self) -> float:
        return self._sim.end

    @property
    def book(self) -> Book:
        return self._sim.book

    @property
    def position(self) -> int:
        return self.account.position

    @property
    def bid(self) -> int | None:
        return _best_of_others(self._sim.book, Side.BUY, self.owner)

    @property
    def ask(self) -> int | None:
        return _best_of_others(self._sim.book, Side.SELL, self.owner)

    @property
    def mid(self) -> float | None:
        bid, ask = self.bid, self.ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def mark(self) -> float:
        mid = self.mid
        if mid is not None:
            self._mark = mid
        return self._mark

    def pnl(self) -> float:
        return self.account.value(self.mark())

    def submit(
        self,
        side: Side,
        price: int,
        qty: int,
        *,
        kind: OrderType = OrderType.LIMIT,
        tif: Tif = Tif.GTC,
        post_only: bool = False,
    ) -> int:
        return self._sim._submit(self, side, price, qty, kind, tif, post_only)

    def buy(self, price: int, qty: int = 1, **kw) -> int:
        return self.submit(Side.BUY, price, qty, **kw)

    def sell(self, price: int, qty: int = 1, **kw) -> int:
        return self.submit(Side.SELL, price, qty, **kw)

    def ahead(self, oid: int) -> int:
        return self._sim.book.ahead(oid)

    def cancel(self, oid: int) -> None:
        if oid in self.orders:
            self._sim._act(self._sim.book.cancel, oid)

    def modify(self, oid: int, price: int, qty: int) -> None:
        if oid in self.orders:
            self._sim._act(self._sim.book.modify, oid, price, qty)

    def cancel_all(self) -> None:
        for oid in list(self.orders):
            self.cancel(oid)

    def set_timer(self, delay: float) -> None:
        self._sim._schedule(self, delay)

    def stop(self) -> None:
        self._sim.stopped = True

    def _record(self, t: float) -> None:
        mark = self.mark()
        self.history.append(Sample(t, mark, self.position, self.account.value(mark)))


class Simulator:
    def __init__(
        self,
        flow: ZeroIntelligence,
        agents: Sequence[Agent] = (),
        *,
        seed: int = 0,
        maker_fee: float = 0.0,
        taker_fee: float = 0.0,
        record_every: float | None = 1.0,
    ) -> None:
        self.book = Book()
        self.tape = Tape()
        self.flow = flow
        self.rng = random.Random(seed)
        self.t = 0.0
        self.end = 0.0
        self.stopped = False
        self.contexts = [
            Context(self, agent, flow.owner + 1 + i, Account(maker_fee, taker_fee))
            for i, agent in enumerate(agents)
        ]
        self._record_every = record_every
        self._next_sample = math.inf
        self._ids = itertools.count(1)
        self._agent_orders: dict[int, Context] = {}
        self._timers: list[tuple[float, int, Context]] = []
        self._timer_seq = itertools.count()
        self._pending: deque = deque()
        self._busy = False
        self._started = False
        self._pre_mid: float | None = None

    def seed(self, bid: int, ask: int) -> None:
        self.flow.seed(self.book, bid, ask, self._ids)
        self.tape.record(self.t, self.book.mid())

    def warm_up(self, duration: float) -> None:
        if self._started:
            raise RuntimeError("warm_up has to come before the first run")
        self._loop(self.t + duration)

    def run(self, duration: float) -> None:
        self.end = self.t + duration
        self.stopped = False
        if not self._started:
            self._started = True
            if self._record_every is not None:
                self._next_sample = self.t
            for ctx in self.contexts:
                self._queue(ctx.agent.on_start, ctx)
            self._drain()
        self._loop(self.end)

    def _loop(self, end: float) -> None:
        while not self.stopped:
            next_timer = self._timers[0][0] if self._timers else math.inf
            stop = min(next_timer, end)
            step = self.flow.step(self.book, self.rng, stop - self.t, self._ids)
            if step is not None:
                dt, cmd = step
                self._advance(self.t + dt)
                self._pre_mid = self.book.mid()
                self._dispatch(cmd(self.book))
                if self._started:
                    for ctx in self.contexts:
                        self._queue(ctx.agent.on_book, ctx)
                self._drain()
                continue
            self._advance(stop)
            if next_timer > end:
                return
            ctx = heapq.heappop(self._timers)[2]
            self._queue(ctx.agent.on_timer, ctx)
            self._drain()

    def _advance(self, t: float) -> None:
        while self._next_sample <= t:
            for ctx in self.contexts:
                ctx._record(self._next_sample)
            self._next_sample += self._record_every
        self.t = t

    def _submit(self, ctx, side, price, qty, kind, tif, post_only) -> int:
        oid = next(self._ids)
        self._agent_orders[oid] = ctx
        self._act(
            self.book.submit,
            oid,
            side,
            price,
            qty,
            owner=ctx.owner,
            kind=kind,
            tif=tif,
            post_only=post_only,
        )
        return oid

    def _act(self, command, *args, **kw) -> None:
        self._pre_mid = self.book.mid()
        self._dispatch(command(*args, **kw))
        self._drain()

    def _dispatch(self, events: list[Event]) -> None:
        for e in events:
            if e.type == EventType.TRADE:
                maker, taker = self._agent_orders.get(e.maker), self._agent_orders.get(e.id)
                if maker is not None:
                    self._fill(maker, e.maker, e.side.opposite, e, taker)
                if taker is not None:
                    self._fill(taker, e.id, e.side, e, maker)
                continue
            ctx = self._agent_orders.get(e.id)
            if ctx is None:
                continue
            match e.type:
                case EventType.ACCEPTED:
                    ctx.orders[e.id] = LiveOrder(e.side, e.price, e.qty, self._joined(e.id))
                case EventType.MODIFIED:
                    o = ctx.orders[e.id]
                    if e.price != o.price or e.qty > o.qty:
                        o.ahead = self._joined(e.id)
                    o.price, o.qty = e.price, e.qty
                case EventType.CANCELED:
                    self._reduce(ctx, e.id, e.qty)
                case EventType.REJECTED if e.id not in ctx.orders:
                    del self._agent_orders[e.id]
        self.tape.record(self.t, self.book.mid())

    def _fill(self, ctx: Context, oid: int, side: Side, e: Event, other: Context | None) -> None:
        maker = oid == e.maker
        ahead = left = None
        if maker:
            ahead = ctx.orders[oid].ahead
            lvl = self.book.level(side, e.price)
            left = 0 if lvl is None else lvl.qty
        who = self.flow.owner if other is None else other.owner
        fill = ctx.account.fill(
            self.t, oid, side, e.price, e.qty, maker, self._pre_mid, who, ahead=ahead, left=left
        )
        self._reduce(ctx, oid, e.qty)
        self._queue(ctx.agent.on_fill, ctx, fill)

    def _joined(self, oid: int) -> int:
        return self.book.ahead(oid) if oid in self.book else 0

    def _reduce(self, ctx: Context, oid: int, qty: int) -> None:
        o = ctx.orders[oid]
        o.qty -= qty
        if o.qty == 0:
            del ctx.orders[oid]
            del self._agent_orders[oid]

    def _schedule(self, ctx: Context, delay: float) -> None:
        heapq.heappush(self._timers, (self.t + delay, next(self._timer_seq), ctx))

    def _queue(self, fn, *args) -> None:
        self._pending.append((fn, args))

    def _drain(self) -> None:
        if self._busy:
            return
        self._busy = True
        while self._pending:
            fn, args = self._pending.popleft()
            fn(*args)
        self._busy = False


def _best_of_others(book: Book, side: Side, owner: int) -> int | None:
    for lvl in book.levels(side):
        if any(o.owner != owner for o in lvl.orders.values()):
            return lvl.price
    return None
