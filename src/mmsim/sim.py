from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from .accounting import Account, Fill
from .book import Book, Event, EventType, OrderType, Side, Tif
from .flow import ZeroIntelligence
from .markout import Tape

Command = Callable[[Book], list[Event]]


class Agent:
    def on_start(self, ctx: Context) -> None:
        pass

    def on_book(self, ctx: Context) -> None: ...

    def on_fill(self, ctx: Context, fill: Fill) -> None:
        pass

    def on_timer(self, ctx: Context) -> None:
        pass


@dataclass(frozen=True)
class Latency:
    entry: float = 0.0
    response: float = 0.0
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if min(self.entry, self.response, self.jitter) < 0:
            raise ValueError("latencies cannot be negative")


class Link:
    __slots__ = ("delay", "jitter", "rng", "last")

    def __init__(self, delay: float, jitter: float, rng: random.Random) -> None:
        self.delay = delay
        self.jitter = jitter
        self.rng = rng
        self.last = -math.inf

    @property
    def instant(self) -> bool:
        return self.delay == 0 and self.jitter == 0

    def arrival(self, t: float) -> float:
        d = self.delay
        if self.jitter:
            d += self.rng.expovariate(1 / self.jitter)
        self.last = max(self.last, t + d)
        return self.last


@dataclass(slots=True)
class LiveOrder:
    side: Side
    price: int
    qty: int
    ahead: int = 0
    acked: bool = False
    canceling: bool = False


@dataclass(slots=True)
class _Resting:
    ctx: Context
    price: int
    qty: int
    ahead: int


class Sample(NamedTuple):
    t: float
    mid: float
    position: int
    pnl: float


class Context:
    def __init__(
        self,
        sim: Simulator,
        agent: Agent,
        owner: int,
        account: Account,
        latency: Latency,
        rng: random.Random,
    ) -> None:
        self._sim = sim
        self.agent = agent
        self.owner = owner
        self.account = account
        self.latency = latency
        self.orders: dict[int, LiveOrder] = {}
        self.history: list[Sample] = []
        self._position = 0
        self._mark = math.nan
        self._entry = Link(latency.entry, latency.jitter, rng)
        self._response = Link(latency.response, latency.jitter, rng)

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
        return self._position

    @property
    def bid(self) -> int | None:
        return _best_of_others(self.book, Side.BUY, self.owner)

    @property
    def ask(self) -> int | None:
        return _best_of_others(self.book, Side.SELL, self.owner)

    @property
    def mid(self) -> float | None:
        bid, ask = self.bid, self.ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def mark(self) -> float:
        book = self._sim.book
        bid = _best_of_others(book, Side.BUY, self.owner)
        ask = _best_of_others(book, Side.SELL, self.owner)
        if bid is not None and ask is not None:
            self._mark = (bid + ask) / 2
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
        oid = next(self._sim._ids)
        self.orders[oid] = LiveOrder(side, 0 if kind == OrderType.MARKET else price, qty)
        owner = self.owner
        self._sim._send(
            self,
            lambda b: b.submit(
                oid, side, price, qty, owner=owner, kind=kind, tif=tif, post_only=post_only
            ),
        )
        return oid

    def buy(self, price: int, qty: int = 1, **kw) -> int:
        return self.submit(Side.BUY, price, qty, **kw)

    def sell(self, price: int, qty: int = 1, **kw) -> int:
        return self.submit(Side.SELL, price, qty, **kw)

    def ahead(self, oid: int) -> int:
        return self.book.ahead(oid)

    def cancel(self, oid: int) -> None:
        o = self.orders.get(oid)
        if o is None or o.canceling:
            return
        o.canceling = True
        self._sim._send(self, lambda b: b.cancel(oid))

    def modify(self, oid: int, price: int, qty: int) -> None:
        o = self.orders.get(oid)
        if o is not None and not o.canceling:
            self._sim._send(self, lambda b: b.modify(oid, price, qty))

    def cancel_all(self) -> None:
        for oid in list(self.orders):
            self.cancel(oid)

    def set_timer(self, delay: float) -> None:
        self._sim._at(self.t + delay, self._sim._queue, self.agent.on_timer, self)

    def stop(self) -> None:
        self._sim.stopped = True

    def _report(self, e: Event, info: Fill | int | None) -> None:
        match e.type:
            case EventType.TRADE:
                self._position += info.qty if info.side == Side.BUY else -info.qty
                self._reduce(info.oid, info.qty)
                self._sim._queue(self.agent.on_fill, self, info)
            case EventType.ACCEPTED:
                o = self.orders[e.id]
                o.acked, o.ahead = True, info
            case EventType.MODIFIED:
                o = self.orders[e.id]
                if info is not None:
                    o.ahead = info
                o.price, o.qty = e.price, e.qty
            case EventType.CANCELED:
                self._reduce(e.id, e.qty)
            case EventType.REJECTED:
                o = self.orders.get(e.id)
                if o is not None and not o.acked:
                    del self.orders[e.id]

    def _reduce(self, oid: int, qty: int) -> None:
        o = self.orders[oid]
        o.qty -= qty
        if o.qty == 0:
            del self.orders[oid]

    def _record(self, t: float) -> None:
        mark = self.mark()
        self.history.append(Sample(t, mark, self.account.position, self.account.value(mark)))


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
        latency: Mapping[Agent, Latency] | None = None,
    ) -> None:
        latency = latency or {}
        self.book = Book()
        self.tape = Tape()
        self.flow = flow
        self.rng = random.Random(seed)
        self._wire = random.Random(f"wire{seed}")
        self.t = 0.0
        self.end = 0.0
        self.stopped = False
        self.contexts = [
            Context(
                self,
                agent,
                flow.owner + 1 + i,
                Account(maker_fee, taker_fee),
                latency.get(agent, Latency()),
                self._wire,
            )
            for i, agent in enumerate(agents)
        ]
        self._record_every = record_every
        self._next_sample = math.inf
        self._ids = itertools.count(1)
        self._agent_orders: dict[int, _Resting] = {}
        self._events: list = []
        self._seq = itertools.count()
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
            next_event = self._events[0][0] if self._events else math.inf
            stop = min(next_event, end)
            step = self.flow.step(self.book, self.rng, stop - self.t, self._ids)
            if step is not None:
                dt, cmd = step
                self._advance(self.t + dt)
                self._execute(None, cmd)
                continue
            self._advance(stop)
            if next_event > end:
                return
            _, _, fn, args = heapq.heappop(self._events)
            fn(*args)
            self._drain()

    def _advance(self, t: float) -> None:
        while self._next_sample <= t:
            for ctx in self.contexts:
                ctx._record(self._next_sample)
            self._next_sample += self._record_every
        self.t = t

    def _at(self, t: float, fn, *args) -> None:
        heapq.heappush(self._events, (t, next(self._seq), fn, args))

    def _send(self, ctx: Context, cmd: Command) -> None:
        if ctx._entry.instant:
            self._execute(ctx, cmd)
        else:
            self._at(ctx._entry.arrival(self.t), self._execute, ctx, cmd)

    def _execute(self, sender: Context | None, cmd: Command) -> None:
        self._pre_mid = self.book.mid()
        for e in cmd(self.book):
            if e.type == EventType.TRADE:
                maker = self._agent_orders.get(e.maker)
                taker = self._agent_orders.get(e.id)
                if maker is not None:
                    self._fill(maker, e.maker, e.side.opposite, e, taker)
                if taker is not None:
                    self._fill(taker, e.id, e.side, e, maker)
            elif sender is not None:
                self._book_event(sender, e)
        self.tape.record(self.t, self.book.mid())
        if self._started and sender is None:
            for ctx in self.contexts:
                self._queue(ctx.agent.on_book, ctx)
        self._drain()

    def _book_event(self, ctx: Context, e: Event) -> None:
        info = None
        match e.type:
            case EventType.ACCEPTED:
                info = self._joined(e.id)
                self._agent_orders[e.id] = _Resting(ctx, e.price, e.qty, info)
            case EventType.MODIFIED:
                r = self._agent_orders[e.id]
                if e.price != r.price or e.qty > r.qty:
                    info = r.ahead = self._joined(e.id)
                r.price, r.qty = e.price, e.qty
            case EventType.CANCELED:
                self._retire(e.id, e.qty)
        self._deliver(ctx, e, info)

    def _fill(self, r: _Resting, oid: int, side: Side, e: Event, other: _Resting | None) -> None:
        ctx = r.ctx
        maker = oid == e.maker
        ahead = left = None
        if maker:
            ahead = r.ahead
            lvl = self.book.level(side, e.price)
            left = 0 if lvl is None else lvl.qty
        o = ctx.orders.get(oid)
        fill = ctx.account.fill(
            self.t,
            oid,
            side,
            e.price,
            e.qty,
            maker,
            self._pre_mid,
            self.flow.owner if other is None else other.ctx.owner,
            ahead=ahead,
            left=left,
            stale=o is not None and o.canceling,
        )
        self._retire(oid, e.qty)
        self._deliver(ctx, e, fill)

    def _deliver(self, ctx: Context, e: Event, info) -> None:
        if ctx._response.instant:
            ctx._report(e, info)
        else:
            self._at(ctx._response.arrival(self.t), ctx._report, e, info)

    def _joined(self, oid: int) -> int:
        return self.book.ahead(oid) if oid in self.book else 0

    def _retire(self, oid: int, qty: int) -> None:
        r = self._agent_orders[oid]
        r.qty -= qty
        if r.qty == 0:
            del self._agent_orders[oid]

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
