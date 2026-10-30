import random

import numpy as np
import pytest

from mmsim import qr
from mmsim.book import Book, EventType, OrderType, Side
from mmsim.bots import SymmetricQuoter
from mmsim.itch import NS_PER_MINUTE, Msg
from mmsim.qr import CANCEL, LIMIT, MARKET, Gap, Params, QueueReactive, invariant
from mmsim.sim import Latency, Simulator

B, S = Side.BUY, Side.SELL


def sec(s, kind, ref, side=B, px=0, qty=0):
    return Msg(round(s * NS_PER_MINUTE / 60), kind, ref, side, px * 100, qty)


def pre_open():
    return [
        sec(1, "A", 1, B, 99, 100),
        sec(2, "A", 2, B, 99, 300),
        sec(3, "A", 3, S, 100, 200),
        sec(4, "A", 4, B, 98, 50),
        sec(5, "A", 6, S, 101, 500),
    ]


def test_hand_worked_fit():
    msgs = [
        *pre_open(),
        sec(70, "A", 5, B, 99, 100),
        sec(75, "A", 8, B, 98, 40),
        sec(80, "D", 2),
        sec(90, "E", 3, qty=200),
        sec(100, "A", 7, B, 100, 100),
        sec(110, "D", 6),
    ]
    f = qr.fit(msgs, 1.0, 3.0, depth=2, bins=10)
    assert f.aes.tolist() == [240, 40]
    lvl, n, kind, qty, regime, pos = f.events.T
    assert lvl.tolist() == [0, 1, 0, 0, 0, 0]
    assert kind.tolist() == [LIMIT, LIMIT, CANCEL, MARKET, LIMIT, CANCEL]
    assert qty.tolist() == [100, 40, 300, -200, 100, -500]
    assert n.tolist() == [2, 2, 3, 1, 0, 3]
    assert pos.tolist() == [-1, -1, 500, -1, -1, -1]
    assert regime[4] == qr.STAYED_MARKET + qr.BASE
    assert f.tau[0].sum() == pytest.approx(2 * 50 / 60)
    assert f.tau[1].sum() == pytest.approx(2 * 50 / 60)
    assert f.counts.sum() == 6


def test_gap_names_the_side_that_left():
    g = Gap()
    g.update(100, 101, False)
    assert not g.wide
    g.update(100, 102, True)
    assert g.side == S and g.market and g.up
    assert g.regime(S) == qr.LEFT_MARKET and g.regime(B) == qr.STAYED_MARKET
    g.update(101, 102, False)
    assert not g.wide and g.up
    g.update(100, 102, False)
    assert g.side == B and not g.market and not g.up
    assert g.regime(B) == qr.LEFT_CANCEL and g.regime(S) == qr.STAYED_CANCEL


def flat(K, N, birth, cancel, market, aes=1, sizes=None, **kw):
    table = tuple(
        (float(birth(n)), float(cancel(n)) if n else 0.0, float(market(n)) if n else 0.0)
        for n in range(N + 1)
    )
    rates = tuple(tuple(table for _ in range(qr.REGIMES)) for _ in range(K))
    if sizes is None:
        sizes = {(k, i): (aes,) for k in (LIMIT, CANCEL, MARKET) for i in range(K)}
    queues = tuple((3 * aes,) for _ in range(K))
    return Params(rates, sizes, (aes,) * K, (2 * aes, 4 * aes), queues, False, **kw)


def test_invariant_closed_form():
    birth = np.array([2.0, 2.0, 2.0, 0.0])
    death = np.array([0.0, 1.0, 1.0, 1.0])
    assert invariant(birth, death) == pytest.approx(np.array([1, 2, 4, 8]) / 15)
    assert invariant(np.array([1.0, 3.0, 0.0]), np.array([0, 2.0, 6.0])) == pytest.approx(
        np.array([1, 0.5, 0.25]) / 1.75
    )


def occupancy(p, minutes, seed):
    book, rng, ids = Book(), random.Random(seed), iter(range(1, 10**9))
    flow = QueueReactive(p, seed=seed)
    flow.seed(book, 100, 101, ids)
    hist = np.zeros(p.N + 1)
    gone = 0
    t = 0.0
    while True:
        lvl = book.level(B, 100)
        q = lvl.qty if lvl else 0
        step = flow.step(book, rng, minutes - t, ids)
        if step is None:
            return hist / hist.sum(), gone / minutes
        t += step[0]
        hist[q] += step[0]
        step[1](book)
        gone += q > 0 and book.level(B, 100) is None


def test_pinned_queue_is_the_birth_death_chain():
    N = 12

    def birth(n):
        return 3.0 / (1 + 0.2 * n) if n < N else 0.0

    def cancel(n):
        return 0.5 * n**0.5

    def market(n):
        return 2.0 / n

    hist, flux = occupancy(flat(1, N, birth, cancel, market), 40_000.0, 11)
    b = np.array([birth(n) for n in range(N + 1)])
    d = np.array([0.0] + [cancel(n) + market(n) for n in range(1, N + 1)])
    pi = invariant(b, d)
    assert hist == pytest.approx(pi, abs=0.01)
    assert flux == pytest.approx(pi[1] * d[1], rel=0.03)


def test_gap_closes_at_the_measured_odds():
    N = 3
    rates = []
    for r in range(qr.REGIMES):
        base = r % qr.BASE
        rows = []
        for n in range(N + 1):
            if base >= qr.STAYED_CANCEL:
                lam = {qr.LEFT_MARKET: 300.0, qr.STAYED_MARKET: 900.0}.get(base, 0.0) * (n == 0)
            else:
                lam = 40.0 if 0 < n < N else 0.0
            rows.append((lam, 0.0, 30.0 if n else 0.0))
        rates.append(tuple(rows))
    sizes = {(k, i): (1,) for k in (LIMIT, CANCEL, MARKET) for i in range(2)}
    p = Params((tuple(rates), tuple(rates)), sizes, (1, 1), (1, 2), ((2,), (2,)), False)
    book, rng, ids = Book(), random.Random(5), iter(range(1, 10**9))
    flow = QueueReactive(p)
    flow.seed(book, 100, 101, ids)
    t = 0.0
    continued = closed = 0
    gone = None
    while t < 1500.0:
        dt, cmd = flow.step(book, rng, 1e9, ids)
        t += dt
        was_bid, was_ask = book.best_bid(), book.best_ask()
        cmd(book)
        bid, ask = book.best_bid(), book.best_ask()
        if gone is None and ask - bid > 1 and was_ask - was_bid == 1:
            gone = S if ask > was_ask else B
        elif gone is not None and ask - bid == 1:
            continued += bid > was_bid if gone == S else ask < was_ask
            closed += 1
            gone = None
    assert closed > 500
    assert abs(continued / closed - 0.75) < 4 * np.sqrt(0.75 * 0.25 / closed)


def test_commands_replay_identically():
    sizes = {(k, i): (40, 100, 300) for k in (LIMIT, CANCEL, MARKET) for i in range(3)}
    p = flat(3, 8, lambda n: 30.0, lambda n: 12.0, lambda n: 6.0, aes=100, sizes=sizes)
    a, b, rng, ids = Book(), Book(), random.Random(9), iter(range(1, 10**9))
    flow = QueueReactive(p)
    flow.seed(a, 100, 101, ids)
    for o in a.orders(B) + a.orders(S):
        b.submit(o.id, o.side, o.price, o.qty)
    for _ in range(4000):
        _, cmd = flow.step(a, rng, 1e9, ids)
        got = [e for e in cmd(a) if e.type == EventType.TRADE]
        again = [e for e in cmd(b) if e.type == EventType.TRADE]
        assert got == again
    assert a.audit() == "" and b.audit() == ""
    assert a.orders(B) == b.orders(B) and a.orders(S) == b.orders(S)


def test_conditioned_sizes_only_empty_a_queue_on_the_atom():
    p = flat(1, 4, lambda n: 1.0, lambda n: 1.0, lambda n: 1.0)
    flow = QueueReactive(Params(p.rates, p.sizes, p.aes, p.edges, p.queues, True))
    assert flow._take(300, 200) == 199
    assert flow._take(50, 200) == 50
    assert flow._take(-120, 200) == 200
    assert flow._take(-500, 200) == 500
    assert flow._take(70, 1) == 1
    assert QueueReactive(p)._take(300, 200) == 300


@pytest.mark.parametrize(
    "back, qty, left",
    [
        (1.0, 100, {1: 100, 2: 100}),
        (0.0, 100, {2: 100, 3: 100}),
        (0.5, 100, {1: 100, 3: 100}),
        (1.0, 250, {1: 50}),
        (0.5, 250, {1: 50}),
        (0.0, 40, {1: 60, 2: 100, 3: 100}),
    ],
)
def test_cancel_starts_at_the_measured_position(back, qty, left):
    book = Book()
    for i in (1, 2, 3):
        book.submit(i, B, 100, 100)
    p = flat(1, 4, lambda n: 1.0, lambda n: 1.0, lambda n: 1.0, back=((back,),))
    QueueReactive(p)._pull(book.level(B, 100), 0, qty, random.Random(0))(book)
    assert {o.id: o.qty for o in book.orders(B)} == left
    assert book.audit() == ""


def test_fit_keeps_where_cancels_came_from():
    msgs = [
        *pre_open(),
        sec(61, "A", 10, B, 99, 100),
        sec(62, "A", 11, B, 99, 100),
        sec(63, "D", 11),
        sec(64, "D", 10),
        sec(65, "D", 1),
        sec(70, "A", 12, B, 98, 10),
        sec(71, "A", 13, S, 102, 10),
    ]
    f = qr.fit(msgs, 1.0, 3.0, depth=2, bins=10)
    assert f.params(min_events=1).back[0] == (1.0, 1.0, 0.0)


def test_rates_reflect_at_the_top_and_fall_back_when_thin():
    n = np.array([[30.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    pooled_n = np.array([[100.0, 0, 0], [0, 50, 50], [10, 40, 0]])
    rates = qr._filled(n, np.ones(3), pooled_n, np.array([2.0, 5.0, 10.0]), 30)
    assert rates == ((30.0, 0.0, 0.0), (0.0, 10.0, 10.0), (0.0, 4.0, 0.0))


def test_unit_sizes_are_one_aes():
    msgs = [*pre_open(), sec(70, "A", 5, B, 99, 100), sec(75, "A", 8, B, 98, 40)]
    p = qr.fit(msgs, 1.0, 3.0, depth=2).params(sizes=qr.UNIT, min_events=1)
    assert p.sizes[LIMIT, 0] == (100,) and p.sizes[LIMIT, 1] == (40,)
    assert not p.conditioned


def flow_params():
    sizes = {(k, i): (5, 10, 30) for k in (LIMIT, CANCEL, MARKET) for i in range(4)}
    return flat(4, 6, lambda n: 40.0, lambda n: 15.0, lambda n: 30.0, aes=10, sizes=sizes)


def test_a_bot_quotes_on_the_flow():
    bot = SymmetricQuoter(0.5, size=5, max_position=50)
    sim = Simulator(QueueReactive(flow_params(), seed=2), [bot], seed=2)
    sim.seed(1000, 1001)
    sim.warm_up(2.0)
    sim.run(20.0)
    assert len(sim.contexts[0].account.fills) > 20
    assert sim.book.audit() == ""
    assert len(sim.tape.t) > 20


def test_flow_never_cancels_a_bot_order():
    book = Book()
    p = flat(1, 4, lambda n: 0.0, lambda n: 60.0, lambda n: 0.0, aes=10, back=((1.0,),))
    flow = QueueReactive(p, seed=1)
    ids = iter(range(10, 10**6))
    flow.seed(book, 100, 101, ids)
    book.submit(1, B, 100, 7, owner=5)
    rng = random.Random(1)
    for _ in range(30):
        flow.step(book, rng, 1e9, ids)[1](book)
    assert book.level(B, 100).qty == 7


def test_atom_sized_to_the_market_misses_a_bot_behind_it():
    book = Book()
    sizes = {(LIMIT, 0): (10,), (CANCEL, 0): (10,), (MARKET, 0): (-1,)}
    p = flat(1, 4, lambda n: 0.0, lambda n: 0.0, lambda n: 60.0, aes=10, sizes=sizes)
    flow = QueueReactive(Params(p.rates, sizes, p.aes, p.edges, p.queues, True), seed=1)
    ids = iter(range(10, 10**6))
    flow.seed(book, 100, 101, ids)
    book.submit(1, B, 100, 7, owner=5)
    book.submit(2, S, 101, 7, owner=5)
    flow.step(book, random.Random(1), 1e9, ids)[1](book)
    assert 1 in book and 2 in book
    assert book.audit() == ""


def test_same_seed_same_tape():
    def run():
        sim = Simulator(QueueReactive(flow_params(), seed=3), seed=3)
        sim.seed(1000, 1001)
        sim.run(5.0)
        return sim.tape

    a, b = run(), run()
    assert a.mid == b.mid and a.t == b.t


def test_flow_runs_under_latency():
    bot = SymmetricQuoter(0.5, size=5, max_position=50)
    lag = Latency(feed=0.001, entry=0.001, response=0.001)
    sim = Simulator(QueueReactive(flow_params(), seed=3), [bot], seed=3, latency={bot: lag})
    sim.seed(1000, 1001)
    sim.run(5.0)
    assert sim.contexts[0].account.fills
    assert sim.book.audit() == "" and sim.contexts[0]._view.audit() == ""


def test_prices_coming_into_range_get_filled():
    p = flat(2, 4, lambda n: 0.0, lambda n: 0.0, lambda n: 0.0, aes=10)
    book = Book()
    ids = iter(range(10, 10**6))
    flow = QueueReactive(p, seed=1)
    flow.seed(book, 100, 101, ids)
    assert book.level(B, 98) is None
    book.submit(1, S, 0, 30, kind=OrderType.MARKET)
    assert book.best_bid() == 99
    rng = random.Random(1)
    assert flow.step(book, rng, 1e9, ids) is None
    book.submit(2, S, 100, 5)
    assert book.best_ask() == 100
    flow.step(book, rng, 1e9, ids)[1](book)
    assert book.level(B, 98).qty == 30
    assert book.level(S, 103) is None
    assert flow.step(book, rng, 1e9, ids) is None


def feed(p, minutes, seed):
    book, rng, ids = Book(), random.Random(seed), iter(range(1, 10**9))
    flow = QueueReactive(p, seed=seed)
    flow.seed(book, 100, 101, ids)
    msgs = [Msg(0, "A", o.id, o.side, o.price * 100, o.qty) for s in Side for o in book.orders(s)]
    t = 0.0
    while (step := flow.step(book, rng, minutes - t, ids)) is not None:
        t += step[0]
        ns = round(t * NS_PER_MINUTE)
        for e in step[1](book):
            if e.type == EventType.TRADE:
                msgs.append(Msg(ns, "E", e.maker, qty=e.qty))
            elif e.type == EventType.CANCELED:
                msgs.append(Msg(ns, "D", e.id))
            elif e.type == EventType.ACCEPTED and e.id in book:
                msgs.append(Msg(ns, "A", e.id, e.side, e.price * 100, e.qty))
    return msgs


def test_fit_recovers_the_rates_that_made_the_feed():
    def birth(n):
        return 30.0 / (1 + 0.3 * n)

    def cancel(n):
        return 6.0 + n

    def market(n):
        return 9.0 / n

    truth = flat(2, 8, birth, cancel, market)
    f = qr.fit(feed(truth, 3000.0, 4), 100.0, 3000.0, depth=2, bins=8)
    assert f.aes.tolist() == [1, 1]
    got = f.params(regimes=False, min_events=1).rates[0][0]
    n_events = f.counts[0].sum(axis=0)
    for n in range(1, 7):
        for k, fn in ((LIMIT, birth), (CANCEL, cancel), (MARKET, market)):
            assert abs(got[n][k] / fn(n) - 1) < 4 / np.sqrt(n_events[n, k])


def regime_params():
    rates = tuple(
        tuple(((1000.0 * i + r + 1, 0.0, 0.0),) * 7 for r in range(qr.REGIMES)) for i in range(2)
    )
    return Params(rates, {}, (10, 10), (20, 40), ((10,), (10,)), False, memory=True)


def regimes_seen(flow, book, bid, ask):
    actions, weights = flow._menu(book, bid, ask)
    return {(a[1], a[2]): int(w) % 1000 - 1 for a, w in zip(actions, weights, strict=True)}


@pytest.mark.parametrize(
    "ask_qty, up, regime",
    [(20, False, qr.THIN), (40, False, qr.USUAL), (41, False, qr.DEEP), (20, True, qr.THIN)],
)
def test_flow_reads_the_opposite_queue_and_last_move(ask_qty, up, regime):
    book = Book()
    book.submit(1, B, 100, 30)
    book.submit(2, S, 101, ask_qty)
    flow = QueueReactive(regime_params())
    flow.gap.up = up
    seen = regimes_seen(flow, book, 100, 101)
    assert seen[B, 0] == regime + (qr.BASE if up else 0)
    assert seen[B, 1] == (qr.BASE if up else 0)
    assert seen[S, 1] == (0 if up else qr.BASE)


def test_flow_reads_the_gap():
    book = Book()
    book.submit(1, B, 100, 30)
    book.submit(2, S, 102, 30)
    flow = QueueReactive(regime_params())
    flow.gap.update(100, 101, False)
    flow.gap.update(100, 102, True)
    seen = regimes_seen(flow, book, 100, 102)
    assert seen[B, 0] == qr.STAYED_MARKET + qr.BASE
    assert seen[S, 0] == qr.LEFT_MARKET
    assert seen[B, 1] == qr.STAYED_MARKET + qr.BASE


def test_fit_and_flow_agree_on_the_state():
    msgs = [
        sec(1, "A", 1, B, 99, 500),
        sec(2, "A", 2, S, 100, 300),
        sec(3, "A", 3, B, 98, 100),
        sec(4, "A", 6, S, 101, 100),
        sec(70, "A", 5, B, 99, 100),
        sec(71, "A", 7, S, 100, 100),
        sec(72, "A", 8, B, 98, 100),
        sec(73, "A", 9, S, 101, 100),
    ]
    f = qr.fit(msgs, 1.0, 3.0, depth=2, bins=10)
    lvl, n, _, _, regime, _ = f.events.T
    book = Book()
    book.submit(1, B, 99, 500)
    book.submit(2, S, 100, 300)
    book.submit(3, B, 98, 100)
    book.submit(6, S, 101, 100)
    rates = tuple(
        tuple(
            tuple((1.0 + 100.0 * b + r * 10000, 0.0, 0.0) for b in range(11))
            for r in range(qr.REGIMES)
        )
        for _ in range(2)
    )
    p = Params(rates, {}, tuple(int(a) for a in f.aes), f.edges, ((1,), (1,)), False, memory=True)
    flow = QueueReactive(p)
    flow.gap.update(99, 100, False)
    flow.gap.up = True
    actions, weights = flow._menu(book, 99, 100)
    got = {(a[1], a[2]): (a[3], a[4]) for a in actions}
    assert (int(n[0]), int(regime[0])) == got[B, 0]
