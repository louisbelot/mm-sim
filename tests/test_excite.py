import dataclasses
import math
import random

import numpy as np
import pytest
from test_qr import feed, flat, flow_params

from mmsim import qr
from mmsim.book import Book, Side
from mmsim.bots import SymmetricQuoter
from mmsim.qr import MARKET, Excite, QueueReactive
from mmsim.sim import Latency, Simulator

TRUTH = (0.4, 0.1, 30.0)


def flat_market():
    return flat(2, 8, lambda n: 30.0 / (1 + 0.3 * n), lambda n: 6.0 + n, lambda n: 4.0)


def hot(own=0.4, cross=0.1, beta=30.0):
    return dataclasses.replace(flat_market(), excite=Excite((own,), (cross,), (beta,)))


@pytest.fixture(scope="module")
def sim_fit():
    p = hot()
    return qr.fit(feed(p, 600.0, 0), 0.0, 600.0, depth=2, bins=8, edges=p.edges)


def test_zero_kernel_gives_the_same_tape():
    def run(p):
        sim = Simulator(QueueReactive(p, seed=3), seed=3)
        sim.seed(1000, 1001)
        sim.run(5.0)
        return sim.tape

    p = flow_params()
    a = run(p)
    b = run(dataclasses.replace(p, excite=Excite((0.0,), (0.0,), (30.0,))))
    assert a.mid == b.mid and a.t == b.t


def test_kick_decays_at_beta():
    p = hot(beta=20.0)
    flow = QueueReactive(p)
    flow.t = 1.0
    flow._kick(Side.BUY)
    assert flow._excess(1.0) == pytest.approx([0.4 * 20.0, 0.1 * 20.0])
    assert flow._excess(1.05) == pytest.approx([8.0 * math.exp(-1.0), 2.0 * math.exp(-1.0)])
    flow.t = 1.05
    flow._kick(Side.SELL)
    e = math.exp(-1.0)
    assert flow._excess(1.05) == pytest.approx([20 * (0.4 * e + 0.1), 20 * (0.1 * e + 0.4)])


def test_branching_sets_the_market_rate(sim_fit):
    rate = len(sim_fit.hits) / sim_fit.T
    mu, rho = 2 * 4.0, 0.5
    sd = math.sqrt(mu / (1 - rho) ** 3 / sim_fit.T)
    assert abs(rate - mu / (1 - rho)) < 4 * sd
    calm = qr.fit(feed(flat_market(), 600.0, 0), 0.0, 600.0, depth=2, bins=8)
    assert len(calm.hits) / calm.T == pytest.approx(mu, rel=0.05)


def test_excited_flow_never_draws_an_unseen_market_size():
    p = hot()
    rates = np.array(p.rates)
    rates[1, ..., MARKET] = 0.0
    sizes = dict(p.sizes)
    sizes[MARKET, 1] = ()
    p = dataclasses.replace(p, rates=tuple(map(tuple, rates.tolist())), sizes=sizes)
    book, rng, ids = Book(), random.Random(1), iter(range(1, 10**9))
    flow = QueueReactive(p, seed=1)
    flow.seed(book, 100, 101, ids)
    for _ in range(3000):
        step = flow.step(book, rng, 1e9, ids)
        step[1](book)
    assert book.audit() == ""


def test_excited_flow_runs_under_latency():
    p = dataclasses.replace(flow_params(), excite=Excite((0.5,), (0.1,), (200.0,)))
    bot = SymmetricQuoter(0.5, size=5, max_position=50)
    lag = Latency(feed=0.001, entry=0.001, response=0.001)
    sim = Simulator(QueueReactive(p, seed=3), [bot], seed=3, latency={bot: lag})
    sim.seed(1000, 1001)
    sim.run(5.0)
    assert sim.contexts[0].account.fills
    assert sim.book.audit() == "" and sim.contexts[0]._view.audit() == ""
