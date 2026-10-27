import copy
import random

import pytest
from reference_book import ReferenceBook
from test_differential import apply, commands, snapshot

from mmsim.book import Book, Side
from mmsim.bots import GLFT, FlowLean, SymmetricQuoter
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence
from mmsim.glft import Model
from mmsim.hawkes import Hawkes
from mmsim.sim import Latency, Simulator

pytest.importorskip("orderbook")
from orderbook import load_library  # noqa: E402

try:
    load_library()
except FileNotFoundError:
    pytest.skip("ob_c shared library not found", allow_module_level=True)

from mmsim.native import NativeBook  # noqa: E402


@pytest.mark.parametrize("seed", range(30))
def test_matches_reference(seed):
    rng = random.Random(seed)
    book, ref = NativeBook(), ReferenceBook()
    for i, cmd in enumerate(commands(rng, ref, 2000)):
        got, want = apply(book, cmd), apply(ref, cmd)
        assert got == want, f"command {i}: {cmd}"
        for side in Side:
            assert snapshot(book, side) == ref.snapshot(side), f"command {i}: {cmd}"
        if i % 100 == 0:
            assert book.audit() == "", f"command {i}: {cmd}"
    assert book.audit() == ""


SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
MODEL = Model(1.0, 2.1, 3.6, 0.3, 5)
SLOW = Latency(feed=0.02, entry=0.01, response=0.01, jitter=0.01)

CASES = {
    "poisson_symmetric": (lambda: ZeroIntelligence(SKY), lambda: SymmetricQuoter(1.0), None),
    "poisson_glft": (lambda: ZeroIntelligence(SKY), lambda: GLFT(MODEL, keep=1), None),
    "hawkes_lean": (
        lambda: HawkesFlow(SKY, HAWKES),
        lambda: GLFT(MODEL, keep=1, signal=FlowLean(HAWKES, 0.3, 0.5)),
        None,
    ),
    "poisson_latency": (lambda: ZeroIntelligence(SKY), lambda: GLFT(MODEL, keep=1), SLOW),
}


def run(case, seed, book):
    flow, bot, latency = CASES[case]
    agent = bot()
    sim = Simulator(
        flow(), [agent], seed=seed, latency={agent: latency} if latency else None, book=book
    )
    sim.seed(1000, 1001)
    sim.warm_up(20)
    sim.run(120)
    acct = sim.contexts[0].account
    fills = [(f.t, f.oid, f.side, f.price, f.qty, f.stale) for f in acct.fills]
    return sim, (acct.position, acct.cash, fills)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", CASES)
def test_session_equals_python_book(case, seed):
    py_sim, py = run(case, seed, Book)
    nat_sim, nat = run(case, seed, NativeBook)
    assert len(py[2]) > 10
    assert nat == py
    assert nat_sim.book.audit() == ""
    assert snapshot(nat_sim.book, Side.BUY) == snapshot(py_sim.book, Side.BUY)
    assert snapshot(nat_sim.book, Side.SELL) == snapshot(py_sim.book, Side.SELL)


def test_replica_keeps_queue_order_and_is_independent():
    book = NativeBook()
    for i, (side, px, owner) in enumerate(
        [(Side.BUY, 99, 1), (Side.BUY, 99, 2), (Side.SELL, 101, 1)]
    ):
        book.submit(i + 1, side, px, 3 + i, owner=owner)
    book.modify(1, 99, 5)
    twin = copy.deepcopy(book)
    assert (
        snapshot(twin, Side.BUY)
        == snapshot(book, Side.BUY)
        == [(2, 2, Side.BUY, 99, 4), (1, 1, Side.BUY, 99, 5)]
    )
    assert twin.ahead(1) == book.ahead(1) == 4
    assert twin.audit() == ""
    twin.cancel(2)
    assert len(book) == 3 and len(twin) == 2
