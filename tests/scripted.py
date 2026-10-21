from collections import deque

from mmsim.book import OrderType, Side
from mmsim.sim import Simulator


class ScriptedFlow:
    owner = 0

    def __init__(self, script):
        self.script = deque(script)
        self.now = 0.0

    def seed(self, book, bid, ask, ids):
        book.submit(next(ids), Side.BUY, bid, 5)
        book.submit(next(ids), Side.SELL, ask, 5)

    def step(self, book, rng, horizon, ids):
        if self.script and self.script[0][0] - self.now <= horizon:
            t, cmd = self.script.popleft()
            dt, self.now = t - self.now, t
            return dt, lambda: cmd(book, ids)
        self.now += horizon
        return None


def market(side, qty):
    return lambda book, ids: book.submit(next(ids), side, 0, qty, kind=OrderType.MARKET)


def limit(side, px, qty=1):
    return lambda book, ids: book.submit(next(ids), side, px, qty)


def make(script, *agents, **kw):
    sim = Simulator(ScriptedFlow(script), agents, **kw)
    sim.seed(99, 102)
    return sim
