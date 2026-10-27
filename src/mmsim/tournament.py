from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .attribution import attribute
from .book import Book
from .flow import ZeroIntelligence
from .sim import Agent, Simulator

Entrants = Mapping[str, Callable[[], Agent]]


@dataclass(frozen=True, slots=True)
class Result:
    pnl: float
    position: int
    fills: int
    spread: float
    adverse: float
    inventory: float


@dataclass(frozen=True, slots=True)
class Standing:
    name: str
    pnl: np.ndarray
    solo: np.ndarray
    position: np.ndarray
    fills: np.ndarray
    wins: float

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def se(self) -> float:
        return float(self.pnl.std(ddof=1) / np.sqrt(len(self.pnl)))

    @property
    def sharpe(self) -> float:
        return float(self.pnl.mean() / self.pnl.std(ddof=1))

    @property
    def crowding(self) -> tuple[float, float]:
        d = self.pnl - self.solo
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def session(
    entrants: Entrants,
    seed: int,
    *,
    flow: Callable[[], ZeroIntelligence],
    warm: float = 30.0,
    minutes: float = 390.0,
    book: Callable[[], Book] = Book,
    horizon: float = 1.0,
) -> dict[str, Result]:
    names = list(entrants)
    random.Random(f"seats{seed}").shuffle(names)
    agents = {name: entrants[name]() for name in names}
    sim = Simulator(flow(), list(agents.values()), seed=seed, book=book)
    sim.seed(10_000, 10_001)
    sim.warm_up(warm)
    sim.run(minutes)
    close = float(sim.tape.at(sim.end))
    out = {}
    for name, ctx in zip(agents, sim.contexts, strict=True):
        a = attribute(ctx.account.fills, sim.tape, horizon, sim.end)
        out[name] = Result(
            ctx.account.value(close),
            ctx.account.position,
            len(ctx.account.fills),
            a.spread,
            a.adverse,
            a.inventory,
        )
    return out


def standings(together: Sequence[dict[str, Result]], alone: Sequence[dict[str, Result]]):
    names = list(together[0])
    pnl = np.array([[r[n].pnl for n in names] for r in together])
    best = pnl == pnl.max(axis=1, keepdims=True)
    wins = (best / best.sum(axis=1, keepdims=True)).mean(axis=0)
    rows = [
        Standing(
            n,
            pnl[:, i],
            np.array([r[n].pnl for r in alone]),
            np.array([r[n].position for r in together]),
            np.array([r[n].fills for r in together]),
            float(wins[i]),
        )
        for i, n in enumerate(names)
    ]
    return sorted(rows, key=lambda s: -s.mean)
