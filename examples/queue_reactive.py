from concurrent.futures import ProcessPoolExecutor
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from itch_replay import (
    AQUA,
    BLUE,
    CLOSE,
    INK,
    OPEN,
    ORANGE,
    ROOT,
    SESSION,
    STARTS,
    SURFACE,
    SYMBOLS,
    bot,
    msgs,
    result,
    style,
)

from mmsim import qr
from mmsim.book import Side
from mmsim.markout import signature
from mmsim.qr import CANCEL, LIMIT, MARKET
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Agent, Simulator

FIRST = SYMBOLS[0]
SEEDS = range(48)
NOON = (OPEN + CLOSE) / 2
RUNGS = {
    "unit sizes": dict(sizes=qr.UNIT, regimes=False, back=False, memory=False),
    "stationary sizes": dict(sizes=qr.STATIONARY, regimes=False, back=False, memory=False),
    "queue sizes": dict(sizes=qr.QUEUE, regimes=False, back=False, memory=False),
    "+ regimes": dict(sizes=qr.QUEUE, regimes=True, back=False, memory=False),
    "+ cancels at back": dict(sizes=qr.QUEUE, regimes=True, back=True, memory=False),
    "+ last move": dict(sizes=qr.QUEUE, regimes=True, back=True, memory=True),
}
FULL = "+ last move"


class Sampler(Agent):
    def __init__(self):
        self.q = []

    def on_start(self, ctx):
        ctx.set_timer(1 / 60)

    def on_book(self, ctx):
        pass

    def on_timer(self, ctx):
        bid, ask = ctx.book.best_bid(), ctx.book.best_ask()
        if bid is not None and ask is not None and ask - bid == 1:
            self.q += [ctx.book.level(Side.BUY, bid).qty, ctx.book.level(Side.SELL, ask).qty]
        ctx.set_timer(1 / 60)


@cache
def fitted(symbol, start, end):
    return qr.fit(msgs(symbol), start, end)


@cache
def params(symbol, rung, start, end):
    return fitted(symbol, start, end).params(**RUNGS[rung])


def row(sim):
    sig = signature(sim.tape, sim.end - SESSION, sim.end)
    return np.concatenate((result(sim, 1)[:5], sig))


def real_session(args):
    symbol, start = args
    sim = Simulator(ItchFlow(msgs(symbol)), [bot("join the touch", 100, 1000)], book=ReplayBook)
    sim.warm_up(start)
    sim.run(SESSION)
    return row(sim)


def model_session(args):
    symbol, rung, seed, start, end = args
    flow = qr.QueueReactive(params(symbol, rung, start, end), seed=seed)
    sim = Simulator(flow, [bot("join the touch", 100, 1000)], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(SESSION)
    return row(sim)


def sample_queue(args):
    symbol, seed = args
    agent = Sampler()
    flow = qr.QueueReactive(params(symbol, FULL, OPEN, CLOSE), seed=seed)
    sim = Simulator(flow, [agent], seed=seed, record_every=None)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(SESSION)
    return agent.q


def summary(rows):
    rows = np.array(rows)
    pnl, vol = rows[:, 0], rows[:, 1]
    per = rows[:, :5].sum(axis=0) / vol.sum()
    se = (pnl - per[0] * vol).std(ddof=1) * np.sqrt(len(rows)) / vol.sum()
    return (*per[[0, 2, 3, 4]], se, vol.mean(), *np.nanmean(rows[:, 5:], axis=0))


def table(title, lines):
    print(f"\n{title}, cents per share traded")
    print("                     pnl (se)        spread  adverse  invent   shares  moves  eta  vol")
    for name, s in lines.items():
        pnl, spread, adverse, inventory, se, vol, moves, eta, sigma = s
        print(
            f"{name:18s} {pnl:6.3f} ({se:5.3f}) {spread:7.3f} {adverse:8.3f} {inventory:7.3f}"
            f" {vol:8.0f} {moves:6.2f} {eta:5.3f} {sigma:5.2f}"
        )


def chart(symbol, f, real, model, ladder, path):
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(13, 4.3), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 1.05]}
    )
    style(a)
    counts, tau = f.counts[0].sum(axis=0), f.tau[0].sum(axis=0)
    n = np.arange(1, 21)
    rates = counts[1:21] / tau[1:21, None]
    for k, label, color in (
        (LIMIT, "limit", BLUE),
        (CANCEL, "cancel", AQUA),
        (MARKET, "market", ORANGE),
    ):
        a.semilogy(n, rates[:, k], color=color, linewidth=2, marker="o", markersize=3, label=label)
    a.set_xlabel(f"queue at the touch, in units of {f.aes[0]} shares", color=INK)
    a.set_ylabel("events per minute, log scale", color=INK)
    a.legend(frameon=False, labelcolor=INK)

    style(b)
    edges = np.arange(0, 12_500, 500)
    mid = (edges[:-1] + edges[1:]) / 2
    b.plot(mid, np.histogram(real, edges, density=True)[0] * 500, color=INK, linewidth=2)
    b.plot(mid, np.histogram(model, edges, density=True)[0] * 500, color=BLUE, linewidth=2)
    top = np.histogram(real, edges, density=True)[0].max() * 500
    b.text(mid[3], top * 1.02, "tape", color=INK, fontsize=9, ha="center")
    b.text(5900, top * 0.42, "model", color=BLUE, fontsize=9)
    b.set_ylim(0, top * 1.12)
    b.set_xlabel("shares at the touch", color=INK)
    b.set_ylabel("share of time, 500-share bins", color=INK)

    style(c)
    names = [r for r in ladder if r != "real"]
    y = np.arange(len(names))[::-1]
    c.axvline(ladder["real"][2], color=INK, linestyle="--", linewidth=1.2)
    c.scatter([ladder[r][2] for r in names], y, s=46, color=ORANGE, zorder=3)
    c.set_yticks(y, names, color=INK)
    c.set_xlabel("one-minute adverse selection, cents per share", color=INK)
    c.grid(False)
    c.grid(color="#e4e3df", linewidth=0.8, axis="x")
    c.text(ladder["real"][2] + 0.008, len(names) - 0.72, "tape", color=INK, fontsize=9)
    c.set_ylim(-0.5, len(names) - 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def model_jobs(symbol, rung, start, end):
    return [(symbol, rung, s, start, end) for s in SEEDS]


if __name__ == "__main__":
    f = fitted(FIRST, OPEN, CLOSE)
    print(FIRST, "aes", f.aes, "opposite queue edges", f.edges)
    print("share of events by kind", (f.counts.sum(axis=(0, 1, 2)) / f.counts.sum()).round(3))
    lvl, _, kind, _, _, pos = f.events.T
    u = pos[(lvl == 0) & (kind == CANCEL) & (pos >= 0)] / qr.POSITIONS
    print(f"touch cancels: mean position {u.mean():.3f}, last 10% of queue {(u >= 0.9).mean():.3f}")
    for name, r in (("market", qr.LEFT_MARKET), ("cancel", qr.LEFT_CANCEL)):
        left = f.counts[0, r, 0, LIMIT] + f.counts[0, r + qr.BASE, 0, LIMIT]
        stay = r - 2
        kept = f.counts[0, stay, 0, LIMIT] + f.counts[0, stay + qr.BASE, 0, LIMIT]
        print(f"after a {name} empties a queue: continues {kept}, reverts {left}")

    with ProcessPoolExecutor() as pool:
        real = list(pool.map(real_session, [(FIRST, s) for s in STARTS]))
        ladder = {"real": summary(real)}
        for r in RUNGS:
            jobs = model_jobs(FIRST, r, OPEN, CLOSE)
            ladder[r] = summary(list(pool.map(model_session, jobs)))
        table(f"{FIRST}, in sample, fit on the whole day", ladder)

        later = [s for s in STARTS if s >= NOON]
        rows = list(pool.map(real_session, [(FIRST, s) for s in later]))
        jobs = model_jobs(FIRST, FULL, OPEN, NOON)
        out = {
            "real, afternoon": summary(rows),
            "fit on morning": summary(list(pool.map(model_session, jobs))),
        }
        table(f"{FIRST}, fit on {OPEN:.0f} to {NOON:.0f}, tape after", out)

        by = {}
        for symbol in SYMBOLS:
            by[f"{symbol} real"] = summary(
                list(pool.map(real_session, [(symbol, s) for s in STARTS]))
            )
            by[f"{symbol} model"] = summary(
                list(pool.map(model_session, model_jobs(symbol, FULL, OPEN, CLOSE)[:12]))
            )
        table("every symbol, 12 sessions a row", by)

        model = [q for qs in pool.map(sample_queue, [(FIRST, s) for s in SEEDS]) for q in qs]
    sampled = f.queues[:, 0]
    path = ROOT / "docs" / "queue_reactive.png"
    chart(FIRST, f, sampled[sampled > 0], np.array(model), ladder, path)
