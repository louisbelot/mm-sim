from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from book_reference import SIGMA, A, K, Q
from itch_replay import (
    BLUE,
    INK,
    MUTED,
    ORANGE,
    ROOT,
    SESSION,
    STARTS,
    SURFACE,
    SYMBOLS,
    msgs,
    result,
    style,
)
from queue_reference import flow

from mmsim.book import Side
from mmsim.bots import GLFT, SymmetricQuoter
from mmsim.glft import Model
from mmsim.informed import Fundamental, InformedTrader
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Simulator

N = 2000
WORLDS = {
    "poisson": (False, False, False),
    "informed": (False, True, False),
    "hawkes": (True, False, False),
    "sizes": (False, False, True),
    "hawkes+sizes": (True, False, True),
}
RULES = {
    "off": (0, 1),
    "alone": (1, 1),
    "under 2": (2, 1),
    "under 3": (3, 1),
    "alone, 2 back": (1, 2),
}
ITCH_RULES = {"off": 0, "alone": 1, "under 2 lots": 200, "under 5 lots": 500}


def play(args):
    world, rule, seed = args
    hawkes, informed, sizes = WORLDS[world]
    thin, back = RULES[rule]
    agents = [GLFT(Model(SIGMA, A, K, 0.08, Q), keep=1, thin=thin, back=back)]
    if informed:
        agents.append(InformedTrader(Fundamental(1.0, seed=2 * seed), 1.0, seed=2 * seed + 1))
    sim = Simulator(flow(hawkes, sizes), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(60)
    close = float(sim.tape.at(sim.end))
    a = sim.contexts[0].account
    edge = np.zeros(2)
    move = np.zeros(2)
    n = np.zeros(2)
    for f in a.fills:
        s = (1 if f.side == Side.BUY else -1) * f.qty
        e = 0.0 if f.mid is None else s * (f.mid - f.price)
        g = int(f.left == 0)
        edge[g] += e
        move[g] += s * (float(sim.tape.at(f.t + 1)) - f.price) - e
        n[g] += f.qty
    pnl = a.value(close)
    return np.concatenate([[pnl, a.volume, edge.sum(), pnl - edge.sum()], edge, move, n])


def simulate(pool):
    out = {}
    for world in WORLDS:
        for rule in RULES:
            jobs = [(world, rule, s) for s in range(N)]
            out[world, rule] = np.array(list(pool.map(play, jobs, chunksize=50)))
        print(f"done {world}", flush=True)
    return out


def sim_table(out):
    print(
        f"\nGLFT gamma 0.08 alone, {N} one-hour sessions per row, same seeds across rules (ticks)"
    )
    print(
        "world          rule           pnl/h   vs off (se)   fills/h  edge/fill  after/h"
        "  cleared  move cleared  move standing"
    )
    for world in WORLDS:
        base = out[world, "off"][:, 0]
        for rule in RULES:
            r = out[world, rule]
            d = r[:, 0] - base
            fills = r[:, 1].sum()
            cleared = r[:, 9].sum() / fills
            mc = r[:, 7].sum() / r[:, 9].sum()
            ms = r[:, 6].sum() / r[:, 8].sum()
            print(
                f"{world:14s} {rule:13s} {r[:, 0].mean():7.2f}"
                f"  {d.mean():+6.2f} ({d.std() / np.sqrt(N):4.2f})"
                f"  {r[:, 1].mean():7.1f}  {r[:, 2].sum() / fills:9.2f}  {r[:, 3].mean():7.2f}"
                f"  {cleared:6.1%}  {mc:12.2f}  {ms:13.2f}"
            )


def replay(args):
    symbol, thin, start = args
    bot = SymmetricQuoter(0.5, size=100, max_position=1000, keep=1, thin=thin)
    sim = Simulator(ItchFlow(msgs(symbol)), [bot], book=partial(ReplayBook, sweep=True))
    sim.warm_up(start)
    sim.run(SESSION)
    return result(sim, 1)


def itch(pool):
    out = {}
    for symbol in SYMBOLS:
        for rule, thin in ITCH_RULES.items():
            out[symbol, rule] = np.array(
                list(pool.map(replay, [(symbol, thin, s) for s in STARTS]))
            )
    print(f"\nITCH replay, keep priority, 100-share quotes, {len(STARTS)} sessions of 30 minutes")
    print("symbol rule           $/session  vs off (se)   shares  cents/share  adverse c/sh")
    for symbol in SYMBOLS:
        base = out[symbol, "off"][:, 0] / 100
        for rule in ITCH_RULES:
            r = out[symbol, rule]
            d = r[:, 0] / 100 - base
            vol = r[:, 1].sum()
            print(
                f"{symbol:6s} {rule:13s} {r[:, 0].mean() / 100:9.2f}"
                f"  {d.mean():+7.2f} ({d.std(ddof=1) / np.sqrt(len(d)):5.2f})"
                f"  {r[:, 1].mean():7.0f}  {r[:, 0].sum() / vol:11.3f}  {r[:, 3].sum() / vol:12.3f}"
            )
    return out


def chart(out, path):
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    style(ax)
    rules = list(RULES)
    x = np.arange(len(WORLDS))
    w = 0.8 / len(rules)
    shades = [MUTED, BLUE, "#7fb1ea", "#c3daf5", ORANGE]
    for k, rule in enumerate(rules):
        mean = [out[wd, rule][:, 0].mean() for wd in WORLDS]
        se = [out[wd, rule][:, 0].std() / np.sqrt(N) for wd in WORLDS]
        ax.bar(
            x + (k - (len(rules) - 1) / 2) * w,
            mean,
            w,
            yerr=2 * np.array(se),
            color=shades[k],
            edgecolor=SURFACE,
            ecolor=INK,
            capsize=2,
            label=rule,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, list(WORLDS), color=INK)
    ax.set_ylabel("GLFT P&L per hour, ticks", color=INK)
    ax.set_title(
        "stepping back when the quote would be alone, bars are 2 se", color=INK, fontsize=10
    )
    ax.legend(frameon=False, labelcolor=INK, loc="lower left", ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        out = simulate(pool)
        sim_table(out)
        chart(out, ROOT / "docs" / "last_share.png")
        itch(pool)
