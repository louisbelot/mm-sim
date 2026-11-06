from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from itch_replay import (
    AQUA,
    BLUE,
    CLOSE,
    INK,
    MUTED,
    OPEN,
    ORANGE,
    ROOT,
    SESSION,
    STARTS,
    SURFACE,
    SYMBOLS,
    bot,
    msgs,
    style,
)
from matplotlib.ticker import NullFormatter
from queue_reactive import NOON, Sampler, real_session, row, summary, table

from mmsim import qr
from mmsim.qr import CANCEL, KINDS, LIMIT, MARKET, SHARED
from mmsim.sim import Simulator

FIRST = SYMBOLS[0]
SEEDS = range(8)
LATER = [s for s in STARTS if s >= NOON]
MODES = {"flat": None, "shared clock": SHARED, "clock by kind": KINDS}


@cache
def day(symbol):
    return qr.fit(msgs(symbol), OPEN, CLOSE)


@cache
def halves(symbol):
    morning = qr.fit(msgs(symbol), OPEN, NOON)
    return morning, qr.fit(msgs(symbol), NOON, CLOSE, aes=morning.aes, edges=morning.edges)


@cache
def params(symbol, window, mode):
    if window == "day":
        return day(symbol).params(clock=mode)
    return halves(symbol)[0].params()


def session(args):
    symbol, window, mode, clock, start, seed = args
    p = params(symbol, window, mode)
    if clock is not None:
        p = replace(p, clock=clock)
    sampler = Sampler()
    flow = qr.QueueReactive(p, seed=seed, start=start - 30.0)
    sim = Simulator(flow, [bot("join the touch", 100, 1000), sampler], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(SESSION)
    return np.append(row(sim), np.median(sampler.q) if sampler.q else np.nan)


def afternoon(symbol, mode):
    morning, after = halves(symbol)
    return qr.rescale(morning.params(), after, mode)


def peers(symbol, clocks):
    s = np.median([np.array(clocks[o].scale)[:, 0] for o in clocks if o != symbol], axis=0)
    return qr.Clock(clocks[symbol].edges, tuple((x, x, x) for x in s))


def clock_label(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def chart(f, by_session, by_symbol, path):
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(13.5, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 0.9]}
    )
    edges = np.array(f.slots)
    hours = np.arange(600, 960, 60)

    style(a)
    shared = np.array(f.clock(SHARED).scale)[:, 0]
    kinds = np.array(f.clock(KINDS).scale)
    for k, label, color in (
        (LIMIT, "limit", BLUE),
        (CANCEL, "cancel", AQUA),
        (MARKET, "market", ORANGE),
    ):
        a.stairs(kinds[:, k], edges, color=color, linewidth=1.4, label=label, baseline=None)
    a.stairs(shared, edges, color=INK, linewidth=2.4, label="shared", baseline=None)
    a.set_yscale("log")
    a.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    a.set_xticks(hours, [clock_label(h) for h in hours])
    a.set_ylabel(f"{FIRST} rate multiplier, log scale", color=INK)
    a.legend(frameon=False, labelcolor=INK, ncol=2, fontsize=9)

    style(b)
    mid = np.array(STARTS) + SESSION / 2
    b.plot(mid, by_session["tape"], color=INK, linewidth=2.2, marker="o", markersize=4)
    for name, color, dash in (
        ("flat", BLUE, "--"),
        ("shared clock", ORANGE, "-"),
        ("clock by kind", AQUA, "-"),
    ):
        b.plot(mid, by_session[name], color=color, linewidth=1.6, linestyle=dash, label=name)
    b.plot([], [], color=INK, linewidth=2.2, marker="o", markersize=4, label="tape")
    b.set_yscale("log")
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
    b.set_xticks(hours, [clock_label(h) for h in hours])
    b.set_ylabel(f"{FIRST} mid moves per minute, log scale", color=INK)
    b.legend(frameon=False, labelcolor=INK, fontsize=9)

    style(c)
    names = list(by_symbol)
    y = np.arange(len(names))[::-1]
    c.axvline(1.0, color=MUTED, linewidth=1)
    for key, label, color in (
        ("flat", "morning rates", BLUE),
        ("peers", "+ peers' afternoon clock", AQUA),
        ("measured", "+ its own afternoon clock", ORANGE),
    ):
        x = [by_symbol[s][key] / by_symbol[s]["tape"] for s in names]
        c.scatter(x, y, s=44, color=color, zorder=3, label=label)
    c.set_xscale("log")
    c.xaxis.set_minor_formatter(NullFormatter())
    c.set_xticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
    c.set_yticks(y, names, color=INK)
    c.set_xlabel("afternoon moves, model over tape, log scale", color=INK)
    c.set_ylim(-0.6, len(names) + 0.9)
    c.grid(False)
    c.grid(color="#e4e3df", linewidth=0.8, axis="x")
    c.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    f = day(FIRST)
    print(f"{FIRST} multipliers by slot, fit on the whole day")
    print("start   shared   limit  cancel  market   events/min")
    rate = f.arrivals.sum(axis=1) / np.diff(f.slots)
    for t, s, k, r in zip(
        f.slots[:-1], f.clock(SHARED).scale, f.clock(KINDS).scale, rate, strict=True
    ):
        print(f"{clock_label(t):5s} {s[0]:8.3f} {k[0]:7.3f} {k[1]:7.3f} {k[2]:7.3f} {r:10.0f}")

    with ProcessPoolExecutor() as pool:
        tape = {
            s: r
            for s, r in zip(
                STARTS, pool.map(real_session, [(FIRST, s) for s in STARTS]), strict=True
            )
        }
        by_session = {"tape": [tape[s][5] for s in STARTS]}
        print(f"\n{FIRST}, fit on the whole day, mid moves per minute in each session")
        print("start    tape   " + "  ".join(f"{m:>13s}" for m in MODES))
        for name, mode in MODES.items():
            jobs = [(FIRST, "day", mode, None, s, k) for s in STARTS for k in SEEDS]
            res = np.array(list(pool.map(session, jobs))).reshape(len(STARTS), len(SEEDS), -1)
            by_session[name] = res[:, :, 5].mean(axis=1)
        for i, s in enumerate(STARTS):
            line = "  ".join(f"{by_session[m][i]:13.2f}" for m in MODES)
            print(f"{clock_label(s):5s} {by_session['tape'][i]:7.2f}   {line}")

        clocks = {sym: afternoon(sym, SHARED) for sym in SYMBOLS}
        own = {
            "fit on morning": None,
            "+ afternoon clock, shared": clocks[FIRST],
            "+ afternoon clock, by kind": afternoon(FIRST, KINDS),
            "+ peers' afternoon clock": peers(FIRST, clocks),
        }
        print(f"\n{FIRST} afternoon multipliers on the morning's rates")
        print("start   shared   limit  cancel  market   peers")
        kinds, near = own["+ afternoon clock, by kind"].scale, own["+ peers' afternoon clock"].scale
        for t, s, k, p in zip(
            clocks[FIRST].edges[:-1], clocks[FIRST].scale, kinds, near, strict=True
        ):
            print(
                f"{clock_label(t):5s} {s[0]:8.3f} {k[0]:7.3f} {k[1]:7.3f} {k[2]:7.3f} {p[0]:7.3f}"
            )

        lines = {"real, afternoon": summary([tape[s] for s in LATER])}
        touch = halves(FIRST)[1].queues[:, 0]
        medians = {"real, afternoon": np.median(touch[touch > 0])}
        for name, clock in own.items():
            jobs = [(FIRST, "morning", None, clock, s, k) for s in LATER for k in SEEDS]
            res = list(pool.map(session, jobs))
            lines[name] = summary([r[:-1] for r in res])
            medians[name] = np.nanmedian([r[-1] for r in res])
        table(f"{FIRST}, fit on {clock_label(OPEN)} to {clock_label(NOON)}, tape after", lines)
        print("\nmedian touch queue, shares")
        for name, q in medians.items():
            print(f"{name:28s} {q:7.0f}")

        by_symbol = {}
        print("\nafternoon mid moves per minute, morning fit, 24 sessions a row")
        print("symbol    tape    flat  measured   peers   adverse tape  measured   peers")
        for sym in SYMBOLS:
            real = summary(list(pool.map(real_session, [(sym, s) for s in LATER])))
            out = {"tape": real}
            for key, clock in (
                ("flat", None),
                ("measured", clocks[sym]),
                ("peers", peers(sym, clocks)),
            ):
                jobs = [(sym, "morning", None, clock, s, k) for s in LATER for k in SEEDS[:4]]
                out[key] = summary([r[:-1] for r in pool.map(session, jobs)])
            by_symbol[sym] = {k: v[6] for k, v in out.items()}
            print(
                f"{sym:6s} {real[6]:7.2f} {out['flat'][6]:7.2f} {out['measured'][6]:9.2f}"
                f" {out['peers'][6]:7.2f}   {real[2]:12.3f} {out['measured'][2]:9.3f}"
                f" {out['peers'][2]:7.3f}"
            )

    chart(f, by_session, by_symbol, ROOT / "docs" / "activity_clock.png")
