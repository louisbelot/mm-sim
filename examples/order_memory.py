import pickle
from concurrent.futures import ProcessPoolExecutor
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
from queue_reactive import FULL, RUNGS, summary, table

from mmsim import qr
from mmsim.markout import signature
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Agent, Simulator

FIRST = SYMBOLS[0]
SEEDS = range(48)
FEW = range(12)
LAGS = np.arange(1, 21)
MODELS = {"queue-reactive": 0, "+ 1 kernel": 1, "+ 2 kernels": 2, "+ 3 kernels": 3}


class Prints(Agent):
    def __init__(self):
        self.p = []

    def on_start(self, ctx):
        pass

    def on_book(self, ctx):
        pass

    def on_trade(self, ctx, p):
        self.p.append((p.t, int(p.side)))


@cache
def day(symbol):
    return qr.fit(msgs(symbol), OPEN, CLOSE)


def stored(symbol, kernels):
    return ROOT / "data" / f"{symbol}.qr{kernels}.pickle"


def store(symbol, kernels):
    p = day(symbol).params(**RUNGS[FULL])
    p = day(symbol).excite(p, kernels) if kernels else p
    stored(symbol, kernels).write_bytes(pickle.dumps(p))
    return p


@cache
def params(symbol, kernels):
    return pickle.loads(stored(symbol, kernels).read_bytes())


def flow_stats(p, end):
    a = np.array(p, dtype=float).reshape(-1, 2)
    s = 2 * a[:, 1] - 1
    acf = [np.mean(s[k:] * s[:-k]) if s.size > k else np.nan for k in LAGS]
    c = np.histogram(a[:, 0], np.linspace(end - SESSION, end, 181))[0]
    return np.array([len(a), c.var() / c.mean() if c.mean() else np.nan, *acf])


def run(sim, prints):
    sig = signature(sim.tape, sim.end - SESSION, sim.end)
    return np.concatenate((result(sim, 1)[:5], sig, flow_stats(prints.p, sim.end)))


def real_session(args):
    symbol, start = args
    prints = Prints()
    sim = Simulator(
        ItchFlow(msgs(symbol)), [bot("join the touch", 100, 1000), prints], book=ReplayBook
    )
    sim.warm_up(start)
    sim.run(SESSION)
    return run(sim, prints)


def model_session(args):
    symbol, kernels, seed = args
    prints = Prints()
    flow = qr.QueueReactive(params(symbol, kernels), seed=seed)
    sim = Simulator(flow, [bot("join the touch", 100, 1000), prints], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(SESSION)
    return run(sim, prints)


def sessions(pool, fn, jobs):
    return np.array(list(pool.map(fn, jobs)))


def residual_stats(f, p):
    r = np.sort(np.concatenate(f.residuals(p)))
    u = 1 - np.exp(-r)
    ks = np.abs(u - (np.arange(r.size) + 0.5) / r.size).max()
    return r, ks


def flows(rows):
    rows = rows[:, 8:]
    return rows[:, 0].mean(), np.nanmean(rows[:, 1]), np.nanmean(rows[:, 2:], axis=0)


def chart(resid, acf, adverse, path):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURFACE)
    colors = {"queue-reactive": MUTED, "+ 1 kernel": AQUA, "+ 3 kernels": BLUE}

    style(a)
    for name, color in colors.items():
        r = resid[name]
        a.semilogy(r, 1 - (np.arange(r.size) + 0.5) / r.size, color=color, linewidth=2)
    x = np.linspace(0, 8, 50)
    a.semilogy(x, np.exp(-x), color=INK, linestyle="--", linewidth=1.2)
    a.set_xlim(0, 8)
    a.set_ylim(1e-3, 1)
    a.set_xlabel("rescaled wait between market orders", color=INK)
    a.set_ylabel("share of waits longer, log scale", color=INK)
    a.text(2.6, 1.6e-2, "exp(1)", color=INK, fontsize=9)

    style(b)
    b.axhline(0, color=MUTED, linewidth=0.8)
    b.plot(LAGS, acf["tape"], color=INK, linewidth=2, marker="o", markersize=3, label="tape")
    for name, color in colors.items():
        b.plot(LAGS, acf[name], color=color, linewidth=2, marker="o", markersize=3, label=name)
    b.set_xticks([1, 5, 10, 15, 20])
    b.set_xlabel("lag, in prints", color=INK)
    b.set_ylabel("autocorrelation of aggressor sign", color=INK)
    b.legend(frameon=False, labelcolor=INK)

    style(c)
    names = list(adverse)
    x = np.arange(len(names))
    w = 0.26
    bars = (("tape", INK), ("queue-reactive", MUTED), ("+ 3 kernels", BLUE))
    for j, (key, color) in enumerate(bars):
        c.bar(x + (j - 1) * w, [adverse[n][key] for n in names], w, color=color, label=key)
    c.set_xticks(x, names, color=INK)
    c.set_ylabel("one-minute adverse selection, cents per share", color=INK)
    c.legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    f = day(FIRST)
    print(FIRST, "market orders in the fit", len(f.hits))
    resid = {}
    print("\nmodel              branching  timescales (ms)          resid var    ks")
    for name, k in MODELS.items():
        p = store(FIRST, k)
        r, ks = residual_stats(f, p)
        resid[name] = r
        e = p.excite
        rho = f"{e.branching:9.3f}" if e else "        -"
        ms = " ".join(f"{60_000 / b:8.2f}" for b in e.beta) if e else "-"
        print(f"{name:16s} {rho}  {ms:24s} {r.var():9.3f} {ks:6.3f}")

    for symbol in SYMBOLS[1:]:
        store(symbol, 0)
        store(symbol, 3)

    with ProcessPoolExecutor() as pool:
        tape = sessions(pool, real_session, [(FIRST, s) for s in STARTS])
        lines = {"tape": summary(tape[:, :8])}
        stats = {"tape": flows(tape)}
        for name, k in MODELS.items():
            rows = sessions(pool, model_session, [(FIRST, k, s) for s in SEEDS])
            lines[name] = summary(rows[:, :8])
            stats[name] = flows(rows)
        table(f"{FIRST}, fit on the whole day, join the touch", lines)
        print("\n                  prints  fano 10s   sign acf at lags 1, 2, 5, 10")
        for name, (n, fano, acf) in stats.items():
            lags = " ".join(f"{acf[i]:6.3f}" for i in (0, 1, 4, 9))
            print(f"{name:16s} {n:7.0f} {fano:9.2f}  {lags}")

        adverse, by = {}, {}
        for symbol in SYMBOLS:
            real = summary(sessions(pool, real_session, [(symbol, s) for s in STARTS])[:, :8])
            plain = summary(sessions(pool, model_session, [(symbol, 0, s) for s in FEW])[:, :8])
            hot = summary(sessions(pool, model_session, [(symbol, 3, s) for s in FEW])[:, :8])
            by[f"{symbol} tape"], by[f"{symbol} qr"], by[f"{symbol} excited"] = real, plain, hot
            adverse[symbol] = {"tape": real[2], "queue-reactive": plain[2], "+ 3 kernels": hot[2]}
        table("every symbol, 3 kernels, 12 sessions a row", by)

    chart(resid, {k: v[2] for k, v in stats.items()}, adverse, ROOT / "docs" / "order_memory.png")
