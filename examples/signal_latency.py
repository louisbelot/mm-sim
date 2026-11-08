import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from alpha_quotes import LOT, Q, calibrate, policies, row, touch
from micro_quotes import CLOSE, NOON, OPEN, ROOT, SESSION, STARTS, load, models, msgs, style

from mmsim.alpha import responses
from mmsim.bots import AlphaTouch, SymmetricQuoter
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Latency, Simulator

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c43d7a", "#7a6a12"]
SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
US = 1e-6 / 60
DELAYS = np.array([0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]) * US
H = 10 / 60
WIRE = (0, 10, 100, 1_000, 10_000)
BOTS = ("mid", "micro 0.0", "cjp alpha")


def edge(symbol):
    top = load(symbol)
    fits = models(symbol)
    out = {}
    for half, (lo, hi) in enumerate(((OPEN, NOON), (NOON, CLOSE))):
        micro = fits[1 - half]
        k = np.flatnonzero((top.t >= lo) & (top.t < hi - H))
        clock = np.arange(lo, hi - H, 1 / 60)
        for name, t, i in (("event", top.t[k], k), ("clock", clock, top.index(clock))):
            g = micro.adjust(top.spread[i], top.bq[i], top.aq[i])
            imb = top.imbalance[i] - 0.5
            ok = ~np.isnan(g)
            for sig, x in (("micro", g), ("imbalance", imb)):
                b = responses(t[ok], x[ok], top.mid_at, [*DELAYS, H])
                out.setdefault((sig, name), []).append(b)
    return {key: 1 - np.mean(v, axis=0)[:-1] / np.mean(v, axis=0)[-1] for key, v in out.items()}


def session(args):
    symbol, name, start, us, fit, sols = args
    m, c, micro = fit
    if name == "mid":
        bot = SymmetricQuoter(0.5, LOT, Q * LOT)
    elif name == "micro 0.0":
        bot = SymmetricQuoter(0.5, LOT, Q * LOT, signal=micro)
    else:
        bot = AlphaTouch(sols[0], micro, m.zeta * c, size=LOT)
    lat = {bot: Latency(feed=us * US, entry=us * US)} if us else None
    sim = Simulator(ItchFlow(msgs(symbol)), [bot], book=ReplayBook, latency=lat)
    sim.warm_up(start)
    sim.run(SESSION)
    return row(sim)


def chart(edges, rows, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax in (a, b):
        style(ax)
    x = DELAYS[1:] / US
    for s, col in zip(SYMBOLS, COLORS, strict=False):
        a.plot(x, edges[s]["micro", "event"][1:], color=col, label=s, linewidth=1.6)
        if s != "AAPL":
            a.plot(x, edges[s]["micro", "clock"][1:], color=col, linewidth=1.0, linestyle="--")
    a.set_xscale("log")
    a.set_ylim(-0.05, 1.05)
    a.set_xlabel("delay before acting, microseconds", color=INK)
    a.set_ylabel("share of the 10 s microprice move still ahead", color=INK)
    a.set_title("solid: at book changes, dashed: once a second", color=INK, fontsize=10)
    a.legend(frameon=False, labelcolor=INK, ncol=2, loc="lower left")
    names = [s for s in SYMBOLS if s != "AAPL"]
    k = np.arange(len(WIRE))
    for nm, col in zip(BOTS, (MUTED, "#9cc3ef", "#2a78d6"), strict=True):
        y = [sum(rows[s, nm, us][:, 0].mean() for s in names) / 100 for us in WIRE]
        se = [
            np.sqrt(sum(rows[s, nm, us][:, 0].var(ddof=1) / len(rows[s, nm, us]) for s in names))
            / 100
            for us in WIRE
        ]
        b.errorbar(k, y, yerr=2 * np.array(se), color=col, label=nm, marker="o", capsize=3)
    b.axhline(0, color=MUTED, linewidth=0.8)
    b.set_xticks(k, [f"{us:,}" for us in WIRE], color=INK)
    b.set_xlabel("one-way latency on feed and entry, microseconds", color=INK)
    b.set_ylabel(f"$ per session, sum of {len(names)} symbols, 2 se", color=INK)
    b.set_title("without AAPL", color=INK, fontsize=10)
    b.legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    half = {t: 0 if t < NOON else 1 for t in STARTS}
    with ProcessPoolExecutor() as pool:
        edges = dict(zip(SYMBOLS, pool.map(edge, SYMBOLS), strict=True))
        base = {s: list(pool.map(touch, [(s, t) for t in STARTS])) for s in SYMBOLS}
        fits = {
            (s, h): calibrate(s, h, [base[s][k][1] for k, t in enumerate(STARTS) if half[t] == h])
            for s in SYMBOLS
            for h in (0, 1)
        }
        sols = dict(zip(fits, pool.map(policies, [f[0] for f in fits.values()]), strict=True))
        jobs = {
            (s, nm, us): pool.map(
                session,
                [(s, nm, t, us, fits[s, 1 - half[t]], sols[s, 1 - half[t]]) for t in STARTS],
            )
            for s in SYMBOLS
            for nm in BOTS
            for us in WIRE
        }
        rows = {k: np.array(list(v)) for k, v in jobs.items()}
    print("share of the 10 s move still ahead after each delay (micro, event / clock sampled)")
    print("symbol  " + " ".join(f"{d / US:>8.0f}" for d in DELAYS))
    for s in SYMBOLS:
        for sig in ("micro", "imbalance"):
            for name in ("event", "clock"):
                v = edges[s][sig, name]
                print(f"{s:6s} {sig:9s} {name:5s} " + " ".join(f"{x:8.3f}" for x in v))
    print("one-way latency, us: $/session (se), adverse c/share, shares/session")
    for s in SYMBOLS:
        for nm in BOTS:
            cells = []
            for us in WIRE:
                r = rows[s, nm, us]
                vol = r[:, 1].sum()
                adv = r[:, 3].sum() / vol if vol else float("nan")
                pnl = r[:, 0] / 100
                se = pnl.std(ddof=1) / np.sqrt(len(r))
                cells.append(f"{pnl.mean():8.1f} ({se:5.1f}) {adv:6.3f} {vol / len(r):6.0f}")
            print(f"{s:6s} {nm:9s} | " + " | ".join(cells))
    with open(ROOT / "data" / "signal_latency.pickle", "wb") as f:
        pickle.dump((edges, rows), f)
    chart(edges, rows, ROOT / "docs" / "signal_latency.png")
