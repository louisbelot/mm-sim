from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
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
    bot,
    msgs,
    style,
)

from mmsim.book import Side
from mmsim.markout import clustered, grouped, markouts
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Simulator

HORIZONS = np.array([1, 5, 30, 60]) / 60
TICKS = ["edge", "1s", "5s", "30s", "1m"]
BUCKETS = ["nobody", "short", "middle", "long"]
NAMES = ["join the touch", "keep priority"]
ONE = TICKS.index("1m")


def session(args):
    symbol, name, start = args
    sim = Simulator(
        ItchFlow(msgs(symbol)), [bot(name, 100, 1000)], book=partial(ReplayBook, sweep=True)
    )
    sim.warm_up(start)
    sim.run(SESSION)
    fills = [f for f in sim.contexts[0].account.fills if f.maker]
    sign = np.array([1.0 if f.side == Side.BUY else -1.0 for f in fills])
    pre = np.array([np.nan if f.mid is None else f.mid for f in fills])
    edge = sign * (pre - [f.price for f in fills])
    m = np.column_stack([edge, markouts(fills, sim.tape, HORIZONS, sim.end)])
    return fills, m, float(sim.tape.at(sim.end))


def queue_groups(rows):
    ahead = np.concatenate([[f.ahead for f in fills] for fills, _, _ in rows])
    cuts = np.quantile(ahead[ahead > 0], [1 / 3, 2 / 3])
    out = []
    for fills, m, price in rows:
        a = np.array([f.ahead for f in fills])
        g = np.where(a == 0, 0, 1 + np.searchsorted(cuts, a, side="left"))
        cleared = np.array([f.left == 0 for f in fills], dtype=int)
        out.append((grouped(fills, m, g, len(BUCKETS)), grouped(fills, m, cleared, 2), price))
    return out, cuts


def collect(pool):
    jobs = {
        (symbol, name): pool.map(session, [(symbol, name, s) for s in STARTS])
        for symbol in SYMBOLS
        for name in NAMES
    }
    out = {}
    for k, v in jobs.items():
        out[k], cuts = queue_groups(list(v))
        print(f"{k[0]:6s} {k[1]:15s} tercile cuts of shares ahead: {cuts[0]:.0f}, {cuts[1]:.0f}")
    return out


def pooled(out, name, which):
    num, den = [], []
    for symbol in SYMBOLS:
        for r in out[symbol, name]:
            n, d = r[which]
            num.append(n / r[2] * 1e4)
            den.append(d)
    return np.array(num), np.array(den)


def row(num, den):
    mean, se = clustered(num, den)
    move = mean - mean[0]
    return mean, se, move


def tables(out):
    curves = {}
    for name in NAMES:
        print(f"\n{name}, six symbols pooled, basis points, se clustered by session")
        print(f"{'':12s} {'shares':>9s} {'edge':>6s}  move 1s  move 1m  1m markout (se)")
        for which, labels in ((0, BUCKETS), (1, ["standing", "cleared"])):
            num, den = pooled(out, name, which)
            for g, label in enumerate(labels):
                if not den[:, g, 0].sum():
                    continue
                mean, se, move = row(num[:, g], den[:, g])
                shares = den[:, g, 0].sum()
                print(
                    f"{label:12s} {shares:9.0f} {mean[0]:6.2f}  {move[1]:7.2f}  {move[ONE]:7.2f}"
                    f"  {mean[ONE]:6.2f} ({se[ONE]:.2f})"
                )
                curves[name, which, g] = mean, se
    print("\nkeep priority, 1-minute markout in bps by symbol: level left standing, level cleared")
    for symbol in SYMBOLS:
        cells = []
        for g in (0, 1):
            num = np.array([r[1][0][g] / r[2] * 1e4 for r in out[symbol, "keep priority"]])
            den = np.array([r[1][1][g] for r in out[symbol, "keep priority"]])
            mean, se = clustered(num, den)
            share = den[:, 0].sum()
            cells.append(f"{mean[ONE]:6.2f} ({se[ONE]:.2f}) {share:8.0f} sh")
        print(f"{symbol:6s} " + "   ".join(cells))
    return curves


def chart(curves, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    style(a)
    x = np.arange(len(BUCKETS))
    w = 0.38
    for k, (name, color) in enumerate(zip(NAMES, (BLUE, ORANGE), strict=True)):
        got = [curves.get((name, 0, g)) for g in range(len(BUCKETS))]
        mean = [c[0][ONE] if c else np.nan for c in got]
        se = [c[1][ONE] if c else np.nan for c in got]
        a.bar(
            x + (k - 0.5) * w,
            mean,
            w,
            yerr=2 * np.array(se),
            color=color,
            edgecolor=SURFACE,
            ecolor=INK,
            capsize=2,
            label=name,
        )
    a.axhline(0, color=MUTED, linewidth=0.8)
    a.set_xticks(x, BUCKETS, color=INK)
    a.set_xlabel("queue ahead at entry, terciles within each symbol", color=INK)
    a.set_ylabel("1-minute markout, basis points", color=INK)
    a.set_title("six symbols pooled, bars are 2 se", color=INK, fontsize=10)
    a.legend(frameon=False, labelcolor=INK, loc="lower right")

    style(b)
    x = np.arange(len(TICKS))
    for g, (label, color) in enumerate((("level left standing", BLUE), ("level cleared", ORANGE))):
        mean, se = curves["keep priority", 1, g]
        b.plot(x, mean, color=color, marker="o", markersize=4, linewidth=1.6, label=label)
        b.fill_between(x, mean - 2 * se, mean + 2 * se, color=color, alpha=0.15, linewidth=0)
    b.axhline(0, color=MUTED, linewidth=0.8)
    b.set_xticks(x, TICKS, color=INK)
    b.set_xlim(-0.3, x[-1] + 0.3)
    b.set_xlabel("time after the fill", color=INK)
    b.set_ylabel("markout, basis points", color=INK)
    b.set_title("keep priority, by what the aggressor left at the price", color=INK, fontsize=10)
    b.legend(frameon=False, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        out = collect(pool)
    curves = tables(out)
    chart(curves, ROOT / "docs" / "queue_fills.png")
