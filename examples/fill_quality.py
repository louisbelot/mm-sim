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
    result,
    style,
)

from mmsim.markout import clustered, markouts, totals
from mmsim.replay import Cause, ItchFlow, ReplayBook, label
from mmsim.sim import Simulator

HORIZONS = np.array([0, 1, 5, 30, 60, 300]) / 60
TICKS = ["0", "1s", "5s", "30s", "1m", "5m"]
RULES = {"no impact": False, "sweep": True}
NAMES = ["join the touch", "keep priority"]
ONE = TICKS.index("1m")


def session(args):
    symbol, name, sweep, start = args
    b = bot(name, 100, 1000)
    sim = Simulator(ItchFlow(msgs(symbol)), [b], book=partial(ReplayBook, sweep=sweep))
    sim.warm_up(start)
    sim.run(SESSION)
    fills = sim.contexts[0].account.fills
    causes = label(sim.book, fills)
    m = markouts(fills, sim.tape, HORIZONS, sim.end)
    price = float(sim.tape.at(sim.end))
    by = np.array(
        [
            totals(
                [f for f, k in zip(fills, causes, strict=True) if k == c], m[np.equal(causes, c)]
            )
            for c in Cause
        ]
    )
    return result(sim, 1), by, price


def collect(pool):
    jobs = {
        (symbol, name, rule): pool.map(session, [(symbol, name, sweep, s) for s in STARTS])
        for symbol in SYMBOLS
        for name in NAMES
        for rule, sweep in RULES.items()
    }
    return {k: list(v) for k, v in jobs.items()}


def rule_table(out):
    print("cents per share traded, 12 sessions of 30 minutes, 100-share quotes")
    print(
        "symbol  bot              rule        shares  swept   gross (se)   adverse  $/session"
        "   swept 1m   queue 1m"
    )
    for (symbol, name, rule), rows in out.items():
        res = np.array([r[0] for r in rows])
        by = np.array([r[1] for r in rows])
        pnl, vol = res[:, 0], res[:, 1]
        mean, se = clustered(pnl, vol)
        adverse = res[:, 3].sum() / vol.sum()
        swept = by[:, Cause.SWEEP, 1, 0].sum() / vol.sum()
        mark = []
        for c in (Cause.SWEEP, Cause.QUEUE):
            num, den = by[:, c, 0, ONE], by[:, c, 1, ONE]
            mark.append(f"{num.sum() / den.sum():10.3f}" if den.sum() else f"{'':>10s}")
        print(
            f"{symbol:7s} {name:16s} {rule:9s} {vol.mean():8.0f} {swept:6.1%} {mean:7.3f}"
            f" ({se:5.3f}) {adverse:8.3f} {pnl.mean() / 100:10.1f} {mark[0]} {mark[1]}"
        )


def bps(rows):
    by = np.array([r[1] for r in rows])
    price = np.array([r[2] for r in rows])
    return by[:, :, 0] / price[:, None, None] * 1e4, by[:, :, 1]


def cause_table(out):
    print("\nmarkout by cause, basis points of price, join the touch, sweep rule, six symbols")
    print(f"cause    shares  {'  '.join(f'{t:>12s}' for t in TICKS)}")
    num, den = [], []
    for symbol in SYMBOLS:
        n, d = bps(out[symbol, "join the touch", "sweep"])
        num.append(n)
        den.append(d)
    num, den = np.concatenate(num), np.concatenate(den)
    curves = {}
    for c in Cause:
        if not den[:, c, 0].any():
            continue
        mean, se = clustered(num[:, c], den[:, c])
        curves[c] = mean, se
        cells = "  ".join(f"{m:6.2f} ({s:4.2f})" for m, s in zip(mean, se, strict=True))
        print(f"{c.name.lower():6s} {den[:, c, 0].sum():8.0f}  {cells}")
    return curves


def chart(out, curves, path):
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.1]}
    )
    style(a)
    x = np.arange(len(SYMBOLS))
    w = 0.38
    for k, (c, color) in enumerate(((Cause.QUEUE, BLUE), (Cause.SWEEP, ORANGE))):
        by = [np.array([r[1] for r in out[s, "join the touch", "sweep"]]) for s in SYMBOLS]
        mean, se = zip(*(clustered(v[:, c, 0, ONE], v[:, c, 1, ONE]) for v in by), strict=True)
        a.bar(
            x + (k - 0.5) * w,
            mean,
            w,
            yerr=2 * np.array(se),
            color=color,
            edgecolor=SURFACE,
            ecolor=INK,
            capsize=2,
            label=f"{c.name.lower()} fills",
        )
    a.axhline(0, color=MUTED, linewidth=0.8)
    a.set_xticks(x, SYMBOLS, color=INK)
    a.set_ylabel("1-minute markout, cents per share", color=INK)
    a.set_title("join the touch, sweep rule, bars are 2 se", color=INK, fontsize=10)
    a.legend(frameon=False, labelcolor=INK, loc="lower left")

    style(b)
    x = np.arange(len(TICKS))
    for c, color in ((Cause.QUEUE, BLUE), (Cause.SWEEP, ORANGE)):
        mean, se = curves[c]
        b.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label=f"{c.name.lower()} fills",
        )
        b.fill_between(x, mean - 2 * se, mean + 2 * se, color=color, alpha=0.15, linewidth=0)
    b.axhline(0, color=MUTED, linewidth=0.8)
    b.set_xticks(x, TICKS, color=INK)
    b.set_xlim(-0.3, x[-1] + 0.3)
    b.set_xlabel("time after the fill", color=INK)
    b.set_ylabel("markout, basis points, six symbols pooled", color=INK)
    b.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        out = collect(pool)
    rule_table(out)
    curves = cause_table(out)
    chart(out, curves, ROOT / "docs" / "fill_quality.png")
