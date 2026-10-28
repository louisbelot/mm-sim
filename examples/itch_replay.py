import sys
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.attribution import attribute
from mmsim.bots import SymmetricQuoter
from mmsim.calibrate import calibrate
from mmsim.flow import HawkesFlow, ZeroIntelligence
from mmsim.hawkes import Hawkes
from mmsim.itch import read
from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = sys.argv[1:] or ["CSCO", "INTC", "CMCSA", "MSFT", "AAPL", "QQQ"]
OPEN, CLOSE, SESSION = 575.0, 955.0, 30.0
STARTS = [OPEN + SESSION * k for k in range(int((CLOSE - OPEN) // SESSION))]
SEEDS = range(len(STARTS))
BOTS = {"join the touch": 0, "keep priority": 1}


@cache
def msgs(symbol):
    return read(ROOT / "data" / f"{symbol}.itch")


def bot(name, size, cap):
    return SymmetricQuoter(0.5, size=size, max_position=cap, keep=BOTS[name])


def result(sim, scale):
    ctx = sim.contexts[0]
    a = ctx.account
    att = attribute(a.fills, sim.tape, 1.0, sim.end)
    pnl = a.value(float(sim.tape.at(sim.end)))
    moves = np.count_nonzero(np.asarray(sim.tape.t) > sim.end - SESSION)
    row = np.array([pnl, a.volume, att.spread, att.adverse, att.inventory]) * scale
    return np.append(row, moves)


def real_session(args):
    symbol, name, start = args
    sim = Simulator(ItchFlow(msgs(symbol)), [bot(name, 100, 1000)], book=ReplayBook)
    sim.warm_up(start)
    sim.run(SESSION)
    return result(sim, 1)


def model_session(args):
    name, flow, seed, cst, hk, unit = args
    f = ZeroIntelligence(cst) if hk is None else HawkesFlow(cst, hk)
    sim = Simulator(f, [bot(name, 1, round(1000 / unit))], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(SESSION)
    return result(sim, unit)


def report(symbol, c, hk):
    print(f"{symbol}, 09:35 to 15:55, {c.T:.0f} minutes")
    print(
        f"limit orders {len(c.limit_sizes)}, mean {c.limit_size:.0f} shares,"
        f" {np.mean(c.limit_sizes == 100):.0%} exactly 100"
    )
    print(
        f"market orders {len(c.market_sizes)}, mean {c.market_sizes.mean():.0f} shares,"
        f" hidden volume {c.hidden_shares} of {c.market_sizes.sum() + c.hidden_shares:.0f}"
    )
    print("i   lam/min   theta/min   mean queue")
    cst = c.cst()
    for i in range(len(cst.lam)):
        print(f"{i + 1:<3d} {cst.lam[i]:7.1f}   {cst.theta[i]:9.3f}   {c.queue[:, i].mean():10.0f}")
    print(f"mu {cst.mu:.2f} units of {c.limit_size:.0f} shares per minute per side")
    print(
        f"hawkes decay {1 / hk.beta.diagonal() * 60_000} ms,"
        f" branching {hk.branching.round(3).tolist()}"
    )


def summarize(rows):
    rows = np.array(rows)
    pnl, vol = rows[:, 0], rows[:, 1]
    per = rows[:, :5].sum(axis=0) / vol.sum()
    se = (pnl - per[0] * vol).std(ddof=1) * np.sqrt(len(rows)) / vol.sum()
    return per, se, pnl.mean() / 100, vol.mean(), rows[:, 5].mean() / SESSION


def table(title, out):
    print(f"\n{title}, {len(STARTS)} sessions of {SESSION:.0f} minutes, cents per share traded")
    print(
        "bot              flow      pnl (se)        spread  adverse  inventory  $/session  shares"
        "  moves/min"
    )
    for (name, flow), (per, se, dollars, vol, moves) in out.items():
        print(
            f"{name:16s} {flow:8s} {per[0]:6.3f} ({se:5.3f}) {per[2]:8.3f} {per[3]:8.3f}"
            f" {per[4]:9.3f} {dollars:10.1f} {vol:7.0f} {moves:9.2f}"
        )


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def chart(symbol, c, rows, path):
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.25]}
    )
    style(a)
    width = 10.0
    edges = np.arange(0, c.T + width, width)
    n, _ = np.histogram(c.market_times, edges)
    clock = (OPEN + edges[:-1]) / 60
    a.bar(
        clock,
        n / width,
        width / 60,
        align="edge",
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1,
        label="real, 10-minute buckets",
    )
    a.axhline(
        len(c.market_times) / c.T, color=INK, linestyle="--", linewidth=1.2, label="Poisson model"
    )
    a.set_xticks(range(10, 17), [f"{h}:00" for h in range(10, 17)], color=INK)
    a.set_ylabel(f"{symbol} market orders per minute", color=INK)
    a.legend(frameon=False, labelcolor=INK, loc="upper center")

    style(b)
    names = list(rows)
    x = np.arange(len(names))
    w = 0.26
    parts = (("spread", 2, BLUE), ("adverse, 1 min", 3, ORANGE), ("inventory", 4, AQUA))
    for k, (label, j, color) in enumerate(parts):
        b.bar(
            x + (k - 1) * w,
            [rows[s][0][j] for s in names],
            w,
            color=color,
            label=label,
            edgecolor=SURFACE,
            linewidth=1,
        )
    net = [rows[s][0][0] for s in names]
    b.scatter(x, net, s=46, color=INK, zorder=3, label="net")
    for xi, v in zip(x, net, strict=True):
        b.text(xi, 0.9, f"net {v:+.2f}", ha="center", color=INK, fontsize=8.5)
    b.axhline(0, color=MUTED, linewidth=0.8)
    b.set_xticks(x, names, color=INK)
    b.set_ylabel("join the touch, cents per share traded", color=INK)
    b.legend(frameon=False, labelcolor=INK, ncol=4, loc="lower left", fontsize=8.5)
    lo = min(min(rows[s][0][3] for s in names), min(net))
    b.set_ylim(lo - 0.45, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    first = SYMBOLS[0]
    c = calibrate(msgs(first), OPEN, CLOSE)
    hk = c.hawkes()
    report(first, c, hk)
    cst = c.cst()
    hk = Hawkes(hk.mu * c.market_sizes.mean() / c.limit_size, hk.alpha, hk.beta)
    with ProcessPoolExecutor() as pool:
        jobs = {}
        for name in BOTS:
            jobs[name, "real"] = pool.map(real_session, [(first, name, s) for s in STARTS])
            for flow, model in (("Poisson", None), ("Hawkes", hk)):
                args = [(name, flow, s, cst, model, c.limit_size) for s in SEEDS]
                jobs[name, flow] = pool.map(model_session, args)
        table(
            f"{first}, real flow against the calibrated models",
            {k: summarize(list(v)) for k, v in jobs.items()},
        )
        jobs = {
            (name, symbol): pool.map(real_session, [(symbol, name, s) for s in STARTS])
            for symbol in SYMBOLS
            for name in BOTS
        }
        out = {k: summarize(list(v)) for k, v in jobs.items()}
    table("real flow by symbol", out)
    touch = {symbol: out["join the touch", symbol] for symbol in SYMBOLS}
    chart(first, c, touch, ROOT / "docs" / "itch_replay.png")
