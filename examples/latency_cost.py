from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import GLFT, SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import markouts
from mmsim.sim import Latency, Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

SESSIONS = 200
SECONDS = (0, 0.5, 1, 2, 5, 10, 20, 30)
H = np.array([0, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30])
LEGS = {
    "feed": (10, 0, 0),
    "entry": (0, 10, 0),
    "response": (0, 0, 10),
    "feed 5 + entry 5": (5, 5, 0),
    "all three": (10, 10, 10),
}
LINES = (
    ("join the touch", True, ORANGE, "-"),
    ("GLFT, gamma 0.3", True, BLUE, "-"),
    ("GLFT, gamma 0.3", False, BLUE, "--"),
)


def make_bot(name):
    if name == "join the touch":
        return SymmetricQuoter(0.5, max_position=5, keep=1)
    return GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1)


def session(args):
    name, legs, informed, seed = args
    bot = make_bot(name)
    agents = [bot]
    if informed:
        agents.append(InformedTrader(Fundamental(1.0, seed=seed), 1.0, seed=seed))
    lat = Latency(*(s / 60 for s in legs))
    sim = Simulator(
        ZeroIntelligence(CSTParams.sky_perfect()), agents, seed=seed, latency={bot: lat}
    )
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    m = markouts(fills, sim.tape, H, sim.end)
    stale = np.array([f.stale for f in fills], dtype=bool)
    worst = max(abs(s.position) for s in ctx.history)
    return ctx.pnl(), m, stale, worst


def summary(r):
    pnl = np.array([x[0] for x in r])
    m = np.concatenate([x[1] for x in r])
    stale = np.concatenate([x[2] for x in r])
    over = np.mean([x[3] > 5 for x in r])
    return pnl, m, stale, over


def line(label, r):
    pnl, m, stale, over = summary(r)
    se = pnl.std() / np.sqrt(len(pnl))
    fresh = np.nanmean(m[~stale][:, [0, 6]], axis=0)
    picked = np.nanmean(m[stale][:, [0, 6]], axis=0) if stale.any() else [np.nan] * 2
    return (
        f"{label} {pnl.mean():8.1f} ({se:.1f}) {pnl.std():7.1f} {len(m) / len(r):6.0f}"
        f" {stale.mean():6.3f}  {fresh[0]:5.2f} {fresh[1]:5.2f}"
        f"  {picked[0]:5.2f} {picked[1]:5.2f} {over:5.2f}"
    )


def main():
    sweep, legs = {}, {}
    with ProcessPoolExecutor() as pool:

        def run(name, lat, informed):
            jobs = [(name, lat, informed, s) for s in range(SESSIONS)]
            return list(pool.map(session, jobs, chunksize=4))

        for name, informed, *_ in LINES:
            for sec in SECONDS:
                sweep[name, informed, sec] = run(name, (sec, sec, sec), informed)
        for informed in (False, True):
            for label, lat in LEGS.items():
                legs[label, informed] = run("GLFT, gamma 0.3", lat, informed)

    head = "mean P&L (se)  sd P&L  fills  stale  fresh mk 0/5  stale mk 0/5  over cap"
    print(f"strategy         informed  latency (s)  {head}")
    for (name, informed, sec), r in sweep.items():
        print(line(f"{name:16s} {'yes' if informed else 'no':9s} {sec:11g} ", r))
    print(f"\nGLFT, gamma 0.3  informed  10 s on      {head}")
    for (label, informed), r in legs.items():
        print(line(f"{'':16s} {'yes' if informed else 'no':9s} {label:16s}", r))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    x = np.array(SECONDS, dtype=float)
    for name, informed, color, style in LINES:
        pnl = [summary(sweep[name, informed, sec])[0] for sec in SECONDS]
        mean = np.array([p.mean() for p in pnl])
        se = np.array([p.std() / np.sqrt(len(p)) for p in pnl])
        left.errorbar(
            x,
            mean,
            yerr=2 * se,
            color=color,
            linestyle=style,
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            label=f"{name}, {'informed flow' if informed else 'noise only'}",
        )
    left.axhline(0, color=MUTED, linewidth=1)
    left.set_xlim(-0.5, 31)
    left.set_xlabel("one-way latency on every leg (seconds)")
    left.set_ylabel("mean P&L per session (ticks, 2 se bars)")
    left.set_title("What latency costs", loc="left", color=INK)
    left.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=9)

    for sec, color in zip((0, 2, 10, 30), (AQUA, BLUE, VIOLET, ORANGE), strict=True):
        m = summary(sweep["GLFT, gamma 0.3", True, sec])[1]
        right.plot(H, np.nanmean(m, axis=0), color=color, linewidth=2, label=f"{sec:g} s")
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xlim(0, 30)
    right.set_xlabel("minutes after the fill")
    right.set_ylabel("mean markout per lot (ticks)")
    right.set_title("GLFT fills with informed flow, by latency", loc="left", color=INK)
    right.legend(loc="upper right", frameon=False, labelcolor=INK, title="latency")

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "latency_cost.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
