from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import GLFT, AvellanedaStoikov, FlowLean, SymmetricQuoter
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.sim import Simulator
from mmsim.tournament import session, standings

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

SESSIONS = 100
SKY = CSTParams.sky_perfect()
TRUE = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
MODEL = Model(1.0, 2.1, 3.6, 0.3, 5)
HORIZON = 0.5
NAMES = ["join the touch", "AS, gamma 0.001", "GLFT", "GLFT + lean"]


def poisson():
    return ZeroIntelligence(SKY)


def hawkes():
    return HawkesFlow(SKY, TRUE)


def make(name, hk, kappa):
    match name:
        case "join the touch":
            return SymmetricQuoter(0.5)
        case "AS, gamma 0.001":
            return AvellanedaStoikov(0.001, 1.0, 3.6)
        case "GLFT":
            return GLFT(MODEL, keep=1)
        case "GLFT + lean":
            return GLFT(MODEL, keep=1, signal=FlowLean(hk, kappa, HORIZON))


def play(args):
    flow, names, hk, kappa, seed = args
    pick = {n: partial(make, n, hk, kappa) for n in names}
    return session(pick, seed, flow=flow)


def calibrate():
    sim = Simulator(HawkesFlow(SKY, TRUE), seed=11)
    sim.seed(10_000, 10_001)
    sim.warm_up(2000.0)
    times = np.array([p.t for p in sim.prints])
    marks = np.array([int(p.side) for p in sim.prints])
    hk = fit(times, marks, 2000.0, 2)
    return hk, FlowLean.fit(hk, times, marks, sim.tape, HORIZON).kappa


def table(label, flow, hk, kappa, pool):
    seeds = range(SESSIONS)
    together = list(pool.map(play, [(flow, NAMES, hk, kappa, s) for s in seeds]))
    solo = {n: list(pool.map(play, [(flow, [n], hk, kappa, s) for s in seeds])) for n in NAMES}
    alone = [{n: solo[n][i][n] for n in NAMES} for i in range(SESSIONS)]
    rows = standings(together, alone)
    print(f"\n{label}, {SESSIONS} sessions, pnl in ticks")
    print("bot               mean (se)     alone    crowding (se)   sd     mean/sd  wins   fills")
    for r in rows:
        c, ce = r.crowding
        print(
            f"{r.name:17s} {r.mean:6.1f} ({r.se:4.1f}) {r.solo.mean():7.1f} {c:9.1f} ({ce:4.1f})"
            f" {r.pnl.std(ddof=1):6.1f} {r.sharpe:7.2f} {r.wins:6.2f} {r.fills.mean():6.0f}"
        )
    return rows


def chart(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    x = np.arange(len(rows))
    w = 0.38
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a))  # noqa: E731
    ax.bar(
        x - w / 2,
        [r.solo.mean() for r in rows],
        w,
        color=BLUE,
        label="alone",
        yerr=[se(r.solo) for r in rows],
        ecolor=INK,
    )
    ax.bar(
        x + w / 2,
        [r.mean for r in rows],
        w,
        color=ORANGE,
        label="all four together",
        yerr=[r.se for r in rows],
        ecolor=INK,
    )
    ax.set_xticks(x, [r.name for r in rows], color=INK)
    ax.set_ylabel("mean P&L per session on Hawkes flow (ticks)", color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    hk, kappa = calibrate()
    print(f"fitted lean kappa {kappa:.3f}")
    with ProcessPoolExecutor() as pool:
        table("Poisson flow", poisson, hk, kappa, pool)
        rows = table("Hawkes flow", hawkes, hk, kappa, pool)
    chart(rows, Path(__file__).resolve().parents[1] / "docs" / "tournament.png")
