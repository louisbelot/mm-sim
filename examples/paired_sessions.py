from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import GLFT, FlowLean, SymmetricQuoter
from mmsim.flow import CSTParams, HawkesFlow, PairedFlow
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

SESSIONS = 300
SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
KAPPAS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
GLFT_NAME, TOUCH = "GLFT, gamma 0.3", "join the touch"
PAIRS = [(GLFT_NAME, 0.4, 0.5), (GLFT_NAME, 0.0, 0.5), (TOUCH, 0.2, 0.4), (TOUCH, 0.0, 0.4)]
KINDS = ("HawkesFlow", "PairedFlow")


def history(T, seed):
    sim = Simulator(HawkesFlow(SKY, HAWKES), seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(T)
    times = np.array([p.t for p in sim.prints])
    marks = np.array([int(p.side) for p in sim.prints])
    return times, marks


def make_bot(name, signal):
    if name == TOUCH:
        return SymmetricQuoter(0.5, max_position=5, keep=1, signal=signal)
    return GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1, signal=signal)


def session(args):
    kind, name, model, kappa, seed = args
    signal = None if kappa == 0 else FlowLean(model, kappa, 0.5)
    flow = HawkesFlow(SKY, HAWKES) if kind == "HawkesFlow" else PairedFlow(SKY, HAWKES)
    sim = Simulator(flow, [make_bot(name, signal)], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    return sim.contexts[0].pnl()


def compute():
    times, marks = history(2000.0, seed=11)
    model = fit(times, marks, 2000.0, 2)
    configs = {(GLFT_NAME, k) for k in KAPPAS} | {(n, k) for n, a, b in PAIRS for k in (a, b)}
    runs = {}
    with ProcessPoolExecutor() as pool:
        for kind in KINDS:
            for name, kappa in sorted(configs):
                jobs = [(kind, name, model, kappa, s) for s in range(SESSIONS)]
                runs[kind, name, kappa] = np.array(list(pool.map(session, jobs, chunksize=4)))
    return runs


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def report(runs):
    print(f"{SESSIONS} sessions per row, the same seeds on both flows, P&L in ticks")
    print("comparison                  flow        corr  difference  paired se  unpaired se  ratio")
    for name, a, b in PAIRS:
        for kind in KINDS:
            x, y = runs[kind, name, a], runs[kind, name, b]
            d = y - x
            unpaired = np.hypot(se(x), se(y))
            print(
                f"{f'{name} {a:g} -> {b:g}':26s}  {kind:10s}  {np.corrcoef(x, y)[0, 1]:4.2f}"
                f"  {d.mean():10.1f}  {se(d):9.2f}  {unpaired:11.2f}"
                f"  {(unpaired / se(d)) ** 2:5.1f}"
            )
    print(f"\n{GLFT_NAME}, difference from kappa 0.5 (paired se)")
    print("kappa   " + "  ".join(f"{k:>12g}" for k in KAPPAS))
    for kind in KINDS:
        ref = runs[kind, GLFT_NAME, 0.5]
        cells = []
        for k in KAPPAS:
            d = runs[kind, GLFT_NAME, k] - ref
            cells.append(f"{d.mean():6.1f} ({se(d):.1f})" if k != 0.5 else f"{'0':>12s}")
        print(f"{kind:10s} " + "  ".join(f"{c:>12s}" for c in cells))
    for kind in KINDS:
        cells = [f"{runs[kind, GLFT_NAME, k].mean():6.1f}" for k in KAPPAS]
        print(f"{kind:10s} level  " + "  ".join(f"{c:>12s}" for c in cells))


def chart(runs, out):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for kind, color in zip(KINDS, (ORANGE, BLUE), strict=True):
        x, y = runs[kind, GLFT_NAME, 0.4], runs[kind, GLFT_NAME, 0.5]
        r = np.corrcoef(x, y)[0, 1]
        left.scatter(x, y, s=9, color=color, alpha=0.55, label=f"{kind}, correlation {r:.2f}")
    lo = min(left.get_xlim()[0], left.get_ylim()[0])
    hi = max(left.get_xlim()[1], left.get_ylim()[1])
    left.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    left.set_xlim(lo, hi)
    left.set_ylim(lo, hi)
    left.set_xlabel("P&L with kappa 0.4 (ticks)")
    left.set_ylabel("P&L with kappa 0.5, same seed (ticks)")
    left.set_title(f"GLFT, the same {SESSIONS} seeds twice", loc="left", color=INK)
    left.legend(loc="upper left", frameon=False, labelcolor=INK)

    k = np.array(KAPPAS)
    for j, (kind, color) in enumerate(zip(KINDS, (ORANGE, BLUE), strict=True)):
        ref = runs[kind, GLFT_NAME, 0.5]
        diffs = [runs[kind, GLFT_NAME, x] - ref for x in KAPPAS]
        means = np.array([d.mean() for d in diffs])
        ses = np.array([se(d) if x != 0.5 else 0.0 for d, x in zip(diffs, KAPPAS, strict=True)])
        right.errorbar(
            k + (j - 0.5) * 0.012,
            means,
            yerr=2 * ses,
            color=color,
            marker="o",
            markersize=5,
            capsize=3,
            label=kind,
        )
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xlabel("kappa, ticks per expected order over the next 0.5 minutes")
    right.set_ylabel("P&L minus P&L at kappa 0.5, same seed (ticks, 2 se bars)")
    right.set_title(f"GLFT sweep, {SESSIONS} sessions per point", loc="left", color=INK)
    right.legend(loc="lower right", frameon=False, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    runs = compute()
    report(runs)
    chart(runs, Path(__file__).resolve().parent.parent / "docs" / "paired_sessions.png")


if __name__ == "__main__":
    main()
