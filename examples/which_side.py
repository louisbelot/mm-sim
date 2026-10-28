import dataclasses
import math
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
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SESSIONS = 300
SKY = CSTParams.sky_perfect()
FLOWS = {
    "one-sided": Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0),
    "two-sided": Hawkes.symmetric(SKY.mu, 0.5, 0.2, 3.0),
}
BASE = Model(1.0, 2.1, 3.6, 0.3, 5)
GRID_A = np.linspace(math.log(0.1), math.log(20.0), 161)
VARIANTS = ("plain", "lean", "sides", "sides+activity", "lean+sides", "lean+sides+activity")
DEAD = (0.1, 0.2, 0.3, 0.5, 0.8)
_TABLES = {}


def tables(model):
    key = (model.sigma, model.A, model.k, model.gamma, model.Q)
    if key not in _TABLES:
        busier = [dataclasses.replace(model, A=model.A * math.exp(g)) for g in GRID_A]
        qs = [m.quotes(math.inf) for m in busier]
        _TABLES[key] = np.array([b for b, _ in qs]), np.array([a for _, a in qs])
    return _TABLES[key]


class Busy(GLFT):
    def __init__(self, model, hawkes, signal, sides):
        super().__init__(model, keep=1, signal=signal, sides=sides)
        self.norm = float(np.sqrt(np.prod(hawkes.mean_rate())))
        self.tb, self.ta = tables(model)

    def targets(self, ctx):
        lam = self.signal.tracker.intensity(ctx.t)
        g = math.log(math.sqrt(lam[0] * lam[1]) / self.norm)
        j = min(max(np.searchsorted(GRID_A, g), 1), len(GRID_A) - 1)
        w = min(max((g - GRID_A[j - 1]) / (GRID_A[j] - GRID_A[j - 1]), 0.0), 1.0)
        self.bids = (1 - w) * self.tb[j - 1] + w * self.tb[j]
        self.asks = (1 - w) * self.ta[j - 1] + w * self.ta[j]
        return super().targets(ctx)


def calibrate(flow):
    sim = Simulator(HawkesFlow(SKY, FLOWS[flow]), seed=11)
    sim.seed(10_000, 10_001)
    sim.warm_up(2000.0)
    times = np.array([p.t for p in sim.prints])
    marks = np.array([int(p.side) for p in sim.prints])
    model = fit(times, marks, 2000.0, 2)
    return model, FlowLean.from_impact(model, times, marks, sim.tape).kappa


def make_bot(bot, variant, hawkes, impact):
    dead = float(variant.split("dead ")[1]) if "dead " in variant else 0.0
    kappa = impact if "lean" in variant or "dead " in variant else 0.0
    signal = None if variant in ("plain", "no lean") else FlowLean(hawkes, kappa, dead=dead)
    if bot == "touch":
        return SymmetricQuoter(0.5, max_position=5, keep=1, signal=signal)
    if "activity" in variant:
        return Busy(BASE, hawkes, signal, "sides" in variant)
    return GLFT(BASE, keep=1, signal=signal, sides="sides" in variant)


def session(args):
    flow, bot, variant, hawkes, impact, seed = args
    agent = make_bot(bot, variant, hawkes, impact)
    sim = Simulator(PairedFlow(SKY, FLOWS[flow]), [agent], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    return sim.contexts[0].pnl()


def compute():
    runs, impacts = {}, {}
    jobs = [(f, "glft", v) for f in FLOWS for v in VARIANTS + tuple(f"dead {d:g}" for d in DEAD)]
    jobs += [("one-sided", "touch", v) for v in ("no lean", "lean")]
    jobs += [("one-sided", "touch", f"dead {d:g}") for d in DEAD]
    calib = {f: calibrate(f) for f in FLOWS}
    with ProcessPoolExecutor() as pool:
        for flow, bot, variant in jobs:
            hawkes, impact = calib[flow]
            impacts[flow] = impact
            args = [(flow, bot, variant, hawkes, impact, s) for s in range(SESSIONS)]
            runs[flow, bot, variant] = np.array(list(pool.map(session, args, chunksize=4)))
    return runs, impacts


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def report(d):
    runs, impacts = d
    for flow, impact in impacts.items():
        print(f"{flow} flow: impact per market order {impact:.3f}")
    print(f"\nGLFT, {SESSIONS} paired sessions, gain over plain GLFT (paired se)")
    print("variant              " + "".join(f"{f:>22s}" for f in FLOWS))
    for v in VARIANTS[1:]:
        cells = []
        for f in FLOWS:
            x = runs[f, "glft", v] - runs[f, "glft", "plain"]
            cells.append(f"{x.mean():+.1f} ({se(x):.1f})")
        print(f"{v:20s} " + "".join(f"{c:>22s}" for c in cells))
    print("\nlean with a dead zone, gain over the plain lean (paired se)")
    rows = [(f, "glft", "lean") for f in FLOWS] + [("one-sided", "touch", "lean")]
    print("dead zone (ticks)       " + "".join(f"{d:>14g}" for d in DEAD))
    for flow, bot, ref in rows:
        cells = []
        for dz in DEAD:
            x = runs[flow, bot, f"dead {dz:g}"] - runs[flow, bot, ref]
            cells.append(f"{x.mean():+.1f} ({se(x):.1f})")
        print(f"{bot} on {flow:10s} " + "".join(f"{c:>14s}" for c in cells))
    print("\nlevels: " + ", ".join(f"{k[0]} {k[1]} {k[2]} {v.mean():.1f}" for k, v in runs.items()))


def chart(d, out):
    runs, _ = d
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    shown = VARIANTS[1:]
    x = np.arange(len(shown))
    w = 0.38
    for j, (flow, color) in enumerate(zip(FLOWS, (BLUE, ORANGE), strict=True)):
        gains = [runs[flow, "glft", v] - runs[flow, "glft", "plain"] for v in shown]
        left.bar(
            x + (j - 0.5) * w,
            [g.mean() for g in gains],
            w,
            yerr=[2 * se(g) for g in gains],
            color=color,
            ecolor=INK,
            capsize=2,
            label=f"{flow} bursts",
        )
    left.axhline(0, color=MUTED, linewidth=1)
    left.set_xticks(x, [v.replace("+", "\n+ ") for v in shown])
    left.set_ylabel("gain over plain GLFT (ticks a session, 2 paired se)")
    left.set_title(f"GLFT on Hawkes flow, {SESSIONS} paired sessions", loc="left", color=INK)
    left.legend(loc="upper right", frameon=False, labelcolor=INK)

    dz = np.array((0.0,) + DEAD)
    for (flow, bot), color, label in (
        (("one-sided", "glft"), BLUE, "GLFT, one-sided bursts"),
        (("two-sided", "glft"), ORANGE, "GLFT, two-sided bursts"),
        (("one-sided", "touch"), AQUA, "join the touch, one-sided bursts"),
    ):
        ref = runs[flow, bot, "lean"]
        diffs = [np.zeros(len(ref))] + [runs[flow, bot, f"dead {d:g}"] - ref for d in DEAD]
        right.errorbar(
            dz,
            [x.mean() for x in diffs],
            yerr=[2 * se(x) if x.any() else 0.0 for x in diffs],
            color=color,
            marker="o",
            markersize=5,
            capsize=3,
            label=label,
        )
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xlabel("dead zone: shifts smaller than this are ignored (ticks)")
    right.set_ylabel("gain over the lean without one (ticks, 2 paired se)")
    right.set_title("Lean only when the drift is worth a tick", loc="left", color=INK)
    right.legend(loc="lower left", frameon=False, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    d = compute()
    report(d)
    chart(d, Path(__file__).resolve().parent.parent / "docs" / "which_side.png")


if __name__ == "__main__":
    main()
