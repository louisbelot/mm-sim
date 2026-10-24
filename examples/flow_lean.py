from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.attribution import attribute
from mmsim.bots import GLFT, FlowLean, SymmetricQuoter
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import markouts
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c8384a"

SESSIONS = 200
SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
HORIZON = 0.5
SPLIT = 1.0
MULTS = (0, 0.5, 1, 1.5, 2, 2.5, 3)
ROWS = [
    (bot, lean, "hawkes", informed)
    for informed in (False, True)
    for bot in ("join the touch", "GLFT, gamma 0.3")
    for lean in (False, True)
] + [("GLFT, gamma 0.3", lean, "poisson", False) for lean in (False, True)]


def history(T, seed):
    flow = HawkesFlow(SKY, HAWKES)
    sim = Simulator(flow, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(T)
    times = np.array([t for t, _ in flow.markets])
    marks = np.array([int(s) for _, s in flow.markets])
    return times, marks, sim.tape


def make_bot(name, signal):
    if name == "join the touch":
        return SymmetricQuoter(0.5, max_position=5, keep=1, signal=signal)
    return GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1, signal=signal)


def session(args):
    name, model, kappa, kind, informed, seed = args
    signal = None if kappa == 0 else FlowLean(model, kappa, HORIZON)
    agents = [make_bot(name, signal)]
    if informed:
        agents.append(InformedTrader(Fundamental(1.0, seed=seed), 1.0, seed=seed))
    flow = HawkesFlow(SKY, HAWKES) if kind == "hawkes" else ZeroIntelligence(SKY)
    sim = Simulator(flow, agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    a = attribute(fills, sim.tape, SPLIT, sim.end)
    m = markouts(fills, sim.tape, [0, 5], sim.end)
    return ctx.pnl(), len(fills), (a.spread, a.adverse, a.inventory), m


def trace(model, kappa, seed, T):
    flow = HawkesFlow(SKY, HAWKES)
    sim = Simulator(flow, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(T)
    grid = np.arange(0, T, 0.002)
    lean = FlowLean(model, kappa, HORIZON)
    shift = np.empty(len(grid))
    k = 0
    for n, t in enumerate(grid):
        while k < len(flow.markets) and flow.markets[k][0] <= t:
            lean.tracker.update(*flow.markets[k])
            k += 1
        shift[n] = lean.shift(t)
    burst = grid[np.argmax(np.abs(shift))]
    sel = (grid >= burst - 2.5) & (grid <= burst + 1.5)
    prints = [(t - burst, int(side)) for t, side in flow.markets if -2.5 <= t - burst <= 1.5]
    mids = sim.tape.at(grid[sel])
    return grid[sel] - burst, mids - mids[0], shift[sel], prints


def main():
    T = 2000.0
    times, marks, tape = history(T, seed=11)
    model = fit(times, marks, T, 2)
    print(f"{len(times)} market orders, fitted branching {model.branching.round(3).tolist()}")
    for h in (0.25, 0.5, 1.0, 2.0):
        print(f"kappa at horizon {h:g} min: {FlowLean.fit(model, times, marks, tape, h).kappa:.3f}")
    kappa = FlowLean.fit(model, times, marks, tape, HORIZON).kappa

    results, sweep = {}, {}
    with ProcessPoolExecutor() as pool:
        for row in ROWS:
            name, lean, kind, informed = row
            k = kappa if lean else 0
            jobs = [(name, model, k, kind, informed, s) for s in range(SESSIONS)]
            results[row] = list(pool.map(session, jobs, chunksize=4))
        for name in ("join the touch", "GLFT, gamma 0.3"):
            for mult in MULTS:
                jobs = [(name, model, mult * kappa, "hawkes", False, s) for s in range(SESSIONS)]
                sweep[name, mult] = np.array([r[0] for r in pool.map(session, jobs, chunksize=4)])

    print(
        "\nstrategy          lean  flow     informed  mean P&L (se)  fills"
        "  spread  adverse  inventory  mk 0/5 min"
    )
    for row, r in results.items():
        name, lean, kind, informed = row
        pnl = np.array([x[0] for x in r])
        parts = np.array([x[2] for x in r]).mean(axis=0)
        m = np.concatenate([x[3] for x in r])
        print(
            f"{name:17s} {'yes' if lean else 'no':5s} {kind:8s} {'yes' if informed else 'no':9s}"
            f" {pnl.mean():7.1f} ({pnl.std() / np.sqrt(len(pnl)):.1f})"
            f" {np.mean([x[1] for x in r]):6.0f} {parts[0]:7.1f} {parts[1]:8.1f} {parts[2]:10.1f}"
            f"  {np.nanmean(m[:, 0]):5.2f} {np.nanmean(m[:, 1]):5.2f}"
        )
    for name, kind, informed in (
        ("join the touch", "hawkes", False),
        ("GLFT, gamma 0.3", "hawkes", False),
        ("GLFT, gamma 0.3", "hawkes", True),
        ("GLFT, gamma 0.3", "poisson", False),
    ):
        d = np.array([x[0] for x in results[name, True, kind, informed]]) - np.array(
            [x[0] for x in results[name, False, kind, informed]]
        )
        tag = f"{name}, {kind}{', informed' if informed else ''}"
        print(f"paired lean gain, {tag}: {d.mean():.1f} (se {d.std() / np.sqrt(len(d)):.1f})")
    print("\nkappa multiple  " + "  ".join(f"{m:>6g}" for m in MULTS))
    for name in ("join the touch", "GLFT, gamma 0.3"):
        print(f"{name:15s} " + "  ".join(f"{sweep[name, m].mean():6.1f}" for m in MULTS))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    left, mid, right = axes

    t, mids, shift, prints = trace(model, kappa, 3, 600.0)
    left.plot(t, mids, color=MUTED, linewidth=1.2, drawstyle="steps-post", label="mid")
    left.plot(t, mids + shift, color=BLUE, linewidth=1.4, label="mid + lean")
    base = min(mids.min(), (mids + shift).min()) - 1.0
    for s, side in prints:
        left.plot(s, base - 0.5 * side, "|", color=(AQUA, RED)[side], markersize=9)
    left.text(-2.45, base + 0.35, "market buys (top), sells (bottom)", color=MUTED, fontsize=9)
    left.set_xlim(-2.5, 1.5)
    left.set_xlabel("minutes from the largest lean in 600 minutes")
    left.set_ylabel("ticks from the start of the window")
    left.set_title("What the lean does in a burst", loc="left", color=INK)
    left.legend(loc="upper right", frameon=False, labelcolor=INK)

    for name, color in (("join the touch", ORANGE), ("GLFT, gamma 0.3", BLUE)):
        means = np.array([sweep[name, m].mean() for m in MULTS])
        ses = np.array([sweep[name, m].std() / np.sqrt(SESSIONS) for m in MULTS])
        mid.errorbar(
            MULTS, means, yerr=2 * ses, color=color, marker="o", markersize=5, capsize=3, label=name
        )
    mid.axhline(0, color=MUTED, linewidth=1)
    mid.axvline(1, color=MUTED, linewidth=1, linestyle="--")
    mid.text(1.05, mid.get_ylim()[1], "regression kappa", color=MUTED, fontsize=9, va="top")
    mid.set_xlabel("lean as a multiple of the fitted kappa")
    mid.set_ylabel("mean P&L per session (ticks, 2 se bars)")
    mid.set_title("How hard to lean, Hawkes flow, no informed trader", loc="left", color=INK)
    mid.legend(loc="lower right", frameon=False, labelcolor=INK)

    shown = [r for r in ROWS if r[2] == "hawkes" and not r[3]]
    x = np.arange(len(shown))
    parts = np.array([np.array([v[2] for v in results[r]]).mean(axis=0) for r in shown])
    totals = [np.mean([v[0] for v in results[r]]) for r in shown]
    w = 0.22
    for j, (label, color) in enumerate(
        (("spread", AQUA), (f"adverse, first {SPLIT:g} min", RED), ("inventory", ORANGE))
    ):
        right.bar(x + (j - 1) * w, parts[:, j], w, color=color, label=label)
    right.plot(x, totals, "D", color=INK, markersize=6, label="P&L")
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_ylim(parts.min() * 1.15, parts.max() * 1.45)
    short = {"join the touch": "touch", "GLFT, gamma 0.3": "GLFT"}
    right.set_xticks(x, [f"{short[r[0]]}\n{'lean' if r[1] else 'no lean'}" for r in shown])
    right.set_ylabel("ticks per session")
    right.set_title("Where the money goes, Hawkes flow", loc="left", color=INK)
    right.legend(loc="upper center", ncol=2, frameon=False, labelcolor=INK, fontsize=9)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "flow_lean.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
