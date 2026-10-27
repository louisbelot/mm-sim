from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.attribution import attribute
from mmsim.book import Side
from mmsim.bots import GLFT, FlowLean, SymmetricQuoter
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import markouts
from mmsim.sim import Print, Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c8384a"

SESSIONS = 200
SWEEP = 1000
HISTORIES = range(100, 140)
SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
SPLIT = 1.0
KAPPAS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
BOTS = ("join the touch", "GLFT, gamma 0.3")
ROWS = [
    (bot, lean, "hawkes", informed)
    for informed in (False, True)
    for bot in BOTS
    for lean in (False, True)
] + [("GLFT, gamma 0.3", lean, "poisson", False) for lean in (False, True)]


def history(T, seed):
    sim = Simulator(HawkesFlow(SKY, HAWKES), seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(T)
    times = np.array([p.t for p in sim.prints])
    marks = np.array([int(p.side) for p in sim.prints])
    return times, marks, sim.tape


def estimates(args):
    model, seed = args
    times, marks, tape = history(2000.0, seed)
    reg = FlowLean.fit(model, times, marks, tape, 0.5).kappa
    return reg, FlowLean.from_impact(model, times, marks, tape).kappa


def horizon_ratio(model, times, marks):
    short, full = FlowLean(model, 1.0, 0.5), FlowLean(model, 1.0)
    x = np.empty((len(times), 2))
    for n, (t, m) in enumerate(zip(times, marks, strict=True)):
        p = Print(t, Side(m), 0, 1)
        short.observe(p)
        full.observe(p)
        x[n] = short.shift(t), full.shift(t)
    return float(x[:, 0] @ x[:, 1] / (x[:, 0] @ x[:, 0]))


def make_bot(name, signal):
    if name == "join the touch":
        return SymmetricQuoter(0.5, max_position=5, keep=1, signal=signal)
    return GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1, signal=signal)


def session(args):
    name, lean, kind, informed, seed = args
    signal = None if lean is None else FlowLean(*lean)
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


def trace(lean, seed, T):
    sim = Simulator(HawkesFlow(SKY, HAWKES), seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(T)
    grid = np.arange(0, T, 0.002)
    lean = FlowLean(*lean)
    shift = np.empty(len(grid))
    k = 0
    for n, t in enumerate(grid):
        while k < len(sim.prints) and sim.prints[k].t <= t:
            lean.observe(sim.prints[k])
            k += 1
        shift[n] = lean.shift(t)
    burst = grid[np.argmax(np.abs(shift))]
    sel = (grid >= burst - 2.5) & (grid <= burst + 1.5)
    prints = [(p.t - burst, int(p.side)) for p in sim.prints if -2.5 <= p.t - burst <= 1.5]
    mids = sim.tape.at(grid[sel])
    return grid[sel] - burst, mids - mids[0], shift[sel], prints


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def compute():
    T = 2000.0
    times, marks, tape = history(T, seed=11)
    model = fit(times, marks, T, 2)
    d = {
        "orders": len(times),
        "model": model,
        "fitted": {h: FlowLean.fit(model, times, marks, tape, h).kappa for h in (0.25, 0.5, 1, 2)},
        "impact": FlowLean.from_impact(model, times, marks, tape).kappa,
        "ratio": horizon_ratio(model, times, marks),
        "results": {},
        "sweep": {},
    }
    lean = (model, d["impact"])
    with ProcessPoolExecutor() as pool:
        d["estimates"] = np.array(list(pool.map(estimates, [(model, s) for s in HISTORIES])))
        for row in ROWS:
            name, leaning, kind, informed = row
            jobs = [(name, lean if leaning else None, kind, informed, s) for s in range(SESSIONS)]
            d["results"][row] = list(pool.map(session, jobs, chunksize=4))
        for name in BOTS:
            for kappa in KAPPAS:
                signal = None if kappa == 0 else (model, kappa, 0.5)
                jobs = [(name, signal, "hawkes", False, s) for s in range(SWEEP)]
                pnl = [r[0] for r in pool.map(session, jobs, chunksize=4)]
                d["sweep"][name, kappa] = np.array(pnl)
    d["trace"] = trace(lean, 3, 600.0)
    return d


def cubic(d):
    glft = [d["sweep"]["GLFT, gamma 0.3", k] for k in KAPPAS]
    return np.polyfit(KAPPAS, [p.mean() for p in glft], 3, w=[1 / se(p) for p in glft])


def report(d):
    print(f"{d['orders']} market orders, fitted branching {d['model'].branching.round(3).tolist()}")
    for h, k in d["fitted"].items():
        print(f"regression kappa at horizon {h:g} min: {k:.3f}")
    print(
        f"impact per market order {d['impact']:.3f}; the full forecast is {d['ratio']:.3f} times"
        f" the 0.5-minute one, so on the 0.5-minute scale that is {d['impact'] * d['ratio']:.3f}"
    )
    reg, imp = d["estimates"][:, 0], d["estimates"][:, 1] * d["ratio"]
    print(f"\nkappa on the 0.5-minute scale, {len(HISTORIES)} independent 2000-minute histories")
    for label, k in (("regression", reg), ("impact", imp)):
        print(
            f"{label:10s} mean {k.mean():.3f}  sd {k.std(ddof=1):.3f}"
            f"  range {k.min():.3f} to {k.max():.3f}"
        )

    results = d["results"]
    print(
        "\nstrategy          lean  flow     informed  mean P&L (se)  fills"
        "  spread  adverse  inventory  mk 0/5 min"
    )
    for row, r in results.items():
        name, leaning, kind, informed = row
        pnl = np.array([x[0] for x in r])
        parts = np.array([x[2] for x in r]).mean(axis=0)
        m = np.concatenate([x[3] for x in r])
        print(
            f"{name:17s} {'yes' if leaning else 'no':5s} {kind:8s} {'yes' if informed else 'no':9s}"
            f" {pnl.mean():7.1f} ({se(pnl):.1f})"
            f" {np.mean([x[1] for x in r]):6.0f} {parts[0]:7.1f} {parts[1]:8.1f} {parts[2]:10.1f}"
            f"  {np.nanmean(m[:, 0]):5.2f} {np.nanmean(m[:, 1]):5.2f}"
        )
    for name, kind, informed in (
        ("join the touch", "hawkes", False),
        ("GLFT, gamma 0.3", "hawkes", False),
        ("GLFT, gamma 0.3", "hawkes", True),
        ("GLFT, gamma 0.3", "poisson", False),
    ):
        diff = np.array([x[0] for x in results[name, True, kind, informed]]) - np.array(
            [x[0] for x in results[name, False, kind, informed]]
        )
        tag = f"{name}, {kind}{', informed' if informed else ''}"
        print(f"paired lean gain, {tag}: {diff.mean():.1f} (se {se(diff):.1f})")

    sweep = d["sweep"]
    print(f"\nkappa ({SWEEP} sessions)  " + "  ".join(f"{k:>11g}" for k in KAPPAS))
    for name in BOTS:
        cells = [f"{sweep[name, k].mean():6.1f} ({se(sweep[name, k]):.1f})" for k in KAPPAS]
        print(f"{name:20s} " + "  ".join(cells))
    c = cubic(d)
    fine = np.linspace(0, 1, 1001)
    peak = fine[np.argmax(np.polyval(c, fine))]
    print(f"GLFT, cubic through the sweep: peak {np.polyval(c, peak):.1f} at kappa {peak:.2f}")
    for label, k in (("regression", reg), ("impact", imp)):
        v = np.sort(np.polyval(c, k))
        print(
            f"  expected P&L with the {label} fit on 2000 minutes: {v.mean():.1f},"
            f" worst tenth {v[: len(v) // 10].mean():.1f}"
        )


def chart(d, out):
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

    t, mids, shift, prints = d["trace"]
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

    k = np.array(KAPPAS)
    for name, color in (("join the touch", ORANGE), ("GLFT, gamma 0.3", BLUE)):
        means = np.array([d["sweep"][name, x].mean() for x in KAPPAS])
        ses = np.array([se(d["sweep"][name, x]) for x in KAPPAS])
        mid.errorbar(
            k, means, yerr=2 * ses, color=color, marker="o", markersize=5, capsize=3, label=name
        )
    fine = np.linspace(0, 1, 201)
    mid.plot(fine, np.polyval(cubic(d), fine), color=BLUE, linewidth=1, alpha=0.5)
    mid.axhline(0, color=MUTED, linewidth=1)
    lo, hi = mid.get_ylim()
    lo -= 0.25 * (hi - lo)
    reg, imp = d["estimates"][:, 0], d["estimates"][:, 1] * d["ratio"]
    for j, (label, kk, color) in enumerate(
        (("kappa from the regression, 40 histories", reg, MUTED), ("from the impact", imp, INK))
    ):
        y = lo + (0.03 + 0.1 * j) * (hi - lo)
        mid.plot(kk, np.full(len(kk), y), "|", color=color, markersize=8)
        mid.text(0.0, y + 0.045 * (hi - lo), label, color=color, fontsize=8, va="center")
    mid.set_ylim(lo, hi)
    mid.set_xlabel("kappa, ticks per expected order over the next 0.5 minutes")
    mid.set_ylabel(f"mean P&L per session (ticks, 2 se bars, {SWEEP} sessions)")
    mid.set_title("How hard to lean, Hawkes flow, no informed trader", loc="left", color=INK)
    mid.legend(loc="upper right", frameon=False, labelcolor=INK)

    shown = [r for r in ROWS if r[2] == "hawkes" and not r[3]]
    x = np.arange(len(shown))
    parts = np.array([np.array([v[2] for v in d["results"][r]]).mean(axis=0) for r in shown])
    totals = [np.mean([v[0] for v in d["results"][r]]) for r in shown]
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
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    d = compute()
    report(d)
    chart(d, Path(__file__).resolve().parent.parent / "docs" / "flow_lean.png")


if __name__ == "__main__":
    main()
