from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import hidden as h

PATHS = 100_000
DEPTH = h.World(phi=0.01, eta=0.05)
PAPER = h.World(
    T=10.0, dt=0.02, A=(2.0, 10.0), k=(25.0, 25.0), sigma=0.1, Q=3, phi=10.0, rate=(5.0, 5.0)
)
RATES = (0.25, 1.0, 4.0, 16.0)
BELIEFS = (0.0, 0.25, 0.5, 0.6, 0.75, 1.0)


def solve(w):
    full = h.informed(w)
    bayes, ce = h.bayes(w), h.ce(w)
    ce = h.Plan(h.evaluate(w, ce.bid, ce.ask), ce.bid, ce.ask)
    return {
        "informed": full,
        "bayes": bayes,
        "ce": ce,
        "qmdp": h.qmdp(w, full),
        "plugin": h.plugin(w, full),
        "blind": h.blind(w, ce),
    }


def sweep(rate):
    w = h.World(phi=0.01, eta=0.05, rate=(rate, rate))
    p = solve(w)
    return {k: h.start(w, v) for k, v in p.items()}


def report(title, w, plans):
    print(f"{title}: A {w.A}, k {w.k}, switching {w.rate}, T {w.T:g}, {PATHS:,} paths")
    cols = f"{'exact':>9s} {'simulated':>19s} {'spread':>7s} {'fills':>6s} {'|q|':>5s}"
    print(f"{'quoter':9s} {cols}")
    base = None
    for name, plan in plans.items():
        quote = h.oracle(plan) if name == "informed" else h.believer(plan)
        run = h.simulate(w, quote, PATHS, 1)
        rew = run.reward
        se = rew.std() / np.sqrt(PATHS)
        diff = ""
        if name == "bayes":
            base = rew
        elif base is not None:
            d = base - rew
            diff = f"  bayes - this {d.mean():.4f} +- {d.std() / np.sqrt(PATHS):.4f}"
        print(
            f"{name:9s} {h.start(w, plan):9.4f} {rew.mean():9.4f} +- {se:6.4f}"
            f" {np.nanmean(run.spread):7.3f} {run.fills.mean():6.1f}"
            f" {np.abs(run.q).mean():5.2f}{diff}"
        )
    G = plans["bayes"].bid.shape[1]
    i = w.Q
    print("bid depth at t = 0, q = 0 by belief in regime 1")
    print("belief   " + " ".join(f"{p:7.2f}" for p in BELIEFS))
    for name in ("bayes", "ce", "qmdp"):
        row = [plans[name].bid[0, round(p * (G - 1)), i] for p in BELIEFS]
        print(f"{name:8s} " + " ".join(f"{x:7.3f}" for x in row))
    full = plans["informed"].bid[0, :, i]
    mix = [(1 - p) * full[0] + p * full[1] for p in BELIEFS]
    print("inf mix  " + " ".join(f"{x:7.3f}" for x in mix))
    print(f"informed regime 0 {full[0]:.3f}, regime 1 {full[1]:.3f}, crossing {w.crossing:.3f}\n")


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        worlds = list(ex.map(solve, (DEPTH, PAPER)))
        rates = list(ex.map(sweep, RATES))
    report("depth regimes", DEPTH, worlds[0])
    report("liquidity regimes, Campi-Zabaljauregui parameters", PAPER, worlds[1])
    print("switching rate sweep, depth regimes, exact values at pi0 = 0.5")
    print("rate   " + " ".join(f"{k:>9s}" for k in rates[0]))
    for rate, row in zip(RATES, rates, strict=True):
        print(f"{rate:6.2f} " + " ".join(f"{v:9.3f}" for v in row.values()))

    w, plans = DEPTH, worlds[0]
    G = plans["bayes"].bid.shape[1]
    pi = np.linspace(0, 1, G)

    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COLORS = {"informed": "#1f9e6e", "bayes": "#2a78d6", "ce": "#eb6834", "blind": "#52514e"}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    left, mid, right = axes

    i = w.Q
    bayes, ce = plans["bayes"].bid[0, :, i], plans["ce"].bid[0, :, i]
    left.plot(pi, bayes, color=COLORS["bayes"], lw=2, label="Bayes (plans to learn)")
    left.plot(pi, ce, color=COLORS["ce"], lw=2, ls="--", label="certainty equivalent")
    full = plans["informed"].bid[0, :, i]
    left.scatter([0, 1], full, color=COLORS["informed"], zorder=3, s=36, label="regime known")
    left.set_xlabel("belief in the thin regime")
    left.set_ylabel("bid depth at t = 0, flat")
    left.set_title("Quotes on the belief", loc="left", color=INK)
    left.legend(loc="upper left", frameon=False)

    d = np.linspace(0, 4, 400)
    for p, ls in ((0.5, "-"), (0.2, "--")):
        mi = h.information(w, p, d, d)
        mid.plot(d, 1e3 * mi, color=INK, lw=1.6, ls=ls, label=f"belief {p:g}")
    mid.axvline(w.crossing, color=MUTED, lw=1, ls=":")
    mid.annotate(
        "fills say nothing",
        (w.crossing, 0),
        xytext=(w.crossing + 0.2, 0.6 * 1e3 * h.information(w, 0.5, 0.0, 0.0)),
        color=MUTED,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
    )
    mid.set_xlabel("depth quoted on both sides")
    mid.set_ylabel("information per step (millinats)")
    mid.set_title("What one step of fills reveals", loc="left", color=INK)
    mid.legend(loc="upper right", frameon=False)

    x = np.array(RATES)
    for name, label in (("bayes", "Bayes"), ("ce", "certainty equivalent"), ("blind", "blind")):
        gap = [r["informed"] - r[name] for r in rates]
        right.plot(x, gap, color=COLORS[name], lw=2, marker="o", ms=4, label=label)
    right.set_xscale("log")
    right.set_xticks(x, [f"{v:g}" for v in x])
    right.set_ylim(bottom=0)
    right.set_xlabel("regime switching rate (per session)")
    right.set_ylabel("reward lost against knowing the regime")
    right.set_title("Cost of not seeing the regime", loc="left", color=INK)
    right.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "hidden_flow.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
