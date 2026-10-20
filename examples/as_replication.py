"""Avellaneda-Stoikov (2008) section 3.3, rerun: tables 1 to 3 and figures 1 and 2.

python examples/as_replication.py  ->  prints the tables, writes docs/as_replication.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import avellaneda as av

N = 10_000
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

print(f"{N} paths per row; the paper used 1000")
print("gamma  strategy   spread  profit  sd(profit)  final q  sd(q)")
for gamma in (0.1, 0.01, 1.0):
    for strategy in ("inventory", "symmetric"):
        r = av.simulate(gamma, strategy, n_paths=N, seed=1)
        print(
            f"{gamma:<6} {strategy:10s} {r.spread:6.2f} {r.profit.mean():7.1f}"
            f" {r.profit.std():11.1f} {r.final_q.mean():8.2f} {r.final_q.std():6.1f}"
        )

inv = av.simulate(0.1, "inventory", n_paths=N, seed=1)
sym = av.simulate(0.1, "symmetric", n_paths=N, seed=1)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

left.plot(inv.t, inv.ask, color=ORANGE, linewidth=1.4)
left.plot(inv.t, inv.mid, color=INK, linewidth=1.4)
left.plot(inv.t, inv.bid, color=BLUE, linewidth=1.4)
for y, label in ((inv.ask[-1], "ask"), (inv.mid[-1], "mid"), (inv.bid[-1], "bid")):
    left.annotate(
        label, (inv.t[-1], y), xytext=(6, 0), textcoords="offset points", va="center", color=INK
    )
left.set_xlim(0, 1.08)
left.set_xlabel("t")
left.set_ylabel("price")
left.set_title("One path, gamma = 0.1: quotes lean against inventory", loc="left", color=INK)

bins = np.arange(0, 132, 3)
counts, _, _ = right.hist(
    inv.profit,
    bins=bins,
    color=BLUE,
    alpha=0.8,
    edgecolor=SURFACE,
    linewidth=0.6,
    label=f"inventory: mean {inv.profit.mean():.1f}, sd {inv.profit.std():.1f}",
)
right.hist(
    sym.profit,
    bins=bins,
    histtype="step",
    color=ORANGE,
    linewidth=2,
    label=f"symmetric: mean {sym.profit.mean():.1f}, sd {sym.profit.std():.1f}",
)
right.legend(loc="upper left", frameon=False, labelcolor=INK)
right.set_xlim(0, 130)
right.set_ylim(0, 1.35 * counts.max())  # room for the legend above the bars
right.set_xlabel("terminal profit")
right.set_ylabel(f"paths (of {N:,})")
right.set_title("Profit distribution, gamma = 0.1", loc="left", color=INK)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "as_replication.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
