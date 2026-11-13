from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import regime as g

PATHS = 100_000
DRIFT = g.World(phi=0.01)
VOL = g.World(b=(0.0, 0.0), sigma=(1.0, 4.0), phi=0.05)


def quoters(w):
    full = g.informed(w)
    plan, _ = g.bayes(w)
    mm, bl = g.minimax(w), g.blind(w)
    AS = g.avellaneda(w, 2 * w.phi)
    return {
        "informed": (g.oracle(full), g.start(w, full.V)),
        "bayes": (g.believer(plan), g.believed(w, plan)),
        "qmdp": (g.qmdp(w, full), None),
        "blind": (g.table(bl.bid, bl.ask), g.start(w, g.evaluate(w, bl.bid, bl.ask))),
        "minimax": (g.table(mm.bid, mm.ask), g.start(w, g.evaluate(w, mm.bid, mm.ask))),
        f"AS gamma {2 * w.phi:g}": (g.table(*AS), g.start(w, g.evaluate(w, *AS))),
    }


for title, w in (("drift regimes b = -5 / +5", DRIFT), ("volatility regimes 1 / 4", VOL)):
    print(f"{title}, switching rate {w.rate[0]:g}, phi {w.phi:g}, {PATHS:,} paths")
    print(f"{'quoter':14s} {'exact':>7s} {'reward':>15s} {'sharpe':>6s} {'spread':>6s} {'|q|':>5s}")
    rows = quoters(w)
    base = None
    for name, (quote, exact) in rows.items():
        run = g.simulate(w, quote, PATHS, 1)
        rew = run.reward
        if name == "bayes":
            base = rew
        diff = ""
        if name == "qmdp":
            d = rew - base
            diff = f"  bayes - qmdp {-d.mean():.3f} +- {d.std() / np.sqrt(PATHS):.3f}"
        ex = f"{exact:7.2f}" if exact is not None else " " * 7
        print(
            f"{name:14s} {ex} {rew.mean():6.2f} +- {rew.std():5.2f} {rew.mean() / rew.std():6.2f}"
            f" {np.nanmean(run.spread):6.3f} {np.abs(run.q).mean():5.2f}{diff}"
        )
    full, bayes, blind = (rows[k][1] for k in ("informed", "bayes", "blind"))
    print(f"share of the value of information captured: {(bayes - blind) / (full - blind):.3f}\n")

rates = np.geomspace(0.25, 64, 13)
curves = {k: [] for k in ("informed", "bayes", "blind", "minimax", "AS")}
for rate in rates:
    w = g.World(phi=0.01, rate=(rate, rate))
    curves["informed"].append(g.start(w, g.informed(w).V))
    curves["bayes"].append(g.believed(w, g.bayes(w)[0]))
    for name, s in (("blind", g.blind(w)), ("minimax", g.minimax(w))):
        curves[name].append(g.start(w, g.evaluate(w, s.bid, s.ask)))
    curves["AS"].append(g.start(w, g.evaluate(w, *g.avellaneda(w, 0.02))))
print("switching rate sweep, drift regimes")
print("rate   " + " ".join(f"{k:>9s}" for k in curves))
for i, rate in enumerate(rates):
    print(f"{rate:6.2f} " + " ".join(f"{curves[k][i]:9.2f}" for k in curves))

w = DRIFT
plan, _ = g.bayes(w)
bl = g.blind(w)
SEED = 4
path = g.simulate(w, g.believer(plan), 1, SEED, keep=True)
flat = g.simulate(w, g.table(bl.bid, bl.ask), 1, SEED, keep=True)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"informed": "#1f9e6e", "bayes": "#2a78d6", "blind": "#52514e", "minimax": "#eb6834"}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

labels = {
    "informed": "sees the regime",
    "bayes": "Bayes on the filter",
    "blind": "best fixed-drift quoter",
    "minimax": "minimax drift quoter",
}
for name, label in labels.items():
    left.plot(rates, curves[name], color=COLORS[name], linewidth=2, label=label)
left.fill_between(rates, curves["blind"], curves["bayes"], color=COLORS["bayes"], alpha=0.12)
left.set_xscale("log")
left.set_xticks([0.25, 1, 4, 16, 64], ["0.25", "1", "4", "16", "64"])
left.set_xlabel("regime switching rate (per session)")
left.set_ylabel("expected reward")
left.set_title("Value of knowing the drift regime", loc="left", color=INK)
left.legend(loc="upper right", frameon=False, labelcolor=INK)

t = np.arange(w.N) * w.dt
s = path.s[:, 0]
edges = np.flatnonzero(np.diff(s)) + 1
for a, b in zip(np.r_[0, edges], np.r_[edges, w.N], strict=True):
    if s[a] == 1:
        right.axvspan(t[a], t[b - 1] + w.dt, color=COLORS["informed"], alpha=0.12, linewidth=0)
right.plot(t, path.pi[:, 0], color=COLORS["bayes"], linewidth=2, label="filter P(up regime)")
right.set_ylim(-0.02, 1.02)
right.set_xlim(0, w.T)
right.set_xlabel("time (session = 1)")
right.set_ylabel("belief")
right.set_title("One session: up regime shaded", loc="left", color=INK)
inv = right.twinx()
inv.plot(t, path.inventory[:, 0], color=INK, linewidth=1.2, label="inventory, Bayes")
inv.plot(
    t, flat.inventory[:, 0], color=MUTED, linewidth=1.2, linestyle="--", label="inventory, blind"
)
inv.set_ylabel("inventory")
lim = max(np.abs(path.inventory).max(), np.abs(flat.inventory).max()) + 2
inv.set_ylim(-lim, lim)
inv.spines["top"].set_visible(False)
lines = right.get_legend_handles_labels()
more = inv.get_legend_handles_labels()
right.legend(lines[0] + more[0], lines[1] + more[1], loc="lower left", frameon=False)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "regime_mm.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
