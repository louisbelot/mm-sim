from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import robust as r

PATHS = 100_000
game = r.Game()
rng = np.random.default_rng(99)
random = r.Market(
    rng.uniform(*game.b, PATHS), rng.uniform(*game.A, PATHS), rng.uniform(*game.k, PATHS)
)

rows = [
    ("symmetric 1/k", *r.symmetric(game)),
    ("AS gamma 0.1", *r.avellaneda(game, 0.1)),
]
for eta, zeta in ((0, 0), (1, 0), (0.5, 0), (0.1, 0), (0.01, 0), (0, 0.01), (0, 0.001)):
    s = r.best_response(r.Game(eta=eta, zeta=zeta))
    rows.append((f"fixed eta {eta} zeta {zeta}", s.bid, s.ask))
for c in ("b", "A", "k", "bAk"):
    s = r.equilibrium(game, c)
    rows.append((f"strategic {c}", s.bid, s.ask))
for eta, zeta in ((1, 0), (0.1, 0), (0, 0.01), (0, 0.001)):
    s = r.equilibrium(r.Game(eta=eta, zeta=zeta))
    rows.append((f"strategic bAk eta {eta} zeta {zeta}", s.bid, s.ask))

print(f"{PATHS:,} paths against the fixed market (b 0, A 140, k 1.5), q0 = 0")
print(
    f"{'market maker':32s} {'wealth':>13s} {'sharpe':>7s} {'final q':>12s} {'spread':>7s}"
    f" {'exact':>7s} {'random':>7s} {'worst b':>8s} {'worst':>7s}"
)
points = {}
for name, bid, ask in rows:
    run = r.simulate(game, bid, ask, paths=PATHS, seed=1)
    rnd = r.simulate(game, bid, ask, random, paths=PATHS, seed=2)
    exact = r.evaluate(game, bid, ask)[0, game.Q]
    worst_b = r.worst_case(game, bid, ask, "b").V[0, game.Q]
    worst = r.worst_case(game, bid, ask).V[0, game.Q]
    w, q = run.wealth, run.q
    points[name] = exact, worst
    print(
        f"{name:32s} {w.mean():6.1f} +- {w.std():4.1f} {w.mean() / w.std():7.2f}"
        f" {q.mean():5.2f} +- {q.std():4.2f} {np.nanmean(run.spread):7.3f}"
        f" {exact:7.2f} {rnd.wealth.mean():7.2f} {worst_b:8.2f} {worst:7.2f}"
    )

DATA = Path(__file__).resolve().parent.parent / "data"
learned = {}
for tag, label in (("fixed", "PPO, fixed market"), ("bAk", "PPO, RARL b A k")):
    for f in sorted(DATA.glob(f"arl_{tag}_*.npz")):
        t = np.load(f)
        v = r.evaluate(game, t["bid"], t["ask"])[0, game.Q]
        learned.setdefault(label, []).append(
            (v, r.worst_case(game, t["bid"], t["ask"]).V[0, game.Q])
        )

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1f9e6e", "#8a5cd1"]
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

tables = {name: (bid, ask) for name, bid, ask in rows}
qs = game.q
for color, (name, label) in zip(
    COLORS,
    (
        ("fixed eta 0 zeta 0", "risk neutral"),
        ("fixed eta 0 zeta 0.01", "zeta 0.01"),
        ("strategic b", "drift adversary"),
        ("strategic bAk", "b, A, k adversary"),
    ),
    strict=True,
):
    bid, ask = tables[name]
    offset = (ask[0] - bid[0]) / 2
    left.plot(qs, offset, color=color, linewidth=2, label=label)
left.axhline(0, color=MUTED, linewidth=0.8)
left.set_xlim(-12, 12)
left.set_ylim(-1.6, 1.6)
left.set_xlabel("inventory q")
left.set_ylabel("reservation offset (ask - bid) / 2")
left.set_title("Quotes at t = 0 lean against inventory", loc="left", color=INK)
left.legend(loc="upper right", frameon=False, labelcolor=INK)

for x, y in points.values():
    right.scatter(x, y, s=36, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
FLOOR = 16
for label in learned:
    color = COLORS[1] if "fixed" in label else COLORS[3]
    xy = np.array(learned[label])
    below = xy[xy[:, 1] < FLOOR]
    xy = xy[xy[:, 1] >= FLOOR]
    if len(below):
        worst = ", ".join(f"{y:.1f}" for y in sorted(below[:, 1], reverse=True))
        right.annotate(
            f"{label}: worst case {worst}, below the axis",
            (0.02, 0.04),
            xycoords="axes fraction",
            color=INK,
        )
        right.scatter(
            below[:, 0],
            np.full(len(below), FLOOR + 0.4),
            s=48,
            marker="v",
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=4,
            clip_on=False,
        )
    right.scatter(
        *xy.T,
        s=48,
        marker="D",
        color=color,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=4,
        label=label,
    )
notes = {
    "symmetric 1/k": ("risk neutral", (-6, 6)),
    "strategic bAk": ("b, A, k equilibrium", (-6, 6)),
    "strategic b": ("drift equilibrium", (6, 4)),
    "AS gamma 0.1": ("AS gamma 0.1", (0, -14)),
    "fixed eta 1 zeta 0": ("terminal penalty 1", (-6, -14)),
    "strategic k": ("k adversary", (-6, -14)),
}
for name, (text, off) in notes.items():
    x, y = points[name]
    ha = "right" if off[0] < 0 else "left"
    right.annotate(text, (x, y), xytext=off, textcoords="offset points", color=INK, ha=ha)
right.set_xlabel("expected wealth, fixed market")
right.set_ylabel("expected wealth, worst-case adversary")
right.set_title("Value against the fixed market vs the guarantee", loc="left", color=INK)
right.set_ylim(FLOOR, 37)
if learned:
    right.legend(loc="center left", frameon=False, labelcolor=INK)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "robust_mm.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
