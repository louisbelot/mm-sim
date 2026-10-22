import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import glosten as gm
from mmsim import kyle

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"

v = np.linspace(60, 140, 321)
prior = np.exp(-0.5 * ((v - 100) / 10) ** 2)
prior /= prior.sum()
n_trades, n_paths = 150, 200

rng = np.random.default_rng(0)
values = rng.choice(v, size=n_paths, p=prior)
spreads, errors = {}, {}
for alpha in (0.1, 0.3, 0.5):
    m = gm.Model(alpha, noise=5.0, eta=0.4)
    s, e, pnl, informed = [], [], [], []
    for i, value in enumerate(values):
        path = gm.simulate(m, v, prior, value, n_trades, seed=i)
        s.append(path.ask - path.bid)
        e.append(np.abs(path.mean[:-1] - value))
        x = path.mm_pnl(value)
        pnl.append(x.sum())
        informed.append(-x[path.informed].sum())
    spreads[alpha], errors[alpha] = np.mean(s, axis=0), np.mean(e, axis=0)
    pnl = np.array(pnl)
    print(
        f"alpha {alpha}: spread {spreads[alpha][0]:.2f} -> {spreads[alpha][-1]:.2f}, "
        f"mean abs error {errors[alpha][0]:.2f} -> {errors[alpha][-1]:.2f}, "
        f"mm pnl per session {pnl.mean():.2f} (se {pnl.std() / math.sqrt(n_paths):.2f}), "
        f"insiders {np.mean(informed):.1f}"
    )

m = gm.Model(0.3, noise=5.0, eta=0.4)
demo = gm.simulate(m, v, prior, 112.0, n_trades, seed=4)

Sigma0, sigma_u = 1.0, 1.0
eqs = {N: kyle.solve(N, Sigma0, sigma_u) for N in (1, 4, 16, 256)}
print("  N   lambda_1   lambda_N   Sigma_N   insider profit")
for N in (1, 2, 4, 16, 64, 256, 4096):
    eq = eqs.get(N) or kyle.solve(N, Sigma0, sigma_u)
    print(f"{N:4d} {eq.lam[0]:10.4f} {eq.lam[-1]:10.4f} {eq.Sigma[-1]:9.4f} {eq.profit:12.4f}")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
(a, b), (c, d) = axes

k = np.arange(n_trades)
a.fill_between(k, demo.bid, demo.ask, color=BLUE, alpha=0.18, linewidth=0, label="bid to ask")
a.plot(np.arange(n_trades + 1), demo.mean, color=BLUE, linewidth=1.8, label="E[V | trades]")
a.axhline(112.0, color=ORANGE, linewidth=1.5, linestyle="--", label="true value V = 112")
a.set_xlim(0, n_trades)
a.set_xlabel("trade number")
a.set_ylabel("price")
a.set_title("Glosten-Milgrom: one session, 30% informed", loc="left", color=INK)
a.legend(loc="lower right", frameon=False, labelcolor=INK)

for (alpha, s), color in zip(spreads.items(), (AQUA, BLUE, ORANGE), strict=True):
    b.plot(k, s, color=color, linewidth=2, label=f"{alpha:.0%} informed")
b.set_xlim(0, n_trades)
b.set_ylim(0, None)
b.set_xlabel("trade number")
b.set_ylabel("ask minus bid")
b.set_title(f"Average spread over {n_paths} sessions", loc="left", color=INK)
b.legend(loc="upper right", frameon=False, labelcolor=INK)

for (N, eq), color in zip(eqs.items(), (MUTED, AQUA, BLUE, ORANGE), strict=True):
    c.step(
        eq.t,
        np.append(eq.lam, eq.lam[-1]),
        where="post",
        color=color,
        linewidth=1.8,
        label=f"N = {N}",
    )
c.axhline(
    math.sqrt(Sigma0) / sigma_u, color=INK, linewidth=1, linestyle=":", label="continuous time"
)
c.set_xlim(0, 1)
c.set_ylim(0, 1.15)
c.set_xlabel("time")
c.set_ylabel("lambda (price impact per unit of order flow)")
c.set_title("Kyle: price impact through the session", loc="left", color=INK)
c.legend(loc="lower left", frameon=False, labelcolor=INK, ncol=2)

for (N, eq), color in zip(eqs.items(), (MUTED, AQUA, BLUE, ORANGE), strict=True):
    d.plot(
        eq.t,
        eq.Sigma / Sigma0,
        color=color,
        linewidth=1.8,
        marker="o" if N < 20 else None,
        markersize=4,
        label=f"N = {N}",
    )
d.plot([0, 1], [1, 0], color=INK, linewidth=1, linestyle=":", label="continuous time")
d.set_xlim(0, 1)
d.set_ylim(0, 1.02)
d.set_xlabel("time")
d.set_ylabel("Var(V | order flow) / Var(V)")
d.set_title("Kyle: information left in the insider's hands", loc="left", color=INK)
d.legend(loc="upper right", frameon=False, labelcolor=INK)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "why_spreads.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
