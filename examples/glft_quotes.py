import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.glft import Model

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

m = Model(sigma=0.3, A=0.9, k=0.3, gamma=0.01, Q=30)
bid, ask = m.quotes(math.inf)
approx_bid, _ = m.approx()

print(f"alpha {m.alpha:.3g}  eta {m.eta:.4f}  half-spread term {m.base:.4f}")
print("  q   exact bid   closed form   exact ask")
for q in (-29, -20, -10, -1, 0, 1, 10, 20, 29):
    i = q + m.Q
    print(f"{q:4d} {bid[i]:10.4f} {approx_bid[i]:12.4f} {ask[i]:11.4f}")
print(f"stationary spread at q = 0: {bid[m.Q] + ask[m.Q]:.4f}")

T = 600.0
t = np.linspace(0, T, 601)
paths = {q: [] for q in (-20, 0, 20)}
for tau in T - t:
    b, _ = m.quotes(tau)
    for q in paths:
        paths[q].append(b[q + m.Q])

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

q = m.q[:-1]
left.plot(q, bid[:-1], color=BLUE, linewidth=2, label="exact (matrix exponential)")
left.plot(q, approx_bid[:-1], color=ORANGE, linewidth=2, linestyle="--", label="closed form")
left.set_xlabel("inventory q (lots)")
left.set_ylabel("mid minus bid (ticks)")
left.set_title("Stationary bid quote, inventory bound Q = 30", loc="left", color=INK)
left.legend(loc="upper left", frameon=False, labelcolor=INK)

for (q, b), color in zip(paths.items(), (AQUA, BLUE, ORANGE), strict=True):
    right.plot(t, b, color=color, linewidth=2)
    right.annotate(f"q = {q}", (t[0], b[0]), xytext=(4, 6), textcoords="offset points", color=INK)
right.axhline(m.base, color=MUTED, linewidth=1, linestyle=":")
right.annotate(
    "terminal quote (1/gamma) ln(1 + gamma/k)",
    (T / 2, m.base),
    xytext=(0, -8),
    textcoords="offset points",
    ha="center",
    va="top",
    color=MUTED,
)
right.set_xlim(0, T)
right.set_ylim(1.5, 5.0)
right.set_xlabel("t (seconds), session ends at T = 600")
right.set_ylabel("mid minus bid (ticks)")
right.set_title("Bid quote through the session", loc="left", color=INK)

fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "glft_quotes.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")

skew = m.gamma * m.sigma**2 * T * 20
print(f"for comparison, AS at t = 0 with q = 20 moves its reservation price by {skew:.1f} ticks")
