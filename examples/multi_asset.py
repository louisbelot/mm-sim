import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.multi import Exact, Logistic, Market, simulate

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

rfq = Logistic(30, 0.7, 30)
N, T, z = (7, 30), 7.0, 1e4
drift = [0.1, -0.1]


def market(rho, mu=drift):
    return Market.of([1.2, 0.6], rho, rfq, 8e-6, z=z, mu=mu)


m = market(0.5)
ex = Exact(m, N, T)
print("two assets, sigma 1.2 and 0.6 a day, rho 0.5, drift +-0.1, gamma 8e-6, lots of 10,000")
print("limits 70,000 and 300,000, T = 7 days; quotes in euros from the reference price")
print("   q1     q2   bid1 exact  closed   ask1 exact  closed", end="")
print("   bid2 exact  closed   ask2 exact  closed")
inv = np.array([[0, 0], [2, 0], [-2, 0], [0, 8], [0, -8], [2, 8]])
eb, ea = ex.policy(T, inv)
ab, aa = m.policy(math.inf, inv)
for j, (n1, n2) in enumerate(inv):
    cols = (eb[j, 0], ab[j, 0], ea[j, 0], aa[j, 0], eb[j, 1], ab[j, 1], ea[j, 1], aa[j, 1])
    print(
        f"{n1 * z:6.0f} {n2 * z:6.0f} "
        + " ".join(f"{c:11.4f}" if i % 2 == 0 else f"{c:7.4f}" for i, c in enumerate(cols))
    )

print(f"\nstationary A (per share^2): {m.stationary().ravel()}")
print(f"target inventory -A^-1 B / 2: {m.target()}")
print(f"Markowitz Sigma^-1 mu / gamma: {np.linalg.solve(m.Sigma, m.mu) / m.gamma}")

rhos = [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.8, 0.9, 0.95]
print("\nvalue of each strategy over 7 days (expected P&L minus the risk penalty, exact)")
print("  rho  drift     optimal  closed form  independent  closed/opt  indep/opt  target q1, q2")
table = {}
for mu in ([0.0, 0.0], drift):
    for rho in rhos:
        mk = market(rho, mu)
        e = Exact(mk, N, T)
        best = e.value()
        joint = e.evaluate(mk.policy)[N]
        alone = e.evaluate(mk.independent().policy)[N]
        target = np.round(mk.target()) + 0.0
        table[mu[0], rho] = (best, joint, alone)
        print(
            f"{rho:5.2f} {mu[0]:6.1f} {best:11.0f} {joint:12.0f} {alone:12.0f}"
            f" {joint / best:11.4f} {alone / best:10.4f}   {target[0]:9.0f} {target[1]:9.0f}"
        )

print("\nMonte Carlo check, rho 0.5 with drift, 2000 paths, dt 0.001 day")
print("strategy      evaluated   simulated   (se)")
for name, pol in (
    ("optimal", ex.policy),
    ("closed form", m.policy),
    ("independent", m.independent().policy),
):
    want = ex.value() if name == "optimal" else ex.evaluate(pol)[N]
    o = simulate(m, pol, T, N, dt=1e-3, paths=2000, seed=5).objective
    print(f"{name:12} {want:10.0f} {o.mean():11.0f} {o.std() / math.sqrt(len(o)):7.0f}")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

n2 = np.arange(-N[1], N[1] + 1)
for n1, color in ((-3, BLUE), (0, INK), (3, ORANGE)):
    inv = np.column_stack([np.full_like(n2, n1), n2])
    eb, _ = ex.policy(T, inv)
    cb, _ = m.policy(math.inf, inv)
    left.plot(n2 * z / 1e3, eb[:, 0], color=color, lw=1.8, label=f"q1 = {n1 * z / 1e3:.0f}k, exact")
    left.plot(n2 * z / 1e3, cb[:, 0], color=color, lw=1.2, ls="--")
left.plot([], [], color=MUTED, lw=1.2, ls="--", label="closed form")
left.set_xlabel("inventory of asset 2 (thousand shares)")
left.set_ylabel("bid quote of asset 1 (euros below reference)")
left.set_title("Asset 1's bid moves with asset 2's inventory (rho 0.5)", color=INK, fontsize=10)
left.legend(frameon=False, fontsize=8)

for mu0, color, tag in ((0.0, BLUE, "no drift"), (0.1, ORANGE, "drift +-0.1")):
    joint = [table[mu0, r][1] / table[mu0, r][0] for r in rhos]
    alone = [table[mu0, r][2] / table[mu0, r][0] for r in rhos]
    right.plot(rhos, joint, color=color, lw=1.8, marker="o", ms=4, label=f"closed form, {tag}")
    right.plot(
        rhos, alone, color=color, lw=1.2, ls="--", marker="s", ms=3.5, label=f"independent, {tag}"
    )
right.set_ylim(0.15, 1.03)
right.set_xlabel("correlation rho")
right.set_ylabel("share of the optimal value")
right.set_title("What ignoring correlation costs", color=INK, fontsize=10)
right.legend(frameon=False, fontsize=8, loc="lower left")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "multi_asset.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"\nwrote {out}")
