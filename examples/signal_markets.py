import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import prediction as pm

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

ALPHA, RHO, TRADES, MARKETS = 0.5, 0.6, 400, 10_000
a_e = ALPHA * (2 * RHO - 1)
b_star = 1 / math.log((1 + a_e) / (1 - a_e))


def ls(v, b0):
    a = v / (2 * math.log(2))
    return lambda n: pm.LSLMSR(v, n, x=b0 / (2 * a))


def run(make):
    return pm.run(make, ALPHA, TRADES, MARKETS, rho=RHO, seed=7)


print(f"binary event, p0 = 0.5, {TRADES} trades, {MARKETS} markets")
print(f"signal bettors: share {ALPHA}, accuracy {RHO}; the rest trade at random")
print(f"to a quoter the flow looks like informed share alpha (2 rho - 1) = {a_e:.2f}")
print(f"LMSR depth with the flow's likelihood ratio: b* = {b_star:.3f}\n")

families = {
    "quoter": [
        (f"quoter, assumes {k:g} x", lambda n, k=k: pm.Quoter(a_e * k, n), k)
        for k in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9)
    ],
    "lmsr": [
        (f"LMSR b = {b:.3g}", lambda n, b=b: pm.LMSR(b, n), b)
        for b in (1.2, 1.5, 2.0, 3.0, b_star, 7.0, 10.0, 15.0)
    ],
    "fpmm": [
        (f"FPMM F = {F:g}", lambda n, F=F: pm.FPMM(F, n), F)
        for F in (1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0)
    ],
    "ls1": [(f"LS-LMSR v 0.01, b0 = {b0:g}", ls(0.01, b0), b0) for b0 in (1.5, 2, 3, 5, 8, 12)],
    "ls3": [(f"LS-LMSR v 0.03, b0 = {b0:g}", ls(0.03, b0), b0) for b0 in (1.5, 2, 3, 5, 8, 12)],
}
print(
    "maker                       subsidy   (se)  Brier   log score  volume  bettor P&L  noise P&L"
)
points, runs = {}, {}
for fam, rows in families.items():
    pts = []
    for name, make, _ in rows:
        r = run(make)
        runs[name] = r
        sub = -r.maker.mean()
        se = r.maker.std(ddof=1) / math.sqrt(MARKETS)
        pts.append((sub, r.brier()))
        with np.errstate(divide="ignore"):
            log = r.log_score()
        print(
            f"{name:27s} {sub:7.3f} {se:6.3f} {r.brier():7.4f} {log:10.4f}"
            f" {r.volume.mean():7.1f} {r.informed.mean():10.2f} {r.noise.mean():10.2f}"
        )
    points[fam] = np.array(pts)
    print()

ref = runs["quoter, assumes 1 x"]
public = ref.brier()
private = ref.brier(ref.bayes)
print(f"Bayes posterior from the public order flow: Brier {public:.4f}")
print(f"Bayes posterior from every private signal:  Brier {private:.4f}")
lm = runs[f"LMSR b = {b_star:.3g}"]
print(f"LMSR at b* against the quoter: max price gap {np.abs(lm.price - ref.price).max():.2e}")
shut = pm.run(lambda n: pm.Quoter(2 * RHO - 1 + 1e-9, n), ALPHA, TRADES, 2000, rho=RHO, seed=7)
print(
    f"quoter assuming 2 rho - 1 = {2 * RHO - 1:.2f}: bettor trades {shut.volume.mean():.1f}"
    f" of {TRADES}, Brier {shut.brier():.4f}"
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, layout="constrained")
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

style = {
    "quoter": (INK, "Glosten-Milgrom quoter, assumed share 0.5x to 1.9x"),
    "lmsr": (BLUE, "LMSR, b from 1.2 to 15"),
    "fpmm": (AQUA, "FPMM, F from 1.5 to 15"),
    "ls1": (VIOLET, "LS-LMSR, vig 1%, b0 from 1.5 to 12"),
    "ls3": (ORANGE, "LS-LMSR, vig 3%"),
}
for fam, pts in points.items():
    c, label = style[fam]
    left.plot(pts[:, 0], pts[:, 1], color=c, linewidth=1.5, marker="o", markersize=3.5, label=label)
left.axhline(public, color=MUTED, linewidth=0.9, linestyle="--")
left.annotate(
    "Bayes on the public order flow",
    (4.4, public),
    xytext=(0, -12),
    textcoords="offset points",
    color=MUTED,
)
left.axvline(0, color=MUTED, linewidth=0.8)
left.set_xlabel("maker subsidy per market ($, negative is a profit)")
left.set_ylabel("Brier score of the final price")
left.set_ylim(0.0165, 0.034)
left.set_xlim(-3, 10)
left.set_title("Accuracy bought per dollar of subsidy", loc="left")
left.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)

curves = {
    "quoter, assumes 1 x": (INK, "quoter at alpha (2 rho - 1), and LMSR at b*"),
    "LMSR b = 2": (BLUE, "LMSR b = 2"),
    "FPMM F = 3": (AQUA, "FPMM F = 3"),
    "LS-LMSR v 0.01, b0 = 2": (VIOLET, "LS-LMSR vig 1%, b0 = 2"),
    "LMSR b = 15": (ORANGE, "LMSR b = 15"),
}
t = np.arange(TRADES + 1)
for name, (c, label) in curves.items():
    right.plot(t, runs[name].error, color=c, linewidth=1.4, label=label)
right.set_xlabel("trades")
right.set_ylabel("mean |price - outcome|")
right.set_xlim(0, TRADES)
right.set_ylim(0, 0.52)
right.set_title("How fast each maker gets there", loc="left")
right.legend(frameon=False, loc="upper right", fontsize=9)

path = Path(__file__).resolve().parent.parent / "docs" / "signal_markets.png"
fig.savefig(path, dpi=160, facecolor=SURFACE)
print(f"\nwrote {path}")
