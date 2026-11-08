import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import prediction as pm

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

ALPHA, TRADES, MARKETS = 0.2, 400, 20_000
b10 = 10 / math.log(2)
makers = {
    "quoter, alpha known": lambda n: pm.Quoter(ALPHA, n),
    "quoter, assumes 0.1": lambda n: pm.Quoter(0.1, n),
    "quoter, assumes 0.4": lambda n: pm.Quoter(0.4, n),
    "LMSR b = 10": lambda n: pm.LMSR(10.0, n),
    "LMSR b = 14.4": lambda n: pm.LMSR(b10, n),
    "FPMM F = 10": lambda n: pm.FPMM(10.0, n),
}

print(f"binary event, p0 = 0.5, informed share {ALPHA}, {TRADES} unit trades, {MARKETS} markets")
print("maker                 maker P&L    (se)  informed   noise   |p - w| at 25   50  100  200")
runs = {}
for name, make in makers.items():
    r = pm.run(make, ALPHA, TRADES, MARKETS, seed=7)
    runs[name] = r
    se = r.maker.std() / math.sqrt(MARKETS)
    errs = " ".join(f"{r.error[t]:.3f}" for t in (25, 50, 100, 200))
    print(
        f"{name:21s} {r.maker.mean():9.2f} {se:7.2f} {r.informed.mean():9.2f}"
        f" {r.noise.mean():7.2f}   {errs}"
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
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

t = np.arange(TRADES + 1)
lines = (
    ("quoter, alpha known", INK, "quoter"),
    ("LMSR b = 10", BLUE, "LMSR b = 10"),
    ("LMSR b = 14.4", AQUA, "LMSR b = 14.4"),
    ("FPMM F = 10", ORANGE, "FPMM F = 10"),
)
spots = {"quoter": 60, "LMSR b = 10": TRADES, "LMSR b = 14.4": TRADES, "FPMM F = 10": TRADES}
for name, color, label in lines:
    e = runs[name].error
    left.plot(t, e, color=color, linewidth=1.6)
    x = spots[label]
    left.annotate(
        label, (x, e[x]), xytext=(6, 0), textcoords="offset points", va="center", color=color
    )
left.set_yscale("log")
left.set_ylim(1e-3, 0.7)
left.set_xlim(0, TRADES * 1.25)
left.set_xlabel("trades")
left.set_ylabel("mean |price - outcome|")
left.set_title("How fast each maker's price finds the outcome", loc="left")

names = list(makers)
noise = np.array([-runs[n].noise.mean() for n in names])
maker = np.array([-runs[n].maker.mean() for n in names])
y = np.arange(len(names))[::-1]
right.barh(y, noise, color=BLUE, height=0.6, label="noise traders")
right.barh(y, maker, left=np.where(maker > 0, noise, 0), color=ORANGE, height=0.6, label="maker")
for yi, n_, m_ in zip(y, noise, maker, strict=True):
    right.annotate(
        f"{n_ + m_:.1f}" if m_ > 0 else f"{n_:.1f} (maker +{-m_:.1f})",
        (max(n_ + m_, n_), yi),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        color=INK,
    )
right.set_yticks(y)
right.set_yticklabels(names)
right.set_xlim(0, 14)
right.legend(frameon=False, loc="upper right")
right.set_xlabel("dollars per market")
right.set_title("Who pays the informed traders", loc="left")

fig.tight_layout()
path = Path(__file__).resolve().parent.parent / "docs" / "prediction_markets.png"
fig.savefig(path, dpi=160, facecolor=SURFACE)
print(f"wrote {path}")
