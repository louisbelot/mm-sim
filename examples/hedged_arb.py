import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from mmsim import lvr

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

sigma, lam = 0.6, 365 * 86400 / 12
gamma = 0.003
base = lvr.lvr_rate(sigma)
sb = sigma / math.sqrt(lam)
print(f"ETH-like: sigma 60% a year, 12 s blocks (sigma_b {sb * 1e4:.2f} bp), pool fee 30 bp")
print("arbitrageur hedges on a quoter at half spread h; rates over LVR = sigma^2/8")
print("  h bp  band bp  P_trade    ARB    FEE  QUOTER    sum   chain ARB  FEE  QUOTER")
hs = [0, 1, 2, 3, 5, 10, 20, 30]
split = []
for hbp in hs:
    h = hbp * 1e-4
    arb, fee = lvr.arb_rate(sigma, lam, gamma, h) / base, lvr.fee_rate(sigma, lam, gamma, h) / base
    hedge = lvr.hedge_rate(sigma, lam, gamma, h) / base
    split.append((hbp, arb, fee, hedge))
    line = (
        f"{hbp:6.0f} {(gamma + h) * 1e4:8.0f} {lvr.p_trade(sigma, lam, gamma + h):8.4f}"
        f" {arb:6.3f} {fee:6.3f} {hedge:7.3f} {arb + fee + hedge:6.3f}"
    )
    if hbp in (0, 5, 30):
        c = lvr.mispricing(sigma, lam, gamma, chains=2000, blocks=4000, burn=500, h=h, seed=hbp)
        line += f"   {c.arb / base:9.3f} {c.fee / base:4.3f} {c.hedge / base:7.3f}"
    print(line)

print("\nfixed against Poisson block times, no hedge; ARB over LVR")
print("  eta  Poisson exact  fixed, zeta formula  fixed, chain   (se)  P_trade fixed  chain")
etas = [0.3, 1, 2, 3, 5, 10, 20, 30]
fixed = []
for e in etas:
    g = e * sb / math.sqrt(2)
    n = max(1500, int(6 * e**2))
    c = lvr.mispricing(sigma, lam, g, chains=2000, blocks=n, burn=300, fixed=True, seed=7)
    row = (
        e,
        lvr.arb_rate(sigma, lam, g) / base,
        lvr.arb_rate_fixed(sigma, lam, g) / base,
        c.arb / base,
        c.arb_se / base,
        lvr.p_trade_fixed(sigma, lam, g),
        c.trade,
    )
    fixed.append(row)
    print(
        f"{e:5.1f} {row[1]:14.4f} {row[2]:20.4f} {row[3]:13.4f} {row[4]:7.4f}"
        f" {row[5]:14.4f} {row[6]:6.4f}"
    )
ratio = abs(lvr.ZETA_HALF) / math.sqrt(math.pi)
print(f"large-eta ratio fixed/Poisson: |zeta(1/2)|/sqrt(pi) = {ratio:.4f}")

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

x = [s[0] for s in split]
arb = [s[1] for s in split]
fee = [s[2] for s in split]
hedge = [s[3] for s in split]
left.stackplot(
    x,
    fee,
    arb,
    hedge,
    colors=[AQUA, ORANGE, BLUE],
    alpha=0.85,
)
left.text(4, 0.35, "pool fees, kept by the LP", color=SURFACE, fontsize=9)
left.text(17, 0.82, "quoter's spread", color=SURFACE, fontsize=9)
left.annotate(
    "arbitrage profit",
    xy=(12, 0.70),
    xytext=(15, 0.42),
    color=SURFACE,
    fontsize=9,
    arrowprops={"arrowstyle": "-", "color": SURFACE, "lw": 0.8},
)
left.set_xlabel("quoter half spread h (bp), pool fee 30 bp")
left.set_ylabel("share of LVR")
left.set_title("Who gets the LP's loss when arbs hedge on a quoter", color=INK, fontsize=10)
left.set_xlim(0, 30)

e = [r[0] for r in fixed]
right.plot(e, [r[1] for r in fixed], color=ORANGE, lw=1.8, label="Poisson blocks, exact")
right.plot(e, [r[2] for r in fixed], color=BLUE, lw=1.8, label="fixed blocks, zeta(1/2) formula")
right.errorbar(
    e,
    [r[3] for r in fixed],
    yerr=[2 * r[4] for r in fixed],
    fmt="o",
    ms=4,
    color=INK,
    label="fixed blocks, simulated chain",
)
right.set_xscale("log")
right.set_yscale("log")
ticks = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
right.yaxis.set_major_locator(FixedLocator(ticks))
right.yaxis.set_minor_locator(NullLocator())
right.set_yticklabels([f"{t:g}" for t in ticks])
right.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30]))
right.xaxis.set_minor_locator(NullLocator())
right.set_xticklabels(["0.3", "1", "3", "10", "30"])
right.set_xlabel("eta = sqrt(2) gamma / sigma_b")
right.set_ylabel("arbitrage profit over LVR")
right.set_title("Fixed blocks leave arbitrageurs less", color=INK, fontsize=10)
right.legend(frameon=False, fontsize=8, loc="lower left")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "hedged_arb.png"
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"\nwrote {out}")
