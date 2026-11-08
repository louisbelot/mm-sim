import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import lvr
from mmsim import venues as vn

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"

sigma, lam = 1.0, 5000.0
etas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
print("constant product pool, sigma 1, lambda 5000; rates over LVR = sigma^2/8")
print("  eta  P_trade  ARB exact  ARB poisson  ARB fixed  FEE exact  FEE poisson")
rows = []
for e in etas:
    g = e * sigma / math.sqrt(2 * lam)
    burn = max(300, int(3 * e**2))
    n = max(1500, int(6 * e**2))
    pois = lvr.mispricing(sigma, lam, g, chains=2000, blocks=n, burn=burn, seed=11)
    fixed = lvr.mispricing(sigma, lam, g, chains=2000, blocks=n, burn=burn, fixed=True, seed=12)
    base = lvr.lvr_rate(sigma)
    row = (
        e,
        lvr.p_trade(sigma, lam, g),
        lvr.arb_rate(sigma, lam, g) / base,
        pois.arb / base,
        fixed.arb / base,
        lvr.fee_rate(sigma, lam, g) / base,
        pois.fee / base,
    )
    rows.append(row)
    print(
        f"{e:5.1f} {row[1]:8.4f} {row[2]:10.4f} {row[3]:12.4f} {row[4]:10.4f}"
        f" {row[5]:10.4f} {row[6]:12.4f}"
    )

eth = vn.Market()
b = eth.sigma / math.sqrt(2 * eth.lam)
zero = lvr.stale_quote_loss(eth.sigma, eth.lam, 0.0)
print(f"\nquoter refreshed every block, ETH-like market, b = sigma sqrt(dt/2) = {b * 1e4:.2f} bp")
print("  h/b  loss exact  loss simulated  (both over the h = 0 loss)")
stale = []
for e in (0.3, 1.0, 2.0, 3.0, 5.0):
    h = e * b
    s = vn.run(eth, 0.2, 1e6, 0.05, vn.Quoter(half=h, limit=1e12), paths=200, seed=21)
    sim = -s.mm.arb.mean() / (eth.lam * eth.hours * eth.size * eth.P0) / zero
    exact = lvr.stale_quote_loss(eth.sigma, eth.lam, h) / zero
    stale.append((e, exact, sim))
    print(f"{e:5.1f} {exact:11.5f} {sim:15.5f}")

fees = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
halves = [None, 5, 15, 30]
print("\npool +-20% around 2000 with $1M, 24 h, 60 noise trades of 1 ETH an hour, 200 paths")
print("quoter: AS reservation price (gamma 3e-4, tau 1 h), fixed half spread, 50 ETH limit")
print("fee bp  quoter bp  LP fees  LP edge     LVR  LP hedged  (se)  pool share  quoter hedged")
grid = {}
for half in halves:
    for fee in fees:
        qt = None if half is None else vn.Quoter(half=half * 1e-4, gamma=3e-4, tau=1.0, limit=50)
        s = vn.run(eth, 0.2, 1e6, fee * 1e-4, qt, paths=200, seed=31)
        lp = s.lp
        hedged = lp.hedged
        se = hedged.std(ddof=1) / math.sqrt(len(hedged))
        mm_vol = s.mm.volume.mean() if s.mm else 0.0
        share = lp.volume.mean() / (lp.volume.mean() + mm_vol)
        mm_h = s.mm.hedged.mean() if s.mm else float("nan")
        grid[half, fee] = (hedged.mean(), se, mm_h)
        print(
            f"{fee:6d} {str(half):>9s} {(lp.noise_fees + lp.arb_fees).mean():8.0f}"
            f" {(lp.noise + lp.arb).mean():8.0f} {s.lvr.mean():7.0f} {hedged.mean():10.0f}"
            f" {se:5.0f} {share:11.2f} {mm_h:14.0f}"
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

xs = np.geomspace(0.08, 40, 200)
gs = xs * sigma / math.sqrt(2 * lam)
arb = [lvr.arb_rate(sigma, lam, g) / lvr.lvr_rate(sigma) for g in gs]
fee = [lvr.fee_rate(sigma, lam, g) / lvr.lvr_rate(sigma) for g in gs]
left.plot(xs, arb, color=BLUE, linewidth=1.6)
left.plot(xs, fee, color=AQUA, linewidth=1.6)
left.plot(xs, np.exp(-xs), color=ORANGE, linewidth=1.6)
r = np.array(rows)
left.scatter(r[:, 0], r[:, 3], color=BLUE, s=22, zorder=3)
left.scatter(r[:, 0], r[:, 4], facecolor=SURFACE, edgecolor=BLUE, s=22, zorder=3)
left.scatter(r[:, 0], r[:, 6], color=AQUA, s=22, zorder=3)
st = np.array(stale)
left.scatter(st[:, 0], st[:, 2], color=ORANGE, s=22, zorder=3)
left.set_xscale("log")
left.set_yscale("log")
left.set_xlim(0.08, 40)
left.set_ylim(1e-3, 2)
left.set_xlabel("fee or half spread over sigma sqrt(dt/2)")
left.set_ylabel("loss rate over the frictionless loss")
left.set_title("Pool loss ~ 1/(1+eta), fresh quote loss ~ e^-eta", loc="left")
left.annotate("pool arbitrage loss", (3, 0.33), xytext=(4, 0.5), color=BLUE)
left.annotate("pool fees from arbs", (3, 0.75), xytext=(0.1, 1.2), color=AQUA)
left.annotate("quoter refreshed\nevery block", (2.0, 0.14), xytext=(0.12, 0.015), color=ORANGE)
left.annotate("hollow: fixed blocks", (0.1, 0.002), color=MUTED)

spots = {5: (3, (0, -18)), 15: (7, (0, 10)), 30: (15, (10, 6))}
for half, color in ((5, ORANGE), (15, VIOLET), (30, BLUE)):
    y = np.array([grid[half, f][0] for f in fees])
    e = np.array([grid[half, f][1] for f in fees])
    right.plot(fees, y, color=color, linewidth=1.6, marker="o", markersize=4)
    right.fill_between(fees, y - 2 * e, y + 2 * e, color=color, alpha=0.15, linewidth=0)
    at, off = spots[half]
    right.annotate(
        f"quoter at {half} bp",
        (at, grid[half, at][0]),
        xytext=off,
        textcoords="offset points",
        color=color,
    )
right.axhline(0, color=MUTED, linewidth=0.8)
right.set_xscale("log")
right.set_xlim(0.8, 130)
right.set_xticks(fees[::2] + [100])
right.set_xticklabels([str(f) for f in fees[::2]] + ["100"])
right.set_xlabel("pool fee (bp)")
right.set_ylabel("LP hedged P&L per day ($)")
right.set_title("Best fee tier depends on the quoter next to the pool", loc="left")

fig.tight_layout()
path = Path(__file__).resolve().parent.parent / "docs" / "amm_lvr.png"
fig.savefig(path, dpi=160, facecolor=SURFACE)
print(f"\nwrote {path}")
