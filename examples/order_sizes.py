from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from itch_replay import (
    AQUA,
    BLUE,
    CLOSE,
    INK,
    MUTED,
    OPEN,
    ORANGE,
    ROOT,
    SURFACE,
    SYMBOLS,
    msgs,
    style,
)

from mmsim.calibrate import calibrate
from mmsim.sizes import ITCH_CAP, Sizes, fit_geometric, fit_zeta, units, vuong, zeta_alpha

CAP = ITCH_CAP


def prints(symbol):
    c = calibrate(msgs(symbol), OPEN, CLOSE)
    return c.limit_size, c.market_sizes


def ccdf(s, k):
    return 1 - np.concatenate([[0.0], np.cumsum(s.pmf)])[k - 1]


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        raw = dict(zip(SYMBOLS, pool.map(prints, SYMBOLS), strict=True))

    print(f"market orders in units of the mean limit order, sizes 1 to {CAP}")
    print("symbol  unit   orders  >1 unit  mean   alpha  geom q   vuong z   P(1) data zeta geom")
    ks = {}
    for symbol, (unit, shares) in raw.items():
        k = units(shares, unit, CAP)
        ks[symbol] = k
        z, g = fit_zeta(k, CAP), fit_geometric(k, CAP)
        vz = vuong(z, g, k)
        print(
            f"{symbol:6s} {unit:5.0f} {len(k):8d}  {np.mean(k > 1):6.1%}  {k.mean():5.2f}"
            f"  {zeta_alpha(z):5.2f}  {g.pmf[1] / g.pmf[0]:6.3f}  {vz:8.1f}"
            f"   {np.mean(k == 1):.3f} {z.pmf[0]:.3f} {g.pmf[0]:.3f}"
        )
    pooled = np.concatenate(list(ks.values()))
    z, g = fit_zeta(pooled, CAP), fit_geometric(pooled, CAP)
    vz = vuong(z, g, pooled)
    sim = Sizes.itch()
    print(
        f"pooled {len(pooled)} orders, alpha {zeta_alpha(z):.3f}, vuong z {vz:.1f},"
        f" mean {pooled.mean():.3f}; simulator alpha {zeta_alpha(sim):.2f}, mean {sim.mean:.3f}"
    )
    print("k   data    zeta    geometric")
    for k in (1, 2, 3, 5, 10, 20):
        print(f"{k:<3d} {np.mean(pooled == k):.4f}  {z.pmf[k - 1]:.4f}  {g.pmf[k - 1]:.4f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    k = np.arange(1, CAP + 1)
    for ax in (a, b):
        style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("market order size, units of the mean limit order", color=INK)
    data = np.array([np.mean(pooled >= j) for j in k])
    a.plot(k, data, "o", color=INK, markersize=4, label="six symbols pooled")
    a.plot(k, ccdf(z, k), color=BLUE, linewidth=1.6, label=f"zeta, alpha {zeta_alpha(z):.2f}")
    a.plot(k, ccdf(g, k), color=ORANGE, linewidth=1.6, label="geometric")
    a.set_ylim(data[data > 0].min() / 3, 1.5)
    a.set_ylabel("P(size >= k)", color=INK)
    a.set_title("pooled sizes and the two one-parameter fits", color=INK, fontsize=10)
    a.legend(frameon=False, labelcolor=INK, loc="lower left")

    colors = [BLUE, ORANGE, AQUA, "#7b5cc4", MUTED, INK]
    for (symbol, ksym), color in zip(ks.items(), colors, strict=True):
        f = fit_zeta(ksym, CAP)
        emp = np.array([np.mean(ksym >= j) for j in k])
        keep = emp > 0
        b.plot(k[keep], emp[keep], "o", color=color, markersize=3)
        b.plot(k, ccdf(f, k), color=color, linewidth=1.2, label=f"{symbol} {zeta_alpha(f):.2f}")
    b.set_ylim(a.get_ylim())
    b.set_title("each symbol with its own zeta fit", color=INK, fontsize=10)
    b.legend(frameon=False, labelcolor=INK, loc="lower left", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "order_sizes.png", dpi=160, facecolor=SURFACE)
