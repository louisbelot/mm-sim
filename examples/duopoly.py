from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from mmsim import duopoly as du

BASE = du.Game(phi=0.5)
TICKS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
BETAS = (0.0, 5.0)
PATHS = 20_000


def touch(sol, Q):
    bid = min(sol.bid1[0, Q, Q], sol.bid2[0, Q, Q])
    ask = min(sol.ask1[0, Q, Q], sol.ask2[0, Q, Q])
    return bid + ask


def row(args):
    tick, beta = args
    g = replace(BASE, tick=tick, beta=beta)
    Q = g.Q[0]
    nash = du.solve(g)
    first = du.solve(replace(g, share=1.0))
    cart = du.cartel(g)
    V, bid, ask = du.monopoly(g)
    run = du.simulate(g, nash.quotes, PATHS, 7)
    r = run.reward1
    return {
        "tick": tick,
        "beta": beta,
        "nash": nash.V1[0, Q, Q],
        "mc": r.mean(),
        "se": r.std() / np.sqrt(PATHS),
        "first": first.V1[0, Q, Q],
        "second": first.V2[0, Q, Q],
        "cartel": (cart.V1 + cart.V2)[0, Q, Q] / 2,
        "monopoly": V[0, Q],
        "spread": touch(nash, Q),
        "cartel spread": touch(cart, Q),
        "monopoly spread": bid[0, Q] + ask[0, Q],
        "trades": run.trades1.mean() + run.trades2.mean(),
        "|q|": np.abs(run.q1).mean(),
        "hard": nash.hard + first.hard,
    }


def winner(sol, n=0):
    b1, b2 = sol.bid1[n], sol.bid2[n]
    return np.where(b1 < b2, 0, np.where(b2 < b1, 2, 1)).astype(float)


if __name__ == "__main__":
    jobs = [(t, b) for b in BETAS for t in TICKS]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(row, jobs))
    heads = ("nash", "mc", "first", "second", "cartel", "monopoly", "spread", "trades", "|q|")
    for beta in BETAS:
        print(f"phi {BASE.phi}, adversary drift bound {beta}, values per maker at q = (0, 0)")
        print(f"{'tick':>5s} " + " ".join(f"{h:>8s}" for h in heads) + "  cartel/monopoly spread")
        for r in (r for r in rows if r["beta"] == beta):
            cells = " ".join(f"{r[h]:8.3f}" for h in heads)
            print(
                f"{r['tick']:5.2f} {cells}  {r['cartel spread']:.3f} {r['monopoly spread']:.3f}"
                f"  (mc se {r['se']:.3f}, no pure eq {r['hard']})"
            )
        print()

    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1f9e6e", "#52514e"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    top = max(r["first"] for r in rows) * 1.08
    for ax, beta, title in zip(
        axes[:2], BETAS, ("No adversary", "Adversary on the joint inventory"), strict=True
    ):
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        rs = [r for r in rows if r["beta"] == beta]
        x = [r["tick"] for r in rs]
        ax.plot(
            x,
            [r["cartel"] for r in rs],
            color=GREEN,
            lw=2,
            marker="o",
            ms=4,
            label="cartel, half the joint value",
        )
        ax.plot(
            x,
            [r["first"] for r in rs],
            color=BLUE,
            lw=1.6,
            ls="--",
            marker="o",
            ms=3,
            label="Nash, maker first in the queue",
        )
        ax.plot(
            x, [r["nash"] for r in rs], color=INK, lw=2, marker="o", ms=4, label="Nash, ties split"
        )
        ax.plot(
            x,
            [r["second"] for r in rs],
            color=ORANGE,
            lw=1.6,
            ls="--",
            marker="o",
            ms=3,
            label="Nash, maker second in the queue",
        )
        ax.set_xscale("log")
        ax.set_xticks(TICKS, [f"{t:g}" for t in TICKS])
        ax.minorticks_off()
        ax.set_ylim(0, top)
        ax.set_xlabel("tick size")
        ax.set_ylabel("value per maker at t = 0, flat")
        ax.set_title(f"{title} (drift bound {beta:g})", loc="left", color=INK)
    axes[0].legend(loc="upper left", frameon=False)

    g = replace(BASE, tick=0.05)
    sol = du.solve(g)
    Q, span = g.Q[0], 8
    w = winner(sol)[Q - span : Q + span + 1, Q - span : Q + span + 1]
    right = axes[2]
    right.imshow(
        w.T,
        origin="lower",
        cmap=ListedColormap([BLUE, "#d9d8d4", ORANGE]),
        vmin=0,
        vmax=2,
        extent=(-span - 0.5, span + 0.5, -span - 0.5, span + 0.5),
    )
    right.plot([-span, span], [-span, span], color=SURFACE, lw=0.8, ls=":")
    right.set_xlabel("inventory of maker 1")
    right.set_ylabel("inventory of maker 2")
    right.set_title("Who sets the best bid (tick 0.05)", loc="left", color=INK)
    for label, color, xy in (
        ("maker 1", SURFACE, (-6.5, 5.5)),
        ("maker 2", SURFACE, (4.0, -7.0)),
        ("tie", INK, (-6.5, -6.0)),
    ):
        right.text(*xy, label, color=color, fontsize=10, weight="bold")
    right.set_xticks(range(-span, span + 1, 2))
    right.set_yticks(range(-span, span + 1, 2))

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "duopoly.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
