import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import duopoly as du
from mmsim import learners as le

BASE = du.Game(tick=0.1, reach=1.5, Q=(5, 5), phi=0.5)
GAMMA = 0.99
TICKS = (0.1, 0.2, 0.4)
SHARES = (0.5, 1.0)
STEPS, RUNS = 1_000_000, 16


def half(quotes, Q):
    b1, a1, b2, a2 = (x[Q, Q] for x in quotes)
    return (min(b1, b2) + min(a1, a2)) / 2


def job(args):
    tick, share, sync = args
    g = replace(BASE, tick=tick, share=share)
    Q = g.Q[0]
    nash = le.markov(g, GAMMA)
    cart = le.collusive(g, GAMMA)
    learned = le.train(g, GAMMA, STEPS, RUNS, sync=sync, seed=11)
    s = le.score(g, GAMMA, learned)
    lo = (nash.V1 + nash.V2)[Q, Q] / 2
    hi = (cart.V1 + cart.V2)[Q, Q] / 2
    v = (s.V1 + s.V2) / 2
    bids = np.array([le.greedy(g, t)[0] for t in np.concatenate([learned.table1, learned.table2])])
    return {
        "tick": tick,
        "share": share,
        "sync": sync,
        "nash": lo,
        "cartel": hi,
        "learned": v,
        "gain": (v - lo) / (hi - lo),
        "gap": np.maximum(s.gap1, s.gap2),
        "v1": s.V1,
        "v2": s.V2,
        "nash1": nash.V1[Q, Q],
        "nash2": nash.V2[Q, Q],
        "half": learned.half.mean(axis=1),
        "nash half": half(nash.quotes, Q),
        "cartel half": half(cart.quotes, Q),
        "learned half": np.array(
            [
                half(le.quotes(g, a, b), Q)
                for a, b in zip(learned.table1, learned.table2, strict=True)
            ]
        ),
        "bids": bids,
        "nash bid": nash.quotes[0][:, Q],
        "cartel bid": np.minimum(cart.quotes[0], cart.quotes[2])[:, Q],
        "eps": nash.eps,
    }


def draw(rows, out):
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#52514e"
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
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    colors = {False: ORANGE, True: BLUE}
    names = {False: "asynchronous", True: "synchronous"}
    base = [r for r in rows if r["tick"] == TICKS[0] and r["share"] == 0.5]

    ax = axes[0]
    for r in base:
        h = r["half"]
        x = (np.arange(h.size) + 1) * STEPS / h.size / 1e6
        ax.plot(x, h, color=colors[r["sync"]], lw=1.6, label=names[r["sync"]])
    ref = base[0]
    for y, label in ((ref["cartel half"], "cartel"), (ref["nash half"], "Nash")):
        ax.axhline(y, color=GREY, lw=1, ls="--")
        ax.text(0.02, y, f" {label}", color=GREY, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("inside half-spread while training")
    ax.set_title(f"Learning to quote (tick {TICKS[0]:g})", loc="left", color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.1), frameon=False)

    ax = axes[1]
    rng = np.random.default_rng(0)
    for k, tick in enumerate(TICKS):
        for r in (r for r in rows if r["tick"] == tick and r["share"] == 0.5):
            off = 0.15 if r["sync"] else -0.15
            g = r["gain"]
            ax.scatter(
                k + off + rng.uniform(-0.05, 0.05, g.size),
                g,
                s=12,
                color=colors[r["sync"]],
                alpha=0.5,
                lw=0,
            )
            ax.plot([k + off - 0.1, k + off + 0.1], [g.mean()] * 2, color=colors[r["sync"]], lw=2.4)
    for y, label in ((0, "Nash"), (1, "cartel")):
        ax.axhline(y, color=GREY, lw=1, ls="--")
        ax.text(1.0, y, f" {label}", color=GREY, va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(TICKS)), [f"{t:g}" for t in TICKS])
    ax.set_xlabel("tick size")
    ax.set_ylabel("profit gain (learned - Nash) / (cartel - Nash)")
    ax.set_title("Where the learners end up, 16 runs each", loc="left", color=INK)

    ax = axes[2]
    Q = BASE.Q[0]
    q = np.arange(-Q, Q + 1)
    for r in base:
        b = np.median(r["bids"], axis=0)
        ax.plot(
            q,
            b,
            color=colors[r["sync"]],
            lw=1.6,
            marker="o",
            ms=3,
            label=f"{names[r['sync']]}, median",
        )
    ax.plot(
        q, ref["nash bid"], color=GREY, lw=1.2, ls="--", marker="s", ms=3, label="Nash (rival flat)"
    )
    ax.plot(
        q,
        ref["cartel bid"],
        color=GREY,
        lw=1.2,
        ls=":",
        marker="^",
        ms=3,
        label="cartel (rival flat)",
    )
    ax.set_xlabel("own inventory")
    ax.set_ylabel("bid depth")
    ax.set_title(f"Learned bid ladder (tick {TICKS[0]:g})", loc="left", color=INK)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    jobs = [(t, s, y) for t in TICKS for s in SHARES for y in (False, True)]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(job, jobs))
    print(f"phi {BASE.phi}, gamma {GAMMA}, Q {BASE.Q}, {RUNS} runs of {STEPS:,} steps")
    print("values per maker at q = (0, 0); gain = (learned - nash) / (cartel - nash)")
    print(
        f"{'tick':>5s} {'share':>5s} {'update':>6s} {'nash':>7s} {'cartel':>7s} {'learned':>14s}"
        f" {'gain':>12s} {'half':>6s} {'nash':>5s} {'cart':>5s} {'exploit':>7s}"
    )
    for r in rows:
        v, gn = r["learned"], r["gain"]
        print(
            f"{r['tick']:5.2f} {r['share']:5.2f} {'sync' if r['sync'] else 'async':>6s}"
            f" {r['nash']:7.3f} {r['cartel']:7.3f} {v.mean():7.3f} +-{v.std():5.3f}"
            f" {gn.mean():5.2f} +-{gn.std():4.2f} {r['learned half'].mean():6.3f}"
            f" {r['nash half']:5.2f} {r['cartel half']:5.2f} {r['gap'].mean():7.3f}"
            f"  (nash eps {r['eps']:.4f})"
        )
    for r in rows:
        if r["share"] == 1.0:
            print(
                f"priority tick {r['tick']}, {'sync' if r['sync'] else 'async'}: first"
                f" {r['v1'].mean():.3f} (nash {r['nash1']:.3f}), second {r['v2'].mean():.3f}"
                f" (nash {r['nash2']:.3f})"
            )
    out = Path(__file__).resolve().parent.parent / "docs" / "learners.png"
    draw(rows, out)
    print(f"wrote {out}")
