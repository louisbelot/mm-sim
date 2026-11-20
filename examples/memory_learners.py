import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import duopoly as du
from mmsim import learners as le
from mmsim import memory as mem

BASE = du.Game(tick=0.1, reach=1.5, Q=(5, 5), phi=0.5)
GAMMA = 0.99
STEPS, RUNS, DECAY, TAU = 3_000_000, 16, 2e-6, 5000.0
if "--quick" in sys.argv:
    STEPS, RUNS, DECAY = 40_000, 2, 1.5e-4
CONFIGS = [
    (0.1, False, False, None),
    (0.1, False, True, None),
    (0.1, True, False, None),
    (0.1, True, False, TAU),
    (0.1, True, True, None),
    (0.1, True, True, TAU),
    (0.2, False, False, None),
    (0.2, False, True, None),
    (0.2, True, False, TAU),
    (0.2, True, True, TAU),
]


def name(sync, memory, tau):
    return f"{'sync' if sync else 'async'}{' memory' if memory else ''}{' 1/n' if tau else ''}"


def job(args):
    tick, sync, memory, tau = args
    g = replace(BASE, tick=tick)
    Q = g.Q[0]
    nash = le.markov(g, GAMMA)
    cart = le.collusive(g, GAMMA)
    lo = (nash.V1 + nash.V2)[Q, Q] / 2
    hi = (cart.V1 + cart.V2)[Q, Q] / 2
    learned = le.train(
        g, GAMMA, STEPS, RUNS, decay=DECAY, sync=sync, memory=memory, tau=tau, seed=7
    )
    rows = []
    for t1, t2 in zip(learned.table1, learned.table2, strict=True):
        if memory:
            qs = mem.profile(g, t1, t2)
        else:
            qs = tuple(np.broadcast_to(x, (9, *x.shape)) for x in le.quotes(g, t1, t2))
        rows.append(mem.assess(g, GAMMA, qs))
    v = np.array([(r.v1 + r.v2) / 2 for r in rows])
    return {
        "tick": tick,
        "name": name(sync, memory, tau),
        "nash": lo,
        "cartel": hi,
        "value": v,
        "gain": (v - lo) / (hi - lo),
        "gap": np.array([max(r.gap1, r.gap2) for r in rows]),
        "half": np.array([r.half for r in rows]),
        "shock": np.array([np.stack(r.shock[:6]) for r in rows]),
        "pays": np.array([r.shock.gain for r in rows]),
        "train half": learned.half.mean(axis=1),
        "nash half": (min(nash.quotes[0][Q, Q], nash.quotes[2][Q, Q]) + nash.quotes[1][Q, Q]) / 2,
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax = axes[0]
    base = [r for r in rows if r["tick"] == 0.1]
    rng = np.random.default_rng(0)
    for k, r in enumerate(base):
        color = BLUE if "memory" in r["name"] else ORANGE
        g = r["gain"]
        ax.scatter(k + rng.uniform(-0.12, 0.12, g.size), g, s=12, color=color, alpha=0.5, lw=0)
        ax.plot([k - 0.25, k + 0.25], [g.mean()] * 2, color=color, lw=2.4)
    for y, label in ((0, "Nash"), (1, "cartel")):
        ax.axhline(y, color=GREY, lw=1, ls="--")
        ax.text(1.0, y, f" {label}", color=GREY, va="center", transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(base)), [r["name"].replace(" ", "\n", 1) for r in base], fontsize=8)
    ax.set_ylabel("profit gain (learned - Nash) / (cartel - Nash)")
    ax.set_title("Where the learners end up (tick 0.1)", loc="left", color=INK)

    ax = axes[1]
    for r, color in ((base[1], ORANGE), (base[5], BLUE)):
        s = r["shock"].mean(axis=0)
        t = np.arange(s.shape[1]) - 1
        ax.plot(t, s[0], color=color, lw=1.8, label=f"{r['name']}: deviator")
        ax.plot(t, s[2], color=color, lw=1.4, ls="--", label=f"{r['name']}: rival")
    ax.axvline(0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("steps after a forced one-tick undercut")
    ax.set_ylabel("mean bid depth")
    ax.set_title("Impulse response, mean of 16 runs", loc="left", color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    ax = axes[2]
    for r in base:
        if "sync" not in r["name"] or "async" in r["name"]:
            continue
        h = r["train half"]
        x = (np.arange(h.size) + 1) * STEPS / h.size / 1e6
        color = BLUE if "memory" in r["name"] else ORANGE
        ax.plot(x, h, color=color, lw=1.4, ls="-" if "1/n" in r["name"] else ":", label=r["name"])
    ax.axhline(base[0]["nash half"], color=GREY, lw=1, ls="--")
    ax.text(
        0.02,
        base[0]["nash half"],
        " Nash",
        color=GREY,
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("inside half-spread while training")
    ax.set_title("Synchronous learners, constant vs 1/n rate", loc="left", color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(len(CONFIGS)) as ex:
        rows = list(ex.map(job, CONFIGS))
    print(f"phi {BASE.phi}, gamma {GAMMA}, Q {BASE.Q}, {RUNS} runs of {STEPS:,} steps")
    print("values per maker at q = (0, 0), memory drawn from where each pair settles")
    print(
        f"{'tick':>4s} {'learner':>18s} {'nash':>6s} {'cartel':>6s} {'gain':>12s} {'half':>6s}"
        f" {'exploit':>7s} {'undercut pays':>13s} {'rival t+1':>9s}"
    )
    for r in rows:
        s = r["shock"]
        react = (s[:, 2, 2] - s[:, 2, 0]).mean() / r["tick"]
        print(
            f"{r['tick']:4.1f} {r['name']:>18s} {r['nash']:6.3f} {r['cartel']:6.3f}"
            f" {r['gain'].mean():5.2f} +-{r['gain'].std():4.2f} {r['half'].mean():6.3f}"
            f" {r['gap'].mean():7.3f} {(r['pays'] > 0).mean():13.2f} {react:+9.3f}"
        )
    out = Path(__file__).resolve().parent.parent / "docs" / "memory_learners.png"
    if "--quick" not in sys.argv:
        draw(rows, out)
        print(f"wrote {out}")
