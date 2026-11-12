import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from rl_market_maker import (
    BLUE,
    GREEN,
    INK,
    MINUTES,
    MUTED,
    ORANGE,
    ROOT,
    SURFACE,
    avst,
    glft,
    learned,
    style,
    touch,
)

from mmsim.env import sky
from mmsim.informed import Fundamental, InformedTrader
from mmsim.tournament import session

SEEDS = range(1000)
PURPLE = "#7b5cc4"


def informed(seed):
    return [InformedTrader(Fundamental(1.0, seed + 1), 0.5, seed=seed)]


def play(args):
    make, toxic, seed = args
    crowd = informed if toxic else None
    return session({"m": make}, seed, flow=sky, minutes=MINUTES, crowd=crowd)["m"].pnl


def chart(path, names, res):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=SURFACE, sharey=True, sharex=True)
    colors = (MUTED, ORANGE, BLUE, PURPLE, GREEN)
    for ax, market in zip(axes, ("calm", "informed trader"), strict=True):
        ys = np.arange(len(names))
        data = [res[market][n] for n in names]
        parts = ax.violinplot(
            data,
            ys,
            orientation="horizontal",
            widths=0.8,
            showextrema=False,
        )
        for body, c in zip(parts["bodies"], colors, strict=False):
            body.set(facecolor=c, edgecolor=c, alpha=0.35)
        for y, d in zip(ys, data, strict=True):
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            ax.plot([m - 2 * se, m + 2 * se], [y, y], color=INK, lw=3, solid_capstyle="butt")
            ax.plot(np.median(d), y, "o", color=SURFACE, mec=INK, ms=5)
        ax.axvline(0, color=MUTED, lw=1, ls=":")
        ax.set_yticks(ys, names)
        ax.invert_yaxis()
        ax.set_xlabel("P&L per 60-minute session (ticks)")
        ax.set_title(f"{market}, {len(SEEDS)} sessions", color=INK, loc="left")
        style(ax)
    fig.text(
        0.99,
        0.98,
        "bar: mean +/- 2 se, dot: median",
        ha="right",
        va="top",
        color=MUTED,
        fontsize=9,
    )
    every = np.concatenate([d for m in res.values() for d in m.values()])
    axes[0].set_xlim(*np.percentile(every, [0.5, 99.5]))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    with open(ROOT / "data" / "ppo_seed0.pkl", "rb") as f:
        policy = pickle.load(f)
    bots = {
        "join the touch": touch,
        "AS, gamma 0.003": partial(avst, 0.003),
        "GLFT, gamma 0.08": partial(glft, 0.08, 1),
        "PPO": partial(learned, policy),
    }
    names = list(bots)
    res = {}
    with ProcessPoolExecutor() as pool:
        for market, toxic in (("calm", False), ("informed trader", True)):
            res[market] = {}
            for n in names:
                jobs = [(bots[n], toxic, s) for s in SEEDS]
                res[market][n] = np.array(list(pool.map(play, jobs, chunksize=20)))
    print("market           strategy           mean    se     sd   p5     p50    p95   P(<0)")
    for market, rows in res.items():
        for n in names:
            d = rows[n]
            p5, p50, p95 = np.percentile(d, [5, 50, 95])
            print(
                f"{market:16s} {n:17s} {d.mean():6.2f} {d.std(ddof=1) / np.sqrt(len(d)):5.2f}"
                f" {d.std(ddof=1):6.2f} {p5:6.1f} {p50:6.1f} {p95:6.1f} {np.mean(d < 0):6.2f}"
            )
    chart(ROOT / "docs" / "strategies.png", names, res)
