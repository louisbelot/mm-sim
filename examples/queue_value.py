from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import GLFT, SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence, prob_fill_first
from mmsim.glft import Model
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import markouts
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SESSIONS = 200
H = np.array([0, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30])
BOTS = ("join the touch", "GLFT, gamma 0.3")
BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, None))


def make_bot(name, keep):
    if name == "join the touch":
        return SymmetricQuoter(0.5, max_position=5, keep=keep)
    return GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=keep)


def session(args):
    name, keep, informed, seed = args
    agents = [make_bot(name, keep)]
    if informed:
        agents.append(InformedTrader(Fundamental(1.0, seed=seed), 1.0, seed=seed))
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    m = markouts(fills, sim.tape, H, sim.end)
    ahead = np.array([f.ahead for f in fills])
    wiped = np.array([f.left == 0 for f in fills])
    return ctx.pnl(), m, ahead, wiped


def main():
    rows = {}
    with ProcessPoolExecutor() as pool:
        for name in BOTS:
            for informed in (False, True):
                for keep in (0, 1, 2):
                    jobs = [(name, keep, informed, s) for s in range(SESSIONS)]
                    rows[name, informed, keep] = list(pool.map(session, jobs, chunksize=4))

    print("strategy          informed  keep  mean P&L (se)  sd P&L  fills  mk at fill  mk 30 min")
    for (name, informed, keep), r in rows.items():
        pnl = np.array([x[0] for x in r])
        m = np.concatenate([x[1] for x in r])
        se = pnl.std() / np.sqrt(len(pnl))
        print(
            f"{name:17s} {'yes' if informed else 'no':8s} {keep:5d} {pnl.mean():8.1f} ({se:.1f})"
            f" {pnl.std():7.1f} {len(m) / len(r):6.0f} {np.nanmean(m[:, 0]):11.2f}"
            f" {np.nanmean(m[:, -1]):10.2f}"
        )

    curves = {}
    for name in BOTS:
        for informed in (False, True):
            r = rows[name, informed, 0]
            m = np.concatenate([x[1] for x in r])
            ahead = np.concatenate([x[2] for x in r])
            wiped = np.concatenate([x[3] for x in r])
            print(f"\n{name}, keep 0, {'informed' if informed else 'noise only'}")
            for lo, hi in BUCKETS:
                sel = (ahead >= lo) & (ahead <= (hi if hi is not None else ahead.max()))
                label = f"{lo}+" if hi is None else f"{lo}" if lo == hi else f"{lo}-{hi}"
                mk = np.nanmean(m[sel][:, [0, 6, 12]], axis=0)
                print(
                    f"  ahead {label:4s} share {sel.mean():.2f}  emptied the level"
                    f" {wiped[sel].mean():.2f}  markout 0/5/30 min {mk.round(2)}"
                )
            for w in (True, False):
                sel = wiped == w
                mk = np.nanmean(m[sel][:, [0, 6, 12]], axis=0)
                print(
                    f"  emptied {w!s:5s} share {sel.mean():.2f}  markout 0/5/30 min {mk.round(2)}"
                )
            if name == BOTS[1]:
                curves[informed] = {w: np.nanmean(m[wiped == w], axis=0) for w in (True, False)}

    sky = CSTParams.sky_perfect()
    ahead = np.arange(0, 11)
    fill = {
        a: [prob_fill_first(a, n + 1, sky.lam[0], sky.mu, sky.theta[0]) for n in ahead]
        for a in (1, 2, 4)
    }
    for a, p in fill.items():
        print(f"ask queue {a}: P(fill first) by lots ahead {np.round(p, 3)}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    left, mid, right = axes

    for (a, p), color in zip(fill.items(), (ORANGE, BLUE, AQUA), strict=True):
        left.plot(
            ahead,
            p,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{a} lot{'s' if a > 1 else ''} at the ask",
        )
    left.set_xlim(0, ahead[-1])
    left.set_ylim(0, 1)
    left.set_xlabel("lots ahead of the order at the best bid")
    left.set_ylabel("P(fill before the ask queue empties)")
    left.set_title("Exact, CST flow, one-tick spread", loc="left", color=INK)
    left.legend(loc="upper right", frameon=False, labelcolor=INK)

    for informed, style in ((False, "--"), (True, "-")):
        for w, color in ((True, ORANGE), (False, BLUE)):
            label = "fill emptied the level" if w else "queue left behind"
            label += ", informed flow" if informed else ", noise only"
            mid.plot(H, curves[informed][w], color=color, linestyle=style, linewidth=2, label=label)
    mid.axhline(0, color=MUTED, linewidth=1)
    mid.set_xlim(0, 30)
    mid.set_xlabel("minutes after the fill")
    mid.set_ylabel("mean markout per lot (ticks)")
    mid.set_title("GLFT fills: did the fill empty its price level?", loc="left", color=INK)
    mid.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=9)

    x = np.arange(3)
    for k, (name, informed) in enumerate([(n, i) for n in BOTS for i in (False, True)]):
        pnl = [np.array([r[0] for r in rows[name, informed, keep]]) for keep in (0, 1, 2)]
        mean = [p.mean() for p in pnl]
        se = [p.std() / np.sqrt(len(p)) for p in pnl]
        color = (BLUE, ORANGE)[BOTS.index(name)]
        right.errorbar(
            x + (k - 1.5) * 0.08,
            mean,
            yerr=2 * np.array(se),
            color=color,
            linestyle="-" if informed else "--",
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            label=f"{name}, {'informed flow' if informed else 'noise only'}",
        )
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xticks(x, ["0 (always requote)", "1", "2"])
    right.set_xlabel("keep: ticks a quote may sit behind its target")
    right.set_ylabel("mean P&L per session (ticks, 2 se bars)")
    right.set_title("Holding queue priority", loc="left", color=INK)
    right.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
        labelcolor=INK,
        fontsize=9,
    )

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "queue_value.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
