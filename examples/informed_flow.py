from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import GLFT, AvellanedaStoikov, SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import markouts
from mmsim.sim import Simulator

SESSIONS = 200
RATE, SIGMA_V = 1.0, 1.0
H = np.array([0, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30])
SIGMA, K, A, Q = 1.0, 3.6, 2.1, 5


def make_bot(name, xi):
    match name:
        case "join the touch":
            return SymmetricQuoter(0.5, max_position=Q)
        case "AS, gamma 0.001":
            return AvellanedaStoikov(0.001, SIGMA, K, max_position=Q)
        case "GLFT, gamma 0.3":
            return GLFT(Model(SIGMA, A, K, 0.3, Q))
        case "GLFT, gamma 1":
            return GLFT(Model(SIGMA, A, K, 1.0, Q))
        case "GLFT, gamma 1, xi":
            return GLFT(Model(SIGMA, A, K, 1.0, Q, xi=xi))


def session(args):
    name, informed, xi, seed = args
    value = Fundamental(SIGMA_V, seed=seed)
    agents = [make_bot(name, xi)]
    if informed:
        agents.append(InformedTrader(value, RATE, seed=seed))
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    sums = {}
    for who in ("noise", "informed"):
        fills = [f for f in ctx.account.fills if (f.counterparty != 0) == (who == "informed")]
        m = markouts(fills, sim.tape, H, sim.end) if fills else np.empty((0, len(H)))
        sums[who] = (np.nansum(m, axis=0), np.nansum(m**2, axis=0), (~np.isnan(m)).sum(axis=0))
    return ctx.pnl(), len(ctx.account.fills), sums


def pooled(results, who):
    s = sum(r[2][who][0] for r in results)
    s2 = sum(r[2][who][1] for r in results)
    n = sum(r[2][who][2] for r in results)
    mean = s / n
    return mean, np.sqrt((s2 / n - mean**2) / n), n[0]


def main():
    names = ["join the touch", "AS, gamma 0.001", "GLFT, gamma 0.3", "GLFT, gamma 1"]
    rows = {}
    with ProcessPoolExecutor() as pool:

        def run(name, informed, xi=0.0):
            jobs = [(name, informed, xi, s) for s in range(SESSIONS)]
            rows[name, informed] = list(pool.map(session, jobs, chunksize=4))

        for name in names:
            for informed in (False, True):
                run(name, informed)

        touch = rows["join the touch", True]
        both = [
            (sum(r[2][w][0] for r in touch), sum(r[2][w][2] for r in touch))
            for w in ("noise", "informed")
        ]
        everything = sum(s for s, _ in both) / sum(n for _, n in both)
        xi = float(everything[0] - everything[-1])
        print(f"xi from the touch quoter's markouts, 0 to {H[-1]:.0f} minutes: {xi:.2f} ticks")
        names.append("GLFT, gamma 1, xi")
        for informed in (False, True):
            run(names[-1], informed, xi)

    print(f"{SESSIONS} sessions of 390 minutes each, P&L in ticks per session")
    print("strategy             informed  mean pnl  (se)    sd pnl  fills")
    for name in names:
        for informed in (False, True):
            pnl = np.array([r[0] for r in rows[name, informed]])
            fills = np.mean([r[1] for r in rows[name, informed]])
            print(
                f"{name:20s} {'yes' if informed else 'no':8s} {pnl.mean():8.1f}"
                f" {pnl.std() / np.sqrt(SESSIONS):6.1f} {pnl.std():8.1f} {fills:6.0f}"
            )

    print("markouts of the touch quoter with informed flow, ticks per lot")
    curves = {who: pooled(touch, who) for who in ("noise", "informed")}
    for who, (mean, se, n) in curves.items():
        print(f"  {who} ({n} fills):", " ".join(f"{h:g}:{m:+.2f}" for h, m in zip(H, mean)))

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 10, "xtick.color": muted, "ytick.color": muted})
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), facecolor=surface, gridspec_kw={"width_ratios": [1, 1.1]}
    )
    for ax in (left, right):
        ax.set_facecolor(surface)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(muted)

    left.grid(color=grid, linewidth=0.8)
    for (who, (mean, se, n)), color in zip(curves.items(), (blue, orange), strict=True):
        left.fill_between(H, mean - 2 * se, mean + 2 * se, color=color, alpha=0.18, linewidth=0)
        left.plot(H, mean, color=color, linewidth=2)
        i = np.searchsorted(H, 15 if who == "noise" else 20)
        left.annotate(
            f"fills against {who} flow ({n:,})",
            (H[i], mean[i]),
            xytext=(0, 10) if who == "noise" else (-10, -8),
            textcoords="offset points",
            ha="center" if who == "noise" else "right",
            va="bottom" if who == "noise" else "top",
            color=ink,
        )
    left.axhline(0, color=muted, linewidth=1)
    left.set_xlim(0, H[-1])
    left.set_xlabel("minutes after the fill", color=ink)
    left.set_ylabel("markout per lot (ticks)", color=ink)
    left.set_title("Joining the touch: who you trade with", loc="left", color=ink)

    right.grid(axis="x", color=grid, linewidth=0.8)
    y = np.arange(len(names))[::-1]
    for dy, informed, color, label in (
        (0.19, False, blue, "noise flow only"),
        (-0.19, True, orange, "with informed traders"),
    ):
        means = [np.mean([r[0] for r in rows[n, informed]]) for n in names]
        right.barh(y + dy, means, height=0.36, color=color, label=label)
        for yi, m in zip(y + dy, means, strict=True):
            right.annotate(
                f"{m:.0f}",
                (m, yi),
                xytext=(4 if m >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if m >= 0 else "right",
                va="center",
                color=ink,
            )
    labels = [n.replace(", xi", f", xi {xi:.2f}") for n in names]
    right.set_yticks(y, labels, color=ink)
    right.axvline(0, color=muted, linewidth=1)
    lo = min(0, *[np.mean([r[0] for r in v]) for v in rows.values()])
    hi = max(np.mean([r[0] for r in v]) for v in rows.values())
    right.set_xlim(lo * 1.35, hi * 1.2)
    right.set_xlabel("mean P&L per session (ticks)", color=ink)
    right.set_title("Same bots, with and without informed flow", loc="left", color=ink)
    right.legend(loc="lower left", frameon=False, labelcolor=ink)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "informed_flow.png"
    fig.savefig(out, dpi=150, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
