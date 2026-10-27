from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mmsim.bots import GLFT, SymmetricQuoter
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence
from mmsim.glft import Model
from mmsim.hawkes import Hawkes, fit
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import markouts
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SESSIONS = 200
SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
H = np.array([0, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30])
BOTS = ("join the touch", "GLFT, gamma 0.3")


def make_bot(name):
    if name == "join the touch":
        return SymmetricQuoter(0.5, max_position=5, keep=1)
    return GLFT(Model(1.0, 2.1, 3.6, 0.3, 5), keep=1)


def make_flow(kind):
    return HawkesFlow(SKY, HAWKES) if kind == "hawkes" else ZeroIntelligence(SKY)


def session(args):
    name, kind, informed, seed = args
    agents = [make_bot(name)]
    if informed:
        agents.append(InformedTrader(Fundamental(1.0, seed=seed), 1.0, seed=seed))
    sim = Simulator(make_flow(kind), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    m = markouts(fills, sim.tape, H, sim.end)
    q = np.array([s.position for s in ctx.history])
    mids = sim.tape.at(np.arange(sim.end - 390, sim.end + 0.5, 1.0))
    return ctx.pnl(), m, np.abs(q).mean(), np.diff(mids).std()


def main():
    rows = {}
    with ProcessPoolExecutor() as pool:
        for name in BOTS:
            for kind in ("poisson", "hawkes"):
                for informed in (False, True):
                    jobs = [(name, kind, informed, s) for s in range(SESSIONS)]
                    rows[name, kind, informed] = list(pool.map(session, jobs, chunksize=4))

    print(
        "strategy          market orders  informed  mean P&L (se)  sd P&L  fills"
        "  mean |q|  sd 1-min mid move  mk 0/5 min"
    )
    for (name, kind, informed), r in rows.items():
        pnl = np.array([x[0] for x in r])
        m = np.concatenate([x[1] for x in r])
        print(
            f"{name:17s} {kind:14s} {'yes' if informed else 'no':9s}"
            f" {pnl.mean():8.1f} ({pnl.std() / np.sqrt(len(pnl)):.1f}) {pnl.std():7.1f}"
            f" {len(m) / len(r):6.0f} {np.mean([x[2] for x in r]):9.2f}"
            f" {np.mean([x[3] for x in r]):18.2f}"
            f"  {np.nanmean(m[:, 0]):5.2f} {np.nanmean(m[:, 6]):5.2f}"
        )

    T = 2000.0
    sim = Simulator(HawkesFlow(SKY, HAWKES), seed=11)
    sim.seed(10_000, 10_001)
    sim.warm_up(T)
    times = np.array([p.t for p in sim.prints])
    marks = np.array([int(p.side) for p in sim.prints])
    fitted = fit(times, marks, T, 2)
    rates = np.bincount(marks, minlength=2) / T
    flat = Hawkes.make(rates, np.zeros((2, 2)), 1.0)
    print(f"\n{len(times)} market orders in {T:g} minutes, {rates.round(3)} per minute")
    print(
        f"fitted branching {fitted.branching.round(3).tolist()}, true {HAWKES.branching.tolist()}"
    )
    print(f"fitted beta {fitted.beta.round(2).tolist()}, mu {fitted.mu.round(3).tolist()}")
    res = {}
    for label, model in (("Hawkes fit", fitted), ("Poisson fit", flat)):
        r = np.concatenate(model.residuals(times, marks))
        res[label] = r
        ks = stats.kstest(r, "expon")
        print(f"{label}: KS statistic {ks.statistic:.4f}, p-value {ks.pvalue:.3g}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    left, mid, right = axes

    lo, hi = 1000.0, 1030.0
    grid = np.linspace(lo, hi, 3000)
    lam = HAWKES.intensity(times, marks, grid)
    for i, (label, color) in enumerate((("buys", BLUE), ("sells", ORANGE))):
        left.plot(grid - lo, lam[:, i], color=color, linewidth=1.4, label=f"market {label}")
        sel = (marks == i) & (times >= lo) & (times < hi)
        left.plot(
            times[sel] - lo,
            np.full(sel.sum(), -0.4 - 0.5 * i),
            "|",
            color=color,
            markersize=9,
        )
    left.axhline(SKY.mu, color=MUTED, linewidth=1, linestyle="--")
    left.text(4.2, SKY.mu + 0.15, "Poisson rate", color=MUTED, ha="left", fontsize=9)
    left.set_xlim(0, 30)
    left.set_xlabel("minutes")
    left.set_ylabel("intensity (market orders per minute)")
    left.set_title("Hawkes market orders, 30 minutes", loc="left", color=INK)
    left.legend(loc="upper right", frameon=False, labelcolor=INK)

    p = (np.arange(1, 501) - 0.5) / 500
    theory = -np.log1p(-p)
    for (label, r), color in zip(res.items(), (AQUA, ORANGE), strict=True):
        mid.plot(theory, np.quantile(r, p), color=color, linewidth=2, label=label)
    mid.plot([0, 7], [0, 7], color=MUTED, linewidth=1, linestyle="--")
    mid.set_xlim(0, 7)
    mid.set_ylim(0, 7)
    mid.set_xlabel("Exp(1) quantile")
    mid.set_ylabel("quantile of the rescaled gaps")
    mid.set_title("Time rescaling: which model explains the flow?", loc="left", color=INK)
    mid.legend(loc="upper left", frameon=False, labelcolor=INK)

    x = np.arange(4)
    labels = []
    for k, name in enumerate(BOTS):
        color = (ORANGE, BLUE)[k]
        means, ses = [], []
        for kind in ("poisson", "hawkes"):
            for informed in (False, True):
                pnl = np.array([r[0] for r in rows[name, kind, informed]])
                means.append(pnl.mean())
                ses.append(pnl.std() / np.sqrt(len(pnl)))
                if k == 0:
                    labels.append(f"{kind}\n{'informed' if informed else 'noise only'}")
        right.errorbar(
            x + (k - 0.5) * 0.12,
            means,
            yerr=2 * np.array(ses),
            color=color,
            linestyle="none",
            marker="o",
            markersize=5,
            capsize=3,
            label=name,
        )
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xticks(x, labels)
    right.set_ylabel("mean P&L per session (ticks, 2 se bars)")
    right.set_title("Same mean rate, clustered or not", loc="left", color=INK)
    right.legend(loc="lower left", frameon=False, labelcolor=INK)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "hawkes_flow.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
