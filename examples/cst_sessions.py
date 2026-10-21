from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import AvellanedaStoikov, SymmetricQuoter
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.sim import Simulator

SESSIONS = 200
GAMMAS = (0.0003, 0.001, 0.003, 0.01)
SIGMA, K = 1.0, 3.6


def session(args):
    gamma, seed = args
    bot = SymmetricQuoter(0.5) if gamma is None else AvellanedaStoikov(gamma, SIGMA, K)
    sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [bot], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    ctx = sim.contexts[0]
    q = np.array([s.position for s in ctx.history])
    return ctx.pnl(), ctx.position, np.abs(q).mean(), len(ctx.account.fills)


def main():
    rows = {}
    with ProcessPoolExecutor() as pool:
        for gamma in (None, *GAMMAS):
            res = np.array(list(pool.map(session, [(gamma, s) for s in range(SESSIONS)])))
            rows[gamma] = res.T

    print(f"{SESSIONS} sessions each; pnl in ticks per session")
    print("strategy        mean pnl   (se)   sd pnl  mean/sd  sd(final q)  mean |q|  fills")
    for gamma, (pnl, q_end, abs_q, fills) in rows.items():
        name = "join the touch" if gamma is None else f"AS gamma={gamma}"
        print(
            f"{name:15s} {pnl.mean():8.1f} {pnl.std() / np.sqrt(SESSIONS):7.1f} {pnl.std():8.1f}"
            f" {pnl.mean() / pnl.std():8.2f} {q_end.std():12.1f} {abs_q.mean():9.2f}"
            f" {fills.mean():6.0f}"
        )

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(muted)
    ax.tick_params(colors=muted)

    sd = [rows[g][0].std() for g in GAMMAS]
    mean = [rows[g][0].mean() for g in GAMMAS]
    ax.plot(sd, mean, color=blue, linewidth=2, marker="o", markersize=8, label="Avellaneda-Stoikov")
    for g, x, y in zip(GAMMAS, sd, mean, strict=True):
        ax.annotate(f"gamma {g}", (x, y), xytext=(8, -4), textcoords="offset points", color=ink)
    pnl = rows[None][0]
    ax.plot(
        pnl.std(),
        pnl.mean(),
        color=orange,
        marker="o",
        markersize=9,
        linestyle="none",
        label="join the touch",
    )
    ax.annotate(
        "join the touch",
        (pnl.std(), pnl.mean()),
        xytext=(-10, -16),
        textcoords="offset points",
        ha="right",
        color=ink,
    )

    ax.set_xlim(0, pnl.std() * 1.12)
    ax.set_ylim(0, max(pnl.mean(), *mean) * 1.2)
    ax.set_xlabel("sd of session P&L (ticks)", color=ink)
    ax.set_ylabel("mean session P&L (ticks)", color=ink)
    ax.set_title(
        f"Risk and return over {SESSIONS} simulated 390-minute sessions", loc="left", color=ink
    )
    ax.legend(loc="lower right", frameon=False, labelcolor=ink)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "cst_sessions.png"
    fig.savefig(out, dpi=150, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
