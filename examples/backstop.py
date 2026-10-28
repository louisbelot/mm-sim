from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.attribution import attribute
from mmsim.bots import GLFT, AvellanedaStoikov, SymmetricQuoter
from mmsim.flow import CSTParams, PairedFlow
from mmsim.glft import Model
from mmsim.hawkes import Hawkes
from mmsim.sim import Simulator

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c8384a"

SESSIONS = 300
SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
MODEL = Model(1.0, 2.1, 3.6, 0.3, 5)
PARTNERS = (
    "alone",
    "touch",
    "touch, cap 20",
    "touch, cap 5",
    "touch, keep 1",
    "AS",
    "GLFT",
)
SOLO = ("touch", "touch, cap 20", "touch, cap 5")


def partner(name):
    match name:
        case "touch":
            return SymmetricQuoter(0.5)
        case "touch, cap 20":
            return SymmetricQuoter(0.5, max_position=20)
        case "touch, cap 5":
            return SymmetricQuoter(0.5, max_position=5)
        case "touch, keep 1":
            return SymmetricQuoter(0.5, keep=1)
        case "AS":
            return AvellanedaStoikov(0.001, 1.0, 3.6)
        case "GLFT":
            return GLFT(MODEL, keep=1)


def play(agents, seed):
    sim = Simulator(PairedFlow(SKY, HAWKES), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(390)
    prints = [p for p in sim.prints if p.t >= 30]
    t = np.array([p.t for p in prints])
    sign = np.array([1.0 if p.side == 0 else -1.0 for p in prints])
    mids = sim.tape.at(np.arange(30.0, 420.0 + 1e-9, 1.0))
    market = dict(
        impact=float(sign @ (sim.tape.at(t) - sim.tape.before(t)) / len(t)),
        vol=float(np.diff(mids).std()),
    )
    return sim, float(sim.tape.at(sim.end)), market


def session(args):
    name, seed = args
    agents = [GLFT(MODEL, keep=1)]
    if name != "alone":
        agents.append(partner(name))
    sim, close, market = play(agents, seed)
    ctx = sim.contexts[0]
    fills = ctx.account.fills
    a = attribute(fills, sim.tape, 1.0, sim.end)
    other = sim.contexts[1] if len(sim.contexts) > 1 else None
    return dict(
        pnl=ctx.account.value(close),
        fills=len(fills),
        parts=(a.spread, a.adverse, a.inventory),
        emptied=sum(f.qty for f in fills if f.maker and f.left == 0),
        other_pnl=other.account.value(close) if other else np.nan,
        other_max=max(abs(s.position) for s in other.history) if other else np.nan,
        **market,
    )


def solo(args):
    name, seed = args
    sim, close, market = play([partner(name)], seed)
    ctx = sim.contexts[0]
    return dict(
        pnl=ctx.account.value(close),
        max=max(abs(s.position) for s in ctx.history),
        **market,
    )


def compute():
    with ProcessPoolExecutor() as pool:
        pairs = {
            name: list(pool.map(session, [(name, s) for s in range(SESSIONS)], chunksize=4))
            for name in PARTNERS
        }
        alone = {
            name: list(pool.map(solo, [(name, s) for s in range(SESSIONS)], chunksize=4))
            for name in SOLO
        }
    return pairs, alone


def report(d):
    runs, alone = d
    base = np.array([r["pnl"] for r in runs["alone"]])
    print(f"GLFT with one other bot, {SESSIONS} sessions on PairedFlow, P&L in ticks")
    print(
        "partner          gain (se)     P&L  fills  spread  adverse  inventory  emptied"
        "  impact  vol    partner P&L  partner max |q|"
    )
    for name, rows in runs.items():
        d = np.array([r["pnl"] for r in rows]) - base
        parts = np.array([r["parts"] for r in rows]).mean(axis=0)

        def mean(key, rows=rows):
            return np.mean([r[key] for r in rows])

        print(
            f"{name:15s} {d.mean():+6.1f} ({d.std(ddof=1) / np.sqrt(len(d)):.1f})"
            f" {mean('pnl'):7.1f} {mean('fills'):6.0f} {parts[0]:7.1f} {parts[1]:8.1f}"
            f" {parts[2]:10.1f} {mean('emptied'):8.1f} {mean('impact'):7.2f} {mean('vol'):5.2f}"
            f" {mean('other_pnl'):13.1f} {mean('other_max'):16.1f}"
        )
    print("\nthe touch bot alone      P&L (se)   max |q|  impact   vol")
    for name, rows in alone.items():
        pnl = np.array([r["pnl"] for r in rows])
        print(
            f"{name:22s} {pnl.mean():7.1f} ({pnl.std(ddof=1) / np.sqrt(len(pnl)):.1f})"
            f" {np.mean([r['max'] for r in rows]):8.1f} {np.mean([r['impact'] for r in rows]):7.2f}"
            f" {np.mean([r['vol'] for r in rows]):5.2f}"
        )


def chart(d, out):
    runs, alone = d
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    names = list(runs)
    x = np.arange(len(names))
    parts = np.array([np.array([r["parts"] for r in runs[n]]).mean(axis=0) for n in names])
    w = 0.26
    for j, (label, color) in enumerate(
        (("spread", AQUA), ("adverse, first minute", RED), ("inventory", ORANGE))
    ):
        left.bar(x + (j - 1) * w, parts[:, j], w, color=color, label=label)
    left.plot(x, [np.mean([r["pnl"] for r in runs[n]]) for n in names], "D", color=INK, label="P&L")
    left.axhline(0, color=MUTED, linewidth=1)
    left.set_xticks(x, [n.replace(", ", "\n") for n in names])
    left.set_ylabel("GLFT, ticks per session")
    left.set_title("GLFT's P&L next to one other bot, Hawkes flow", loc="left", color=INK)
    left.legend(loc="upper right", ncol=2, frameon=False, labelcolor=INK, fontsize=9)

    caps = [("touch, cap 5", 5), ("touch, cap 20", 20), ("touch", 40)]
    cx = [c for _, c in caps]
    base = np.array([r["pnl"] for r in runs["alone"]])
    gain = [np.mean([r["pnl"] for r in runs[n]]) - base.mean() for n, _ in caps]
    paid = [np.mean([r["other_pnl"] for r in runs[n]]) for n, _ in caps]
    impact = [np.mean([r["impact"] for r in runs[n]]) for n, _ in caps]
    solo_pnl = [np.mean([r["pnl"] for r in alone[n]]) for n, _ in caps]
    right.plot(cx, gain, "o-", color=BLUE, label="GLFT's gain over trading alone")
    right.plot(cx, paid, "o-", color=ORANGE, label="the touch bot's P&L next to GLFT")
    right.plot(cx, solo_pnl, "o--", color=AQUA, label="the touch bot's P&L on its own")
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xticks(cx, ["5 lots", "20 lots", "no cap"])
    right.set_xlabel("the touch bot's position limit")
    right.set_ylabel("ticks per session")
    for c, g, i in zip(cx, gain, impact, strict=True):
        right.annotate(
            f"impact {i:.2f}",
            (c, g),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            color=MUTED,
            fontsize=8,
        )
    right.set_xlim(0, 45)
    right.set_title("A bot with room to absorb bursts helps the others", loc="left", color=INK)
    right.legend(loc="upper left", frameon=False, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    runs = compute()
    report(runs)
    chart(runs, Path(__file__).resolve().parent.parent / "docs" / "backstop.png")


if __name__ == "__main__":
    main()
