from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim import duopoly as du
from mmsim import learners as le
from mmsim import trigger as tr

BASE = du.Game(tick=0.1, reach=1.5, Q=(5, 5), phi=0.5)
GAMMA = 0.99
LENGTHS = (1, 2, 3, 5, 10, 20, 50, 100, None)
TOL = 1e-9


def label(L):
    return "grim" if L is None else str(L)


def sweep(tick):
    g = replace(BASE, tick=tick)
    Q = g.Q[0]
    nash = le.markov(g, GAMMA)
    cart = le.collusive(g, GAMMA)
    lo = (nash.V1 + nash.V2)[Q, Q] / 2
    hi = (cart.V1 + cart.V2)[Q, Q] / 2
    tol = TOL + nash.eps
    marks = range(g.J + 1)
    gain = np.zeros(len(marks))
    held = np.zeros((len(LENGTHS), len(marks)), dtype=bool)
    split = tr.shared(g, cart.quotes)
    cartel, share = [], []
    for i, L in enumerate(LENGTHS):
        for m in marks:
            res = tr.check(g, GAMMA, tr.widen(g, nash.quotes, m), nash.quotes, L)
            gain[m] = ((res.V1 + res.V2)[0, Q, Q] / 2 - lo) / (hi - lo)
            held[i, m] = res.gain <= tol
        for out, coop in ((cartel, cart.quotes), (share, split)):
            res = tr.check(g, GAMMA, coop, nash.quotes, L)
            G = np.maximum(res.gain1, res.gain2)
            v = ((res.V1 + res.V2)[0, Q, Q] / 2 - lo) / (hi - lo)
            out.append((G[0].max(), G[1:].max(), v, res.gain <= tol))
    war = tr.check(g, GAMMA, split, tr.war(g), None)
    W = np.maximum(war.gain1, war.gain2)
    return {
        "tick": tick,
        "nash": lo,
        "cartel": hi,
        "gain": gain,
        "held": held,
        "cartel gains": cartel,
        "shared": share,
        "war": (W[0].max(), W[1].max()),
        "eps": nash.eps,
    }


def patience(tick, gammas):
    g = replace(BASE, tick=tick)
    Q = g.Q[0]
    out = []
    for gamma in gammas:
        nash = le.markov(g, gamma)
        cart = le.collusive(g, gamma)
        lo = (nash.V1 + nash.V2)[Q, Q] / 2
        hi = (cart.V1 + cart.V2)[Q, Q] / 2
        res = tr.check(g, gamma, tr.shared(g, cart.quotes), nash.quotes, None)
        G = np.maximum(res.gain1, res.gain2)
        v = ((res.V1 + res.V2)[0, Q, Q] / 2 - lo) / (hi - lo)
        best = 0.0
        for m in range(g.J + 1):
            w = tr.check(g, gamma, tr.widen(g, nash.quotes, m), nash.quotes, None)
            if w.gain <= TOL + nash.eps:
                best = max(best, ((w.V1 + w.V2)[0, Q, Q] / 2 - lo) / (hi - lo))
        out.append((gamma, G[0].max(), G[1].max(), v, best, nash.eps))
    return out


def draw(rows, out):
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COLORS = {0.1: "#2a78d6", 0.2: "#eb6834"}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax = axes[0]
    x = np.arange(len(LENGTHS))
    for r in rows:
        best = np.where(r["held"], r["gain"][None, :], 0.0).max(axis=1)
        split = np.array([v if ok else 0.0 for _, _, v, ok in r["shared"]])
        ax.plot(
            x,
            np.maximum(best, split),
            color=COLORS[r["tick"]],
            lw=1.8,
            marker="o",
            ms=4,
            label=f"tick {r['tick']:g}",
        )
    ax.set_xticks(x, [label(L) for L in LENGTHS])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("punishment length after a deviation (steps of Nash quoting)")
    ax.set_ylabel("best sustainable profit gain")
    ax.set_title("How long a punishment collusion needs", loc="left", color=INK)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    for r in rows:
        first = np.where(r["held"].any(axis=0), r["held"].argmax(axis=0), len(LENGTHS))
        m = np.arange(r["gain"].size)
        ok = first < len(LENGTHS)
        ax.plot(
            m[ok],
            first[ok],
            color=COLORS[r["tick"]],
            lw=1.6,
            marker="o",
            ms=4,
            drawstyle="steps-mid",
            label=f"tick {r['tick']:g}",
        )
    ax.set_yticks(range(len(LENGTHS)), [label(L) for L in LENGTHS])
    ax.set_xlabel("markup over the Nash quotes (ticks)")
    ax.set_ylabel("shortest punishment that holds it")
    ax.set_title("Wider markups need longer memories", loc="left", color=INK)
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    rows = [sweep(t) for t in (0.1, 0.2)]
    print(f"phi {BASE.phi}, gamma {GAMMA}, Q {BASE.Q}; on path: Nash quotes widened by m ticks")
    print("profit gain = (value - Nash) / (cartel - Nash) per maker at q = (0, 0)")
    for r in rows:
        print(f"tick {r['tick']:g}: nash {r['nash']:.3f}, cartel {r['cartel']:.3f}")
        print(f"  Markov Nash eps {r['eps']:.4f} (the tolerance for every check)")
        print("  gain by markup: " + " ".join(f"{m}:{v:.2f}" for m, v in enumerate(r["gain"])))
        rest = zip(LENGTHS, r["held"], r["cartel gains"], r["shared"], strict=True)
        for L, held, (gc, gp, _, _), (sc, sp, sv, _) in rest:
            ms = np.flatnonzero(held)
            best = r["gain"][ms].max() if ms.size else 0.0
            span = f"{ms.min()}-{ms.max()}" if ms.size else "none"
            print(
                f"  length {label(L):>4s}: markups held {span:>5s} (best {best:.2f});"
                f" deviation gain on path / in punishment: cartel {gc:+.3f} / {gp:+.4f},"
                f" shared cartel ({sv:.2f}) {sc:+.3f} / {sp:+.4f}"
            )
        gc, gp = r["war"]
        print(f"  shared cartel with a permanent war: on path {gc:+.4f}, in the war {gp:+.4f}")
    print("grim Nash reversion at tick 0.1 by discount factor (shared cartel, best markup):")
    for gamma, gc, gp, v, best, eps in patience(0.1, (0.5, 0.7, 0.8, 0.9, 0.95)):
        print(
            f"  gamma {gamma:.2f}: shared cartel ({v:.2f}) on path {gc:+.4f}, in punishment"
            f" {gp:+.6f}; best markup held {best:.2f} (nash eps {eps:.4f})"
        )
    out = Path(__file__).resolve().parent.parent / "docs" / "trigger_strategies.png"
    draw(rows, out)
    print(f"wrote {out}")
