import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from itch_replay import AQUA, BLUE, INK, MUTED, ROOT, STARTS, SURFACE, SYMBOLS, bot, style
from order_memory import Prints, day, flows, real_session, run, sessions, store, stored
from queue_reactive import summary, table

from mmsim import qr
from mmsim.qr import CANCEL, LIMIT
from mmsim.sim import Simulator

FIRST = SYMBOLS[0]
SEEDS = range(48)
FEW = range(12)
MODELS = ("market kernels", "+ touch reaction")
KINDS = {"cancels": CANCEL, "limits": LIMIT}
EDGES = np.geomspace(1e-5, 10.0, 31) / 60
WARM, LONG = 30.0, 150.0


def reacting(symbol):
    return ROOT / "data" / f"{symbol}.qr3r.pickle"


@cache
def params(symbol, model):
    path = stored(symbol, 3) if model == MODELS[0] else reacting(symbol)
    if not path.exists():
        p = store(symbol, 3)
        path.write_bytes(pickle.dumps(day(symbol).react(p)))
    return pickle.loads(path.read_bytes())


def model_session(args):
    symbol, model, seed = args
    prints = Prints()
    flow = qr.QueueReactive(params(symbol, model), seed=seed)
    sim = Simulator(flow, [bot("join the touch", 100, 1000), prints], seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(30.0)
    return run(sim, prints)


def response(f, kind):
    src, by = f.t[f.hits[:, 0]], f.hits[:, 1]
    q = f.quotes[f.quotes[:, 2] == kind]
    base = len(q) / 2 / (f.t[-1] - f.t[0])
    out = []
    for same in (True, False):
        n = np.zeros(EDGES.size - 1)
        for s in (0, 1):
            x = f.t[q[q[:, 1] == (s if same else 1 - s), 0]]
            c = np.searchsorted(x, src[by == s][:, None] + EDGES, "right")
            n += np.diff(c, axis=1).sum(axis=0)
        out.append(n / src.size / np.diff(EDGES) / base)
    return np.array(out)


def model_response(args):
    symbol, model, seed = args
    f = day(symbol)
    msgs = qr.messages(params(symbol, model), LONG, seed)
    g = qr.fit(msgs, WARM, LONG, aes=f.aes, edges=f.edges)
    return np.array([response(g, k) for k in KINDS.values()])


def resid_stats(f, p, kind):
    r = np.sort(np.concatenate(f.residuals(p, kind)))
    u = 1 - np.exp(-r)
    return r.size, r.var(), np.abs(u - (np.arange(r.size) + 0.5) / r.size).max()


def chart(curves, moves, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURFACE)
    lag = np.sqrt(EDGES[1:] * EDGES[:-1]) * 60_000
    colors = {"tape": INK, MODELS[0]: MUTED, MODELS[1]: BLUE}
    for ax, (j, name) in zip(axes[:2], enumerate(KINDS), strict=True):
        style(ax)
        ax.axhline(1, color=MUTED, linewidth=0.8)
        for src, color in colors.items():
            own, cross = curves[src][j]
            ax.loglog(lag, own, color=color, linewidth=2, label=f"{src}, hit side")
            ax.loglog(lag, cross, color=color, linewidth=1.4, linestyle="--", label="other side")
        ax.set_xlabel("time after a market order, ms", color=INK)
        ax.set_ylabel(f"rate of touch {name} / its average", color=INK)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=8, ncol=1)

    c = axes[2]
    style(c)
    names = list(moves)
    x = np.arange(len(names))
    w = 0.36
    c.axhline(1, color=INK, linewidth=1)
    for j, (key, color) in enumerate(((MODELS[0], MUTED), (MODELS[1], AQUA))):
        ratio = [moves[n][key] / moves[n]["tape"] for n in names]
        c.bar(x + (j - 0.5) * w, ratio, w, color=color, label=key)
    c.set_xticks(x, names, color=INK)
    c.set_ylim(0, 2.3)
    c.set_ylabel("mid moves per minute, model / tape", color=INK)
    c.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    f = day(FIRST)
    hot = params(FIRST, MODELS[1])
    e = hot.excite
    print(FIRST, "market orders", len(f.hits), "kernel timescales (ms)", end=" ")
    print(" ".join(f"{60_000 / b:.2f}" for b in e.beta))
    print("\ntouch     side     per kernel                 total   baseline scale")
    for name, kind in KINDS.items():
        r = e.react[kind]
        for side, w in (("hit", r.own), ("other", r.cross)):
            ks = " ".join(f"{v:7.3f}" for v in w)
            print(f"{name:9s} {side:6s} {ks}   {sum(w):7.2f}   {r.rest:6.3f}")

    print("\ntouch     model              events  resid var     ks   loglik gain per event")
    for name, kind in KINDS.items():
        cold = params(FIRST, MODELS[0])
        gain = (f.loglik(hot, kind) - f.loglik(cold, kind)) / resid_stats(f, hot, kind)[0]
        for model, p in ((MODELS[0], cold), (MODELS[1], hot)):
            n, var, ks = resid_stats(f, p, kind)
            extra = f"{gain:10.3f}" if p is hot else ""
            print(f"{name:9s} {model:17s} {n:7d} {var:10.2f} {ks:6.3f} {extra}")

    for symbol in SYMBOLS:
        for model in MODELS:
            params(symbol, model)

    with ProcessPoolExecutor() as pool:
        curves = {"tape": np.array([response(f, k) for k in KINDS.values()])}
        for model in MODELS:
            runs = list(pool.map(model_response, [(FIRST, model, s) for s in range(6)]))
            curves[model] = np.mean(runs, axis=0)
        print("\nrate after a market order / average, at 0.1 ms, 1 ms, 10 ms, 100 ms")
        lag = np.sqrt(EDGES[1:] * EDGES[:-1]) * 60_000
        at = [np.argmin(np.abs(np.log(lag / x))) for x in (0.1, 1.0, 10.0, 100.0)]
        for src, c in curves.items():
            for j, name in enumerate(KINDS):
                for side, row in (("hit", c[j, 0]), ("other", c[j, 1])):
                    vals = " ".join(f"{row[i]:7.2f}" for i in at)
                    print(f"{src:17s} {name:8s} {side:6s} {vals}")

        tape = sessions(pool, real_session, [(FIRST, s) for s in STARTS])
        lines, stats = {"tape": summary(tape[:, :8])}, {"tape": flows(tape)}
        for model in MODELS:
            rows = sessions(pool, model_session, [(FIRST, model, s) for s in SEEDS])
            lines[model], stats[model] = summary(rows[:, :8]), flows(rows)
        table(f"{FIRST}, fit on the whole day, join the touch", lines)
        print("\n                  prints  fano 10s   sign acf at lags 1, 2")
        for name, (n, fano, acf) in stats.items():
            print(f"{name:17s} {n:6.0f} {fano:9.2f}  {acf[0]:6.3f} {acf[1]:6.3f}")

        moves, by = {}, {}
        for symbol in SYMBOLS:
            real = summary(sessions(pool, real_session, [(symbol, s) for s in STARTS])[:, :8])
            by[f"{symbol} tape"] = real
            moves[symbol] = {"tape": real[6]}
            for model in MODELS:
                jobs = [(symbol, model, s) for s in FEW]
                row = summary(sessions(pool, model_session, jobs)[:, :8])
                by[f"{symbol} {model[:6]}"], moves[symbol][model] = row, row[6]
        table("every symbol, 12 sessions a row", by)

    chart(curves, moves, ROOT / "docs" / "cancel_race.png")
