import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmsim.bots import GLFT, AvellanedaStoikov, SymmetricQuoter
from mmsim.env import DEPTHS, Learned, MarketMakingEnv, sky
from mmsim.glft import Model
from mmsim.ppo import Config, train
from mmsim.tournament import session, standings

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2f9e6b"

ROOT = Path(__file__).resolve().parents[1]
PHI, TERM, MINUTES, Q = 0.02, 0.25, 60.0, 5
SIGMA, A, K = 1.0, 2.1, 3.6
CFG = Config(episodes=48, batch=512, epochs=8, lr=1e-3, entropy=0.001, lam=0.9)
ITERS = 400
SEEDS = (0, 1, 2)
TUNE = range(10_000_000, 10_000_300)
TEST = range(1000)
GAMMAS = (0.0003, 0.001, 0.003, 0.01, 0.02, 0.04, 0.08, 0.16, 0.3)

make_env = partial(MarketMakingEnv, phi=PHI, terminal=TERM, minutes=MINUTES, Q=Q)


def const(a, obs):
    return a


def touch():
    return SymmetricQuoter(0.5, max_position=Q)


def fixed(a):
    return Learned(partial(const, a), Q, MINUTES, 1.0)


def avst(g):
    return AvellanedaStoikov(g, SIGMA, K, max_position=Q)


def glft(g, keep, inside=True):
    return GLFT(Model(SIGMA, A, K, g, Q), keep=keep, inside=inside)


class Mirror:
    flip = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    def __init__(self, policy):
        self.policy = policy

    def greedy(self, obs):
        i, j = divmod(self.policy.greedy(obs * self.flip), len(DEPTHS))
        return j * len(DEPTHS) + i


def learned(policy):
    return Learned(policy.greedy, Q, MINUTES, 1.0)


def score(r):
    return r.pnl - PHI * r.exposure - TERM * r.position**2


def solo(args):
    bot, seed = args
    r = session({"bot": bot}, seed, flow=sky, minutes=MINUTES)["bot"]
    return score(r), r.pnl, r.fills, r.exposure


def together(args):
    bots, seed = args
    return session(bots, seed, flow=sky, minutes=MINUTES)


def apart(args):
    bots, seed = args
    return {n: session({n: b}, seed, flow=sky, minutes=MINUTES)[n] for n, b in bots.items()}


def evaluate(pool, bot, seeds):
    return np.array(list(pool.map(solo, [(bot, s) for s in seeds], chunksize=25)))


def tune(pool, candidates):
    return max(candidates, key=lambda c: evaluate(pool, c[1], TUNE)[:, 0].mean())


def tuned(pool):
    a = tune(pool, [(a, partial(fixed, a)) for a in range(len(DEPTHS) ** 2)])[0]
    g = tune(pool, [(g, partial(avst, g)) for g in GAMMAS])[0]
    gk = tune(pool, [((g, k), partial(glft, g, k)) for g in GAMMAS for k in (0, 1)])[0]
    return a, g, gk


def label(a):
    i, j = divmod(a, len(DEPTHS))
    name = lambda d: "off" if d is None else f"{d}"  # noqa: E731
    return f"bid {name(DEPTHS[i])}, ask {name(DEPTHS[j])}"


def policy_map(policy):
    rows = []
    for imb in (-0.6, 0.0, 0.6):
        acts = [policy.greedy(np.array([q / Q, 0.5, 0.0, imb, 0.0, 0.0])) for q in range(-Q, Q + 1)]
        rows.append(acts)
    return np.array(rows)


def quotes_vs_q(policy):
    qs = np.arange(-Q, Q + 1)
    bid, ask = [], []
    for q in qs:
        i, j = divmod(policy.greedy(np.array([q / Q, 0.5, 0.0, 0.0, 0.0, 0.0])), len(DEPTHS))
        db, da = DEPTHS[i], DEPTHS[j]
        bid.append(np.nan if db is None or q == Q else -0.5 - db)
        ask.append(np.nan if da is None or q == -Q else 0.5 + da)
    return qs, np.array(bid), np.array(ask)


def glft_vs_q(model):
    bids, asks = model.quotes(np.inf)
    qs = np.arange(-Q, Q + 1)
    bid = [np.nan if np.isnan(b) else min(np.floor(-b + 0.5), 0) - 0.5 for b in bids]
    ask = [np.nan if np.isnan(a) else max(np.ceil(a + 0.5), 1) - 0.5 for a in asks]
    return qs, np.array(bid), np.array(ask)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def chart(curves, refs, policy, model, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax in (left, right):
        style(ax)
    w = 20
    for k, c in enumerate(curves):
        sm = np.convolve(c, np.ones(w) / w, mode="valid")
        left.plot(
            np.arange(w, len(c) + 1), sm, color=BLUE, alpha=0.5 + 0.25 * (k == 0), linewidth=1.4
        )
    for (name, y), color in zip(refs.items(), (ORANGE, GREEN, MUTED), strict=True):
        left.axhline(y, color=color, linestyle="--", linewidth=1.2, label=name)
    left.legend(frameon=False, labelcolor=INK, loc="lower right")
    left.set_xlabel("PPO iteration (48 sessions each)", color=INK)
    left.set_ylabel("score per hour (ticks), 20-iteration mean", color=INK)
    left.set_title("PPO training runs, 3 seeds", color=INK, loc="left")

    qs, pb, pa = quotes_vs_q(policy)
    _, gb, ga = glft_vs_q(model)
    kw = dict(linestyle="none", markersize=11, markerfacecolor="none", markeredgewidth=1.6)
    right.plot(qs, gb, "s", color=ORANGE, label="tuned GLFT", **kw)
    right.plot(qs, ga, "s", color=ORANGE, **kw)
    right.plot(qs, pb, "o", color=BLUE, label="PPO seed 0", markersize=5)
    right.plot(qs, pa, "o", color=BLUE, markersize=5)
    right.axhline(0, color=MUTED, linewidth=0.8)
    right.set_xlabel("inventory q", color=INK)
    right.set_ylabel("quote minus mid (ticks), 1-tick spread", color=INK)
    right.set_title("quotes by inventory (missing: side not quoted)", color=INK, loc="left")
    right.set_xticks(qs)
    right.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        pmap = partial(pool.map, chunksize=4)
        agents, curves = [], []
        for s in SEEDS:
            agent, curve = train(make_env, 6, len(DEPTHS) ** 2, ITERS, CFG, seed=s, pmap=pmap)
            agents.append(agent)
            curves.append(curve)
            print(
                f"seed {s}: last 20 iterations {np.mean(curve[-20:]):.2f}, "
                f"{time.perf_counter() - t0:.0f} s"
            )
        (ROOT / "data").mkdir(exist_ok=True)
        for s, agent in zip(SEEDS, agents, strict=True):
            with open(ROOT / "data" / f"ppo_seed{s}.pkl", "wb") as f:
                pickle.dump(agent.policy, f)

        best_a, best_as, best_gl = tuned(pool)
        print(
            f"tuned on {len(TUNE)} sessions: fixed {label(best_a)}, AS gamma {best_as}, "
            f"GLFT gamma {best_gl[0]} keep {best_gl[1]}"
        )

        bots = {
            "join the touch": touch,
            f"best fixed ({label(best_a)})": partial(fixed, best_a),
            f"AS, gamma {best_as}": partial(avst, best_as),
            f"GLFT, gamma {best_gl[0]}": partial(glft, *best_gl),
        }
        for s, agent in zip(SEEDS, agents, strict=True):
            bots[f"PPO seed {s}"] = partial(learned, agent.policy)
        res = {name: evaluate(pool, bot, TEST) for name, bot in bots.items()}
        gl = res[f"GLFT, gamma {best_gl[0]}"][:, 0]
        print(
            f"\n{len(TEST)} held-out sessions of {MINUTES:.0f} min, "
            f"score = pnl - {PHI} * int q^2 dt - {TERM} * q_T^2"
        )
        print("bot                         score (se)    pnl     fills  int q^2   vs GLFT (se)")
        for name, r in res.items():
            d = r[:, 0] - gl
            n = len(r)
            print(
                f"{name:27s} {r[:, 0].mean():6.2f} ({r[:, 0].std(ddof=1) / n**0.5:4.2f})"
                f" {r[:, 1].mean():6.2f} {r[:, 2].mean():7.1f} {r[:, 3].mean():8.1f}"
                f"   {d.mean():+6.2f} ({d.std(ddof=1) / n**0.5:4.2f})"
            )

        print("\nmirrored policy: q, imbalance, move and flow negated, bid and ask swapped")
        for s, agent in zip(SEEDS, agents, strict=True):
            m = evaluate(pool, partial(learned, Mirror(agent.policy)), TEST)[:, 0]
            d = res[f"PPO seed {s}"][:, 0] - m
            print(
                f"seed {s}: mirror {m.mean():6.2f}, policy minus mirror {d.mean():+.2f}"
                f" ({d.std(ddof=1) / len(d) ** 0.5:.2f})"
            )

        print("\nPPO seed 0 action by inventory (columns q = -5..5), tau 0.5, 1-tick spread")
        for imb, row in zip((-0.6, 0.0, 0.6), policy_map(agents[0].policy), strict=True):
            print(f"imbalance {imb:+.1f}: " + " | ".join(label(a) for a in row))

        room = {
            "GLFT": partial(glft, *best_gl),
            "join the touch": touch,
            "AS": partial(avst, best_as),
            "PPO": partial(learned, agents[0].policy),
        }
        shared = list(pool.map(together, [(room, s) for s in TEST[:300]], chunksize=10))
        alone = list(pool.map(apart, [(room, s) for s in TEST[:300]], chunksize=10))
        print("\nall four in one market, 300 sessions")
        print("bot              score (se)   alone   pnl    fills")
        for st in standings(shared, alone):
            sc = np.array([score(r[st.name]) for r in shared])
            sa = np.array([score(r[st.name]) for r in alone])
            print(
                f"{st.name:15s} {sc.mean():6.2f} ({sc.std(ddof=1) / len(sc) ** 0.5:4.2f})"
                f" {sa.mean():7.2f} {st.mean:6.2f} {st.fills.mean():6.1f}"
            )

    refs = {
        "tuned GLFT, held out": gl.mean(),
        "best fixed depth": res[f"best fixed ({label(best_a)})"][:, 0].mean(),
        "join the touch": res["join the touch"][:, 0].mean(),
    }
    chart(
        curves,
        refs,
        agents[0].policy,
        Model(SIGMA, A, K, best_gl[0], Q),
        ROOT / "docs" / "rl_market_maker.png",
    )
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")
