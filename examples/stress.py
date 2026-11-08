import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from rl_market_maker import (
    CFG,
    INK,
    ITERS,
    MINUTES,
    MUTED,
    PHI,
    ROOT,
    SURFACE,
    TERM,
    Q,
    avst,
    fixed,
    glft,
    learned,
    style,
    touch,
    tuned,
)

from mmsim.env import DEPTHS, MarketMakingEnv
from mmsim.flow import CSTParams, RegimeFlow, ZeroIntelligence
from mmsim.informed import Fundamental, InformedTrader
from mmsim.ppo import train
from mmsim.sim import Simulator
from mmsim.tournament import session

SKY = CSTParams.sky_perfect()
TEST = range(600)


def calm():
    return ZeroIntelligence(SKY)


def spike():
    return RegimeFlow([(0.0, SKY), (50.0, SKY.scale(theta=2.0, mu=3.0)), (70.0, SKY)])


def thin():
    return RegimeFlow([(0.0, SKY), (60.0, SKY.scale(lam=0.5))])


def informed(seed):
    return [InformedTrader(Fundamental(1.0, seed + 1), 0.5, seed=seed)]


SCENARIOS = {
    "calm": (calm, None),
    "volatility spike": (spike, None),
    "thin book": (thin, None),
    "informed flow": (calm, informed),
}


def env_for(flow, crowd):
    return MarketMakingEnv(flow, phi=PHI, terminal=TERM, minutes=MINUTES, Q=Q, crowd=crowd)


class Mixture:
    def __init__(self):
        self.envs = [env_for(f, c) for f, c in SCENARIOS.values()]

    def reset(self, *, seed=None):
        self.env = self.envs[seed % len(self.envs)]
        return self.env.reset(seed=seed)

    def step(self, a):
        return self.env.step(a)


def score(args):
    bot, scenario, seed = args
    flow, crowd = SCENARIOS[scenario]
    r = session({"bot": bot}, seed, flow=flow, minutes=MINUTES, crowd=crowd)["bot"]
    return r.pnl - PHI * r.exposure - TERM * r.position**2, r.pnl, r.adverse


def volatility(args):
    scenario, seed = args
    flow, crowd = SCENARIOS[scenario]
    sim = Simulator(flow(), [] if crowd is None else crowd(seed), seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30.0)
    sim.run(MINUTES)
    mids = sim.tape.at(np.arange(30.0, 30.0 + MINUTES + 1))
    return np.diff(mids)


def chart(table, path):
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="x", visible=False)
    bots = list(table)
    colors = ["#9b9a96", "#c7a43f", "#8a5fbf", "#eb6834", "#f2a98a", "#2a78d6", "#2f9e6b"]
    w = 0.8 / len(bots)
    x = np.arange(len(SCENARIOS))
    for k, (name, color) in enumerate(zip(bots, colors, strict=True)):
        m = [table[name][s][0] for s in SCENARIOS]
        e = [table[name][s][1] for s in SCENARIOS]
        ax.bar(
            x + (k - (len(bots) - 1) / 2) * w,
            m,
            w,
            yerr=e,
            color=color,
            label=name,
            ecolor=INK,
            capsize=2,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, list(SCENARIOS), color=INK)
    ax.set_ylabel("score per hour (ticks), held-out sessions", color=INK)
    ax.legend(
        frameon=False, labelcolor=INK, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    with open(ROOT / "data" / "ppo_seed0.pkl", "rb") as f:
        calm_policy = pickle.load(f)
    with ProcessPoolExecutor() as pool:
        for s in SCENARIOS:
            d = np.concatenate(list(pool.map(volatility, [(s, i) for i in range(100)])))
            print(
                f"{s:17s} 1-min mid change sd {d.std():.2f}, over 3 ticks {np.mean(abs(d) > 3):.3f}"
            )

        mixed, curve = train(
            Mixture, 6, len(DEPTHS) ** 2, ITERS, CFG, seed=0, pmap=partial(pool.map, chunksize=4)
        )
        print(f"mixture PPO, last 20 iterations {np.mean(curve[-20:]):.2f}")
        with open(ROOT / "data" / "ppo_mixture.pkl", "wb") as f:
            pickle.dump(mixed.policy, f)

        a, g_as, (g_gl, keep) = tuned(pool)
        bots = {
            "join the touch": touch,
            "best fixed": partial(fixed, a),
            "AS": partial(avst, g_as),
            "GLFT": partial(glft, g_gl, keep),
            "GLFT, no improving": partial(glft, g_gl, keep, False),
            "PPO, calm": partial(learned, calm_policy),
            "PPO, mixture": partial(learned, mixed.policy),
        }
        table = {}
        raw = {}
        for name, bot in bots.items():
            table[name], raw[name] = {}, {}
            for s in SCENARIOS:
                r = np.array(list(pool.map(score, [(bot, s, i) for i in TEST], chunksize=25)))
                raw[name][s] = r
                table[name][s] = (r[:, 0].mean(), r[:, 0].std(ddof=1) / len(r) ** 0.5)

    print(f"\nscore per hour, {len(TEST)} held-out sessions per cell, se in brackets")
    print(f"{'bot':19s}" + "".join(f"{s:>22s}" for s in SCENARIOS))
    for name in bots:
        print(f"{name:19s}" + "".join(f"{m:14.2f} ({e:4.2f})" for m, e in table[name].values()))
    print("\nminus GLFT, paired")
    for name in bots:
        cells = []
        for s in SCENARIOS:
            d = raw[name][s][:, 0] - raw["GLFT"][s][:, 0]
            cells.append(f"{d.mean():+14.2f} ({d.std(ddof=1) / len(d) ** 0.5:4.2f})")
        print(f"{name:19s}" + "".join(cells))
    print("\nadverse term (ticks per hour)")
    for name in bots:
        print(f"{name:19s}" + "".join(f"{raw[name][s][:, 2].mean():21.2f}" for s in SCENARIOS))
    chart(table, ROOT / "docs" / "stress.png")
