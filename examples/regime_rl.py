import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from mmsim import regime as g
from mmsim.ppo import Config, train

ITERS = 600
CFG = Config(lr=3e-3, episodes=32)
WORLD = g.World(phi=0.01)
OUT = Path(__file__).resolve().parent.parent / "data"


def job(args):
    belief, seed = args
    t = time.time()
    agent, curve = train(partial(g.Env, WORLD, belief), 2 + belief, 81, ITERS, CFG, seed)
    return belief, seed, agent.policy, curve, (time.time() - t) / 60


def report(name, run):
    rew = run.reward
    print(
        f"{name:22s} {rew.mean():6.2f} +- {rew.std():5.2f} {rew.mean() / rew.std():6.2f}"
        f" {np.nanmean(run.spread):6.3f} {np.abs(run.q).mean():5.2f}"
    )


if __name__ == "__main__":
    w = WORLD
    jobs = [(b, s) for b in (True, False) for s in range(3)]
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(job, jobs))

    plan, _ = g.bayes(w)
    blind = g.blind(w)
    print(f"PPO, {ITERS} iterations of {CFG.episodes} episodes, 100,000 test paths each")
    print(f"{'quoter':22s} {'reward':>15s} {'sharpe':>6s} {'spread':>6s} {'|q|':>5s}")
    report("bayes", g.simulate(w, g.believer(plan), 100_000, 1))
    report("blind", g.simulate(w, g.table(blind.bid, blind.ask), 100_000, 1))
    report("avellaneda 0.02", g.simulate(w, g.table(*g.avellaneda(w, 0.02)), 100_000, 1))
    curves = {}
    for belief, seed, policy, curve, minutes in results:
        name = f"ppo {'belief' if belief else 'no belief'} {seed} ({minutes:.0f} min)"
        report(name, g.simulate(w, g.learned(w, policy, belief), 100_000, 1))
        curves[f"{int(belief)}_{seed}"] = np.array(curve)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "regime_rl_curves.npz", **curves)
