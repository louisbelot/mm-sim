import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mmsim import arl
from mmsim import robust as r
from mmsim.ppo import Config

ROUNDS, ITERS = 30, 20
CFG = Config(lr=3e-3, episodes=32)
OUT = Path(__file__).resolve().parent.parent / "data"


def job(args):
    controls, seed = args
    game = r.Game()
    t = time.time()
    rounds, iters = (1, ROUNDS * ITERS) if not controls else (ROUNDS, ITERS)
    mm, adv = arl.rarl(game, rounds, controls, iters, iters, CFG, seed, start=10)
    bid, ask = arl.quotes(mm.policy, game)
    out = {"bid": bid, "ask": ask, "minutes": (time.time() - t) / 60}
    if adv is not None:
        out["adv"] = arl.market(adv.policy, game, controls)
    return controls, seed, out


if __name__ == "__main__":
    jobs = [(c, s) for c in ("", "b", "bAk") for s in range(3)]
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(job, jobs))

    game = r.Game()
    print(f"PPO, {ROUNDS * ITERS} market-maker iterations of {CFG.episodes} episodes each")
    print(
        f"{'trained against':16s} {'seed':>4s} {'wealth':>13s} {'sharpe':>7s} {'sd q':>5s}"
        f" {'spread':>6s} {'exact':>6s} {'worst b':>8s} {'worst':>6s} {'adv gap':>7s} {'min':>5s}"
    )
    for controls, seed, out in results:
        bid, ask = out["bid"], out["ask"]
        np.savez(OUT / f"arl_{controls or 'fixed'}_{seed}.npz", bid=bid, ask=ask)
        run = r.simulate(game, bid, ask, paths=100_000, seed=1)
        w = run.wealth
        exact = r.evaluate(game, bid, ask)[0, game.Q]
        worst_b = r.worst_case(game, bid, ask, "b").V[0, game.Q]
        worst = r.worst_case(game, bid, ask).V[0, game.Q]
        gap = ""
        if "adv" in out:
            value = r.equilibrium(game, controls).V[0, game.Q]
            gap = f"{r.best_response(game, out['adv']).V[0, game.Q] - value:7.2f}"
        print(
            f"{controls or 'fixed':16s} {seed:4d} {w.mean():6.1f} +- {w.std():4.1f}"
            f" {w.mean() / w.std():7.2f} {run.q.std():5.2f} {np.nanmean(run.spread):6.3f}"
            f" {exact:6.2f} {worst_b:8.2f} {worst:6.2f} {gap:>7s} {out['minutes']:5.1f}"
        )
