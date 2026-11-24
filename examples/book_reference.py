import json
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from mmsim.book import Side
from mmsim.bots import GLFT, AvellanedaStoikov
from mmsim.env import sky
from mmsim.glft import Model
from mmsim.sim import Simulator
from mmsim.tournament import session

N = 2000
MINUTES = 60.0
SIGMA, A, K, Q = 1.0, 2.1, 3.6, 5


def glft(gamma):
    return GLFT(Model(SIGMA, A, K, gamma, Q), keep=1)


def avst(gamma):
    return AvellanedaStoikov(gamma, SIGMA, K, max_position=Q)


ENTRANTS = {
    "glft": {"glft": partial(glft, 0.08)},
    "as": {"as": partial(avst, 0.003)},
    "both": {"glft": partial(glft, 0.08), "as": partial(avst, 0.003)},
}


def flow_only(seed):
    sim = Simulator(sky(), seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    start = sim.tape.mid[-1]
    spread, depth = [], []
    for _ in range(int(MINUTES)):
        sim.warm_up(1)
        bid, ask = sim.book.best_bid(), sim.book.best_ask()
        if bid is not None and ask is not None:
            spread.append(ask - bid)
            depth.append(sim.book.level(Side.BUY, bid).qty)
    return sim.tape.mid[-1] - start, np.mean(spread), np.mean(depth)


def play(args):
    name, seed = args
    res = session(ENTRANTS[name], seed, flow=sky, minutes=MINUTES)
    return {k: (r.pnl, r.fills, r.position) for k, r in res.items()}


def summary(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "sd": float(xs.std())}


if __name__ == "__main__":
    sim = Simulator(sky())
    sim.seed(10_000, 10_001)
    seeded = {
        side.name.lower(): [[lvl.price, lvl.qty] for lvl in sim.book.levels(side)] for side in Side
    }

    models = []
    for sigma, a, k, gamma, q, mu, xi in [
        (1.0, 2.1, 3.6, 0.08, 5, 0.0, 0.0),
        (1.0, 2.1, 3.6, 0.003, 5, 0.0, 0.0),
        (0.3, 1.4, 2.0, 0.5, 8, 0.0, 0.0),
        (1.0, 2.1, 3.6, 0.08, 5, 0.02, 0.0),
        (1.0, 2.1, 3.6, 0.2, 4, 0.0, 0.4),
    ]:
        m = Model(sigma, a, k, gamma, q, mu, xi)
        bids, asks = m.quotes(math.inf)
        ab, aa = m.approx()
        models.append(
            {
                "params": {
                    "sigma": sigma,
                    "A": a,
                    "k": k,
                    "gamma": gamma,
                    "Q": q,
                    "mu": mu,
                    "xi": xi,
                },
                "bids": [None if math.isnan(x) else float(x) for x in bids],
                "asks": [None if math.isnan(x) else float(x) for x in asks],
                "approxBids": [float(x) for x in ab],
                "approxAsks": [float(x) for x in aa],
            }
        )

    with ProcessPoolExecutor() as pool:
        dmid, spread, depth = np.array(list(pool.map(flow_only, range(N), chunksize=50))).T
        flow = {
            "n": N,
            "dmid": summary(dmid),
            "spread": summary(spread),
            "depth": summary(depth),
        }
        print("flow", flow)
        games = {}
        for name in ENTRANTS:
            rows = list(pool.map(play, [(name, s) for s in range(N)], chunksize=50))
            games[name] = {
                bot: {
                    stat: summary([r[bot][j] for r in rows])
                    for j, stat in enumerate(("pnl", "fills", "position"))
                }
                for bot in ENTRANTS[name]
            }
            print(name, json.dumps(games[name]))

    out = Path(__file__).resolve().parent.parent / "web" / "test" / "book_reference.json"
    ref = {"seeded": seeded, "models": models, "flow": flow, "games": games, "n": N}
    out.write_text(json.dumps(ref, indent=1) + "\n")
