import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from book_reference import ENTRANTS, MINUTES, N, summary

from mmsim.book import Side
from mmsim.flow import CSTParams, HawkesFlow, ZeroIntelligence
from mmsim.hawkes import Hawkes
from mmsim.informed import Fundamental, InformedTrader
from mmsim.sim import Simulator

SKY = CSTParams.sky_perfect()
HAWKES = Hawkes.symmetric(SKY.mu, 0.6, 0.1, 6.0)
WORLDS = {
    "informed": (False, True),
    "hawkes": (True, False),
    "hawkes+informed": (True, True),
}
STATS = ("pnl", "fills", "position", "front", "markout")


def flow(hawkes):
    return HawkesFlow(SKY, HAWKES) if hawkes else ZeroIntelligence(SKY)


def hawkes_only(seed):
    f = flow(True)
    sim = Simulator(f, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    start, n0 = sim.tape.mid[-1], len(f.markets)
    sim.warm_up(MINUTES)
    return sim.tape.mid[-1] - start, len(f.markets) - n0


def play(args):
    world, game, seed = args
    hawkes, informed = WORLDS[world]
    names = list(ENTRANTS[game])
    random.Random(f"seats{seed}").shuffle(names)
    agents = [ENTRANTS[game][n]() for n in names]
    if informed:
        value = Fundamental(1.0, seed=2 * seed)
        agents.append(InformedTrader(value, 1.0, seed=2 * seed + 1))
    sim = Simulator(flow(hawkes), agents, seed=seed)
    sim.seed(10_000, 10_001)
    sim.warm_up(30)
    sim.run(MINUTES)
    close = float(sim.tape.at(sim.end))
    out = {}
    for name, ctx in zip(names, sim.contexts[: len(names)], strict=True):
        fills = ctx.account.fills
        mark = sum(
            (1 if f.side == Side.BUY else -1) * f.qty * (float(sim.tape.at(f.t + 1)) - f.price)
            for f in fills
        )
        front = sum(f.ahead == 0 for f in fills)
        out[name] = (ctx.account.value(close), len(fills), ctx.account.position, front, mark)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        dmid, markets = np.array(list(pool.map(hawkes_only, range(N), chunksize=50))).T
        hawkes = {"dmid": summary(dmid), "markets": summary(markets)}
        print("hawkes flow", hawkes)
        games = {}
        for world in WORLDS:
            for game in ("glft", "both"):
                jobs = [(world, game, s) for s in range(N)]
                rows = list(pool.map(play, jobs, chunksize=50))
                games[f"{world}/{game}"] = {
                    bot: {stat: summary([r[bot][j] for r in rows]) for j, stat in enumerate(STATS)}
                    for bot in ENTRANTS[game]
                }
                print(world, game, json.dumps(games[f"{world}/{game}"]))

    out = Path(__file__).resolve().parent.parent / "web" / "test" / "queue_reference.json"
    ref = {"n": N, "hawkes": hawkes, "games": games}
    out.write_text(json.dumps(ref, indent=1) + "\n")
