from concurrent.futures import ProcessPoolExecutor

import numpy as np
from itch_replay import SESSION, STARTS, SYMBOLS, bot, msgs, result

from mmsim.replay import ItchFlow, ReplayBook
from mmsim.sim import Latency, Simulator

US = 1 / 60_000_000
DELAYS = (0, 100, 1000, 10_000)
REBATES = (0.18, 0.305)


def session(args):
    symbol, us, start = args
    b = bot("join the touch", 100, 1000)
    lag = Latency(feed=us * US, entry=us * US) if us else None
    sim = Simulator(
        ItchFlow(msgs(symbol)), [b], book=ReplayBook, latency={b: lag} if lag else None
    )
    sim.warm_up(start)
    sim.run(SESSION)
    return result(sim, 1)


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        jobs = {
            (symbol, us): pool.map(session, [(symbol, us, s) for s in STARTS])
            for symbol in SYMBOLS
            for us in DELAYS
        }
        print(
            "symbol  one-way us   gross (se)       spread  adverse   shares  $/session"
            "   net low  net high  $ low  $ high"
        )
        for (symbol, us), rows in jobs.items():
            rows = np.array(list(rows))
            pnl, vol = rows[:, 0], rows[:, 1]
            per = rows[:, :5].sum(axis=0) / vol.sum()
            se = (pnl - per[0] * vol).std(ddof=1) * np.sqrt(len(rows)) / vol.sum()
            net = [per[0] + r for r in REBATES]
            dollars = [n * vol.mean() / 100 for n in net]
            print(
                f"{symbol:7s} {us:10d} {per[0]:7.3f} ({se:5.3f}) {per[2]:8.3f} {per[3]:8.3f}"
                f" {vol.mean():8.0f} {pnl.mean() / 100:10.1f} {net[0]:9.3f} {net[1]:9.3f}"
                f" {dollars[0]:6.1f} {dollars[1]:7.1f}"
            )
