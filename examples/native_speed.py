import time

from mmsim.book import Book
from mmsim.bots import GLFT
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.glft import Model
from mmsim.native import NativeBook
from mmsim.sim import Simulator

SKY = CSTParams.sky_perfect()
MODEL = Model(1.0, 2.1, 3.6, 0.3, 5)

for name, book in (("python", Book), ("c++ engine", NativeBook)):
    best = float("inf")
    for seed in range(3):
        bot = GLFT(MODEL, keep=1)
        sim = Simulator(ZeroIntelligence(SKY), [bot], seed=seed, book=book)
        sim.seed(10_000, 10_001)
        sim.warm_up(30)
        t0 = time.perf_counter()
        sim.run(390)
        best = min(best, time.perf_counter() - t0)
        n = len(sim.contexts[0].account.fills)
    print(f"{name:10s} {best:6.2f} s per 390 min session, {n} fills")
