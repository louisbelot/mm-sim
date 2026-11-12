from pathlib import Path

import numpy as np

from mmsim import irl
from mmsim import robust as r

game = r.Game()
DATA = Path(__file__).resolve().parent.parent / "data"

bots = [
    ("planted zeta .004 nu .01 eta .3", r.best_response(r.Game(zeta=0.004, nu=0.01, eta=0.3))),
    ("fixed zeta 0.01", r.best_response(r.Game(zeta=0.01))),
    ("AS gamma 0.1", r.avellaneda(game, 0.1)),
    ("strategic b", r.equilibrium(game, "b")),
    ("strategic bAk", r.equilibrium(game, "bAk")),
]
bots = [(name, (b.bid, b.ask) if isinstance(b, r.Solution) else b) for name, b in bots]
for f in sorted(DATA.glob("arl_*.npz")):
    t = np.load(f)
    bots.append((f"PPO {f.stem[4:]}", (t["bid"], t["ask"])))

fits = (("zeta",), ("zeta", "nu"), ("zeta", "nu", "eta"), ("zeta", "nu", "A", "k"))
print("penalties fitted in the fixed market; rms is the quote error at visited states")
print(
    f"{'bot':32s} {'fit':18s} {'zeta':>8s} {'nu':>7s} {'eta':>6s} {'A':>6s} {'k':>6s} {'rms':>7s}"
)
for name, (bid, ask) in bots:
    visits = r.simulate(game, bid, ask, paths=5000, seed=3).visits
    for names in fits:
        f = irl.match(game, bid, ask, visits, names)
        p = f.params
        cell = [
            f"{p[k]:{w}.{d}f}" if k in p else " " * w
            for k, w, d in (("zeta", 8, 5), ("nu", 7, 4), ("eta", 6, 3), ("A", 6, 1), ("k", 6, 3))
        ]
        print(f"{name:32s} {' '.join(names):18s} {' '.join(cell)} {np.sqrt(f.loss):7.4f}")
