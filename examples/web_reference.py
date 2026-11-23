import json
from pathlib import Path

from mmsim import avellaneda as av

N = 20_000

quotes = []
for s, q, tau, gamma, sigma, k in [
    (100.0, 0, 1.0, 0.1, 2.0, 1.5),
    (100.0, 3, 0.5, 0.1, 2.0, 1.5),
    (101.3, -4, 0.25, 0.01, 2.0, 1.5),
    (99.0, 7, 0.9, 1.0, 2.0, 1.5),
    (100.0, -2, 0.0, 0.5, 1.0, 3.0),
]:
    bid, ask = av.quotes(s, q, tau, gamma, sigma, k)
    quotes.append(
        {
            "s": s,
            "q": q,
            "tau": tau,
            "gamma": gamma,
            "sigma": sigma,
            "k": k,
            "r": float(av.reservation_price(s, q, tau, gamma, sigma)),
            "spread": float(av.spread(tau, gamma, sigma, k)),
            "bid": float(bid),
            "ask": float(ask),
        }
    )

sims = []
for gamma in (0.1, 0.01, 1.0):
    for strategy in ("inventory", "symmetric"):
        r = av.simulate(gamma, strategy, n_paths=N, seed=1)
        sims.append(
            {
                "gamma": gamma,
                "strategy": strategy,
                "n": N,
                "spread": r.spread,
                "profit": float(r.profit.mean()),
                "sdProfit": float(r.profit.std()),
                "q": float(r.final_q.mean()),
                "sdQ": float(r.final_q.std()),
            }
        )
        print(gamma, strategy, round(sims[-1]["profit"], 2), round(sims[-1]["sdProfit"], 2))

out = Path(__file__).resolve().parent.parent / "web" / "test" / "reference.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"quotes": quotes, "sims": sims}, indent=1) + "\n")
