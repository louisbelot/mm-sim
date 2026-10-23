# mm-sim

A market-making research lab in Python. It has a limit order book with
price-time priority, background order flow from the Cont-Stoikov-Talreja model,
an event-driven simulator where bots quote against that flow, the
Avellaneda-Stoikov and Gueant-Lehalle-Fernandez-Tapia inventory models (each
checked against its paper), informed traders with a private view of the
fundamental value, markout analysis to measure what they cost a market maker,
the Glosten-Milgrom and Kyle models of why spreads and price impact exist, and
queue-position tracking with a measurement of what priority is worth. The C++ [orderbook](https://github.com/louisbelot/orderbook) engine and
real NASDAQ data come next.

![Avellaneda-Stoikov replication](docs/as_replication.png)

## Avellaneda-Stoikov, rerun

Avellaneda and Stoikov (2008) quote around a reservation price
r = s - q gamma sigma^2 (T - t) that leans against the inventory q, with a total
spread of gamma sigma^2 (T - t) + (2 / gamma) ln(1 + gamma / k). Section 3.3 of
the paper compares this with a symmetric strategy that quotes the same average
spread around the mid. `avellaneda.simulate` reruns that experiment with the
paper's parameters (s = 100, T = 1, sigma = 2, dt = 0.005, k = 1.5, A = 140).
10,000 paths per row here, 1000 in the paper, paper values in brackets:

| gamma | strategy  | spread      | mean profit | sd profit   | mean final q   | sd final q |
|-------|-----------|-------------|-------------|-------------|----------------|------------|
| 0.1   | inventory | 1.49 (1.49) | 64.9 (65.0) | 6.6 (6.6)   | 0.01 (0.08)    | 2.9 (2.9)  |
| 0.1   | symmetric | 1.49 (1.49) | 68.2 (68.4) | 13.3 (12.7) | -0.03 (0.26)   | 8.4 (8.4)  |
| 0.01  | inventory | 1.35 (1.35) | 68.4 (68.6) | 8.9 (8.7)   | -0.01 (0.12)   | 5.2 (5.1)  |
| 0.01  | symmetric | 1.35 (1.35) | 68.6 (68.8) | 13.5 (12.8) | -0.10 (0.09)   | 8.7 (8.7)  |
| 1     | inventory | 3.02 (3.02) | 31.4 (31.4) | 4.9 (5.0)   | 0.01 (0.02)    | 1.7 (1.7)  |
| 1     | symmetric | 3.02 (3.02) | 43.8 (44.0) | 11.0 (11.0) | -0.03 (0.00)   | 5.2 (5.1)  |

At gamma = 0.1 the inventory strategy gives up 3.3 of mean profit and halves the
standard deviation. At gamma = 1 it gives up 12.4 and the standard deviation goes
from 11.0 to 4.9. At gamma = 0.01 the trader is close to risk neutral: mean
profits are within 0.2 of each other and the gap in risk narrows, as the paper
says it should. The tests check every
row against the published one, with a tolerance that covers the sampling error of
both runs.

## The same idea on an order book

The paper's fills arrive at a rate A exp(-k delta) that only depends on the
distance from the mid. On a real book a quote fills when the queue in front of it
clears. In this simulator, stepping back one tick from the best quote cuts the
fill rate from 0.348 to 0.009 lots per minute, because the order now waits behind
a whole queue and loses its place every time the bot moves it. The chart below
puts a bot that joins the best bid and ask next to the Avellaneda-Stoikov bot,
with sigma = 1 tick per sqrt(minute) and k = 3.6 per tick, both measured on this
flow. Each point is 200 sessions of 390 minutes.

![risk and return of the quoting bots](docs/cst_sessions.png)

| strategy         | mean P&L (se) | sd P&L | mean / sd | sd final q | mean abs q | fills |
|------------------|---------------|--------|-----------|------------|------------|-------|
| join the touch   | 128.9 (10.9)  | 153.9  | 0.84      | 13.8       | 7.80       | 274   |
| AS, gamma 0.0003 | 106.5 (3.4)   | 48.0   | 2.22      | 7.3        | 2.48       | 253   |
| AS, gamma 0.001  | 81.4 (1.8)    | 25.9   | 3.14      | 4.5        | 1.08       | 220   |
| AS, gamma 0.003  | 39.7 (1.1)    | 15.8   | 2.51      | 3.7        | 0.45       | 128   |
| AS, gamma 0.01   | 13.2 (0.6)    | 8.7    | 1.52      | 2.5        | 0.14       | 47    |

P&L is in ticks per session, marked at the mid of everyone else's quotes, with
no fees. Joining the touch makes the most money and ends the day with an
inventory whose standard deviation is 14 lots. At gamma = 0.001 the AS bot keeps
63% of that mean with a sixth of the risk. Past that point its quotes sit far
enough back that it barely trades. This flow has no informed traders, so the
only adverse selection is mechanical (a fill often means the queue just
emptied and the price is about to move through it). The informed traders
below change that.

## Gueant-Lehalle-Fernandez-Tapia, solved exactly

Avellaneda-Stoikov leans on a horizon: its skew is gamma sigma^2 (T - t) per lot,
so at the open of a long session it is huge and at the close it is zero. Gueant,
Lehalle and Fernandez-Tapia (2013) solve the same problem with a hard inventory
bound |q| <= Q. The substitution u = -exp(-gamma (x + q s)) v_q(t)^(-gamma/k)
turns the HJB equation into a linear system v' = M v, where M is tridiagonal with
alpha q^2 on the diagonal and -eta off it (alpha = k gamma sigma^2 / 2,
eta = A (1 + gamma/k)^-(1 + k/gamma)). So v(t) = exp(-M (T - t)) 1 and the
quotes are

```
delta_bid(q) = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k)
delta_ask(q) = (1/k) ln(v_q / v_{q-1}) + (1/gamma) ln(1 + gamma/k)
```

with no bid at q = Q and no ask at q = -Q. `glft.Model` computes this for any t,
the long-horizon limit (the ground state of M), the paper's closed-form
approximation, and its two extensions: a drift mu in the mid and a price impact xi
per fill, which is the paper's model of adverse selection.

![GLFT quotes](docs/glft_quotes.png)

This is the paper's example (sigma = 0.3 tick per sqrt(s), A = 0.9 per s, k = 0.3
per tick, gamma = 0.01 per tick, T = 600 s). Far from the end the quotes barely
depend on t (at t = 0 they are within 4e-7 ticks of the infinite-horizon limit), and the stationary spread is 6.6261 ticks against 6.626 from
the closed form. The closed form is within 0.002 ticks for |q| <= 10 and drifts
off near the bound: at q = 29 the exact bid sits 6.13 ticks below the mid, the
closed form says 5.28. For comparison, Avellaneda-Stoikov at t = 0 with q = 20
would move its reservation price by 10.8 ticks; GLFT moves its bid by 1.35.

The checks: the quotes match a direct solve of the nonlinear HJB system (Radau,
in log space) to 1e-6 ticks, with and without drift and impact. Near the end
they reduce to the Avellaneda-Stoikov expansion, the paper's footnote 8. Taking
the plotted window [-30, 30] out of a bound of Q = 50 reproduces the axis ranges
of the paper's figures 1 to 3. And an exact Monte Carlo (Poisson fills by
thinning, the Brownian part integrated out analytically, Q = 4 and T = 50) earns
a certainty equivalent of 63.93 +- 0.08 ticks against the value function's
63.90, while moving every quote 0.3 ticks in or out loses about 0.8.

## Informed traders and markouts

`informed.py` adds a fundamental value V that follows a Brownian motion nobody
else can see, and informed traders who wake up at Poisson times and send a
market order whenever V is beyond the best quote. They are the only link between
V and the book, so the mid follows V with a lag, and every informed trade is
profitable at the moment it happens. Fills now record who the counterparty was,
and the simulator keeps a tape of every change in the mid, so `markout.py` can
measure, for each fill, how far the mid has moved for or against it h minutes
later.

![markouts and P&L with informed flow](docs/informed_flow.png)

One informed trader at 1 order per minute, V moving 1 tick per sqrt(minute),
200 sessions of 390 minutes per row, every bot capped at 5 lots:

| strategy           | informed flow | mean P&L (se) | sd P&L | fills |
|--------------------|---------------|---------------|--------|-------|
| join the touch     | no            | 86.9 (4.0)    | 56.3   | 252   |
| join the touch     | yes           | -73.2 (5.1)   | 72.6   | 336   |
| AS, gamma 0.001    | no            | 74.7 (2.0)    | 28.7   | 218   |
| AS, gamma 0.001    | yes           | 8.7 (2.8)     | 40.0   | 295   |
| GLFT, gamma 0.3    | no            | 62.5 (1.3)    | 17.9   | 202   |
| GLFT, gamma 0.3    | yes           | 32.6 (1.8)    | 25.7   | 285   |
| GLFT, gamma 1      | no            | 18.3 (0.8)    | 10.6   | 83    |
| GLFT, gamma 1      | yes           | 13.5 (1.1)    | 15.6   | 159   |
| GLFT, gamma 1, xi  | no            | 4.1 (0.3)     | 4.4    | 13    |
| GLFT, gamma 1, xi  | yes           | 5.3 (0.6)     | 8.7    | 36    |

The markouts tell the story. Against noise flow the touch quoter earns its half
spread and keeps it. Against informed flow it earns the same 0.54 ticks at the
instant of the fill, is underwater after 2.5 minutes and is down 2.96 ticks per
lot after 30. Three fills in ten are informed, and that is enough to turn +87
ticks a session into -73.

The noise curve rising to +1.01 is a side effect of the position cap: once
informed buying has pushed the quoter to -5 lots it only shows a bid, so its
remaining noise fills are buys made while the price is still catching up with V.
Without the cap the curve stays flat (0.60 at the fill, 0.54 after 30 minutes,
over 40 sessions).

Inventory control turns out to be the defence. Informed flow is one-sided, so it
builds inventory, and a bot that leans against its inventory stops feeding it.
GLFT with gamma = 0.3 keeps more than half its P&L. The bot's sigma, k and A were
measured on noise flow only, so it does not know the informed traders exist.

The paper's own answer to adverse selection is xi, a permanent price impact per
fill. It widens both quotes by xi / 2 and makes the inventory penalty steeper.
Measured from the touch quoter's markouts (fill to 30 minutes, all
counterparties) it is 0.72 ticks, which moves the flat-inventory quotes from 0.57
to 1.24 ticks off the mid. The spread on this book is one tick 74% of the time,
so the bot now sits a tick behind the best price whenever it is flat, where the
fill rate is about 40 times lower. It drops from 83 fills a session to 13 and
makes 4 to 5 ticks with or without the informed traders. On a tick grid, leaning
on inventory did far more than widening.

## Why spreads exist: Glosten-Milgrom and Kyle

The models above get their spread from inventory risk and an assumed fill-rate
curve. The two classic information models get one from a risk-neutral market
maker that holds no inventory risk at all.

In Glosten and Milgrom (1985) a competitive market maker meets one trader at a
time. With probability alpha the trader is informed; otherwise the trader buys
or sells with probability eta each, whatever the price. The market maker sets
the ask to E[V | the next trader buys] and the bid to E[V | the next trader
sells], so it breaks even on average: what it loses to insiders it gets back
from everyone else. With V either lo or hi and P(V = hi) = mu,

```
ask = lo + (hi - lo) mu (alpha + u) / (mu alpha + u)      u = (1 - alpha) eta
bid = lo + (hi - lo) mu u / ((1 - mu) alpha + u)
```

and with an even prior the spread is exactly alpha (hi - lo). Each trade updates
the belief, and only the net number of buys matters: with eta = 1/2 every buy
multiplies the odds by (1 + alpha) / (1 - alpha).

`glosten.py` also solves a version on a grid of values, with insiders who see V
plus Gaussian noise and stay out when their estimate is inside the quotes (close
to Das 2005). The quotes are then fixed points, ask = E[V | a buy at that ask],
found by root-finding, and a no-trade is information too. With a two-point grid
and noise near zero it reproduces the closed form to 1e-9.

![Glosten-Milgrom and Kyle](docs/why_spreads.png)

`examples/why_spreads.py` draws V from a normal prior (sd 10; insiders' noise sd
5, eta 0.4) and runs 200 sessions of 150 trades for each alpha:

| informed | first spread | spread after 150 trades | market maker P&L per session (se) | insiders' P&L per session |
|----------|--------------|-------------------------|-----------------------------------|---------------------------|
| 10%      | 1.75         | 0.95                    | -5.2 (7.5)                        | 81.6                      |
| 30%      | 5.18         | 0.38                    | -2.8 (3.7)                        | 58.3                      |
| 50%      | 8.84         | 0.18                    | -1.9 (2.5)                        | 32.9                      |

More insiders make a wider spread at the open and a faster collapse, because
the price learns faster. As a group the insiders earn the most when they are
rare. The market maker's P&L is zero within its standard error in every row, so
the uninformed pay for all of it.

Kyle (1985) has one insider who knows V ~ N(p0, Sigma0), noise traders who send
N(0, sigma_u^2 dt) to each of N auctions, and market makers who set the price to
E[V | order flow so far]. In equilibrium the insider trades beta_n (V - p) dt and
the price moves lambda_n per unit of net order flow. With one auction
lambda = sqrt(Sigma0) / (2 sigma_u) and the insider makes sqrt(Sigma0) sigma_u / 2.
For N auctions `kyle.solve` runs Kyle's backward recursion, where each step
solves a cubic for lambda_n on (0, 1 / (2 alpha_n)) given the next period's
value-function coefficient alpha_n. The recursion is homogeneous in Sigma, so it
runs once with Sigma_N = 1 and again rescaled to hit Sigma0. With
Sigma0 = sigma_u = 1:

| N    | lambda, first auction | lambda, last auction | Sigma_N | insider profit |
|------|-----------------------|----------------------|---------|----------------|
| 1    | 0.5000                | 0.5000               | 0.5000  | 0.5000         |
| 4    | 0.7648                | 0.6642               | 0.2206  | 0.7307         |
| 16   | 0.9012                | 0.7707               | 0.0742  | 0.8860         |
| 256  | 0.9881                | 0.8437               | 0.0056  | 0.9869         |
| 4096 | 0.9989                | 0.8530               | 0.0004  | 0.9988         |

As N grows the solution converges to continuous time: lambda constant at 1,
Sigma falling linearly to 0, and a profit of 1, twice the one-shot value. The
insider trades just slowly enough that the price only reaches V at the close.
The last auction stays different (lambda about 0.85 of the continuous value)
because there is no later auction to save information for.

The checks do not reuse the recursion. lambda_n equals the Kalman gain of the
market makers' filter given beta_n, to 1e-10. The value function
alpha_0 Sigma0 + delta_0 equals the insider's expected profit computed forward
from the strategy. Moving any single beta_n by 5% either way lowers that profit,
so the strategy is a best response. A Monte Carlo of 200,000 paths matches
Sigma_n to 2% and the profit within 4 standard errors, and its price changes are
uncorrelated with each other and with the remaining pricing error.

## Queue position

The simulator records where each order joined its queue. `LiveOrder.ahead` is
the quantity in front of the order when it joined its level, and
`ctx.ahead(oid)` gives the live value. A modify keeps the old number only when
it shrinks the order in place, the same rule the book uses for time priority.
Maker fills carry `ahead` and `left`, the quantity still resting at the fill
price once the incoming order is done. `left == 0` means the fill emptied the
level and the price went through it.

![queue position](docs/queue_value.png)

The left panel is exact. In the CST model with a one-tick spread, the
probability that an order at the best bid fills before the ask queue empties
falls quickly with the lots in front of it: against two lots at the ask it is
0.70 at the front, 0.47 two lots back and 0.25 ten lots back. It is the same
linear system as `prob_fill_first`, which a simulator test already checks.

In the simulator (200 sessions of 390 minutes per row, the informed traders from
the previous section), the fills that emptied their level are the bad ones. For
the GLFT bot the markout at the fill is 0.36 ticks when the fill took out the
level and 0.57 when it left a queue behind, with or without informed flow. Most
of those fills come from orders that stepped alone into a new price level
(`ahead == 0`); those get hit first and the level empties 67 to 73% of the
time. Position further back in the queue makes little difference to the longer
markouts compared to the noise in these runs.

That points to a policy. `Quoter` takes `keep`: a live quote stays put when it
sits at most `keep` ticks behind the new target, so the bot holds its place
instead of chasing every one-tick move. A quote never stays ahead of its target,
so the bot never quotes more aggressively than its model asks.

| strategy        | informed flow | keep | mean P&L (se) | sd P&L | fills | markout at fill |
|-----------------|---------------|------|---------------|--------|-------|-----------------|
| join the touch  | no            | 0    | 86.9 (4.0)    | 56.3   | 252   | 0.57            |
| join the touch  | no            | 1    | 85.8 (4.0)    | 56.0   | 184   | 0.69            |
| join the touch  | no            | 2    | 75.8 (4.4)    | 61.6   | 143   | 0.75            |
| join the touch  | yes           | 0    | -73.2 (5.1)   | 72.6   | 336   | 0.57            |
| join the touch  | yes           | 1    | -44.9 (5.3)   | 74.4   | 256   | 0.73            |
| join the touch  | yes           | 2    | -31.1 (5.2)   | 73.7   | 204   | 0.81            |
| GLFT, gamma 0.3 | no            | 0    | 62.5 (1.3)    | 17.9   | 202   | 0.52            |
| GLFT, gamma 0.3 | no            | 1    | 67.0 (1.3)    | 18.9   | 150   | 0.68            |
| GLFT, gamma 0.3 | no            | 2    | 57.3 (1.3)    | 18.6   | 111   | 0.77            |
| GLFT, gamma 0.3 | yes           | 0    | 32.6 (1.8)    | 25.7   | 285   | 0.52            |
| GLFT, gamma 0.3 | yes           | 1    | 58.2 (1.9)    | 26.9   | 221   | 0.73            |
| GLFT, gamma 0.3 | yes           | 2    | 51.0 (2.2)    | 30.5   | 172   | 0.84            |

The keep = 0 rows reproduce the informed-trader table exactly, rerun after the
queue changes. Against informed flow, keep = 1 takes GLFT from 32.6 to 58.2
ticks a session. It trades less, each fill is worth about 0.2 ticks more at the
instant it happens, and it takes about a quarter fewer fills from the informed
trader. Without informed flow it is close to neutral (+4.5 for GLFT, -1.1 for
the touch quoter), and at keep = 2 the bot starts missing fills that were worth
having.

## Install

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # .venv/bin/pip on Linux and macOS
.venv/Scripts/python -m pytest
```

Requires Python 3.10+, numpy and scipy. matplotlib is only needed for the
examples.

## Usage

```python
from mmsim.bots import AvellanedaStoikov
from mmsim.flow import CSTParams, ZeroIntelligence
from mmsim.sim import Simulator

bot = AvellanedaStoikov(gamma=0.001, sigma=1.0, k=3.6)
sim = Simulator(ZeroIntelligence(CSTParams.sky_perfect()), [bot], seed=1)
sim.seed(10_000, 10_001)  # best bid and ask, in ticks
sim.warm_up(30)           # minutes of background flow before the bot starts
sim.run(390)
ctx = sim.contexts[0]
ctx.pnl(), ctx.position, len(ctx.account.fills)  # (27.0, 2, 208)
```

A bot is a subclass of `Agent` with any of `on_start`, `on_book` (after every
background event), `on_fill` and `on_timer`. Its `ctx` gives the clock, the book,
everyone else's best bid, ask and mid, its own open orders and account, and
`buy`, `sell`, `cancel`, `modify` and `set_timer`.

```python
from mmsim import avellaneda as av
from mmsim.flow import prob_mid_up

run = av.simulate(0.1, "inventory", n_paths=10_000, seed=1)
run.profit.mean(), run.profit.std()               # (64.87, 6.58)
bid, ask = av.quotes(100.0, 3, 0.5, 0.1, 2.0, 1.5)   # 98.65, 100.15: long, so both lower
prob_mid_up(2, 1, lam=1.85, mu=0.94, theta=0.71)   # 0.335; CST Table 3 says 0.336
```

```python
from mmsim.bots import GLFT
from mmsim.glft import Model
from mmsim.informed import Fundamental, InformedTrader
from mmsim.markout import curve, markouts

m = Model(sigma=0.3, A=0.9, k=0.3, gamma=0.01, Q=30)
bid, ask = m.quotes(float("inf"))   # mid minus bid and ask minus mid, q = -30..30
bid[30] + ask[30]                   # 6.626, the spread at q = 0

value = Fundamental(sigma=1.0, seed=1)
bot = GLFT(Model(sigma=1.0, A=2.1, k=3.6, gamma=0.3, Q=5))
flow = ZeroIntelligence(CSTParams.sky_perfect())
sim = Simulator(flow, [bot, InformedTrader(value, rate=1.0, seed=1)], seed=1)
sim.seed(10_000, 10_001)
sim.warm_up(30)
sim.run(390)
fills = sim.contexts[0].account.fills
curve(fills, markouts(fills, sim.tape, [0, 1, 5, 30], sim.end))
# mean markout per lot 0.50, 0.42, 0.23, 0.11 over 271 fills
```

```python
import numpy as np

from mmsim import glosten as gm
from mmsim import kyle

gm.binary_quotes(0.5, 0.3, 90.0, 110.0)   # (97.0, 103.0): the spread is alpha (hi - lo)
v = np.linspace(60, 140, 321)
prior = np.exp(-0.5 * ((v - 100) / 10) ** 2)
prior /= prior.sum()
gm.quotes(gm.Model(0.3, noise=5.0, eta=0.4), v, prior)   # (97.41, 102.59)

eq = kyle.solve(16, Sigma0=1.0, sigma_u=1.0)
eq.lam[0], eq.Sigma[-1], eq.profit                     # (0.901, 0.074, 0.886)
```

`python examples/as_replication.py`, `cst_sessions.py`, `glft_quotes.py`,
`informed_flow.py`, `why_spreads.py` and `queue_value.py` redraw the six charts.
`informed_flow.py` and `queue_value.py` run 2000 and 2400 sessions and take
90 to 100 seconds each here.

## Notes

**Matching engine.** `book.py` ports the rules of the C++ engine: limit and
market orders, IOC, FOK, post-only, and modify, which keeps time priority only
when an order shrinks at the same price. It emits the same event stream, so the
C++ engine can replace it later without touching the bots. Levels are keyed by
sign times price, with bids negated, so the best level of either side is the
smallest key and one matching loop serves both. A differential test runs 30 seeds
of 2000 random commands against a deliberately naive reference book (one flat
list, sorted on every take) and compares every event and every resting order.
Six hand-injected bugs each failed between 22 and 30 of the seeds.

**Order flow.** In Cont, Stoikov and Talreja (2010) every limit order, market
order and cancel is a Poisson arrival with a rate that depends only on the
distance from the opposite best quote, so the whole book is a continuous-time
Markov chain. `ZeroIntelligence.step` simulates it exactly: the waiting time is
exponential in the total rate and the event is drawn in proportion to its own
rate. When a bot's timer comes first, the draw is thrown away and redrawn later,
which is exact because the chain is memoryless. Parameters are the paper's
estimates for Sky Perfect Communications on the Tokyo Stock Exchange (rates per
minute).

The check against the paper: with a one-tick spread, the probability that the
mid moves up next is a race between two independent birth-death queues to empty
first. `prob_mid_up` solves it as a sparse linear system with one equation per
pair of queue sizes, instead of the paper's Laplace inversion. All 25 entries of
the paper's Table 3 match within 0.001, and simulated races through the real
book land within sampling error of the solve. Run for 20,000 minutes, the flow
has a one-tick spread 74% of the time and an average depth profile that peaks two
ticks from the opposite quote, the hump the paper describes.

**Simulator.** Every change to the book comes back as the book's own events, and
those update accounts, open orders and fills. Bots see `ctx.mid`, the mid of
everyone else's quotes. A bot that quoted around the full book's mid would be
quoting around its own orders, and those quotes would stay put when the market
moved. Callbacks never nest: all the events from one book change are booked
before any callback runs, and orders a bot sends from a callback are queued
behind it.

**Avellaneda-Stoikov on a book.** The model allows a quote on the far side of
the mid; on a book that is a market order. The bot rounds the model's prices
away from the reservation price and then keeps them passive, so a bid never
reaches past one tick below everyone else's ask. Inventory risk shows up as the
bot stepping back on the side that would add to its position, which on a queue
mostly means that side stops filling.

**Solving GLFT without losing the tails.** The quotes depend on ratios of
neighbouring entries of v = exp(-M tau) 1, and those entries can span many orders
of magnitude: with the order-book calibration (gamma = 1, Q = 10) the smallest is
1e-18 of the largest. An eigendecomposition gets the tails wrong, because the
eigenvectors are only accurate to about 1e-16 of their largest entry. But -M tau
has nonnegative off-diagonal entries, so shifting it by its smallest diagonal
entry makes every entry nonnegative. The Taylor series of a nonnegative matrix
has no cancellation, and neither does repeated squaring, so every entry of the
exponential comes out with full relative precision (the same idea as
uniformization for Markov chains). The long-horizon quotes are this exponential
at tau = 80 / (spectral gap), by which point the other eigenvectors have decayed
by e^-80.

**Markouts.** A markout is side times (mid at t + h minus fill price). The tape
keeps the last mid of each instant, so h = 0 already includes the book's
reaction to the fill. Horizons that run past the end of the session are left
out. Summed to the close, a fill's markouts are exactly its contribution to
P&L, which a test pins down.

## License

MIT
