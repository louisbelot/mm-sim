# mm-sim devlog

## 2026-10-19: Simulation core and Avellaneda-Stoikov

First real day: the book, the simulator, CST order flow, the bot API, accounting,
a baseline bot and Avellaneda-Stoikov.

Built:

- Scaffold: hatchling `src/` layout, ruff, pytest, GitHub Actions on Python 3.10
  and 3.12. Runtime dependencies are numpy and scipy.
- `book.py`: limit order book ported from the C++ engine's rules and event
  stream (limit, market, IOC, FOK, post-only, cancel, modify). No self-trade
  prevention yet. Levels are an `OrderedDict` per price, keyed by sign times
  price so the best level of either side is the smallest key.
- `flow.py`: Cont-Stoikov-Talreja zero-intelligence flow with the paper's Sky
  Perfect parameters, simulated exactly (Gillespie). `prob_mid_up` and
  `prob_fill_first` solve the queue races as sparse linear systems.
- `sim.py`: event-driven simulator. Agents get `on_start`, `on_book`, `on_fill`
  and `on_timer`, and a `Context` with the clock, everyone else's quotes, open
  orders, account, timers and `stop`.
- `accounting.py`: cash, position, maker and taker fees, mark-to-market.
- `bots.py`: a `Quoter` base that keeps one post-only order per side and leaves
  a quote alone while its price is unchanged, `SymmetricQuoter`, and
  `AvellanedaStoikov` for the live book.
- `avellaneda.py`: the closed forms and a vectorised rerun of the paper's section
  3.3 experiment.
- Two charts: the AS replication and risk against return on the CST book.

Checks (all measured today):

- 104 tests, 12 s for the whole suite.
- Differential test: 30 seeds of 2000 commands against a naive reference book,
  comparing every event and every resting order. 24% of accepted limit orders
  trade, 8% of commands are rejected, 24 orders rest on average. Six
  hand-injected bugs (equal-size modify losing priority, FOK needing strictly
  more, newest-first matching, post-only allowed to touch, trades at the taker's
  price, modify dropping post-only) failed between 22 and 30 seeds each.
- CST Table 3 (probability the mid moves up next, one-tick spread): my linear
  system matches all 25 Laplace-transform entries within 0.0008. The paper's
  table only matches with the measured lam(1) = 1.85, not the power-law fit
  1.92, which settles which one they used. Simulated races through the full book
  and flow, 4000 per cell on four cells, agree with the solve within 1.3
  standard errors.
- With no limit orders and no cancels both solvers reduce to a race between two
  Poisson streams, which has a binomial closed form; they match it exactly.
- Limit orders arrive at a state-independent rate, so their count over 2000
  minutes is Poisson with mean 2 sum(lam) T. The simulated count lands inside
  4 standard deviations, which checks the clock as well as the event mix.
- A 20,000-minute run: 37.5 events per minute, 17 microseconds per event in
  the flow alone. Spread is one tick 74% of the time, two ticks 19%. Average
  depth at 1, 2 and 3 ticks from the opposite quote is 1.56, 1.63 and 1.57
  lots, the hump at two ticks from the paper's figure 2. The mid moves about 1
  tick per sqrt(minute).
- Avellaneda-Stoikov: all six rows of tables 1 to 3 reproduce within the
  sampling error of both runs (for gamma = 0.1, inventory strategy: profit 64.9
  and sd 6.6 against 65.0 and 6.6). The reservation price matches the
  frozen-inventory indifference price found by root-finding on the utility, and
  the spread term matches a numerical maximisation of A exp(-k d)(1 - exp(-gamma d)).
  Four injected modelling bugs (full horizon in r, fills priced off r, mid moving
  before fills, half the skew) each failed the table tests.
- Fill rate on the CST book for a quote held at distance d from the mid: 0.348,
  0.009 and 0.001 lots per minute per side at d = 0.5, 1.5 and 2.5 ticks
  (4000 minutes each). ln(0.348 / 0.009) gives k of about 3.6 per tick.
- 200 sessions of 390 minutes: joining the touch makes 128.9 ticks a session
  with an sd of 153.9. AS with gamma = 0.001 makes 81.4 with an sd of 25.9, the
  best mean-to-sd ratio of the gammas tried (full table in the README).

Design notes and things that went wrong:

- The first `ZeroIntelligence.step` applied the event and then returned its
  waiting time, so the simulator sampled P&L after the book had already changed.
  The fixed-grid sampling test caught it. `step` now returns the waiting time and
  a function that applies the event, and the simulator moves its clock in
  between.
- Fill callbacks first ran as soon as each trade was booked. A market order
  that hit two of a bot's orders called `on_fill` before the second fill was
  in the account. Callbacks now go on a queue that drains after the whole event
  batch is booked; a test pins it.
- Python gotcha in the first draft of the flow: `case (_LIMIT, side):` in a
  `match` statement does not compare against the constant `_LIMIT`. A bare name
  in a pattern is a capture, so the first case matched everything. Plain
  `if`/`elif` now.
- Bots quote around the mid of everyone else's orders. Around the full book's
  mid, a bot alone at the touch quotes around itself and never follows the
  market.
- The paper reports 3.02 as the average spread for gamma = 1. Averaging the
  spread over the 200 quote times gives 3.03; 3.02 is the average over [0, T].
  The symmetric benchmark now uses the latter.
- I could not reproduce CST Table 4 (probability that a bid fills before the
  mid moves). The paper's formula for the time to execution is a sum of
  exponentials with rates mu + (i - 1) theta, and my solve uses exactly that,
  but for one lot at the bid and two at the ask it gives 0.698 against their
  0.641. Two other readings of the setup did not match either. The simulator
  agrees with my solve, so that is what the test checks.

Next: the Gueant-Lehalle-Fernandez-Tapia solution with inventory
limits (exact quotes from a matrix exponential, compared with their closed-form
approximation and with AS), then informed traders and markouts on
the CST book.
