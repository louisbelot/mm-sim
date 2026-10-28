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

## 2026-10-21: GLFT, informed traders and markouts

The Gueant-Lehalle-Fernandez-Tapia model solved exactly, a bot that quotes it,
then informed traders and the markout tooling to see what they cost.

Built:

- `glft.py`: `Model(sigma, A, k, gamma, Q, mu, xi)` with the tridiagonal matrix
  M, v(tau) = exp(-M tau) v(T), quotes at any time to the close or in the
  long-horizon limit, and the paper's closed-form approximation. It covers both
  extensions in the paper: a drift mu (diagonal gets -k mu q) and a price impact
  xi per fill (off-diagonal scaled by exp(-k xi / 2), terminal v_q = exp(-k xi
  q^2 / 2), quotes shifted by xi / 2). The paper only treats them one at a time;
  the HJB check below confirms they combine as expected.
- `bots.GLFT`: quotes the stationary table for the current inventory, rounds
  away from the mid, keeps quotes passive, drops the side that would break Q.
- `Fill.counterparty`: the owner of the other order (the background flow is 0).
  Self-trades book both sides with the right maker flags.
- `Simulator.tape`: every change in the full book mid, keeping only the last
  value of each instant. `markout.py` has `markouts` (side times mid at t + h
  minus price, NaN past the end of the session) and `curve` (qty-weighted mean
  and standard error per horizon).
- `informed.py`: `Fundamental`, a Brownian value sampled lazily and exactly at
  whatever times it is asked for, and `InformedTrader`, who wakes at Poisson
  times and takes the touch when V is beyond it.
- Two examples and charts: `glft_quotes.py` (paper example) and
  `informed_flow.py` (2000 sessions, about 90 seconds).

Checks (all measured today):

- 135 tests, 17 s for the whole suite.
- The quotes match `scipy.linalg.expm` to 1e-11 relative on the paper's example,
  and a Radau solve of the nonlinear HJB system for theta = log value (derived
  separately, not from the linear system) to 1e-6 ticks in four cases: plain,
  drift, impact, and drift plus impact with the order-book calibration.
- Near the close the quote slopes in time are (2q + 1) gamma sigma^2 / 2, the
  Avellaneda-Stoikov expansion (paper footnote 8), to 0.07% at tau = 1e-4, for
  every q whose neighbours are both interior. The quotes at q = Q - 1 differ at
  first order because v_Q has only one neighbour, which is why the test stops one
  short of the bound.
- Long-horizon quotes equal the ones built from the ground state of M (numpy
  eigh, Q = 12) to 1e-9, and the t = 0 quotes for T = 600 are within 4e-7 ticks
  of them.
- Paper example: stationary spread 6.6261 against the closed form 6.626; closed
  form within 0.002 ticks for |q| <= 10, off by 0.85 at q = 29.
- Exact Monte Carlo of the paper's model (fills by thinning, mid integrated out
  since fills do not depend on it: E[exp(-gamma q dS)] = exp(gamma^2 sigma^2
  q^2 dt / 2)), 20,000 paths: certainty equivalent 63.93 +- 0.08 against
  ln(v_0(0)) / k = 63.90. Shifting all quotes by -0.3 and +0.3 ticks with the
  same random numbers gives 63.04 and 63.15.
- A Bernoulli-per-step version of that Monte Carlo was biased low: 0.43 ticks
  at dt = 0.05, 0.28 at 0.025, 0.20 at 0.0125. Switching to thinning removed the
  bias rather than chasing it with smaller steps.
- Informed traders: every informed fill is on the right side of V at that
  instant (checked over six sessions). With a constant V of 104 and asks at 102
  (5 lots), 103 and 104, the trader buys exactly 6 lots and stops.
- `Fundamental` sampled on a 2-point or a 50-point random grid gives the same
  N(0, sigma^2 T) increment over 3000 seeds.
- Summed to the close, markouts equal account value at the last mid (300 random
  fills).
- Experiment (200 sessions per row, one informed trader at 1 order a minute, V
  at 1 tick per sqrt(minute), all bots capped at 5 lots): joining the touch goes
  from +86.9 to -73.2 ticks a session, AS gamma 0.001 from 74.7 to 8.7, GLFT
  gamma 0.3 from 62.5 to 32.6, GLFT gamma 1 from 18.3 to 13.5. Markouts of the
  touch quoter: +0.54 at the fill against informed flow, -2.96 after 30 minutes
  (19,691 fills); against noise flow +0.58 rising to +1.01 (47,441 fills).

Design notes and things that went wrong:

- The first plan was an eigendecomposition of M. For the order-book bot (gamma =
  1, Q = 10) v spans 18 orders of magnitude and eigenvectors are only good to
  1e-16 of their largest entry, so the tails would be noise. -tau M is a
  Metzler matrix; shifted by its smallest diagonal it is nonnegative, and a
  Taylor series plus repeated squaring of a nonnegative matrix never cancels.
  Each squaring is renormalised by its largest entry since only ratios matter.
  The eigenvalues are still used, for the spectral gap that decides how long
  "infinite" is (80 / gap).
- The paper's figures 1 to 3 do not fit Q = 30: the exact bid at q = 29 is 6.13
  ticks, above the top of their axis (5.5), and the spread there is 7.26 against
  their 6.64. With Q = 40 or more and the window [-30, 30], everything fits. I
  read that as the figures plotting the middle of a wider bound, and the test
  uses Q = 50.
- The noise markouts rising over 30 minutes looked like a bug. It is the position
  cap: a touch quoter at -5 lots only shows a bid, so its noise fills are buys
  made while informed buying is still pulling the price up. Rerunning without the
  cap (40 sessions) gives a flat 0.60 to 0.54.
- xi measured from markouts (0.72 ticks) made GLFT sit a tick behind the touch
  whenever flat; 13 fills a session. The continuous model has no idea the book
  has ticks.
- A first draft of the informed test asserted the informed trader's P&L marked
  at the final V was positive. It failed on a seed: each trade has positive edge
  when it is made, but the inventory it accumulates then rides the Brownian
  motion, which has no sign. The test now checks the edge at the time of each
  trade.
- The first test sessions all shared flow seed 0 by accident; fixed in its own
  commit.
- The `charts` commit went in with two ruff errors in the experiment script
  (an unused loop variable and a zip without strict). A `;` in the shell chain
  let the commit run after the lint failure. Fixed in the next commit.

Next: Glosten-Milgrom and Kyle as small closed-form demos of why spreads exist,
then queue-position modelling, since the xi result says the tick grid and the
queue matter more than the continuous models admit.

## 2026-10-22: Glosten-Milgrom, Kyle and queue position

Two information models that explain why spreads and price impact exist, then
queue position: tracking it, measuring what it does to fills, and a quoting
rule that holds priority.

Built:

- `glosten.py`: the closed-form Glosten-Milgrom quotes for a two-point value,
  and a grid version where insiders see V plus Gaussian noise and stay out when
  their estimate is inside the quotes. Quotes are fixed points
  (ask = E[V | buy at ask]) found with brentq; `update` is Bayes on buy, sell or
  no trade; `simulate` runs a session against a true value.
- `kyle.py`: the N-auction Kyle equilibrium by backward recursion (a cubic for
  lambda_n at each step), the insider's expected profit for any linear strategy
  against the equilibrium lambdas, and a vectorised Monte Carlo.
- Queue position in the simulator: `LiveOrder.ahead` (quantity in front when the
  order joined its level), `ctx.ahead(oid)`, and `Fill.ahead` and `Fill.left`
  (quantity still resting at the fill price after the incoming order). Trade
  booking moved into `Simulator._fill`.
- `Quoter(keep=k)`: leave a live quote alone when it is at most k ticks behind
  the new target. All three bots pass it through.
- Two examples and charts: `why_spreads.py` and `queue_value.py` (2400
  sessions, about 100 seconds).

Checks (all measured today):

- 162 tests, 20 s for the whole suite.
- Glosten-Milgrom: the grid solver on a two-point grid with zero noise equals
  the closed form to 1e-9 for three (mu, alpha, eta) sets. Every ordering of
  four buys and two sells gives the same belief, equal to the net-flow formula.
  The posterior mean after the next trade averages back to the current mean,
  no-trade included, to 1e-10. Over 600 sessions with V drawn from the prior the
  market maker's P&L is zero within 4 standard errors and the insiders' is
  positive by more than 8.
- Kyle: N = 1 matches the closed forms. lambda_n equals the Kalman gain given
  beta_n to 1e-10 relative, for N = 2, 5 and 40 with T = 3. The value function
  equals the profit computed forward from the strategy to 1e-12. Bumping any
  beta_n by 5% either way lowers the profit (N = 6). N = 4000 is within 0.2% of
  the continuous-time lambda and profit. 200,000 Monte Carlo paths match Sigma_n
  to 2%.
- Kyle with Sigma0 = sigma_u = 1: insider profit 0.5, 0.7307, 0.8860, 0.9869,
  0.9988 for N = 1, 4, 16, 256, 4096. The last auction's lambda converges to
  about 0.853, not 1.
- Glosten-Milgrom sessions (200 per alpha, 150 trades): spread 1.75 to 0.95,
  5.18 to 0.38 and 8.84 to 0.18 for 10%, 30% and 50% informed; market maker
  P&L -5.2 (se 7.5), -2.8 (3.7), -1.9 (2.5).
- Queue tests are scripted: two of the bot's orders behind a 5-lot order,
  checked through two market orders, a sweep, three kinds of modify and a taker
  fill.
- The keep = 0 rows of the queue experiment reproduce the informed-flow table
  from 2026-10-21 to the decimal, which says the queue bookkeeping did not
  change any behaviour.
- With informed flow, keep = 1 takes GLFT gamma 0.3 from 32.6 (se 1.8) to 58.2
  (se 1.9) ticks a session and the touch quoter from -73.2 to -44.9. Without it,
  +4.5 and -1.1. Markout at the fill rises from 0.52 to 0.73 for GLFT.
- Fills that emptied their level: GLFT markout at the fill 0.36 against 0.57.
  Orders that started alone at a level (ahead = 0) emptied it 67 to 73% of the
  time they filled.

Design notes and things that went wrong:

- First Kyle draft had beta = lambda Sigma / sigma_u^2 instead of
  lambda sigma_u^2 / Sigma, which made 1 - beta lambda dt negative and Sigma_0
  negative. The N = 1 closed form caught it right away.
- Solving the cubic with brentq on (0, 1 / (2 alpha_n)) needs no case analysis:
  f(0) = -1 and f at the right end is positive, and Kyle proves the root is
  unique there. The final auction (alpha_N = 0) is a square root.
- The Sigma_t = Sigma0 (1 - t) check first used a relative tolerance and failed
  near the close, where the discrete solution stays a fixed 0.004 above a target
  that goes to zero. It now uses an absolute tolerance of 0.2% of Sigma0.
- Glosten-Milgrom with noise exactly zero on a fine grid has discontinuous quote
  equations, so the grid tests use noisy insiders and the zero-noise check uses
  two points, where the equation is continuous between them.
- I expected queue position at entry to sort the 30-minute markouts. It barely
  does at this sample size; what separates good fills from bad is whether the
  fill emptied the level. That is what the README reports.
- Why keep helps: most one-tick moves of a target come from the spread
  flickering open and shut. Chasing them means stepping alone into new levels,
  which are the first thing an informed market order takes out.

Next: latency in the loop (delayed market data and order acknowledgements, so
a bot quotes on a stale book and its cancels can arrive after the fill), then
plugging in the C++ engine.

## 2026-10-23: Latency in the loop and Hawkes order flow

Two things the simulator assumed away: that a bot sees the book and reaches it
instantly, and that market orders arrive independently.

Built:

- Commands as data. `flow.step` now returns a function of the book instead of a
  closure over it, with order ids drawn when the event is drawn. The same
  command can then be applied to more than one book.
- `Latency(feed, entry, response, jitter)` per agent, following hftbacktest's
  split. Each leg is a FIFO `Link`; jitter is exponential and the FIFO clamp
  keeps messages in order. Everything delayed (orders in flight, fill and ack
  reports, feed updates, timers) goes through one heap keyed by time and a
  sequence number.
- The agent's view is now separate from the exchange's. `ctx.orders` holds
  orders from the moment they are sent, with `acked` and `canceling`;
  `ctx.position` moves when fill reports arrive. The exchange side keeps its own
  record of each resting agent order (owner, price, quantity, queue position at
  entry) and books fills into the account at exchange time. `Fill.stale` marks a
  fill on an order whose cancel was already in flight.
- Delayed feed by replication: a lagged agent gets a deep copy of the book at
  the start of the run and every command is replayed on it `feed` later.
- `hawkes.py`: multivariate exponential Hawkes. Ogata thinning, log-likelihood
  and its gradient, maximum likelihood fit, compensator and time-rescaling
  residuals. Decayed sums are computed in blocks so nothing overflows.
- `HawkesFlow`: CST limit orders and cancels, Hawkes market orders, simulated by
  thinning inside `step` with its own clock.
- `InformedTrader` waits for a two-sided book before starting the value process.
- `examples/latency_cost.py` (6800 sessions, 8 minutes) and
  `examples/hawkes_flow.py` (1600 sessions plus a fit), with charts.

Checks (all measured today):

- 187 tests, about 28 s for the whole suite.
- Zero latency changes nothing: six fixed sessions (two bots, three seeds,
  informed flow) give the same P&L, fills, positions and fill times before and
  after every commit, and the latency sweep's zero row equals the keep = 1 row
  from 2026-10-22 (GLFT 58.2, touch -44.9). The README usage example still
  prints (27.0, 2, 208).
- Scripted latency tests: an order reaches the book after `entry`, the ack and
  fill reports after `response`, a cancel sent 0.05 before a market order loses
  (stale fill, then a reject) and one sent 0.15 before wins, double cancels send
  one message, rejected submits leave the view when the report arrives.
- Feed replica: over an hour of CST flow the lagged agent's view at its k-th
  update equals the exchange book at the instant of the k-th flow event, five
  levels on both sides, and its timestamps are exactly 0.05 later.
- A 300-minute GLFT session with every leg delayed and jittered, then a quiet
  minute to drain: no events left, view position equals the account, the bot's
  view of its orders equals what rests on the exchange, and the replica equals
  the book order for order.
- Three injected bugs (no replay of agent commands, stale never set, FIFO clamp
  removed) each fail two of the latency tests.
- Hawkes: log-likelihood equals the O(N^2) double sum to 1e-11 relative; the
  analytic gradient matches central differences; blocked decay sums match the
  naive sums to 1e-11 for beta T up to 200,000; univariate mean rate and the
  variance of counts in 5-minute windows match the closed form
  (Var N(w) = Lambda w + 2 c (w / kappa - (1 - exp(-kappa w)) / kappa^2)) within
  6%; the rescaled gaps pass a KS test under the true model and fail under a
  Poisson model with the same rates (p < 1e-6); the fit recovers the branching
  matrix to 0.04 on 20,000 minutes; refitting on HawkesFlow's own market orders
  recovers it to 0.06.
- Results in the README: GLFT against informed flow goes 58.2, 49.0, 37.4,
  22.4, 0.7, -33.3 ticks a session at 0, 1, 2, 5, 10 and 30 seconds one-way.
  10 s on the feed alone costs 37.6, on entry alone 36.4, on responses alone
  5.4. With Hawkes market orders at the same mean rate GLFT makes -63.0 (se 5.3)
  without any informed trader, against +67.0 under Poisson.

Design notes and things that went wrong:

- The old simulator ran a bot's order inside `ctx.submit`. Keeping that path
  for zero-latency links, instead of pushing everything through the heap with
  delay 0, is what keeps old sessions identical: an extra heap entry would change
  the horizons the flow sees and so its random draws.
- Events other than trades are routed to whoever sent the command, not looked
  up by order id. A cancel that arrives after the fill has to go back to the
  bot, and by then the exchange has forgotten the order.
- I thought about rebuilding the lagged book from the event stream the way an
  ITCH feed handler would. Replaying commands on a copy is shorter and exact
  because matching is deterministic; the event-stream version belongs to the
  C++ engine.
- With feed latency only, the stale flag stays at zero even though the loss is
  as large as with entry latency. The flag only sees cancel races; decisions on
  old data are invisible to it.
- First version of `fit` used numerical gradients and took over two minutes on
  40,000 events. The analytic gradient needs a second decayed sum,
  sum (t - t_l) exp(-beta (t - t_l)), carried through the same blocks. The
  blocks started at e^50 and were widened to e^300 for speed; values below about
  1e-190 then underflow to zero, which the test now allows.
- A Hawkes session crashed when a burst emptied the ask side right at the
  start, so the informed trader started its value at `None`. Fixed in
  `InformedTrader`, with a test.
- The Hawkes loss is large enough that I checked the book's static shape for an
  artifact first: spread, depth and one-sided time are the same under both
  flows.

Next: a quoting rule that reads the market-order intensity off the trades it
sees and leans away from the side that was just hit, then the C API on the C++
engine so it can replace the Python book here.

## 2026-10-23: Leaning on Hawkes flow, P&L attribution

The Hawkes run this morning left GLFT losing 63 ticks a session to clustered
market orders with no informed trader anywhere. This entry gives the bots a way
to read the clustering off the tape, and adds the attribution needed to see
what that changes.

Built:

- Public trade prints. `Agent.on_trade(ctx, Print)` with one print per
  aggressive order (time, aggressor side, worst price, total quantity). Prints
  are queued before the fills and the book update from the same command, so a
  bot's signal is current when it requotes. Agents with feed latency get them
  from the replica book when the command is replayed there, stamped with the
  exchange time. Nothing is printed during warm-up.
- `Hawkes.forecast(h)`: the d x d^2 matrix that maps the excitation above its
  stationary level to the expected excess count of each event type over the
  next h (or infinite) minutes, from the linear ODE the expectation follows.
  `Hawkes.stationary_state()` is the mean of the decayed sums.
- `Tracker`: decayed sums updated event by event, starting from the stationary
  state so a bot that joins mid-session has no opinion yet.
- `FlowLean`: shift = kappa (excess buys - excess sells). `FlowLean.fit`
  regresses the mid change over the next h minutes on the forecast at each
  market order. Any `Quoter` takes a `signal` and quotes around `fair(ctx)`,
  the mid plus the shift. `SymmetricQuoter` now clamps its quotes to stay
  passive like the other two, since a shifted mid can otherwise cross.
- `attribution.py`: spread, adverse and inventory terms per session for a
  chosen horizon, plus fees, adding up exactly to the account value at the
  closing tape mid.
- `examples/flow_lean.py`: 4800 sessions (strategy table and a kappa sweep) and
  the chart, about 6 minutes.

Checks (all measured today):

- 208 tests. Eight fixed sessions (two flows, two bots, two seeds, one seed
  with every latency leg on and jitter) give the same P&L, position, fill
  count and sum of fill times after every commit. The rows without a lean in
  the new table equal the Hawkes table from this morning to the decimal.
- Forecast: the univariate closed form x (1 - e^-(beta - alpha) h) /
  (beta - alpha) and its infinite-horizon limit; mean counts of 20,000
  simulated two-dimensional paths with unequal betas match the forecast from an
  empty start within 4 standard errors, and differ from the stationary count by
  more than 10. Two injected bugs (wrong row map in M, transposed stationary
  state) each fail that test.
- Tracker intensity equals the event sum from `Hawkes.intensity` plus the
  decayed starting state to 1e-12 relative.
- `FlowLean.fit` recovers a planted 0.4-tick permanent impact within 10% and
  gives |kappa| < 0.1 on a random walk.
- Attribution: a hand-worked two-fill example, the markout identity, the
  no-mid case, and the sum against the account in sessions with a maker rebate
  at three horizons.
- Results: the lean is worth 38.3 (se 9.0) ticks a session to the touch quoter
  and 28.9 (se 6.4) to GLFT on Hawkes flow, paired by seed. It costs GLFT 11.7
  (se 1.8) on Poisson flow. Most of the gain shows up in the inventory term
  (-93.1 to -52.2 for GLFT), not in the first-minute adverse term.

Design notes and things that went wrong:

- I wanted the lean to work for any quoter, so it is a fair-value shift rather
  than a change to GLFT's model. GLFT has a drift parameter, but the drift here
  is state-dependent and decays in seconds; resolving the HJB per state was more
  machinery than the first-order shift needs.
- First version of the quoter test failed because `SymmetricQuoter` had no
  passive clamp. A two-tick down shift put its ask on the bid, the post-only
  order was rejected and the bot quoted one side.
- The kappa regression is over every market order, but fills cluster in
  bursts, which is where the forecast is largest and the move after it is
  largest too. The sweep shows GLFT wants about 1.5 times the fitted kappa. A
  fit conditioned on the bot's own fills, or on the forecast being large, is
  the obvious next try.
- The print that fills the bot is often the first of a burst. Nothing before it
  predicts it, so the first-minute adverse term barely moves with the lean.

Next: replace the unconditional kappa with a fit weighted the way fills are,
then move to the C API on the C++ engine so it can replace the Python book.

## 2026-10-27: The C++ engine as the book, and a tournament

The orderbook repo now has a C API and a ctypes package, so the simulator can
match on the real engine. The same day adds the first multi-bot comparison.

Built:

- `Simulator(book=...)` takes a book factory. The default is still the Python
  book.
- `NativeBook` subclasses the Python `Book` and sends submit, cancel and modify to
  the engine. It then applies the returned events to an order-level view (rest
  the taker if it was accepted, not canceled and has quantity left, reduce
  makers on trades, remove canceled orders, rebuild the order on a modify that
  loses priority). The C API gives depth by level and lookup by id but no
  per-order queue, and the bots and the flow need owner and queue position, so
  the view is what `ahead`, `levels` and `orders` read. `audit()` checks the view
  against the engine's own audit, depth, order count and `find` for every order.
  `deepcopy` (used for the feed-latency replica) builds a fresh engine book and
  resubmits the resting orders in queue order.
- `tournament.py`: `session` runs several bots in one session with seats shuffled
  by seed and everyone marked at the closing tape mid, returning P&L, position,
  fills and the spread/adverse/inventory split. `standings` ranks by mean, pairs
  each bot with its solo run for the crowding cost, and counts wins (ties split).
- `examples/tournament.py` and `examples/native_speed.py`.

Checks (measured today):

- 256 tests (was 208). 30 seeds of 2000 random commands give the same events
  and resting orders as the reference book from `NativeBook`, with `audit`
  every 100 commands. 12 whole sessions (four bot and flow combinations, three
  seeds, one with all latency legs and jitter) give the same position, cash and
  fills on the Python book and `NativeBook`, and equal resting books at the end.
- Two injected bugs in the view (ignoring partial fills of a taker, keeping
  priority when a modify raises the quantity) fail 42 and 31 of the 43 native
  tests.
- Speed: a 390-minute GLFT session takes 0.29 s on the Python book and 0.32 s
  through the engine, 169 fills on both. The session cost is Python, not the
  book.
- Tournament, 100 sessions each: on Poisson flow every bot earns less in company
  (touch 146 alone to 96, GLFT 69 to 15). On Hawkes flow the touch bot goes from
  160 alone to -107 together and the leaning GLFT is the only positive row (3.9,
  se 2.7), while GLFT and AS also do better in company than alone.

Design notes and things that went wrong:

- The orderbook repo is private, so mm-sim's CI cannot fetch it with the default
  token. The native tests skip when the package or the shared library is
  missing, which means CI covers only the Python path and the native checks run
  locally.
- First version of the tournament example crashed because `pool.map` returns a
  one-shot iterator and I read it once per seed. Materialising the lists fixed
  it.
- Each bot reads the others' quotes as the market, since its own orders are the
  only ones hidden from `ctx.bid` and `ctx.ask`. That is why crowding is not just
  a split of the same fills: a tight competitor changes the spread another bot
  measures. I have not separated that effect from queue sharing.

Next: replay real ITCH flow as the background market on this engine, then fit the
arrival intensities from it. After that the fill-weighted kappa for the lean.

## 2026-10-27: Calibrating the lean

The plan was a kappa fitted the way fills are weighted, because the sweep on
2026-10-23 said GLFT wanted about 1.5 times the regression kappa. Before
building it I checked whether the gap was real. It was a timing bug plus a low
draw of a noisy estimate, and fill weighting turned out not to matter. What
replaces it is an estimate of the same quantity that is six times tighter, and
two flows and two bots that show how far a fair-value shift gets you.

Real NASDAQ data was first on the list, but this session's network could not
reach NASDAQ's or LOBSTER's servers, so that is still next.

Built:

- `Simulator.prints`: every aggressive order as a `Print`, stamped on the
  simulator's clock, from the first command on (warm-up included).
- `HawkesFlow` no longer keeps a clock or a `markets` list. The examples fit
  their Hawkes models and kappas from `sim.prints`.
- `Tape.before(t)`, the last mid strictly before t, next to `Tape.at`.
- `FlowLean.from_impact`: kappa is the average signed change in the mid across
  each market order, with the forecast running to an infinite horizon.
- `examples/flow_lean.py` rewritten: both estimators over 40 independent
  2000-minute histories, a sweep of both bots at 1000 sessions per point on an
  absolute kappa scale, the expected P&L of each calibration, and the strategy
  table with the impact lean. About 20,000 sessions, 100 minutes on four cores.
  It is split into `compute`, `report` and `chart` so the chart can be redrawn
  from saved results.
- `examples/tournament.py`: the leaning entrant uses `from_impact`.

Checks (measured today):

- 215 tests (was 213). 24 fixed sessions (Poisson and Hawkes flow, five bots,
  half of them with every latency leg, jitter and an informed trader) give
  bit-identical P&L, cash, fills and tapes before and after each library commit,
  and the README usage example still prints (27.0, 2, 208).
- The clock: on the 2000-minute history behind the old tables, 34.6% of
  `HawkesFlow`'s market-order times were early against the simulator's and 54.5%
  late, by up to 1.6e-12. 15.8% of orders read the pre-order mid off the tape.
  Regression kappa at h = 0.5 with the flow's times and with the simulator's:
  0.256 and 0.194 on seed 11, 0.288 and 0.275 on seed 12, 0.396 and 0.331 on
  seed 13. The Hawkes fit itself does not notice (3570 orders, the same
  branching matrix and KS statistic as before).
- The new print test fails if prints are stamped one ulp early. Three injected
  bugs in `from_impact` (reading the mid 1e-6 before the order, using the size
  of each move without its sign, comparing the tape with itself) each fail one
  of its two tests; the sign bug only fails the random-walk one, which is why
  there are two.
- 40 histories of 2000 minutes (143,320 market orders), the seed-11 Hawkes fit
  throughout: regression kappa 0.384, 0.393, 0.388 and 0.374 at h = 0.25, 0.5, 1
  and 2, standard deviations 0.11 to 0.17. Impact per order 0.369, sd 0.019. The
  ratio of the half-minute mid change to the forecast is 0.36 to 0.42 in nine
  deciles of the forecast out of ten. Measured from the mid just before the
  order with a side intercept, the slope is 0.381 against 0.392 from the mid
  after.
- The full forecast is 1.361 times the half-minute one over the states seen in
  the history (correlation 0.995), so the impact lean is 0.514 on the
  half-minute scale.
- Sweep on Hawkes flow, 1000 sessions per point, kappa 0 to 1.0 on the
  half-minute scale. GLFT: -61.5, -49.5, -41.8, -31.0, -33.1, -24.9, -27.4, -31.4,
  -36.0 (se 1.7 to 2.5), and -45.7 at 1.2 in a separate run. A cubic peaks at
  0.57 with -27.2 (chi-squared 8.1 on 5 degrees of freedom). The touch quoter:
  -34.6, 7.1, 10.7, 9.1, 13.7, -0.8, -3.1, -12.2, -23.4 (se 3.1 to 3.6). The first
  200 seeds of GLFT's kappa = 0 point reproduce the Hawkes table (-62.98 against
  -63.0).
- GLFT with the impact lean from seed 11 (0.378, infinite horizon): -26.7 (se
  1.7), 166 fills. Averaged over the 40 histories and read off the cubic, the
  regression fit gives -30.9 (worst tenth -38.3) and the impact fit -27.5
  (worst tenth -28.0).
- The example's sweep sessions equal the exploratory runs to the last bit on
  every cell checked (three seeds at two kappas), and its full run reproduces
  every GLFT point.
- Second flow (0.5 own, 0.2 cross, beta 3): impact 0.404, regression 0.143 on
  seed 11, full forecast 1.486 times the half-minute one. GLFT at 0, 0.5, 1, 1.5
  and 2 times the impact lean: 50.8, 52.7, 47.5, 43.1, 35.4 (se 1.1 to 1.3).
  Attribution, lean minus none, 400 sessions: spread -14.1, adverse -1.5,
  inventory +11.8 there, against -48.3, +33.3, +46.3 on the main flow.
- Fill weighting: 219,312 market orders filled the GLFT bot over 1000 sessions.
  Slope of the half-minute move from the pre-order mid on the forecast before
  the order, with a side intercept: 0.640 on those orders, 0.619 on all 729,494.
- With GLFT in the book the impact of a market order falls to 0.204 (from 0.369
  without it) and rises back as the bot leans harder (0.233 at kappa 0.1, 0.266
  at 0.2). The regression on the bot's own sessions gives 0.517 at h = 0.5 and
  keeps rising with h (0.94 at 5 minutes) on the full book's mid. Leaning at 0.5,
  over 200 sessions: 0.551 on the full book's mid and 0.752 on everyone else's.
  GLFT's holding time by Little's law is 1.9 minutes.
- Strategy table with the impact lean, 200 sessions per row: paired gains of
  35.8 (se 10.2) for the touch quoter and 38.6 (6.3) for GLFT on Hawkes flow,
  33.1 (6.6) for GLFT with the informed trader, -21.0 (1.9) for GLFT on Poisson
  flow. The rows without a lean reproduce the Hawkes table's means.
- Tournament, 100 sessions: on Hawkes flow the leaning GLFT makes 22.9 (se 2.6)
  together and -19.3 alone, against 3.9 and -38.2 before. The solo runs of the
  other three bots are unchanged.

Design notes and things that went wrong:

- The bug was two clocks. The flow kept its own by adding each thinning step,
  the simulator added the step's total, and floating-point addition is not
  associative. Keeping the flow's clock in step would have been possible
  (subtraction is exact when the two values are within a factor of two, so
  t + (stop - t) is stop in practice) but not guaranteed. The simulator owns
  time, so it records the prints, and the flow does not need a clock at all.
  The old test of the flow's clock compared it with `approx`, which is how the
  drift survived.
- The README's reading of the old numbers (kappa falling with the horizon, so
  part of the impact must be transient; leaning harder than the fit because
  fills come in bursts) explained noise. The population kappa is flat in h.
- 200 sessions per point cannot find the top of a flat curve: at kappa 0.3 the
  first 200 seeds give -38.3 and 1000 give -31.0. The sweep now runs 1000.
- For a while I thought the answer was a fixed point: fit kappa on sessions
  where the bot trades with that kappa. The first numbers agreed with the peak
  (0.52), but they were read on the full book's mid, which includes the bot's
  own requote at the same instant. On everyone else's mid the fit is 0.75 while
  leaning at 0.5, and following it overshoots.
- The touch quoter rounds fair minus and plus half a tick outward. At a one-tick
  spread any positive shift moves its ask a tick back and any negative one its
  bid, so at small kappa it follows only the sign of the forecast and kappa
  matters once the shift passes a whole tick. The sweep shows exactly that: the
  first 0.1 is worth 42 ticks, then the curve is flat to 0.4.
- The impact lean is the right order of magnitude on both flows and the optimum
  on neither by construction: GLFT wants 1.1 times it here, the touch quoter 0.2
  to 0.8 times, GLFT on the second flow 0.2 to 0.5 times. On a tick grid GLFT
  steps a whole tick back once the shift passes 0.09 ticks, and a shift of the
  mid has no idea how often each side is about to be hit.

Next: real ITCH data as the background flow, once the data can be fetched. Then
quotes that use the intensity of market orders on each side, not just a shift
of the mid, following Cartea, Jaimungal and Ricci.

## 2026-10-28: Paired sessions

Yesterday's sweep needed 1000 sessions a point because pairing sessions by seed
bought nothing: over the same seeds, GLFT's P&L at two nearby kappas correlated
at about 0.1. This entry adds a flow that keeps the market the same whatever the
bots do, so that comparisons can be paired.

Built:

- `PairedFlow(params, hawkes=None)` in `flow.py`: CST limit orders and
  cancellations with Poisson or Hawkes market orders, from two generators seeded
  off the simulator's rng (so it is still a zero-argument factory for the
  tournament). Market and limit arrivals are kept pending rather than redrawn.
  Each flow order draws an Exp(1) threshold when it is placed, and is cancelled
  when its accumulated hazard reaches it.
- The CST exactness tests and the Hawkes refit now run on both flows, plus three
  tests of the pairing itself.
- `examples/paired_sessions.py` and its chart: both flows on the same 300 seeds,
  for two bots and the GLFT kappa sweep. About half an hour on four cores.

Checks (measured today):

- 232 tests (was 215). The 24 fixed sessions are bit-identical, since the old
  flows are untouched.
- On `PairedFlow`: CST Table 3 through the book (four cells), the queue races
  (three cells), the long-run limit-order rate, one-lot market orders, never
  cancelling another owner's order, and the Hawkes refit (branching within
  0.06).
- A bot that does nothing but set a timer every 0.05 minutes leaves the prints
  and the resting book identical to a session without it, for Poisson and Hawkes
  market orders; with `ZeroIntelligence` the two sessions differ. Two GLFT bots
  with different `keep` share more than 99% of their market-order times. A bot
  that sweeps the bids every minute leaves more than 95% of the flow's prints at
  the calm session's times.
- Four injected bugs each fail one of those tests: skipping the hazard during a
  wait for a timer, redrawing the pending limit arrival every step, exciting the
  Hawkes process only when the order trades, and using theta one level too far
  out.
- The market-order generator alone, 400,000 minutes on each of 20 seeds: 1.8821
  orders a minute (se 0.0017) against 1.8800.
- Prototype stages, GLFT at kappa 0, 0.4 and 0.5 over 200 seeds, correlation of
  the pairs (0, 0.5) and (0.4, 0.5): `HawkesFlow` 0.06 and 0.15; separate market
  and limit streams with cancels from a third stream 0.43 and 0.52; per-order
  thresholds 0.57 and 0.69; market orders that count even when they cannot
  trade 0.67 and 0.79. That last step took the share of market-order times two
  sessions have in common from 74% to 99.8%, and the correlation of their
  one-minute mid moves from 0.85 to 0.95.
- Same model: over 1000 sessions GLFT makes -63.8 (se 2.6) without a lean and
  -27.8 (1.9) with the impact lean on `PairedFlow`, against -61.5 (2.5) and
  -24.9 (1.8) on `HawkesFlow`; fills 216 and 166 against 220 and 166. The lean's
  gain, paired over those 1000 seeds, is 35.9 (se 1.9).
- `paired_sessions.py`, 300 seeds: correlation 0.12 against 0.85 and variance
  ratio 6.6 for GLFT at 0.4 and 0.5; 3.0 for GLFT with and without the lean, 3.2
  and 1.7 for the touch quoter at 0.2 against 0.4 and at 0 against 0.4. The
  paired sweep's differences from kappa 0.5 have standard errors of 2.0 to 3.5,
  against 2.5 to 3.1 from 1000 independent sessions, and peak near 0.6.
- Tournament on paired Hawkes flow, 100 sessions: crowding standard errors 6.6,
  6.7, 8.1 and 32.1 (leaning GLFT, GLFT, AS, touch) against 5.3, 9.2, 23.5 and
  34.7.

Design notes and things that went wrong:

- A cancel clock redrawn every step is exact, but the number of steps depends on
  the bots, so the cancel stream comes apart and the queues with it. A threshold
  per order is the time-change construction of the same clock, and it needs no
  draws after the order is placed. It is exact here because the simulator calls
  the flow after every change to the book, so the rates are constant between
  calls.
- The first version thinned Hawkes market orders by "is the other side
  non-empty", which matches `HawkesFlow` exactly but desynchronises the process
  for good the first time one session's book is one-sided and the other's is
  not. Bursts empty a side far more often than the 0.15% time average suggests:
  14 of 24 sessions had at least one such moment. Market orders now count either
  way.
- The flow keeps its own time, but only to order its pending arrivals. Prints
  are stamped by the simulator; yesterday's bug was exactly a flow clock leaking
  into the tape.
- My first sweeper test asserted that every flow print of the swept session also
  appears in the calm one. It failed on one buy: arrivals are common, but either
  session can drop one, and the calm session's ask side happened to be empty at
  that moment during a burst of buys.
- Pairing helps most where the two sessions stay close. For big changes, and for
  the tournament, the bots' own trading moves the books apart, and nothing in
  the flow can share that. It also does nothing for levels, only differences.

Next: quotes that use the intensity of market orders on each side (Cartea,
Jaimungal and Ricci), compared with the lean on paired sessions. Real ITCH data
still needs a network that can reach it.
