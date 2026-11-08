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

## 2026-10-28: Real NASDAQ flow, and calibrating the models from it

The bots now quote against a real day of NASDAQ ITCH data (30 December 2019,
NASDAQ's public sample), and the order flow models get their parameters from
the same feed.

Built:

- `itch.py`: decodes the order messages of ITCH 5.0 (A, F, E, C, X, D, U, P, Q)
  into `Msg` tuples with struct layouts that check the length of every message.
  `extract` walks the 8.3 GB file once through mmap and copies each wanted
  stock's messages, found by its stock locate code, to its own file
  (`python -m mmsim.itch FILE SYM... --out data`). 68 s for six symbols.
- `replay.py`: `ReplayBook` is a `Book` where the real orders live under negative
  ids and follow the feed exactly, and the bot's orders share their queues. A
  real execution fills every bot order with priority over the executed order (a
  better price, or earlier at the same price), up to the executed size. A real
  add at or through a bot quote fills it. A bot that takes liquidity fills
  against the displayed real orders without removing them. `best` and `mid`
  read only real orders, so the tape and the markouts use the market's mid.
  `ItchFlow` is a flow like `ZeroIntelligence`, so `Simulator`, latency, the
  tournament code and every bot run on it unchanged. Executions with the same
  nanosecond timestamp are one batch and one aggressor, with one aggressor id per
  side. `on_grid` converts prices to cents and drops orders off the cent grid
  with their later messages, turning a replace across the grid into a delete or
  an add.
- `calibrate.py`: Cont-Stoikov-Talreja estimates over a time window: lambda(i)
  counted at the distance from the opposite best quote at the moment of the add,
  theta(i) as cancelled shares per minute over the mean resting shares at i
  (queues sampled on a one-second grid), mu as market shares per minute over the
  mean limit order size. A replace counts as a cancel and an add. `cst()` gives
  `CSTParams`, `hawkes(unit)` fits the bivariate Hawkes process on market order
  times, in seconds by default, and converts back to minutes.
- `examples/itch_replay.py`: calibration report, real tape against the Poisson
  and Hawkes models on CSCO, real tape across six symbols, and the chart.

Checks (all measured today):

- 287 tests (was 256).
- The C++ engine as the reference: `ob-replay --itch` wrote the ITCH for its
  example session (790 bytes, now `tests/data/session.itch`), and replaying it
  gives the same final book and the same 1200 traded shares at a value of
  120,600 cents that `ob-itch` reports. By hand, the top five CSCO levels at
  noon match `ob-itch --until 12:00` in price, size and order count (5700 x
  47.58 | 47.59 x 200 at the top). The 11281 vs 11284 live orders are the three
  $0.0001 stub bids that `on_grid` drops.
- Twelve random feeds of 3000 messages, generated from a `Book` with adds, sweeps,
  partial cancels, replaces and deletes, replayed with a quoting bot of size 10,
  100 or 300: the real orders at the end equal the generating book's, in queue
  order, and every seed has more than 20 bot fills.
- Injected bugs: ignoring the queue position, letting the taking bot, a crossing
  add or an execution move the real book, and not batching sweeps. Each fails at
  least one test. For calibration: appending the market order per message
  instead of per batch, an off-by-one distance, forgetting to sample the queue
  after the last message, and dropping the unit conversion of beta.
- Calibration recovers a zero-intelligence model it did not see: a 1500-minute
  simulation from known CSTParams gives lambda within 4 standard errors, theta
  within 8% and mu within 4 standard errors. A simulated Hawkes process with a
  25 ms decay comes back with mu within 15%, branching within 0.04 and beta
  within 15%, in minutes.
- A hand-worked feed checks every number exactly: rates, both queue profiles,
  theta, a two-order sweep as one market order of 150, hidden shares.

Results (CSCO, 9:35 to 15:55, twelve 30-minute sessions per row):

- lambda(1) is 184 orders a minute per side and theta(1) 14.2 per share per
  minute, against 2927 shares resting at the touch. 54% of limit orders are
  exactly 100 shares, the mean is 272. 4008 market orders, mean 399 shares. 17%
  of traded shares hit hidden orders.
- The calibrated Poisson model's mid moves 0.01 times a minute; the tape's moves
  11.2 times. A touch quoter makes exactly 0.500 cents a share in the model and
  -0.117 (se 0.112) on the tape: 0.514 of spread, -0.727 of one-minute adverse
  selection, 0.096 of inventory.
- Across CSCO, INTC, CMCSA, MSFT, AAPL and QQQ the touch quoter loses on all but
  CMCSA, whose +0.112 is less than its inventory term. AAPL is worst at -0.962
  (se 0.254), adverse -1.962.

Design notes and things that went wrong:

- I considered pushing the feed through the normal matching engine, translating
  each execution into a market order. The trouble is that whenever the bot
  absorbs part of an aggressor, the real order it saved stays in the book, and a
  fully executed real order never gets a delete. That leftover liquidity would
  sit at a stale price for the rest of the day. Keeping the real book exact and
  the bot's fills on top (no market impact) avoids that, at the cost of
  flattering a bot that trades in size.
- First calibration run gave 10,933 market orders with a median gap of 0 ms. The
  append for each batch was indented inside the per-message loop, so a sweep
  through five orders logged five market orders with running totals. 4008 after
  the fix. The hand-worked test now pins it.
- The Hawkes fit starting from second units and from minute units lands in
  different optima. Both are dominated by sub-millisecond clustering, and the
  minute start adds a slow cross term with branching 0.46 and 0.69 but a lower
  likelihood. One exponential cannot fit both scales; the residual KS statistic
  is about 0.12 either way.
- The side field of `P` messages is `B` on all 22,264 of them in the six
  files, so hidden trades carry no direction.
- Off-grid prices first raised an error. They turned out to be stub orders at
  $0.0001 to $0.27, up to ten per symbol, so now they are dropped.
- `C` executions use the resting order's price for the print, not the execution
  price. They are mostly the opening cross, which the sessions skip.
- A run with the "one tick back" quoter was dropped from the example: on a
  one-tick stock it got 9 shares a session on the tape and none in the models.

Next: a queue-reactive model (Huang, Lehalle, Rosenbaum 2015), where the arrival
and cancel rates depend on the queue size, calibrated from the same feed, to see
whether it can reproduce the 11 mid moves a minute. Then latency on the replay
and the fill-weighted kappa for the lean.

## 2026-10-30: Queue-reactive flow, latency and rebates on the tape

A queue-reactive model fitted to the CSCO feed makes 10.26 mid moves a minute
against 11.16 on the tape (the Poisson model made 0.01), and a touch quoter on it
loses 0.155 cents a share against 0.117, with one-minute adverse selection of
-0.606 against -0.727. Nothing was fitted to those targets. The replay also gained
measured latency and rebates.

Built:

- `qr.py`: `fit` takes one pass over the feed and records the ladder of queues
  before each event plus the events, then bins everything with numpy, so the
  average event size and the opposite-queue terciles can be computed first. Rates
  are events over exposure time per (level, regime, queue bin). `Params` holds the
  rates, the empirical sizes, sampled queue volumes and cancel positions.
  `QueueReactive` is a flow like the others, so bots, latency and the tape work
  unchanged. Sparse cells fall back to the pooled rate at that bin, then the
  nearest lower bin. The top bin reflects.
- `Gap` is the small piece of memory shared by the fit and the flow: which side
  left a wide spread, whether a market order did it, and the direction of the
  last mid move.
- `markout.signature`: moves per minute, eta and one-minute volatility of a tape.
- `examples/queue_reactive.py` and `examples/replay_costs.py`.

Checks (all measured today):

- 323 tests (was 287).
- A pinned queue matches the closed-form birth-death invariant within 0.01 at every
  n and its depletion flux within 3%, over 40,000 simulated minutes.
- Rates that generated a feed come back from `fit` within 4 standard errors.
- Gaps close at the set 75% odds, within 4 standard errors, over more than 500 gaps.
- Every command replays identically on a second book. This is what lets the flow
  run under feed latency: all randomness and ids are drawn in `step`, never in the
  command.
- On random feeds with latency the feed replica ends on the exact real book.
- 19 injected bugs in `qr.py` all fail a test. The first pass caught 16. The three
  survivors were real holes: nothing forced a cancel onto a bot's position, and
  nothing tied the flow's regime choice to the fit's, which is the riskiest
  contract in the file. Added a test that the two agree on one book state.
- `examples/queue_reactive.py` run twice gives byte-identical output.

Results are in the README: the ladder, six symbols, fit on the morning against the
afternoon, and latency with rebates.

Design notes and things that went wrong:

- Measuring before modelling set the whole design. I expected queues of 3000 shares
  being eaten. The median queue that ends has 200 shares.
- The first version overshot 2.6 times, and adding detail made it worse. The top
  bin held 42% of the time against 2.8% on the tape. Cancels removed a random whole
  order while adds used conditional sizes, so the drift at each queue size was
  wrong. Conditioning cancel sizes fixed most of it. Then the top bin, whose rates
  come from about 21 side-minutes with almost no drift, needed a reflecting barrier.
- I guessed wrong twice on why the bot fills 1.5 times too often. Back-of-queue
  cancels are a large measured effect but barely changed fills. Blinding the
  aggressor to the bot did not change them either, though it fixed the moves (14.2
  to 10.2). Decomposing settled it: 25% more volume times an 18% bigger slice. I
  have not fixed that.
- The cleanup rename of `depth` to `queues` added a `.T` and silently fixed a real
  bug: `Params` had been storing time snapshots instead of per-level samples, so
  deep levels were seeded from the wrong array. The stationary rung's moves went
  from 12.2 to 18.4 when I reran it, which is how I noticed. Added a test that
  fails without the fix. Every number in the README is from after it.
- The first costs table looked wrong: latency cut adverse selection and QQQ's fills
  went up fourfold. I added the spread term and dollars to test the idea that stale
  fills move the loss between columns. They do, and dollars get worse on all six.
- A test hung because a limit size of -1 made `_lump` count upward. That was the
  test's fault: limit sizes are never signed in real parameters.
- Two experiments started at the same moment killed one process pool at startup
  (WinError 6). They now run one after the other.
- I could not find Nasdaq's fee schedule for December 2019, so rebates are labelled
  scenarios.

Next: time-varying intensities, since the fit on the morning overstates the
afternoon's moves 4.5 times while eta and adverse carry over. Then order-flow memory
beyond one bit, to close the adverse gap, and the fill-weighted kappa for the lean.

## 2026-11-06: An activity clock for the queue-reactive flow

Built:

- `qr.Clock`: slot edges in minutes after midnight and one multiplier per slot and
  event kind. `Clock.slot(t)` gives the slot and the time left to its edge, clamped
  at both ends.
- `fit` now records exposure by slot as well as by state (`Fit.exposure`, `Fit.slots`,
  `Fit.arrivals`), and takes `aes` and `edges` so a second window can be binned
  exactly like the first.
- `Fit.clock(mode)`: the joint maximum likelihood of state rates times slot
  multipliers, solved by alternating the two score equations until the multipliers
  move by less than 1e-10. `KINDS` fits one clock per event kind, `SHARED` one for
  all three. Each clock has a time-weighted mean of one, so the state rates stay
  all-day rates. `Fit.params(clock=...)` returns rates fitted against the clocked
  exposure and attaches the clock.
- `qr.rescale(params, later_fit)`: holds fitted rates fixed and solves only the
  multiplier equation on another window. It refuses a fit binned differently.
- `QueueReactive` keeps its own clock time (`start=`) and samples through slot edges:
  an exponential at the current slot's total rate, and if it crosses the edge, a new
  draw from the edge.
- `examples/activity_clock.py` and a README section with the chart.

Checks:

- 332 tests (was 323).
- Synthetic counts from known rates and multipliers come back within 4 standard
  errors, and at the solution every slot's expected count equals its observed count
  to 1e-8, per kind and for the shared clock.
- A feed generated by a three-slot clocked flow is fitted back to that clock's ratios
  within 4 standard errors, and `rescale` from the first window to the second
  recovers the ratio too.
- Event counts per slot match rate times length, and a draw from a nearly silent
  slot lands past the edge with the next slot's mean wait (1/12 minute here).
- A clock of all ones gives the same tape as no clock.
- Nine injected bugs all fail a test: no normalization, a draw allowed to jump the
  edge, the wait not carried across an edge, clock time not advanced on a quiet
  horizon, the slot search not clamped at the first slot, every kind using the limit
  multiplier, the shared clock fitted per kind, params ignoring the clock, and the
  slot index off by one.

Results (CSCO, 2019-12-30):

- Multipliers fall from 3.95 in the first slot to 0.29 at 13:35. Cancels swing the
  most (5.3 to 0.18). Market orders rise to 2.27 in the last slot.
- Fit on the whole day, the flat model makes 10.72 moves a minute in every session.
  The tape goes from 56.3 to 2.6. The clock by kind gives 52.7 at the open and
  stays within a factor of about two all day.
- Morning rates with the afternoon's own shared clock: 3.48 moves a minute against
  3.66, and 5,137 shares a session against 5,935. Morning rates alone gave 16.82
  and 23,934. The clock by kind gets the median touch queue (2,950 against 3,300,
  shared 2,050) at the cost of 9,151 shares.
- Six symbols: the flat morning rates run 1.2 to 3.6 times too fast. The own clock is
  within 20% on CSCO, INTC and MSFT and gives CMCSA 2.21 against 5.13. The peers'
  median clock is within 20% on INTC and CMCSA and twice too fast on CSCO.

Design notes and what went wrong:

- The estimator is the proportional-hazards shape: a baseline that depends on the
  book and a multiplier that depends on the clock. Alternating the score equations
  is coordinate ascent on a log-likelihood that is concave in the log parameters,
  so it cannot stop anywhere but the maximum. Exposure intervals that straddle a
  slot edge go to the later slot; with 30-minute slots and events milliseconds
  apart that is invisible.
- The flow has to keep its own clock rather than read the simulator's, because the
  simulator starts at zero and the clock is in minutes after midnight. It stays in
  sync because the simulator always either executes the drawn event or advances to
  the horizon it passed in.
- Matching event counts is not matching moves. A shared multiplier scales adds and
  removals together, so it cannot change how often a touch queue runs out, and
  CMCSA's afternoon changes exactly that balance. The clock by kind can, but it is
  fitted to counts too.
- `rescale` uses the afternoon's own counts. It shows how much of the morning's
  misfit is the clock (most of it on three of six symbols); it is not a forecast,
  and with one day of data there is nothing earlier to forecast from.
- The first version of the chart drew `stairs` with a baseline, which drops each
  curve to zero at both ends, and labelled log-scale minor ticks on top of each
  other. Both fixed before committing.
- My mutation run swapped `qr.py` on disk while the experiment's worker processes
  were importing it, so I threw that run away and reran it clean (15 minutes).

Next: order-flow memory beyond one bit, since adverse selection on the model is
still short of the tape on five of six symbols, and the 25% excess volume. Then the
replay fill rule that lets an aggressor reach the bot's displayed shares.

## 2026-11-06: The fills the replay hides, and markouts by cause

Built:

- `ReplayBook(sweep=True)`: a second fill rule. When one aggressor's executions
  (one ITCH timestamp) leave a price level with no real orders, the bot orders still
  resting there fill at that price, with the same aggressor id, so they join the
  print. A level emptied by a cancel or only partly executed fills nobody. The
  flow now hands the whole batch to `ReplayBook.play`, which notes the executed
  levels before applying the messages and checks them after.
- Fill causes: `ReplayBook.causes` logs every bot maker fill as it is emitted
  (`queue`, `add`, `sweep`, `cross`), and `replay.label(book, fills)` lines the log
  up with an account's fills, giving `take` to the bot's own aggressive fills.
- `markout.totals` (share-weighted markout sums per horizon, skipping horizons past
  the end) and `markout.clustered` (the ratio estimator with a standard error
  clustered by session, the same formula the replay tables already used inline).
- `examples/fill_quality.py`: both rules, both bots, six symbols, twelve sessions
  each, then markout curves by cause at 0, 1 s, 5 s, 30 s, 1 min and 5 min, and
  `docs/fill_quality.png`.

Checks:

- 350 tests (was 332).
- Hand-built feeds: an aggressor that exactly clears the bid leaves the bot alone
  under no impact and fills it under sweep, as one 550-share print; a partial
  execution or a level emptied by a cancel fills nobody; a deeper quote is left
  alone; a sweep through two levels fills the first level's remainder by queue
  and the rest by sweep.
- Eight random feeds with a quoting bot under the sweep rule: the real book still
  ends equal to the generating book, the cause log matches the maker fills order
  by order, and every swept fill leaves its level empty.
- `clustered` with equal weights is the mean of cluster means with the textbook
  standard error, and its 95% interval covers the truth in 400 simulated
  experiments at a rate between 90% and 98%.
- Ten injected bugs all fail a test: sweeping a level that still has real orders,
  sweeping on a non-execution batch, mislabelling sweep or add fills, dropping
  the cross log, looking only at the last executed level, the wrong aggressor id,
  labelling taker fills from the maker log, no n/(n-1) correction, and a totals
  denominator that ignores missing horizons.
- The no-impact rows reproduce the earlier six-symbol table to the cent, so the
  batch refactor changed nothing. Running the script twice gives identical output.

Results:

- Under the sweep rule 21% to 49% of the bot's volume is fills the no-impact
  replay never produces (CMCSA 23%, AAPL 49% for join the touch). Dollars per
  session are worse on all twelve symbol and bot pairs. CSCO join the touch goes
  from -$13.5 to -$36.9, and CMCSA, the only profitable symbol, turns negative on
  both bots.
- At one minute a swept fill marks out two to five times worse than a queue fill
  on CSCO, INTC, CMCSA and MSFT. QQQ is mixed and AAPL is reversed (queue fills
  -1.49 cents, swept -1.04).
- Pooled in basis points: queue fills start at +0.19 and swept fills at -0.01,
  because the swept level is gone in the same instant. At 1 s they are -0.25 and
  -0.32, and by a minute they are within one standard error.

Design notes:

- The two rules are bounds, not a model. ITCH gives the executed size, never the
  aggressor's intended size, so the truth for a level-clearing aggressor sits
  somewhere between sized-to-the-real-queue and sized-to-everything-displayed.
  The existing rules already cover the cases the feed does show: an aggressor
  that continues to a deeper level reaches the bot by price priority, and one
  that rests its remainder crosses the bot as an add.
- The cause log lives in the book because only the book knows why it emitted a
  trade. Lining it up with the fills works because the simulator turns the book's
  maker trades into fills in emission order, and order ids are unique, so
  filtering the log by a bot's ids gives that bot's sequence. The random-feed test
  checks the alignment directly.
- Pooling in basis points weights by shares, so AAPL and MSFT dominate the curve;
  the per-symbol cents in the table are the better comparison.

Next: order-flow memory beyond one bit for the queue-reactive model, since adverse
selection is still short of the tape on five of six symbols, then the
fill-weighted kappa for the flow lean.

## 2026-11-07: Hawkes memory on the queue-reactive market orders

The queue-reactive flow now remembers its own market orders. On CSCO the lag-1
autocorrelation of aggressor signs goes from 0.26 to 0.43 (tape 0.43), the bot's
volume excess from 44% to 7%, and one-minute adverse selection from -0.606 to
-0.666 against -0.727 on the tape.

Built:

- `qr.fit` keeps per-row data: row times, slot, the touch cell (level, regime,
  queue bin) on each side, and the market orders as (row, side).
- `qr.Excite` (own, cross, beta per kernel) on `Params.excite`. The flow keeps
  decayed counts of its market orders per kernel and side, adds the excitation to
  the market actions, and samples by thinning against the current rate.
- `Fit.excite(params, kernels, min_events)`: joint maximum likelihood of the
  kernels and one baseline scale per book state with enough market orders (one
  shared scale for the rest), L-BFGS on log parameters with an analytic gradient.
  Returns params with rescaled market rates and the kernels.
- `Fit.residuals(params)`: time-rescaled waits between market orders per side.
- `qr._decay`: the exponential sums by summing the last m events directly when
  that is cheaper than `hawkes.decayed_at`'s overflow buckets.
- `examples/order_memory.py`, a README section and `docs/order_memory.png`.

Checks:

- 369 tests (was 350). Every intermediate commit passes on its own.
- The log-likelihood and its gradient match an O(n^2) brute force with dead rows
  and tied timestamps to 1e-9; residuals match a brute force too.
- Twelve simulated 600-minute feeds with own 0.4, cross 0.1, beta 30 recover
  0.397, 0.0996 and 30.2 (standard errors 0.003, 0.002, 0.4). The market rate
  matches mu / (1 - branching) within 4 standard deviations.
- A zero kernel gives the same tape as no kernel, draw for draw.
- 14 injected bugs: 13 failed a test on the first pass. Residuals that ignored dead
  time survived because the simulated feed has 0.3% dead time; the brute-force
  residual test on a hand-built fit with 15% dead rows catches it.
- `examples/order_memory.py` run twice gives identical output.

Results (CSCO, 2019-12-30):

- One kernel: branching 0.40 at 0.33 ms. Three kernels: 0.51 over 0.20 ms, 5.6 ms
  and 0.60 s. Residual variance 5.09 to 1.40, KS distance 0.418 to 0.048 (5%
  critical value 0.021, so still rejected).
- Sign autocorrelation at lags 1 and 2 matches the tape with three kernels, and the
  10-second Fano factor matches with two (4.31). Mid moves fall from 10.26 to 9.08
  a minute against 11.16.
- Six symbols: adverse moves toward the tape everywhere, closing 75% of the gap on
  INTC and CMCSA, 45% on MSFT, 17% on CSCO and QQQ, and overshooting on AAPL
  (-2.40 against -1.96). Bot volume within 5% of the tape on four symbols.
- The bot's P&L gets worse, not closer: -0.258 cents a share on CSCO against
  -0.117, through the inventory term.

Design notes and what went wrong:

- One scale on the old baseline was the first version. On a simulated feed with a
  flat baseline of 4 it returned 5.9 at queue size one: the old rates already held
  the bursts, because a market order leaves a small queue behind it. Fitting a
  baseline per state fixed it (4.0 to 4.3).
- Then the kernel came back 3.5% low across twelve seeds, five standard errors.
  The same estimator on a plain Hawkes simulation was unbiased, and residuals
  under the true parameters averaged 1.005, so the flow and the likelihood
  disagreed. The flow cannot send a market order on a side whose touch is beyond
  the modelled depth, and that happens right after a market order, when the
  excitation is largest. Integrating the excitation only over time at risk took
  the bias to zero.
- Doing that integral over all 486k CSCO rows took 107 s a fit. Only the rare dead
  rows differ from the whole-window integral, so the code subtracts those: 6 s.
- The brute-force test caught two end-point bugs: a market order on the last row
  counted one unit of excitation mass, and a zero-length dead interval after a
  tied timestamp picked up one too. The second showed up on real data as one
  negative residual.
- On the other symbols some market orders landed in states whose queue-reactive
  rate is zero (sparse cells with no lower bin to borrow from), which makes the
  likelihood minus infinity. Those states now get the level's pooled rate. A
  related crash: the excited flow tried market orders at depths where the tape
  never had one and had no sizes to draw.
- The fitted timescales are sub-millisecond, where `decayed_at` puts one event in
  each overflow bucket and runs a Python loop per event. AAPL took over 20 minutes;
  summing the last m events as vectors brings it to 3.
- I killed two experiment runs: one because I stashed `qr.py` while its workers
  could be importing it, and one that was still running pre-fix code.

Next: let the excitation reach cancels and limit orders at the touch, since moves
are now short of the tape and the excited model overshoots on AAPL, and let a
bot's own market orders excite the flow. Then the fill-weighted kappa for the
flow lean.

## 2026-11-07: Market orders excite the touch

After a market order on CSCO, touch cancels arrive at 2,350 times their average
rate on the hit side and 16,000 times on the other side within 0.1 ms. The flow
now has that reaction, fitted by maximum likelihood, and it matches the tape's
burst to within about a factor of two where the old model was 30 times short. In
sessions it overshoots: mid moves go from 9.1 a minute to 18.2 against 11.2 on
the tape, because the model draws the pull and the repost independently.

Built:

- `qr.fit` records the limit orders and cancels at each side's best queue as
  `Fit.quotes` (row, side, kind).
- `qr.Reaction` (own, cross per kernel, a baseline scale for the touch cells with
  enough events and one shared scale for the rest) on `Excite.react`, keyed by
  kind.
- `Fit.react(params)`: maximum likelihood for cancels and limits at the touch,
  sources are the flow's market orders, betas fixed from `Fit.excite`. With the
  betas fixed the design matrix of decayed counts and the integrated kernels are
  computed once, and the problem is concave in the scales and weights, so
  L-BFGS-B with nonnegative bounds on the raw parameters. 1.5 s on CSCO.
- `Fit.loglik(params, kind)` and `Fit.residuals(params, kind)` for any of the
  three kinds; the market path became `_path(params, kind)`.
- `QueueReactive` adds the excitation to touch cancels and limits and scales
  their baselines; the thinning step covers all excited actions.
- `qr.messages(params, minutes)`: the flow written out as ITCH messages,
  including partial cancels as X messages (the test helper it replaces dropped
  them). Fitting the model's own output with `qr.fit` gives the event study on
  model and tape with the same code.
- `examples/cancel_race.py`, a README section and `docs/cancel_race.png`.

Checks:

- 384 tests (was 369). Every intermediate commit passes on its own.
- `loglik` matches an O(n^2) brute force for markets and touch cancels, with dead
  rows, tied timestamps and per-cell scales, to 1e-9.
- At the fitted reaction, scaling every parameter by 1 +- 0.001 lowers the
  likelihood and the slope is zero to 1e-4 of the event count (at the maximum the
  compensator equals the number of events).
- Replaying `qr.messages` through `ReplayBook` rebuilds the flow's book order for
  order, partial cancels included.
- A zero reaction gives the same tape as no reaction, draw for draw. Feeds where
  quotes ignore trades give own + cross under 0.02.
- Twelve 600-minute simulated feeds with cancels (0.3, 0.05) and limits
  (0.2, 0.1) recover 0.295, 0.0445, 0.1995, 0.1017 (se about 0.003, 0.0016,
  0.004, 0.004). Touch baselines come back within 7% at level 0. The cancel cross
  term is 11% low, 3.4 standard errors. Fitting own alone (0.297) or cross alone
  (0.305) is unbiased, apart from the usual positive bias of the other term at
  its zero bound (0.003 to 0.005). I did not find where the joint fit loses it.

Results (CSCO, 2019-12-30):

- Per market order: 3.6 cancels and 2.8 limits on the hit side's touch, 10.7 and
  16.3 on the other side, the other side's limits almost all inside 0.2 ms. That
  is 44% of touch cancels and 55% of touch limits. Replace messages are 27k of
  250k removals, so this is plain delete-and-add.
- Residual variance for touch cancels 9.38 -> 4.25 (KS 0.541 -> 0.362), limits
  15.25 -> 6.51 (0.614 -> 0.434). Log-likelihood gain 1.65 and 2.52 per event.
- Sessions: P&L -0.088 against -0.117 on the tape (was -0.258), inventory term
  0.003 against 0.096. Moves 18.16 against 11.16, eta 0.32 against 0.60, adverse
  -0.615 against -0.727 (was -0.666).
- Six symbols: moves go from short to far over on CSCO, INTC and CMCSA, closer on
  MSFT and AAPL, further on QQQ. The AAPL adverse overshoot grows to -3.03
  against -1.96.

Design notes and what went wrong:

- The first recovery test compared the fitted baseline scale with 1 and got 0.83.
  The queue-reactive rates were fitted on the excited feed, so they already
  contain the average excitation; the right check is scale times fitted rate
  against the true rate, cell by cell.
- The cell scales only apply at the touch. A cell (level, regime, queue) can be
  the touch in one row and behind it in another (level 1 is the touch only when
  the spread is two ticks), so rescaling the shared rate table would have changed
  cancels deeper in the book too.
- Dead cells are decided on the table rate before the time-of-day clock, so a slot
  with no cancels does not make the fit and the simulator disagree about where an
  excited cancel can happen.
- The overshoot is the main finding. On the tape a reaction mostly pulls a quote
  and puts it back, so the queue size barely moves. Two independent streams with
  their own size distributions empty thin touches and refill them a tick away.

Next: model the reaction as a paired event, a pull and a repost of the same size
drawn together, and check that it keeps the event study while bringing moves and
eta back. Then let bot market orders excite the flow, and the fill-weighted kappa.

## 2026-11-08: Microprice and order-flow imbalance

Two signals from the microstructure literature, measured on the replayed tape
out of sample, and a quoter that uses the first. Quoting around Stoikov's
microprice instead of the mid takes the touch quoter's adverse selection on CSCO
from -0.727 cents a share to +0.029, at the cost of 89% of its volume.

Built:

- `micro.Tops` and `micro.tops(msgs)`: the top of book (time, bid, ask, sizes)
  after every batch that changes it, from the same `ReplayBook`. 305k rows for
  CSCO; the six symbols take 25 s and are cached in `data/<SYMBOL>.tops.npz`.
- `micro.fit(top, n_imb, n_spread)`: Stoikov (2018). States are spread x
  imbalance bucket, one step per top-of-book change, data symmetrized, then
  G1 = (1 - Q)^-1 T K and B = (1 - Q)^-1 R. `Chain.gstar` sums G1 + B G1 + ...
  with the fundamental matrix, (1 - B + 1 pi)^-1 G1, which is exact once
  pi G1 = 0; `Chain.series` is the truncated sum for comparison.
- `micro.spreads(top, end)`: the smallest number of spread states covering 95%
  of the time (4 for AAPL, 2 elsewhere).
- `Microprice.lean(ctx)` and a `Signal` protocol in `bots.py`, so a quoter's
  fair value is mid plus whatever the signal says. `FlowLean` got `lean` too.
- `ofi.py`: Cont-Kukanov-Stoikov e_n, cumulative flow at any time, binned
  OFI / mid change / time-weighted depth, OLS with White or clustered standard
  errors, per-window fits and the log-log depth scaling.
- `examples/microprice.py`, `examples/ofi.py`, `examples/micro_quotes.py`, three
  charts and three README sections.

Checks:

- 408 tests (was 384); every commit passes on its own.
- Two two-state chains against closed forms: g* = (u - d) / (1 - stay + swap)
  when the post-move state is uniform, and that divided by 1 - rho with a
  persistent imbalance. The fundamental-matrix sum matches 200 terms of the
  series.
- A random symmetric 5-state chain: g* against the mean 150-step drift of 40,000
  simulated paths per starting state, all within 4 standard errors.
- `fit` on 200,000 quotes simulated from a known 3-state chain: 0.0559 against
  the true 0.0551.
- The imbalance bucket rule is mirror-symmetric (b(1 - I) = n - 1 - b(I)) and
  needs an odd n, so I = 0.5 maps to the centre.
- `lean` leaves the bot's own shares out: a 40-share bot bid on a 5 against 5
  book still reads as balanced.
- OFI: each of the six cases by hand, telescoping with fixed prices, a mirrored
  book has negated flow, the depth integral against a 200k-point Riemann sum.
  Clustered standard errors on rows repeated 9 times equal the plain White ones.
- The mid-quoter rows of the quoting experiment reproduce the earlier six-symbol
  table exactly (CSCO -0.117, 11,550 shares).

Results (2019-12-30, fit on one half of the day, tested on the other):

- CSCO g* runs from -0.383 to +0.383 ticks at one tick spread; the weighted mid
  would say +-0.45. Four of the other five end within 0.37 to 0.39.
- Out-of-sample R^2 is a few percent at 10 s and after two mid moves, and the
  microprice is about as good as a straight line in imbalance. The weighted mid
  is much worse on AAPL (-0.277 after two moves).
- Calibration: slope of the realized change after k moves on g* is about 1 on
  QQQ (1.07, 0.96, 0.92 at k = 2, 5, 20) and MSFT (0.89, 0.88, 0.94), but 1.5 to
  1.75 on INTC and CMCSA at k = 2 to 5 and 0.4 to 0.6 on AAPL.
- OFI explains 73 to 84% of 10 s mid changes in 30-minute windows; impact
  scales as depth^-1.13 (se 0.02) across 72 windows; beta x depth 0.21 to 0.34.
  Scaling by depth turns CSCO's afternoon R^2 from -0.49 to 0.79. Lagged OFI
  predicts nothing.
- Quoting: adverse term on the large-tick names from about -0.7 to between
  +0.03 and -0.39, volume down 86 to 90%, dollars better on five of six
  (MSFT -149 to +9 a session). Per-share P&L standard errors (0.3 to 0.76)
  are as large as the gains.

Design notes and what went wrong:

- The first test of `adjust` failed because my hand-made quotes put the up move
  after the low-imbalance state; the estimator was right.
- With an even bucket count a balanced book falls on a bucket edge. On AAPL 11%
  of updates have equal sizes, and they would all have taken the positive drift
  of the upper bucket.
- Stoikov's notebook ignores spread changes inside Q (it is block diagonal by
  spread). Here Q is indexed by the full state, and transitions into states never
  seen as a starting point are dropped and the rows renormalized.
- The first quoting run fitted AAPL with two spread states, so the signal was
  off whenever its spread was 3 or 4 ticks; `spreads` fixed it.
- One-second samples overlap heavily at long horizons, so the calibration
  standard errors are clustered by 30-minute block. White ones were 2 to 6 times
  smaller.
- A morning OFI beta is worse than nothing in the afternoon on CSCO because its
  depth goes from 2,124 to 3,671 shares. The depth scaling in the paper is what
  makes the relation portable.

Next: put the drift into the quotes properly (Cartea, Jaimungal and Ricci skew
the reservation price by the expected drift instead of the hard one-tick step
used here), give the microprice chain a memory of the last move to fix the
large-tick calibration, and measure how much of each signal survives the latency
legs.

## 2026-11-08: A market maker that prices its alpha, and how fast the alpha goes stale

The microprice quoter from this morning used a hard rule (step back a tick when
g* points against you). This entry solves for the rule instead, with the
Cartea-Jaimungal-Penalva market maker with short-term alpha, calibrates it from
the replay's own fills, and then measures how much of the signal a slower bot
still gets.

Built:

- `alpha.Model` and `alpha.solve(model, T, policy=None)`: the at-the-touch
  dynamic programming equation of CJP section 10.4.2 (after Cartea, Jaimungal and
  Ricci 2014) on an (alpha, q) grid, 201 x 21 by default. alpha follows
  d alpha = -zeta alpha dt + eta dB + eps (dM+ - dM-); each side is posted or not;
  a posted order fills on the next market order on its side (rate lambda).
  One term added to the book's model: the mid jumps `impact` against the bot at
  each fill, so the posting gain is spread/2 - impact + h(alpha +- eps, q -+ 1) -
  h(alpha +- eps, q). With `policy` given it evaluates that policy instead of
  maximizing.
- Time stepping: drift and diffusion in alpha implicit (upwind, a tridiagonal
  M-matrix solved with `solve_banded`), the jumps and the max explicit with
  lambda dt <= 0.05. Monotone, so it converges to the viscosity solution. The
  first, fully explicit version needed zeta width / d alpha in the step rate,
  which for QQQ (zeta = 338 a minute) meant over a million steps; now 69k steps
  and 10 s. Jumps that land between grid nodes are interpolated linearly, and
  linearly extrapolated past the edge.
- `alpha.simulate`: Monte Carlo of the model under a stored policy, exact OU
  transitions between Poisson arrivals and the conditional mean of the running
  integral (unbiased by the tower property).
- Calibration: `responses` (slope of the mid change h later on a signal),
  `decay` (fit c (1 - e^-zeta h) with 1/h weights), `implied_eta` (from
  Var alpha = (eta^2 + 2 lambda eps^2) / (2 zeta)) and `fill_fit` (impact = mid
  move against the fill within delta, net of the pre-fill alpha's prediction;
  eps = alpha at delta minus the decayed pre-fill alpha).
- `bots.AlphaTouch`: joins the best bid and ask as the policy says, with alpha =
  scale x the signal's lean and q = position in lots.
- `examples/alpha_quotes.py` and `examples/signal_latency.py`, two charts, two
  README sections.

Checks:

- 432 tests (was 408); each commit passes its own tests in a clean worktree.
- Always-post closed form: h(0, a, q) = lambda spread T + a q (1 - e^-zeta T)/zeta
  - (2 lambda eps/zeta)(T - (1 - e^-zeta T)/zeta) - (spread/2) E|q + K| with K
  Skellam(lambda T, lambda T). I first left out the terminal Skellam term and
  the solver looked 1.8 off; the Monte Carlo agreed with the solver, not with me.
  Now within 0.02 at three (a, q) points.
- Policy evaluation against 80k simulated paths: gaps 0.049, 0.020, 0.007 at 30,
  60, 120 nodes per side (Monte Carlo se 0.009). Optimal value >= any fixed
  policy's value everywhere; mirror symmetry; ask posting monotone in alpha;
  impact costs exactly 2 lambda T impact when always posting; with no alpha the
  bot posts iff spread/2 > impact.
- Policy at 20 minutes to go equals the one at 40 for the CSCO, MSFT and QQQ
  fits; doubling the grid moves the q = 0 thresholds by under 0.002.
- Four injected bugs (jump sign, impact left out of the solver, impact left out
  of the simulator, a shifted band in the implicit matrix) each fail a test.
- Estimators: `decay` recovers (1.3, 25) exactly from a clean curve and (1, 30)
  within 5% and 10% from a simulated OU drift; `implied_eta` within 5% from a
  simulated jump-OU; `fill_fit` recovers impact and eps from synthetic fills.
- The mid and micro rows of both experiments reproduce the earlier tables
  exactly, and the zero-latency mid rows match the latency table.

Results (2019-12-30, each half's sessions use parameters fitted on the other):

- Signal half-lives from 0.12 s (QQQ morning) to 4.8 s (CSCO afternoon); c is
  about 1 on CSCO, INTC and CMCSA, 0.7 to 0.85 on MSFT and QQQ, 0.41 to 0.56 on
  AAPL, which matches the calibration slopes found this morning.
- Fill impact 0.40 to 0.67 ticks on the one-tick names, 1.5 to 1.9 on AAPL. On
  CSCO, INTC and AAPL it exceeds half the spread, so the flat model never posts.
- CJP with alpha against the microprice rule, $ per session: CSCO 4.2 vs 7.6,
  INTC 1.7 vs 12.2, CMCSA 0.7 vs 14.5, MSFT -0.9 vs 8.5 (all inside 2 se), QQQ
  4.0 vs -7.0, AAPL 0 (no trades) vs -553.8.
- At book changes, 1.5 to 3% of the 10 s microprice move happens within 10 us,
  12 to 21% within 100 us and 31 to 48% within 1 ms. Sampled once a second, at
  least 99% is still ahead at 1 ms.
- Summed over the five large-tick names, the microprice quoter goes from +35.7
  a session at zero latency to -64.8 at 1 ms and -152.0 at 10 ms; the CJP bot
  from +9.8 to -136.9 and -188.5. Both trade more when slow (QQQ CJP 8,688 ->
  54,901 shares) because their pulls arrive after the fills.

Design notes and what went wrong:

- The textbook model has no instant price move at a fill. Markouts of an
  always-post bot on CSCO are 0.28 ticks at the fill's own timestamp and 0.75
  half a second later, then flat for a minute; the tick-through comes in the
  next ITCH batch. Without `impact` the model saw a 0.27-tick edge where the
  replay has a loss.
- Fitting impact and eps jointly on the markout curve is collinear once
  (1 - e^-zeta h)/zeta is flat over the horizons, which it is for half-lives
  under a second. AAPL came out with impact -0.55 and eps +502. Reading eps off
  the signal itself, one second after the fill, is always identified.
- eps comes out negative on 9 of 12 fits: after the tick-through the new touch
  is thin and the microprice expects some of the move back.
- A single exponential is a rough fit to the response curves, which keep rising
  slowly after the fast part (a mix of time scales). The 1/h weights fit the fast
  part, which is what the quotes act on.
- Why CJP trails the hard rule on the large-tick names: it can only post at the
  touch or pull, and a pulled order rejoins at the back. The rule steps one tick
  back and keeps a place in that queue, and its fills are less toxic (INTC
  adverse -0.39 vs -0.51). lambda and impact also come from a bot that never
  pulls.

Next: depth as a control (touch or one tick back) with fill rate and impact
calibrated per depth from the bot's own fills, a microprice with memory of the
last move, and the Gymnasium environment for an RL market maker.

## 2026-11-08: Memory in the microprice chain

The microprice is miscalibrated on the large-tick names (the mid travels 1.5 to
1.75 times g* on INTC and CMCSA, about 0.5 times on AAPL). The obvious guess was
that the chain needs to know which way the mid last moved. It does not help.

Built:

- `micro.last_move(top, ttl)`: 0 (down), 1 (neither) or 2 (up) for the last mid
  move, forgotten after ttl top-of-book updates.
- `fit(..., memory=ttl)`: states become (spread, imbalance bucket, flag), the
  mirror maps down to up, and `Microprice.adjust` takes the flag.
- `examples/micro_memory.py`: four memory lengths against the plain chain,
  calibration slopes with 30-minute clustered errors, out-of-sample R^2 at 10 s
  and two moves, and a linear baseline in imbalance and last-move sign.

Checks:

- 436 tests. A 6-state chain with momentum built into the kernel, simulated
  for 300,000 steps: the memory fit's g* is within 0.03 of the chain's closed
  form, and the momentum is large (0.2 ticks between the up and down states).
  Swapping the mirror's flag order or the flag itself fails these tests.
- The real-data gap below survives fitting without the mirror.

Results (fitted on one half, tested on the other):

- A permanent flag makes the chain believe in momentum (+0.20 ticks for a
  balanced CSCO book right after an up move) that the tape does not have: from
  those states the next move averages +0.04 ticks after a down move, against a
  chain prediction of -0.13. Out of sample its slopes fall to 0.3 to 0.9 and its
  R^2 goes negative on CSCO and INTC at 10 s.
- Remembering only the move into the current state keeps the chain Markov; INTC
  and CMCSA slopes at two moves go from 1.63 and 1.46 to 1.37 and 1.25, R^2 is
  unchanged within 0.01.
- Ten updates brings INTC and CMCSA to 0.95 and 0.83 while pulling MSFT from 0.89
  to 0.69. No version gains more than 0.02 of R^2 anywhere.

Design notes: a flag that lasts longer than one transition is hidden state, so
the fitted chain averages states seen just after a move with the same states
seen long after and then propagates that average forward. That is what produced
the momentum. The remaining miscalibration looks like a scale problem rather
than a missing direction: the slope is above one on the names with deep queues
relative to their trading and below one on AAPL.

Next: condition the chain on time since the last move or on the Hawkes intensity
of recent market orders instead of the direction, and see whether either moves
the slopes without hurting MSFT and QQQ.
