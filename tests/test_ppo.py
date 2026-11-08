import numpy as np
import pytest

from mmsim.ppo import PPO, Config, Policy, gae, gradients, log_softmax, rollout, train


def test_log_softmax_normalizes_and_is_shift_free():
    z = np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]])
    lp = log_softmax(z)
    assert np.allclose(np.exp(lp).sum(axis=1), 1)
    assert np.allclose(lp[0], log_softmax(z[0] - 1000))
    assert np.allclose(lp[1], -np.log(3))


def test_gae_limits():
    rew = np.array([1.0, -2.0, 0.5, 3.0])
    val = np.array([0.3, -0.1, 0.8, 1.2])
    adv, ret = gae(rew, val, 0.0, 1.0, 1.0)
    assert np.allclose(ret, np.cumsum(rew[::-1])[::-1])
    assert np.allclose(adv, ret - val)
    adv, _ = gae(rew, val, 2.0, 0.9, 0.0)
    assert np.allclose(adv, rew + 0.9 * np.append(val[1:], 2.0) - val)


def test_gae_matches_weighted_sum_of_n_step_advantages():
    rng = np.random.default_rng(0)
    rew, val = rng.standard_normal(6), rng.standard_normal(6)
    g, lam, last = 0.95, 0.7, 0.4
    v = np.append(val, last)
    n = len(rew)

    def n_step(t, k):
        k = min(k, n - t)
        return sum(g**j * rew[t + j] for j in range(k)) + g**k * v[t + k] - val[t]

    want = [
        (1 - lam) * sum(lam ** (k - 1) * n_step(t, k) for k in range(1, n - t))
        + lam ** (n - t - 1) * n_step(t, n - t)
        for t in range(n)
    ]
    adv, _ = gae(rew, val, last, g, lam)
    assert np.allclose(adv, want)


def _loss(policy, obs, act, logp_old, adv, ret, clip, ent):
    lp = log_softmax(policy.pi(obs))
    p = np.exp(lp)
    ratio = np.exp(lp[np.arange(len(act)), act] - logp_old)
    surr = np.minimum(ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv)
    H = -(p * lp).sum(axis=1)
    return -surr.mean() - ent * H.mean(), 0.5 * ((policy.v(obs)[:, 0] - ret) ** 2).mean()


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    policy = Policy(4, 3, (6,), rng)
    for net in (policy.pi, policy.v):
        net.W[-1] *= 30
    n = 9
    obs = rng.standard_normal((n, 4))
    act = rng.integers(0, 3, n)
    lp = log_softmax(policy.pi(obs))[np.arange(n), act]
    logp_old = lp + np.array([0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0.05, -0.05])
    adv = np.array([1.0, -1.0, 0.5, 1.0, -1.0, 1.0, -1.0, 2.0, -2.0])
    ret = rng.standard_normal(n)
    clip, ent = 0.2, 0.05

    stats, g_pi, g_v = gradients(policy, obs, act, logp_old, adv, ret, clip, ent)
    lpi, lv = _loss(policy, obs, act, logp_old, adv, ret, clip, ent)
    assert stats.policy == pytest.approx(lpi)
    assert stats.value == pytest.approx(lv)
    assert stats.clipped == pytest.approx(4 / 9)

    eps = 1e-6
    for k, (net, grads) in enumerate(((policy.pi, g_pi), (policy.v, g_v))):
        for p, g in zip(net.params, grads, strict=True):
            for i in np.ndindex(p.shape):
                keep = p[i]
                p[i] = keep + eps
                up = _loss(policy, obs, act, logp_old, adv, ret, clip, ent)[k]
                p[i] = keep - eps
                down = _loss(policy, obs, act, logp_old, adv, ret, clip, ent)[k]
                p[i] = keep
                assert g[i] == pytest.approx((up - down) / (2 * eps), rel=1e-4, abs=1e-8)


class Bandit:
    def reset(self, *, seed=None):
        self.rng = np.random.default_rng(seed)
        self.n = 0
        return self._obs(), {}

    def _obs(self):
        self.ctx = int(self.rng.integers(2))
        return np.array([1.0 - 2 * self.ctx, 1.0])

    def step(self, a):
        r = 1.0 if a == self.ctx else -0.5 if a == 2 else 0.0
        self.n += 1
        return self._obs(), r + self.rng.normal(0, 0.3), self.n == 8, False, {}


class Delayed:
    def reset(self, *, seed=None):
        self.n, self.first = 0, 0.0
        return np.array([0.0, 0.0]), {}

    def step(self, a):
        if self.n == 0:
            self.first = 1.0 if a == 1 else -1.0
        self.n += 1
        r = self.first if self.n == 6 else 0.0
        return np.array([self.n / 6, self.first]), r, self.n == 6, False, {}


def test_ppo_learns_a_contextual_bandit():
    cfg = Config(episodes=16, batch=64, lr=3e-3, hidden=(16,))
    agent, curve = train(Bandit, 2, 3, 40, cfg, seed=0)
    pol = agent.policy
    assert pol.greedy(np.array([1.0, 1.0])) == 0
    assert pol.greedy(np.array([-1.0, 1.0])) == 1
    assert np.mean(curve[-5:]) > 7.0 > np.mean(curve[:3]) + 3


def test_ppo_assigns_credit_to_an_early_action():
    cfg = Config(episodes=16, batch=48, lr=3e-3, hidden=(16,))
    agent, _ = train(Delayed, 2, 2, 30, cfg, seed=2)
    assert agent.policy.probs(np.array([0.0, 0.0]))[1] > 0.9
    assert agent.policy.value(np.array([0.5, 1.0])) > agent.policy.value(np.array([0.5, -1.0]))


def test_training_is_reproducible():
    cfg = Config(episodes=4, batch=16, hidden=(8,))
    a, ca = train(Bandit, 2, 3, 3, cfg, seed=5)
    b, cb = train(Bandit, 2, 3, 3, cfg, seed=5)
    assert ca == cb
    assert all(
        np.array_equal(x, y) for x, y in zip(a.policy.pi.params, b.policy.pi.params, strict=True)
    )


class Truncated(Delayed):
    def step(self, a):
        o, r, _, _, info = super().step(a)
        return o, r, False, self.n == 3, info


def test_rollout_bootstraps_only_on_truncation():
    pol = PPO(2, 2, Config(hidden=(4,)), seed=0).policy
    done = rollout(Delayed(), pol, 0)
    cut = rollout(Truncated(), pol, 0)
    assert done.last == 0.0 and len(done.rew) == 6
    assert len(cut.rew) == 3
    assert cut.last == pol.value(np.array([0.5, 1.0 if cut.act[0] == 1 else -1.0]))
