import numpy as np
import pytest

from mmsim.nn import MLP, Adam, clip_norm, orthogonal


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (5, 5)])
def test_orthogonal_init(shape):
    W = orthogonal(*shape, 2.0, np.random.default_rng(0))
    small = min(shape)
    gram = W.T @ W if shape[0] >= shape[1] else W @ W.T
    assert W.shape == shape
    assert np.allclose(gram, 4.0 * np.eye(small))


def test_forward_matches_call():
    net = MLP([3, 8, 8, 2], np.random.default_rng(1))
    x = np.random.default_rng(2).standard_normal((5, 3))
    out, _ = net.forward(x)
    assert np.allclose(out, net(x))
    assert net(x[0]).shape == (2,)


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(3)
    net = MLP([3, 7, 5, 2], rng, head=0.5)
    for b in net.b:
        b += rng.standard_normal(b.shape)
    x = rng.standard_normal((4, 3))
    c = rng.standard_normal((4, 2))

    def loss():
        return float((net(x) ** 2 * c).sum())

    out, cache = net.forward(x)
    grads = net.backward(cache, 2 * out * c)
    eps = 1e-6
    for p, g in zip(net.params, grads, strict=True):
        num = np.zeros_like(p)
        for i in np.ndindex(p.shape):
            keep = p[i]
            p[i] = keep + eps
            up = loss()
            p[i] = keep - eps
            down = loss()
            p[i] = keep
            num[i] = (up - down) / (2 * eps)
        assert np.allclose(g, num, rtol=1e-5, atol=1e-7)


def test_adam_first_step_is_lr_times_sign():
    p = [np.array([1.0, -2.0, 3.0])]
    Adam(p, 0.1, eps=0.0).step(p, [np.array([5.0, -1e-3, 0.2])])
    assert np.allclose(p[0], [0.9, -1.9, 2.9])


def test_adam_minimizes_a_quadratic():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    target = np.array([1.0, -1.0])
    x = [np.zeros(2)]
    opt = Adam(x, 0.05)
    for _ in range(2000):
        opt.step(x, [A @ (x[0] - target)])
    assert np.allclose(x[0], target, atol=1e-3)


def test_clip_norm_rescales_only_above_the_limit():
    g = [np.array([3.0]), np.array([[4.0]])]
    assert clip_norm(g, 10.0) == 5.0
    assert g[0][0] == 3.0
    assert clip_norm(g, 1.0) == 5.0
    assert np.isclose(np.sqrt(g[0][0] ** 2 + g[1][0, 0] ** 2), 1.0)
    assert np.isclose(g[0][0] / g[1][0, 0], 0.75)
