"""正則化項の argmin の正当性(KKT・総当たり比較)。"""

import numpy as np
import pytest

from onlopt.learners import L2, NegativeEntropy, TsallisEntropy
from onlopt.utils import make_rng


def brute_force_argmin(reg, theta, n_grid=200):
    """2次元単体上のグリッド探索による近似解(検算用)。"""
    ps = np.linspace(1e-6, 1 - 1e-6, n_grid)
    best_val, best_x = np.inf, None
    for p in ps:
        x = np.array([p, 1 - p])
        val = reg.value(x) - theta @ x
        if val < best_val:
            best_val, best_x = val, x
    return best_x


@pytest.mark.parametrize(
    "reg", [NegativeEntropy(), TsallisEntropy(0.5), TsallisEntropy(0.3), L2()]
)
class TestArgminLinear:
    def test_feasibility(self, reg):
        rng = make_rng(0)
        for _ in range(30):
            theta = rng.normal(size=rng.integers(2, 20)) * 5
            x = reg.argmin_linear(theta)
            assert np.all(x >= -1e-12)
            assert np.isclose(x.sum(), 1.0)

    def test_matches_brute_force_2d(self, reg):
        rng = make_rng(1)
        for _ in range(10):
            theta = rng.normal(size=2) * 3
            x = reg.argmin_linear(theta)
            xb = brute_force_argmin(reg, theta)
            assert np.allclose(x, xb, atol=0.02)

    def test_optimality_vs_random_points(self, reg):
        """射影点の目的値がランダム実行可能点以下であること。"""
        rng = make_rng(2)
        theta = rng.normal(size=6)
        x = reg.argmin_linear(theta)
        fx = reg.value(x) - theta @ x
        for _ in range(200):
            y = rng.dirichlet(np.ones(6))
            fy = reg.value(y) - theta @ y
            assert fx <= fy + 1e-8


class TestSpecificForms:
    def test_negative_entropy_is_softmax(self):
        theta = np.array([1.0, 2.0, -0.5])
        x = NegativeEntropy().argmin_linear(theta)
        # R(x) = sum x log x に対する閉形式は softmax(theta - 1) = softmax(theta)
        w = np.exp(theta - theta.max())
        assert np.allclose(x, w / w.sum())

    def test_tsallis_kkt(self):
        """KKT: x_i = kappa (z - theta_i)^{-1/(1-a)} かつ sum = 1。"""
        reg = TsallisEntropy(0.5)
        rng = make_rng(3)
        theta = rng.normal(size=5) * 2
        x = reg.argmin_linear(theta)
        # 停留条件 grad R(x) - theta = -z * 1(全成分が等しい)
        station = reg.grad(x) - theta
        assert np.allclose(station, station[0], atol=1e-6)

    def test_ftrl_argmin_uniform_at_zero_loss(self):
        for reg in [NegativeEntropy(), TsallisEntropy(0.5), L2()]:
            x = reg.argmin(np.zeros(4), eta=1.0)
            assert np.allclose(x, 0.25, atol=1e-8)

    def test_tsallis_invalid_alpha(self):
        with pytest.raises(ValueError):
            TsallisEntropy(1.0)
        with pytest.raises(ValueError):
            TsallisEntropy(0.0)
