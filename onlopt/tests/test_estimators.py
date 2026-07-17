"""損失推定の不偏性テスト(モンテカルロ)。"""

import numpy as np
import pytest

from onlopt.core import Feedback
from onlopt.learners import IdentityEstimator, ImportanceWeighted
from onlopt.utils import make_rng


def test_identity_returns_full_loss():
    est = IdentityEstimator()
    loss = np.array([0.1, 0.5, 0.9])
    fb = Feedback(action=1, loss=0.5, full_loss=loss)
    assert np.allclose(est.estimate(fb, np.ones(3) / 3, 3), loss)


def test_identity_requires_full_info():
    with pytest.raises(ValueError):
        IdentityEstimator().estimate(Feedback(action=0, loss=0.5), np.ones(2) / 2, 2)


def test_importance_weighted_unbiased():
    """E[hat{l}] = l をモンテカルロで確認する。"""
    rng = make_rng(0)
    K = 5
    dist = np.array([0.4, 0.25, 0.15, 0.15, 0.05])
    true_loss = np.array([0.2, 0.9, 0.5, 0.0, 1.0])
    est = ImportanceWeighted()

    n = 200_000
    acc = np.zeros(K)
    arms = rng.choice(K, size=n, p=dist)
    for a in arms:
        fb = Feedback(action=int(a), loss=float(true_loss[a]))
        acc += est.estimate(fb, dist, K)
    mean_est = acc / n
    # 標準誤差の数倍以内で一致すること
    assert np.allclose(mean_est, true_loss, atol=0.05)


def test_importance_weighted_support():
    est = ImportanceWeighted()
    dist = np.array([0.5, 0.5])
    out = est.estimate(Feedback(action=0, loss=0.8), dist, 2)
    assert out[0] == pytest.approx(1.6)
    assert out[1] == 0.0
