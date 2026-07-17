"""学習率スケジュールの値のテスト。"""

import math

import pytest

from onlopt.learners import DataDependentLR, FixedLR, InverseSqrtLR


def test_fixed():
    lr = FixedLR(0.3)
    assert lr.eta(0, {}) == 0.3
    assert lr.eta(1000, {"cum_sq_est_norm": 99.0}) == 0.3


def test_inverse_sqrt():
    lr = InverseSqrtLR(c=2.0)
    assert lr.eta(0, {}) == pytest.approx(2.0)
    assert lr.eta(3, {}) == pytest.approx(1.0)
    assert lr.eta(99, {}) == pytest.approx(0.2)


def test_data_dependent():
    lr = DataDependentLR(c=1.0)
    assert lr.eta(0, {}) == pytest.approx(1.0)
    assert lr.eta(5, {"cum_sq_est_norm": 3.0}) == pytest.approx(0.5)
    assert lr.eta(5, {"cum_sq_est_norm": 99.0}) == pytest.approx(0.1)
    custom = DataDependentLR(c=2.0, key="my_stat")
    assert custom.eta(0, {"my_stat": 15.0}) == pytest.approx(0.5)


def test_positive_validation():
    with pytest.raises(ValueError):
        FixedLR(0.0)
    with pytest.raises(ValueError):
        InverseSqrtLR(-1.0)
    with pytest.raises(ValueError):
        DataDependentLR(0.0)


def test_monotone_decreasing():
    lr = InverseSqrtLR(1.0)
    vals = [lr.eta(t, {}) for t in range(100)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[0] == pytest.approx(1.0)
    assert vals[99] == pytest.approx(1.0 / math.sqrt(100))
