"""プロット・集計のスモークテスト。"""

import matplotlib

matplotlib.use("Agg")  # ヘッドレス環境用

import numpy as np
from matplotlib.figure import Figure

from onlopt.evaluation import aggregate_curves, plot_regret


def make_agg(T=100, n=5, seed=0):
    rng = np.random.default_rng(seed)
    base = np.sqrt(np.arange(1, T + 1))
    curves = base[None, :] * (1 + 0.1 * rng.standard_normal((n, 1)))
    return aggregate_curves(curves)


def test_plot_single_curve_returns_figure():
    fig = plot_regret(make_agg())
    assert isinstance(fig, Figure)


def test_plot_multiple_with_references_loglog():
    curves = {"hedge": make_agg(seed=0), "tsallis": make_agg(seed=1)}
    fig = plot_regret(curves, loglog=True, references=("sqrt", "log"))
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    # 2 曲線 + 2 参照線
    assert len(ax.lines) == 4


def test_bootstrap_aggregation():
    rng = np.random.default_rng(0)
    curves = rng.random((20, 50)).cumsum(axis=1)
    agg = aggregate_curves(curves, method="bootstrap", confidence=0.9)
    assert np.all(agg.lower <= agg.mean + 1e-12)
    assert np.all(agg.mean <= agg.upper + 1e-12)
    assert agg.method == "bootstrap0.9"
