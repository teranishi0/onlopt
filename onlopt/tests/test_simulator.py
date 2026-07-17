"""Simulator・環境・再現性のテスト。"""

import numpy as np
import pytest

from onlopt import (
    CorruptedStochasticEnv,
    FeedbackType,
    Hedge,
    LoadBalancingEnv,
    ObliviousAdversary,
    Simulator,
    StochasticEnv,
    TsallisINF,
)
from onlopt.environments.load_balancing import uniform_jobs
from onlopt.evaluation import cumulative_regret
from onlopt.experiments import ExperimentConfig, run_many


def test_same_seed_reproducible():
    sim = Simulator()
    env_f = lambda: StochasticEnv.from_gap(5, gap=0.2)
    r1 = sim.run(Hedge(5, horizon=500), env_f(), T=500, seed=42)
    r2 = sim.run(Hedge(5, horizon=500), env_f(), T=500, seed=42)
    assert np.array_equal(r1.losses, r2.losses)
    assert np.array_equal(r1.cum_regret, r2.cum_regret)
    assert r1.actions == r2.actions


def test_different_seed_differs():
    sim = Simulator()
    env = StochasticEnv.from_gap(5, gap=0.2)
    r1 = sim.run(Hedge(5, horizon=500), env, T=500, seed=1)
    r2 = sim.run(Hedge(5, horizon=500), env, T=500, seed=2)
    assert not np.array_equal(r1.losses, r2.losses)


def test_reset_between_runs():
    """同一インスタンスの再利用でも reset により結果が一致すること。"""
    sim = Simulator()
    learner = Hedge(4, horizon=300)
    env = StochasticEnv.from_gap(4, gap=0.3)
    r1 = sim.run(learner, env, T=300, seed=7)
    r2 = sim.run(learner, env, T=300, seed=7)
    assert np.array_equal(r1.cum_regret, r2.cum_regret)


def test_regret_matches_posthoc_computation():
    """ストリーミング計算と事後計算(evaluation.regret)の一致。"""
    T, K = 200, 4
    rng = np.random.default_rng(0)
    matrix = rng.random((T, K))
    env = ObliviousAdversary.from_matrix(matrix)
    sim = Simulator()
    res = sim.run(Hedge(K, horizon=T), env, T=T, seed=3)
    incurred = np.array(
        [matrix[t, a] for t, a in enumerate(res.actions)]
    )
    assert np.allclose(res.losses, incurred)
    assert np.allclose(res.cum_regret, cumulative_regret(incurred, matrix))


def test_pseudo_regret_only_for_stochastic():
    sim = Simulator()
    res_sto = sim.run(
        Hedge(3, horizon=100), StochasticEnv.from_gap(3, 0.2), T=100, seed=0
    )
    assert res_sto.cum_pseudo_regret is not None
    assert res_sto.cum_pseudo_regret.shape == (100,)
    assert np.all(np.diff(res_sto.cum_pseudo_regret) >= -1e-12)  # 単調非減少

    adv = ObliviousAdversary(3, lambda t, rng: rng.random(3))
    res_adv = sim.run(Hedge(3, horizon=100), adv, T=100, seed=0)
    assert res_adv.cum_pseudo_regret is None


def test_out_of_range_loss_warns():
    env = ObliviousAdversary(2, lambda t, rng: np.array([1.5, -0.2]))
    with pytest.warns(UserWarning, match="outside"):
        Simulator().run(Hedge(2, horizon=10), env, T=10, seed=0)


def test_bandit_feedback_masked():
    """bandit 学習者には full_loss が渡らないこと。"""

    class SpyTsallis(TsallisINF):
        def update(self, feedback):
            assert feedback.full_loss is None
            assert isinstance(feedback.loss, float)
            super().update(feedback)

    Simulator().run(
        SpyTsallis(4), StochasticEnv.from_gap(4, 0.2), T=50, seed=0
    )


def test_corruption_budget_recorded():
    base = StochasticEnv.from_gap(3, gap=0.3)
    env = CorruptedStochasticEnv(base, budget=20.0)
    res = Simulator().run(TsallisINF(3), env, T=100, seed=0)
    used = res.config["corruption_used"]
    assert 0.0 < used <= 20.0
    assert res.config["corruption_budget"] == 20.0


def test_corruption_budget_not_exceeded():
    base = StochasticEnv.from_gap(2, gap=0.3)
    env = CorruptedStochasticEnv(base, budget=5.0)
    Simulator().run(TsallisINF(2), env, T=1000, seed=1)
    assert env.corruption_used <= 5.0


def test_load_balancing_env():
    env = LoadBalancingEnv(3, job_size=uniform_jobs(0.5, 1.0))
    res = Simulator().run(Hedge(3, horizon=200), env, T=200, seed=0)
    meta_loads = np.array(res.config["final_loads"])
    # 損失は [0,1]、負荷は全ジョブの合計に一致するはず(最終ジョブは未反映)
    assert np.all(res.losses >= 0) and np.all(res.losses <= 1)
    assert meta_loads.sum() > 0
    assert res.config["makespan"] == pytest.approx(meta_loads.max())


def test_run_many_and_config_roundtrip(tmp_path):
    out = run_many(
        lambda: Hedge(4, horizon=200),
        lambda: StochasticEnv.from_gap(4, 0.25),
        T=200,
        seeds=[0, 1, 2],
    )
    assert out.aggregate.mean.shape == (200,)
    assert out.aggregate.n_runs == 3
    assert out.pseudo_aggregate is not None
    assert len(out.results) == 3
    assert np.all(out.aggregate.lower <= out.aggregate.upper + 1e-12)

    cfg = ExperimentConfig(
        name="demo", T=200, seeds=[0, 1, 2], learner={"class": "Hedge"}
    )
    path = tmp_path / "cfg.json"
    cfg.save_json(path)
    assert ExperimentConfig.load_json(path) == cfg
