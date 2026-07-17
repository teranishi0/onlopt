"""理論整合性テスト(仕様 8.2)とレジーム挙動テスト(仕様 8.3)。"""

import math

import numpy as np
import pytest

from onlopt import (
    FTRL,
    OGD,
    OMD,
    FeedbackType,
    Hedge,
    InverseSqrtLR,
    NegativeEntropy,
    ObliviousAdversary,
    Simulator,
    StochasticEnv,
    TsallisINF,
    loglog_slope,
)
from onlopt.experiments import run_many


class TestHedgeTheory:
    def test_regret_below_theoretical_bound(self):
        """eta 最適時の Hedge のリグレットが sqrt(T ln K / 2) を下回ること
        (複数シード平均、T = 10^4)。"""
        T, K = 10_000, 10
        bound = math.sqrt(T * math.log(K) / 2.0)
        out = run_many(
            lambda: Hedge(K, horizon=T),
            lambda: StochasticEnv.from_gap(K, gap=0.2),
            T=T,
            seeds=list(range(10)),
        )
        assert out.aggregate.mean[-1] < bound

    def test_regret_below_bound_adversarial(self):
        """oblivious な敵対的損失でも上界を下回ること。"""
        T, K = 10_000, 5
        bound = math.sqrt(T * math.log(K) / 2.0)

        def loss_fn(t, rng):
            return rng.random(K)

        out = run_many(
            lambda: Hedge(K, horizon=T),
            lambda: ObliviousAdversary(K, loss_fn),
            T=T,
            seeds=list(range(10)),
        )
        assert out.aggregate.mean[-1] < bound


class TestFTRLEquivalence:
    def test_ftrl_negentropy_equals_hedge(self):
        """FTRL + 負エントロピー + 固定学習率は Hedge と同一の分布を生む。"""
        from onlopt.core import Feedback
        from onlopt.learners import FixedLR

        K, eta = 5, 0.1
        ftrl = FTRL(K, NegativeEntropy(), FixedLR(eta))
        hedge = Hedge(K, eta=eta)
        rng = np.random.default_rng(0)
        for _ in range(50):
            loss = rng.random(K)
            assert np.allclose(ftrl.distribution, hedge.distribution, atol=1e-10)
            fb = Feedback(action=0, loss=float(loss[0]), full_loss=loss)
            ftrl.update(fb)
            hedge.update(fb)

    def test_omd_negentropy_equals_hedge(self):
        """OMD + 負エントロピー(固定学習率)= 乗算型重み更新 = Hedge。"""
        from onlopt.core import Feedback
        from onlopt.learners import FixedLR

        K, eta = 4, 0.2
        omd = OMD(K, NegativeEntropy(), FixedLR(eta))
        hedge = Hedge(K, eta=eta)
        rng = np.random.default_rng(1)
        for _ in range(50):
            loss = rng.random(K)
            assert np.allclose(omd.distribution, hedge.distribution, atol=1e-8)
            fb = Feedback(action=0, loss=float(loss[0]), full_loss=loss)
            omd.update(fb)
            hedge.update(fb)


class TestOGD:
    def test_ogd_sublinear_regret_on_simplex(self):
        T, K = 5000, 5
        sim = Simulator()
        res = sim.run(
            OGD(K, InverseSqrtLR(c=0.5)),
            StochasticEnv.from_gap(K, gap=0.3),
            T=T,
            seed=0,
        )
        # 線形リグレット(=学習していない)でないこと
        assert res.cum_regret[-1] < 0.05 * T
        assert res.cum_pseudo_regret is not None


@pytest.mark.slow
class TestRegimeBehavior:
    """Tsallis-INF の BOBW 挙動: 確率的で log T、敵対的で sqrt T スケール。"""

    def test_stochastic_regime_slope(self):
        """確率的環境で擬似リグレットの log-log 傾きが 0.5 より十分小さいこと。"""
        T = 30_000
        out = run_many(
            lambda: TsallisINF(4),
            lambda: StochasticEnv.from_gap(4, gap=0.25),
            T=T,
            seeds=list(range(8)),
        )
        assert out.pseudo_aggregate is not None
        slope = loglog_slope(out.pseudo_aggregate.mean, t_start=T // 4)
        # log T スケールなら傾き ~ 1/log(T) << 0.5(許容誤差付き)
        assert slope < 0.35, f"stochastic slope too steep: {slope:.3f}"

    def test_adversarial_regime_slope(self):
        """一様ランダム損失(実現ベスト腕比較)ではリグレットが
        sqrt(T) スケールになること。"""
        T = 30_000
        K = 4

        def loss_fn(t, rng):
            return rng.random(K)

        out = run_many(
            lambda: TsallisINF(K),
            lambda: ObliviousAdversary(K, loss_fn),
            T=T,
            seeds=list(range(8)),
        )
        slope = loglog_slope(out.aggregate.mean, t_start=T // 4)
        assert 0.3 < slope < 0.7, f"adversarial slope off: {slope:.3f}"
