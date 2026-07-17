"""FTRL(Follow-the-Regularized-Leader)汎用フレームワーク。

FTRL = Regularizer + LRSchedule + LossEstimator の合成:

    x_t = argmin_{x in simplex} <hat{L}_{t-1}, x> + R(x) / eta_t

Tsallis-INF はこの枠組みで
「Tsallis(alpha=1/2) + 適応学習率 + importance weighting」
の組み合わせとして表現される(tsallis_inf.py 参照)。
"""

from __future__ import annotations

import numpy as np

from onlopt.core.learner import Feedback, FeedbackType, Learner
from onlopt.learners.estimators import (
    IdentityEstimator,
    ImportanceWeighted,
    LossEstimator,
)
from onlopt.learners.lr_schedules import LRSchedule
from onlopt.learners.regularizers import Regularizer


def sample_from(dist: np.ndarray, rng: np.random.Generator) -> int:
    """分布 dist から腕インデックスをサンプルする(cumsum + 二分探索)。"""
    u = rng.random()
    return int(np.searchsorted(np.cumsum(dist), u * dist.sum()))


class FTRL(Learner):
    """確率単体上の FTRL。full-info / bandit の両フィードバックに対応。"""

    def __init__(
        self,
        n_actions: int,
        regularizer: Regularizer,
        lr_schedule: LRSchedule,
        loss_estimator: LossEstimator | None = None,
        feedback_type: FeedbackType = FeedbackType.FULL_INFO,
    ) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        self.n_actions = n_actions
        self.regularizer = regularizer
        self.lr_schedule = lr_schedule
        self.feedback_type = feedback_type
        if loss_estimator is None:
            loss_estimator = (
                IdentityEstimator()
                if feedback_type is FeedbackType.FULL_INFO
                else ImportanceWeighted()
            )
        self.loss_estimator = loss_estimator
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._cum_est = np.zeros(self.n_actions, dtype=np.float64)
        self._state: dict[str, float] = {"cum_sq_est_norm": 0.0, "cum_est_sq": 0.0}
        self._dist: np.ndarray | None = None

    @property
    def distribution(self) -> np.ndarray:
        if self._dist is None:
            self._dist = self._compute_distribution()
        return self._dist

    def _compute_distribution(self) -> np.ndarray:
        eta = self.lr_schedule.eta(self._t, self._state)
        return self.regularizer.argmin(self._cum_est, eta)

    def predict(self, rng: np.random.Generator) -> int:
        return sample_from(self.distribution, rng)

    def update(self, feedback: Feedback) -> None:
        est = self.loss_estimator.estimate(
            feedback, self.distribution, self.n_actions
        )
        self._cum_est += est
        sq = float(est @ est)
        self._state["cum_sq_est_norm"] += sq
        self._state["cum_est_sq"] += sq
        self._t += 1
        self._dist = None  # 次ラウンドで再計算

    def config(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "n_actions": self.n_actions,
            "regularizer": self.regularizer.config(),
            "lr_schedule": self.lr_schedule.config(),
            "loss_estimator": self.loss_estimator.config(),
            "feedback_type": self.feedback_type.name,
        }
