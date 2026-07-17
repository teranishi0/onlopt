"""OMD(Online Mirror Descent)汎用フレームワーク。

FTRL と鏡像(Regularizer)を共有する。更新は

    theta_t = grad R(x_t) - eta_t * hat{l}_t
    x_{t+1} = argmin_{x in simplex} R(x) - <theta_t, x>

で、第2式は Bregman 射影に一致する。
"""

from __future__ import annotations

import numpy as np

from onlopt.core.learner import Feedback, FeedbackType, Learner
from onlopt.learners.estimators import (
    IdentityEstimator,
    ImportanceWeighted,
    LossEstimator,
)
from onlopt.learners.ftrl import sample_from
from onlopt.learners.lr_schedules import LRSchedule
from onlopt.learners.regularizers import Regularizer


class OMD(Learner):
    """確率単体上の OMD。full-info / bandit の両フィードバックに対応。"""

    def __init__(
        self,
        n_actions: int,
        regularizer: Regularizer,
        lr_schedule: LRSchedule,
        loss_estimator: LossEstimator | None = None,
        feedback_type: FeedbackType = FeedbackType.FULL_INFO,
        min_prob: float = 1e-12,
    ) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        self.n_actions = n_actions
        self.regularizer = regularizer
        self.lr_schedule = lr_schedule
        self.feedback_type = feedback_type
        self.min_prob = min_prob
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
        self._x = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)
        self._state: dict[str, float] = {"cum_sq_est_norm": 0.0, "cum_est_sq": 0.0}

    @property
    def distribution(self) -> np.ndarray:
        return self._x

    def predict(self, rng: np.random.Generator) -> int:
        return sample_from(self._x, rng)

    def update(self, feedback: Feedback) -> None:
        est = self.loss_estimator.estimate(feedback, self._x, self.n_actions)
        eta = self.lr_schedule.eta(self._t, self._state)
        # grad R は単体の内部でのみ定義されるため、数値的な 0 を回避する
        x_safe = np.maximum(self._x, self.min_prob)
        theta = self.regularizer.grad(x_safe) - eta * est
        self._x = self.regularizer.argmin_linear(theta)
        sq = float(est @ est)
        self._state["cum_sq_est_norm"] += sq
        self._state["cum_est_sq"] += sq
        self._t += 1

    def config(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "n_actions": self.n_actions,
            "regularizer": self.regularizer.config(),
            "lr_schedule": self.lr_schedule.config(),
            "loss_estimator": self.loss_estimator.config(),
            "feedback_type": self.feedback_type.name,
        }
