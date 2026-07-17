"""Hedge(指数重み法)。full-info の基準線。

horizon T が既知の場合、理論上界 sqrt(T ln K / 2) を達成する
最適学習率 eta = sqrt(8 ln K / T) を自動設定する。
"""

from __future__ import annotations

import math

import numpy as np

from onlopt.core.learner import Feedback, FeedbackType, Learner
from onlopt.learners.ftrl import sample_from


class Hedge(Learner):
    feedback_type = FeedbackType.FULL_INFO

    def __init__(
        self,
        n_actions: int,
        eta: float | None = None,
        horizon: int | None = None,
    ) -> None:
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        if eta is None:
            if horizon is None:
                raise ValueError("specify either eta or horizon")
            eta = math.sqrt(8.0 * math.log(n_actions) / horizon)
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.n_actions = n_actions
        self.eta = eta
        self.reset()

    def reset(self) -> None:
        self._cum_loss = np.zeros(self.n_actions, dtype=np.float64)
        self._dist: np.ndarray | None = None

    @property
    def distribution(self) -> np.ndarray:
        if self._dist is None:
            logits = -self.eta * self._cum_loss
            w = np.exp(logits - logits.max())
            self._dist = w / w.sum()
        return self._dist

    def predict(self, rng: np.random.Generator) -> int:
        return sample_from(self.distribution, rng)

    def update(self, feedback: Feedback) -> None:
        if feedback.full_loss is None:
            raise ValueError("Hedge requires full-information feedback")
        self._cum_loss += feedback.full_loss
        self._dist = None

    def config(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "n_actions": self.n_actions,
            "eta": self.eta,
        }
