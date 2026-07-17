"""OLO 用射影勾配型アルゴリズム(OGD)。負荷分散応用の足がかり。

線形損失 <l_t, x> に対する射影付き勾配降下:

    x_{t+1} = Proj(x_t - eta_t * l_t)

決定集合は射影 callable で指定する(既定は確率単体)。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from onlopt.core.learner import Feedback, FeedbackType, Learner
from onlopt.geometry.projections import project_simplex
from onlopt.learners.lr_schedules import LRSchedule


class OGD(Learner):
    feedback_type = FeedbackType.FULL_INFO

    def __init__(
        self,
        dim: int,
        lr_schedule: LRSchedule,
        projection: Callable[[np.ndarray], np.ndarray] = project_simplex,
        x0: np.ndarray | None = None,
    ) -> None:
        self.dim = dim
        self.lr_schedule = lr_schedule
        self.projection = projection
        self._x0 = (
            np.asarray(x0, dtype=np.float64)
            if x0 is not None
            else self.projection(np.zeros(dim, dtype=np.float64))
        )
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._x = self._x0.copy()
        self._state: dict[str, float] = {"cum_sq_est_norm": 0.0}

    def predict(self, rng: np.random.Generator) -> np.ndarray:
        return self._x

    def update(self, feedback: Feedback) -> None:
        if feedback.full_loss is None:
            raise ValueError("OGD requires full-information feedback")
        g = np.asarray(feedback.full_loss, dtype=np.float64)
        eta = self.lr_schedule.eta(self._t, self._state)
        self._x = self.projection(self._x - eta * g)
        self._state["cum_sq_est_norm"] += float(g @ g)
        self._t += 1

    def config(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "dim": self.dim,
            "lr_schedule": self.lr_schedule.config(),
        }
