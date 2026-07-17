"""バンディットフィードバックからの損失推定。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from onlopt.core.learner import Feedback


class LossEstimator(ABC):
    """観測フィードバックから損失ベクトルの推定量を構成する。"""

    @abstractmethod
    def estimate(
        self, feedback: Feedback, distribution: np.ndarray, n_actions: int
    ) -> np.ndarray:
        """損失ベクトルの推定値 (n_actions,) を返す。"""

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__}


class IdentityEstimator(LossEstimator):
    """full-info 用の恒等写像。"""

    def estimate(
        self, feedback: Feedback, distribution: np.ndarray, n_actions: int
    ) -> np.ndarray:
        if feedback.full_loss is None:
            raise ValueError("IdentityEstimator requires full-information feedback")
        return np.asarray(feedback.full_loss, dtype=np.float64)


class ImportanceWeighted(LossEstimator):
    """importance weighting による不偏推定 hat{l}_i = (l_a / p_a) 1{i = a}。"""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def estimate(
        self, feedback: Feedback, distribution: np.ndarray, n_actions: int
    ) -> np.ndarray:
        a = feedback.action
        if not isinstance(a, (int, np.integer)):
            raise TypeError("ImportanceWeighted requires an arm-index action")
        est = np.zeros(n_actions, dtype=np.float64)
        est[a] = float(feedback.loss) / max(float(distribution[a]), self.eps)
        return est
