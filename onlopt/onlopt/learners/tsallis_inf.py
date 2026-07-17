"""Tsallis-INF(Zimmert & Seldin, 2021)。

FTRL + Tsallis エントロピー(alpha=1/2)+ 適応学習率 + IW 推定の
組み合わせとして定義する(仕様 4.2 の受け入れ条件)。
確率的・敵対的の両レジームで最適な Best-of-Both-Worlds アルゴリズム。
"""

from __future__ import annotations

from onlopt.core.learner import FeedbackType
from onlopt.learners.estimators import ImportanceWeighted
from onlopt.learners.ftrl import FTRL
from onlopt.learners.lr_schedules import InverseSqrtLR, LRSchedule
from onlopt.learners.regularizers import TsallisEntropy


class TsallisINF(FTRL):
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.5,
        c: float = 2.0,
        lr_schedule: LRSchedule | None = None,
    ) -> None:
        super().__init__(
            n_actions=n_actions,
            regularizer=TsallisEntropy(alpha),
            lr_schedule=lr_schedule if lr_schedule is not None else InverseSqrtLR(c),
            loss_estimator=ImportanceWeighted(),
            feedback_type=FeedbackType.BANDIT,
        )
