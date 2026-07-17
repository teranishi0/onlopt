"""実験ループ(Simulator)と実行結果(RunResult)。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from onlopt.core.environment import Environment, History
from onlopt.core.learner import Action, Feedback, FeedbackType, Learner
from onlopt.utils.rng import make_rng


@dataclass
class RunResult:
    """1シード分の実行結果。

    Attributes:
        actions: 各ラウンドの選択行動。
        losses: 各ラウンドの被った損失 (T,)。
        cum_regret: 累積リグレット系列 (T,)。実現損失の累積から、
            その時点までの事後最良固定行動の累積損失を引いたもの。
        cum_pseudo_regret: 擬似リグレット系列 (T,)。環境が期待損失を
            提供する場合のみ計算され、それ以外は None。
        config: 実験設定のスナップショット。
        seed: 使用シード。
    """

    actions: list[Action]
    losses: np.ndarray
    cum_regret: np.ndarray
    cum_pseudo_regret: np.ndarray | None
    config: dict[str, object] = field(default_factory=dict)
    seed: int = 0


class Simulator:
    """Learner と Environment を突き合わせて T ラウンド実行する。"""

    def run(
        self,
        learner: Learner,
        env: Environment,
        T: int,
        seed: int,
        record_actions: bool = True,
    ) -> RunResult:
        rng = make_rng(seed)
        learner.reset()
        env.reset()

        n = env.n_actions
        history = History()
        actions: list[Action] = []
        incurred = np.empty(T, dtype=np.float64)
        cum_regret = np.empty(T, dtype=np.float64)

        mu = env.mean_loss
        pseudo: np.ndarray | None = None
        mu_star = 0.0
        if mu is not None:
            mu = np.asarray(mu, dtype=np.float64)
            mu_star = float(mu.min())
            pseudo = np.empty(T, dtype=np.float64)

        cum_arm_loss = np.zeros(n, dtype=np.float64)  # 座標ごとの累積損失
        cum_incurred = 0.0
        cum_pseudo = 0.0
        warned = False

        for t in range(T):
            action = learner.predict(rng)
            loss_vec = env.get_loss(t, history, rng)

            if not warned and (loss_vec.min() < 0.0 or loss_vec.max() > 1.0):
                warnings.warn(
                    f"round {t}: loss outside [0, 1] "
                    f"(min={loss_vec.min():.4g}, max={loss_vec.max():.4g}). "
                    "losses should be normalized to [0, 1].",
                    stacklevel=2,
                )
                warned = True

            if isinstance(action, np.ndarray):
                loss_t = float(loss_vec @ action)
            else:
                loss_t = float(loss_vec[action])

            feedback = self._make_feedback(
                learner.feedback_type, action, loss_t, loss_vec
            )

            if pseudo is not None:
                assert mu is not None
                dist = learner.distribution
                if dist is not None:
                    exp_loss = float(dist @ mu)
                elif isinstance(action, np.ndarray):
                    exp_loss = float(action @ mu)
                else:
                    exp_loss = float(mu[action])
                cum_pseudo += exp_loss - mu_star
                pseudo[t] = cum_pseudo

            learner.update(feedback)
            history.append(action, loss_t)

            incurred[t] = loss_t
            cum_incurred += loss_t
            cum_arm_loss += loss_vec
            cum_regret[t] = cum_incurred - float(cum_arm_loss.min())
            if record_actions:
                actions.append(action)

        config: dict[str, object] = {
            "learner": learner.config(),
            "environment": env.config(),
            "T": T,
            "seed": seed,
        }
        config.update(env.run_metadata())

        return RunResult(
            actions=actions,
            losses=incurred,
            cum_regret=cum_regret,
            cum_pseudo_regret=pseudo,
            config=config,
            seed=seed,
        )

    @staticmethod
    def _make_feedback(
        feedback_type: FeedbackType,
        action: Action,
        loss_t: float,
        loss_vec: np.ndarray,
    ) -> Feedback:
        """Environment の損失ベクトルを Learner の要求する型にマスクする。"""
        if feedback_type is FeedbackType.FULL_INFO:
            return Feedback(action=action, loss=loss_t, full_loss=loss_vec)
        if feedback_type is FeedbackType.BANDIT:
            return Feedback(action=action, loss=loss_t, full_loss=None)
        # SEMI_BANDIT: 選択した組合せ的行動(0/1 マスク)の成分のみ観測
        if not isinstance(action, np.ndarray):
            raise TypeError("semi-bandit feedback requires an ndarray action mask")
        return Feedback(action=action, loss=loss_vec * action, full_loss=None)
