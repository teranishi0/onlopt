"""Learner 基底クラスとフィードバック型の定義。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import numpy as np

# v0.1 では行動は腕インデックス(int)または決定集合上の点(np.ndarray)
Action = Union[int, np.ndarray]


class FeedbackType(Enum):
    FULL_INFO = auto()  # 損失ベクトル全体を観測
    BANDIT = auto()  # 選択した行動の損失のみ観測
    SEMI_BANDIT = auto()  # 選択した組合せ的行動の成分ごとの損失を観測


@dataclass
class Feedback:
    """Simulator が Learner に渡す観測情報。

    Attributes:
        action: 学習者が選択した行動。
        loss: 観測された損失。bandit ではスカラー、semi-bandit では
            選択成分のみ非マスクのベクトル、full-info では被った損失スカラー。
        full_loss: 損失ベクトル全体(full-info のときのみ非 None)。
    """

    action: Action
    loss: float | np.ndarray
    full_loss: np.ndarray | None = None


class Learner(ABC):
    """オンライン学習アルゴリズムの基底クラス。

    サブクラスはクラス属性またはインスタンス属性として
    ``feedback_type`` を宣言し、``predict`` / ``update`` を実装する。
    乱数は Simulator から渡された Generator のみを使用すること。
    """

    feedback_type: FeedbackType

    @abstractmethod
    def predict(self, rng: np.random.Generator) -> Action:
        """現ラウンドの行動(または行動上の分布からのサンプル)を返す。"""

    @abstractmethod
    def update(self, feedback: Feedback) -> None:
        """観測したフィードバックで内部状態を更新する。"""

    def reset(self) -> None:
        """内部状態を初期化する(複数シード実行用)。"""

    @property
    def distribution(self) -> np.ndarray | None:
        """現在の行動分布。分布を明示的に持たない学習者は None。

        擬似リグレット計算に使用される。
        """
        return None

    def config(self) -> dict[str, object]:
        """実験設定スナップショット用の自己記述。"""
        return {"class": type(self).__name__}
