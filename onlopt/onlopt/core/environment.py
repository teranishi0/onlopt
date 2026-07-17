"""Environment 基底クラスと履歴型の定義。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from onlopt.core.learner import Action


@dataclass
class History:
    """学習者の過去の行動列と被った損失列(適応的敵対者が参照する)。"""

    actions: list[Action] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def append(self, action: Action, loss: float) -> None:
        self.actions.append(action)
        self.losses.append(loss)

    def __len__(self) -> int:
        return len(self.actions)


class Environment(ABC):
    """損失系列を生成する環境の基底クラス。

    損失は [0, 1] に正規化することを規約とする(範囲外の場合は
    Simulator が警告を出す)。

    注記: 仕様書 3.3 の ``get_loss(t, history)`` に対し、再現性規約
    (6.4: 環境は乱数を自前で保持しない)を満たすため、本実装では
    Simulator から渡される ``rng`` を第3引数に取る。
    """

    n_actions: int

    @abstractmethod
    def get_loss(
        self, t: int, history: History | None, rng: np.random.Generator
    ) -> np.ndarray:
        """ラウンド t の損失ベクトルを返す。

        適応的敵対者は history(学習者の過去の行動列)を参照してよい。
        非適応的環境は history を無視する。
        """

    def reset(self) -> None:
        """内部状態を初期化する(複数シード実行用)。"""

    @property
    def mean_loss(self) -> np.ndarray | None:
        """期待損失ベクトル。確率的環境のみ提供し、擬似リグレット計算に使う。"""
        return None

    def best_fixed_action(self, T: int) -> tuple[Action, float] | None:
        """T ラウンドでの事後最良固定行動とその累積損失を返す。

        None を返した場合、Simulator が実現損失ベクトルから
        座標ごとの最小累積損失として計算する(単体型の決定集合で有効)。
        """
        return None

    def run_metadata(self) -> dict[str, object]:
        """実行後に RunResult.config へ併合される環境側の記録
        (例: 実際に消費した汚染量)。"""
        return {}

    def config(self) -> dict[str, object]:
        """実験設定スナップショット用の自己記述。"""
        return {"class": type(self).__name__}
