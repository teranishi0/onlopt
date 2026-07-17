"""複数シード・複数設定の一括実行。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from onlopt.core.environment import Environment
from onlopt.core.learner import Learner
from onlopt.core.simulator import RunResult, Simulator
from onlopt.evaluation.aggregate import (
    AggMethod,
    AggregateResult,
    aggregate_regret,
)

LearnerFactory = Callable[[], Learner]
EnvFactory = Callable[[], Environment]


@dataclass
class ManyRunResult:
    """複数シード実行の集計結果と生データ。"""

    aggregate: AggregateResult  # ラウンドごとの平均リグレットと帯
    pseudo_aggregate: AggregateResult | None  # 擬似リグレット(計算可能な場合)
    results: list[RunResult]  # 全 RunResult(生データ保持)


def run_many(
    learner_factory: LearnerFactory,
    env_factory: EnvFactory,
    T: int,
    seeds: Sequence[int],
    aggregation: AggMethod = "std",
    confidence: float = 0.95,
    record_actions: bool = False,
) -> ManyRunResult:
    """複数シードで実験を実行し、集計結果と全 RunResult を返す。

    シードごとに factory から新しい Learner / Environment を生成する。
    """
    sim = Simulator()
    results = [
        sim.run(
            learner_factory(),
            env_factory(),
            T,
            seed,
            record_actions=record_actions,
        )
        for seed in seeds
    ]
    agg = aggregate_regret(
        results, key="cum_regret", method=aggregation, confidence=confidence
    )
    pseudo: AggregateResult | None = None
    if all(r.cum_pseudo_regret is not None for r in results):
        pseudo = aggregate_regret(
            results,
            key="cum_pseudo_regret",
            method=aggregation,
            confidence=confidence,
        )
    return ManyRunResult(aggregate=agg, pseudo_aggregate=pseudo, results=results)


def run_grid(
    settings: Mapping[str, tuple[LearnerFactory, EnvFactory]],
    T: int,
    seeds: Sequence[int],
    aggregation: AggMethod = "std",
    confidence: float = 0.95,
) -> dict[str, ManyRunResult]:
    """名前付きの (learner_factory, env_factory) 群を一括実行する。

    戻り値はプロット関数にそのまま渡せる名前付き辞書。
    """
    return {
        name: run_many(
            lf, ef, T, seeds, aggregation=aggregation, confidence=confidence
        )
        for name, (lf, ef) in settings.items()
    }
