"""onlopt: オンライン最適化研究用モジュール。

Best-of-Both-Worlds(BOBW)アルゴリズム、リグレット最小化、
オンライン線形最適化(OLO)の数値実験基盤。
"""

from onlopt.core import (
    Action,
    Environment,
    Feedback,
    FeedbackType,
    History,
    Learner,
    RunResult,
    Simulator,
)
from onlopt.environments import (
    AdaptiveAdversary,
    CorruptedStochasticEnv,
    LoadBalancingEnv,
    ObliviousAdversary,
    StochasticEnv,
)
from onlopt.evaluation import (
    AggregateResult,
    aggregate_regret,
    loglog_slope,
    plot_regret,
)
from onlopt.experiments import ExperimentConfig, ManyRunResult, run_grid, run_many
from onlopt.learners import (
    FTRL,
    L2,
    OGD,
    OMD,
    DataDependentLR,
    FixedLR,
    Hedge,
    InverseSqrtLR,
    NegativeEntropy,
    TsallisEntropy,
    TsallisINF,
)
from onlopt.utils import make_rng, spawn_rngs

__version__ = "0.1.0"

__all__ = [
    "FTRL",
    "L2",
    "OGD",
    "OMD",
    "Action",
    "AdaptiveAdversary",
    "AggregateResult",
    "CorruptedStochasticEnv",
    "DataDependentLR",
    "Environment",
    "ExperimentConfig",
    "Feedback",
    "FeedbackType",
    "FixedLR",
    "Hedge",
    "History",
    "InverseSqrtLR",
    "Learner",
    "LoadBalancingEnv",
    "ManyRunResult",
    "NegativeEntropy",
    "ObliviousAdversary",
    "RunResult",
    "Simulator",
    "StochasticEnv",
    "TsallisEntropy",
    "TsallisINF",
    "aggregate_regret",
    "loglog_slope",
    "make_rng",
    "plot_regret",
    "run_grid",
    "run_many",
    "spawn_rngs",
]
