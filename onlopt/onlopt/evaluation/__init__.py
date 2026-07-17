from onlopt.evaluation.aggregate import (
    AggregateResult,
    aggregate_curves,
    aggregate_regret,
)
from onlopt.evaluation.plotting import plot_regret
from onlopt.evaluation.regret import (
    cumulative_pseudo_regret,
    cumulative_regret,
    loglog_slope,
)

__all__ = [
    "AggregateResult",
    "aggregate_curves",
    "aggregate_regret",
    "cumulative_pseudo_regret",
    "cumulative_regret",
    "loglog_slope",
    "plot_regret",
]
